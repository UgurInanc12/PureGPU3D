"""Acceptance tests for desktop batch_size selection, validation, protocol, and worker execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance() or QApplication(["-platform", "offscreen"])

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.desktop.controller import DesktopController, DesktopState
from puregpu3d.desktop.window import MainWindow
from puregpu3d.runtime.protocol import (
    MessageType,
    PipelineRoute,
    Stage,
    WorkerCommand,
)

FIXTURE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"
CHECKPOINT_DIR_SMALL = REPO_ROOT / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"
WORKER_SCRIPT = REPO_ROOT / "src" / "puregpu3d" / "runtime" / "worker.py"


class TestBatchProtocol(unittest.TestCase):
    """Test WorkerCommand batch_size field, legacy missing-key default, and strict validation."""

    def test_default_batch_size_is_1(self) -> None:
        cmd = WorkerCommand(
            job_id="test_default",
            input_path="in.mp4",
            output_path="out.mp4",
        )
        self.assertEqual(cmd.batch_size, 1)

    def test_explicit_valid_batch_size(self) -> None:
        for b in (1, 2, 5, 20):
            cmd = WorkerCommand(
                job_id=f"test_{b}",
                input_path="in.mp4",
                output_path="out.mp4",
                batch_size=b,
            )
            self.assertEqual(cmd.batch_size, b)

    def test_roundtrip_json_serialization(self) -> None:
        cmd = WorkerCommand(
            job_id="test_roundtrip",
            input_path="in.mp4",
            output_path="out.mp4",
            batch_size=5,
        )
        data = json.loads(cmd.to_json())
        self.assertIn("batch_size", data)
        self.assertEqual(data["batch_size"], 5)

        restored = WorkerCommand.from_json(cmd.to_json())
        self.assertEqual(restored.batch_size, 5)

    def test_legacy_missing_key_defaults_to_1(self) -> None:
        data = {
            "job_id": "legacy_job",
            "input_path": "in.mp4",
            "output_path": "out.mp4",
        }
        cmd = WorkerCommand.from_dict(data)
        self.assertEqual(cmd.batch_size, 1)

    def test_strict_type_validation_rejects_bool(self) -> None:
        # bool is subclass of int in Python, so isinstance(True, int) is True
        # but WorkerCommand must strictly reject bool
        with self.assertRaises(TypeError):
            WorkerCommand(
                job_id="test_bool",
                input_path="in.mp4",
                output_path="out.mp4",
                batch_size=True,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            WorkerCommand.from_dict({
                "job_id": "test_bool",
                "input_path": "in.mp4",
                "output_path": "out.mp4",
                "batch_size": False,
            })

    def test_strict_type_validation_rejects_float_and_str(self) -> None:
        for invalid in (1.5, "5", None, [2], {"b": 1}):
            with self.assertRaises(TypeError):
                WorkerCommand(
                    job_id="test_inv",
                    input_path="in.mp4",
                    output_path="out.mp4",
                    batch_size=invalid,  # type: ignore[arg-type]
                )
            with self.assertRaises(TypeError):
                WorkerCommand.from_dict({
                    "job_id": "test_inv",
                    "input_path": "in.mp4",
                    "output_path": "out.mp4",
                    "batch_size": invalid,
                })

    def test_range_validation_rejects_out_of_bounds(self) -> None:
        for invalid in (0, -1, -5, 21, 100):
            with self.assertRaises(ValueError):
                WorkerCommand(
                    job_id="test_range",
                    input_path="in.mp4",
                    output_path="out.mp4",
                    batch_size=invalid,
                )
            with self.assertRaises(ValueError):
                WorkerCommand.from_dict({
                    "job_id": "test_range",
                    "input_path": "in.mp4",
                    "output_path": "out.mp4",
                    "batch_size": invalid,
                })


class TestBatchController(unittest.TestCase):
    """Test DesktopController batch_size property, validation setter, and job serialization."""

    def setUp(self) -> None:
        self.controller = DesktopController()

    def test_controller_default_batch_size_is_1(self) -> None:
        self.assertEqual(self.controller.batch_size, 1)

    def test_controller_set_batch_size_valid(self) -> None:
        signals = []
        self.controller.batch_size_changed.connect(lambda b: signals.append(b))

        self.controller.set_batch_size(5)
        self.assertEqual(self.controller.batch_size, 5)
        self.assertEqual(signals, [5])

        self.controller.set_batch_size(20)
        self.assertEqual(self.controller.batch_size, 20)
        self.assertEqual(signals, [5, 20])

    def test_controller_setter_rejects_bool(self) -> None:
        with self.assertRaises(TypeError):
            self.controller.set_batch_size(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.controller.set_batch_size(False)  # type: ignore[arg-type]
        self.assertEqual(self.controller.batch_size, 1)

    def test_controller_setter_rejects_float_str_none(self) -> None:
        for invalid in (2.5, "4", None, [3]):
            with self.assertRaises(TypeError):
                self.controller.set_batch_size(invalid)  # type: ignore[arg-type]
        self.assertEqual(self.controller.batch_size, 1)

    def test_controller_setter_rejects_out_of_range(self) -> None:
        for invalid in (0, -1, 21, 50):
            with self.assertRaises(ValueError):
                self.controller.set_batch_size(invalid)
        self.assertEqual(self.controller.batch_size, 1)

    def test_controller_serializes_batch_size_to_job(self) -> None:
        if not FIXTURE_VIDEO.is_file():
            self.skipTest(f"Fixture video not found at {FIXTURE_VIDEO}")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "out.mp4"
            self.controller.set_input_path(FIXTURE_VIDEO)
            self.controller.set_output_path(out_file)
            self.controller.set_selected_model("DA3-SMALL")
            self.controller.set_batch_size(4)

            # Start conversion mock to inspect command written to disk
            with patch.object(self.controller, "_get_worker_launch_spec", return_value=(sys.executable, ["-c", "import sys; sys.exit(0)"])):
                started, reason = self.controller.start_conversion()
                self.assertTrue(started, f"Start conversion failed: {reason}")
                cmd_file = self.controller._command_file
                self.assertIsNotNone(cmd_file)
                assert cmd_file is not None
                self.assertTrue(cmd_file.exists())
                cmd_data = json.loads(cmd_file.read_text(encoding="utf-8"))
                self.assertEqual(cmd_data.get("batch_size"), 4)


class TestBatchWindowUI(unittest.TestCase):
    """Test MainWindow UI controls for batch_size: spinbox, range, label, caveat, and running state."""

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_spinbox_initial_state_and_range(self) -> None:
        self.assertTrue(hasattr(self.window, "batch_spin"))
        self.assertEqual(self.window.batch_spin.minimum(), 1)
        self.assertEqual(self.window.batch_spin.maximum(), 20)
        self.assertEqual(self.window.batch_spin.value(), 1)
        self.assertEqual(self.window.controller.batch_size, 1)

    def test_spinbox_syncs_with_controller(self) -> None:
        self.window.batch_spin.setValue(8)
        self.assertEqual(self.window.controller.batch_size, 8)

        self.window.controller.set_batch_size(3)
        self.assertEqual(self.window.batch_spin.value(), 3)

    def test_batch_control_disabled_while_running(self) -> None:
        self.assertTrue(self.window.batch_spin.isEnabled())
        self.window._on_state_changed(DesktopState.CONVERTING.value)
        self.assertFalse(self.window.batch_spin.isEnabled())
        self.window._on_state_changed(DesktopState.COMPLETED.value)
        self.assertTrue(self.window.batch_spin.isEnabled())

    def test_batch_label_and_vram_caveat(self) -> None:
        # Check that UI explains GPU route and VRAM caveat
        self.assertTrue(hasattr(self.window, "batch_hint_label"))
        hint_text = self.window.batch_hint_label.text().lower()
        self.assertIn("vram", hint_text)
        self.assertIn("gpu", hint_text)


class TestWorkerBatchExecution(unittest.TestCase):
    """Test worker handling of batch_size, explicit rejection of Compatible route with batch>1, and Auto fallback."""

    def test_worker_rejects_explicit_compatible_route_with_batch_gt_1(self) -> None:
        """Worker must reject upfront if pipeline_route is compatible and batch_size > 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "cmd.json"
            out_file = Path(tmpdir) / "out.mp4"

            payload = {
                "job_id": "test_compat_batch",
                "input_path": str(FIXTURE_VIDEO),
                "output_path": str(out_file),
                "model_id": "DA3-SMALL",
                "pipeline_route": PipelineRoute.COMPATIBLE,
                "batch_size": 4,
                "overwrite": True,
            }
            cmd_file.write_text(json.dumps(payload), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-B", str(WORKER_SCRIPT), "--command-file", str(cmd_file)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            parsed = [json.loads(line) for line in stdout_lines]
            types = [m["type"] for m in parsed]
            self.assertIn("error", types)

            err_msg = next(m for m in parsed if m["type"] == "error")
            err_text = err_msg["error"].lower()
            # Must mention incompatible / unsupported and actionable advice (batch 1 or GPU route)
            self.assertTrue(
                "gpu" in err_text or "batch size 1" in err_text or "batch_size 1" in err_text,
                f"Error should guide user to GPU route or batch 1: {err_msg['error']}",
            )

    def test_worker_rejects_auto_fallback_with_batch_gt_1(self) -> None:
        """Worker in AUTO mode must reject if GPU unsupported and batch_size > 1 instead of silently processing 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "cmd.json"
            out_file = Path(tmpdir) / "out.mp4"

            payload = {
                "job_id": "test_auto_fallback_batch",
                "input_path": str(FIXTURE_VIDEO),
                "output_path": str(out_file),
                "model_id": "DA3-SMALL",
                "pipeline_route": PipelineRoute.AUTO,
                "batch_size": 3,
                "overwrite": True,
            }
            cmd_file.write_text(json.dumps(payload), encoding="utf-8")

            # Run worker with environment simulating unsupported GPU
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""  # No CUDA devices available

            proc = subprocess.run(
                [sys.executable, "-B", str(WORKER_SCRIPT), "--command-file", str(cmd_file)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertNotEqual(proc.returncode, 0)
            stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            parsed = [json.loads(line) for line in stdout_lines]
            types = [m["type"] for m in parsed]
            self.assertIn("error", types)

            err_msg = next(m for m in parsed if m["type"] == "error")
            self.assertTrue(
                "batch" in err_msg["error"].lower() or "compatible" in err_msg["error"].lower(),
                f"Expected error describing batch incompatibility on fallback: {err_msg['error']}",
            )


class TestRealShortGpuBatch5Acceptance(unittest.TestCase):
    """Real acceptance test: execute short conversion with batch_size=5 on GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from puregpu3d.video.gpu_convert import check_gpu_pipeline_support
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        supported, reason = check_gpu_pipeline_support()
        if not supported:
            raise unittest.SkipTest(f"GPU pipeline unsupported: {reason}")
        if not (CHECKPOINT_DIR_SMALL / "READY").exists():
            raise unittest.SkipTest(f"DA3 Small checkpoint not ready at {CHECKPOINT_DIR_SMALL}")
        if not FIXTURE_VIDEO.is_file():
            raise unittest.SkipTest(f"Fixture video not found at {FIXTURE_VIDEO}")

    def test_real_gpu_batch_5_via_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "cmd.json"
            out_file = Path(tmpdir) / "out_b5.mp4"

            payload = {
                "job_id": "test_real_b5",
                "input_path": str(FIXTURE_VIDEO),
                "output_path": str(out_file),
                "model_id": "DA3-SMALL",
                "pipeline_route": PipelineRoute.GPU,
                "batch_size": 5,
                "overwrite": True,
            }
            cmd_file.write_text(json.dumps(payload), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-B", str(WORKER_SCRIPT), "--command-file", str(cmd_file)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, f"Worker failed: {proc.stderr}\n{proc.stdout}")
            stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            parsed = [json.loads(line) for line in stdout_lines]
            completed_events = [m for m in parsed if m["type"] == MessageType.COMPLETED]
            self.assertEqual(len(completed_events), 1)

            res = completed_events[0]["result"]
            self.assertEqual(res.get("batch_size"), 5)
            self.assertIn("peak_memory_mb", res)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
