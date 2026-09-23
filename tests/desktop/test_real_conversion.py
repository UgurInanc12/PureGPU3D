"""Real headless acceptance test for PureGPU3D Desktop UI controller and worker.

Executes actual Depth Anything 3 Small model conversion on the short
data/verification/video/synthetic_1080p_moving.mp4 fixture into unique temporary
outputs, verifying exact 3840x1080 Full-SBS dimensions, AAC audio stream preservation,
and controller-to-worker JSON-lines execution.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

app = QApplication.instance() or QApplication(["-platform", "offscreen"])

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from puregpu3d.desktop.controller import DesktopController, DesktopState
from puregpu3d.desktop.window import MainWindow
from puregpu3d.models.catalog import load_catalog
from puregpu3d.video.probe import probe_video

FIXTURE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"
CHECKPOINT_DIR = REPO_ROOT / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"
BASE_CHECKPOINT_DIR = REPO_ROOT / "models" / "DA3-BASE" / "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"


class TestDesktopRealConversion(unittest.TestCase):
    """Offscreen acceptance testing of desktop controller and worker on real 1080p video."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available; real GPU desktop conversion test skipped.")
        if not (CHECKPOINT_DIR / "READY").exists():
            raise unittest.SkipTest(f"DA3 Small checkpoint not ready at {CHECKPOINT_DIR}")
        if not FIXTURE_VIDEO.is_file():
            raise unittest.SkipTest(f"Fixture video not found at {FIXTURE_VIDEO}")

    def test_real_controller_worker_1080p_conversion(self) -> None:
        """Execute real end-to-end conversion via DesktopController and worker child QProcess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "desktop_real_3840x1080_sbs.mp4"

            controller = DesktopController()

            # Set input video
            success, msg = controller.set_input_path(FIXTURE_VIDEO)
            self.assertTrue(success, f"Failed to set input: {msg}")
            self.assertIsNotNone(controller.input_probe)
            assert controller.input_probe is not None
            self.assertEqual(controller.input_probe.width, 1920)
            self.assertEqual(controller.input_probe.height, 1080)
            self.assertTrue(controller.input_probe.has_audio)

            # Set unique output path
            success_out, msg_out = controller.set_output_path(out_file)
            self.assertTrue(success_out, f"Failed to set output: {msg_out}")

            # Ensure Small model is selected
            controller.set_selected_model("DA3-SMALL")
            valid, reason = controller.validate_for_conversion()
            self.assertTrue(valid, f"Validation failed: {reason}")

            # Collect events
            stages_visited = []
            frames_recorded = []
            completed_result = {}
            error_recorded = {}

            def on_status(stage: str, message: str) -> None:
                stages_visited.append((stage, message))

            def on_progress(frame: int, total: int, pct: float, fps: float, eta: Any) -> None:
                frames_recorded.append((frame, total, pct, fps))

            def on_completed(res: dict) -> None:
                completed_result.update(res)

            def on_failed(err: str, stg: str) -> None:
                error_recorded["error"] = err
                error_recorded["stage"] = stg

            controller.status_updated.connect(on_status)
            controller.conversion_progress.connect(on_progress)
            controller.conversion_completed.connect(on_completed)
            controller.conversion_failed.connect(on_failed)

            # Start real worker process
            started, err = controller.start_conversion(overwrite_confirmed=True)
            self.assertTrue(started, f"Failed to start conversion: {err}")

            # Run event loop until completed or failed, with generous 60s timeout
            loop = QEventLoop()
            timer = QTimer()
            timer.setSingleShot(True)

            def finish_loop() -> None:
                if loop.isRunning():
                    loop.quit()

            controller.conversion_completed.connect(lambda res: finish_loop())
            controller.conversion_failed.connect(lambda err, stg: finish_loop())
            controller.conversion_cancelled.connect(lambda: finish_loop())
            timer.timeout.connect(lambda: finish_loop())

            timer.start(60000)  # 60 seconds
            loop.exec()

            # Verify completion
            self.assertEqual(
                controller.state,
                DesktopState.COMPLETED,
                f"Conversion did not complete. Error: {error_recorded.get('error')}",
            )
            self.assertGreater(len(frames_recorded), 0, "No progress updates recorded from worker.")
            self.assertGreater(len(completed_result), 0, "No completion payload received.")

            # Verify physical output file on disk
            self.assertTrue(out_file.is_file(), f"Output file does not exist: {out_file}")
            self.assertGreater(out_file.stat().st_size, 0, "Output file is empty.")

            # Probe generated media with ffprobe
            probe_out = probe_video(out_file, strict_sdr_cfr=False)
            self.assertEqual(probe_out.width, 3840, f"Expected 3840 width, got {probe_out.width}")
            self.assertEqual(probe_out.height, 1080, f"Expected 1080 height, got {probe_out.height}")
            self.assertEqual(probe_out.frame_count, controller.input_probe.frame_count)
            self.assertTrue(probe_out.has_audio, "Expected audio in output Full-SBS video.")
            self.assertEqual(probe_out.audio_streams[0].codec_name, "aac")
            self.assertAlmostEqual(probe_out.duration, controller.input_probe.duration, places=1)

            # Verify execution metrics
            self.assertEqual(completed_result.get("total_frames_processed"), controller.input_probe.frame_count)
            self.assertGreater(completed_result.get("wall_clock_seconds", 0), 0)
            self.assertGreater(completed_result.get("effective_fps", 0), 0)

    def test_real_controller_worker_base_conversion(self) -> None:
        """Execute real end-to-end conversion with DA3 Base model via DesktopController."""
        if not BASE_CHECKPOINT_DIR.exists():
            raise unittest.SkipTest(f"DA3 Base checkpoint directory not found at {BASE_CHECKPOINT_DIR}")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "desktop_base_3840x1080_sbs.mp4"

            controller = DesktopController()

            # Set input video
            success, msg = controller.set_input_path(FIXTURE_VIDEO)
            self.assertTrue(success, f"Failed to set input: {msg}")
            self.assertIsNotNone(controller.input_probe)
            assert controller.input_probe is not None

            # Set unique output path
            success_out, msg_out = controller.set_output_path(out_file)
            self.assertTrue(success_out, f"Failed to set output: {msg_out}")

            # Select Base model
            controller.set_selected_model("DA3-BASE")
            self.assertEqual(controller.selected_model_id, "DA3-BASE")
            valid, reason = controller.validate_for_conversion()
            self.assertTrue(valid, f"Validation failed for Base: {reason}")

            # Collect events
            stages_visited = []
            frames_recorded = []
            completed_result = {}
            error_recorded = {}

            def on_status(stage: str, message: str) -> None:
                stages_visited.append((stage, message))

            def on_progress(frame: int, total: int, pct: float, fps: float, eta: Any) -> None:
                frames_recorded.append((frame, total, pct, fps))

            def on_completed(res: dict) -> None:
                completed_result.update(res)

            def on_failed(err: str, stg: str) -> None:
                error_recorded["error"] = err
                error_recorded["stage"] = stg

            controller.status_updated.connect(on_status)
            controller.conversion_progress.connect(on_progress)
            controller.conversion_completed.connect(on_completed)
            controller.conversion_failed.connect(on_failed)

            # Start real worker process
            started, err = controller.start_conversion(overwrite_confirmed=True)
            self.assertTrue(started, f"Failed to start conversion: {err}")

            # Run event loop until completed or failed
            loop = QEventLoop()
            timer = QTimer()
            timer.setSingleShot(True)

            def finish_loop() -> None:
                if loop.isRunning():
                    loop.quit()

            controller.conversion_completed.connect(lambda res: finish_loop())
            controller.conversion_failed.connect(lambda err, stg: finish_loop())
            controller.conversion_cancelled.connect(lambda: finish_loop())
            timer.timeout.connect(lambda: finish_loop())

            timer.start(90000)  # 90 seconds
            loop.exec()

            # Verify completion
            self.assertEqual(
                controller.state,
                DesktopState.COMPLETED,
                f"Base conversion did not complete. Error: {error_recorded.get('error')}",
            )
            self.assertEqual(controller.selected_model_id, "DA3-BASE")
            self.assertGreater(len(frames_recorded), 0, "No progress updates recorded from worker.")
            self.assertGreater(len(completed_result), 0, "No completion payload received.")

            # Verify physical output file on disk
            self.assertTrue(out_file.is_file(), f"Output file does not exist: {out_file}")
            self.assertGreater(out_file.stat().st_size, 0, "Output file is empty.")

            # Probe generated media with ffprobe
            probe_out = probe_video(out_file, strict_sdr_cfr=False)
            self.assertEqual(probe_out.width, 3840, f"Expected 3840 width, got {probe_out.width}")
            self.assertEqual(probe_out.height, 1080, f"Expected 1080 height, got {probe_out.height}")
            self.assertEqual(probe_out.frame_count, controller.input_probe.frame_count)
            self.assertTrue(probe_out.has_audio, "Expected audio in output Full-SBS video.")
            self.assertEqual(probe_out.audio_streams[0].codec_name, "aac")
            self.assertAlmostEqual(probe_out.duration, controller.input_probe.duration, places=1)

            # Verify execution metrics
            self.assertEqual(completed_result.get("total_frames_processed"), controller.input_probe.frame_count)
            self.assertGreater(completed_result.get("wall_clock_seconds", 0), 0)
            self.assertGreater(completed_result.get("effective_fps", 0), 0)

    def test_ui_window_browse_and_controls_offscreen(self) -> None:
        """Test MainWindow offscreen with mocked file dialog browse callbacks."""
        controller = DesktopController()
        window = MainWindow(controller=controller)

        # Mock input browse dialog
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(FIXTURE_VIDEO), "")):
            window._on_browse_input()
            self.assertEqual(window.input_edit.text(), str(FIXTURE_VIDEO))
            self.assertIn("1920x1080", window.input_info_label.text())
            self.assertIn("3840x1080", window.output_geometry_label.text())

        # Test model selection change to unsupported model (Large 1.1)
        large_idx = window.model_combo.findData("DA3-LARGE-1.1")
        self.assertGreaterEqual(large_idx, 0)
        window.model_combo.setCurrentIndex(large_idx)

        # Must show unsupported notice and disable Convert button
        self.assertIn("not yet integrated", window.model_notice_label.text().lower())
        self.assertFalse(window.convert_btn.isEnabled())

        # Test model selection change to supported model (Base)
        base_idx = window.model_combo.findData("DA3-BASE")
        self.assertGreaterEqual(base_idx, 0)
        window.model_combo.setCurrentIndex(base_idx)
        self.assertIn("supported", window.model_notice_label.text().lower())
        self.assertTrue(window.convert_btn.isEnabled())

        # Switch back to Small
        small_idx = window.model_combo.findData("DA3-SMALL")
        window.model_combo.setCurrentIndex(small_idx)
        self.assertIn("supported", window.model_notice_label.text().lower())
        self.assertTrue(window.convert_btn.isEnabled())

        # Test depth strength controls synchronization
        window.strength_spin.setValue(0.005)
        self.assertEqual(window.strength_slider.value(), 5)
        self.assertAlmostEqual(controller.disparity_strength, 0.005)

        window.strength_slider.setValue(2)
        self.assertAlmostEqual(window.strength_spin.value(), 0.002)
        self.assertAlmostEqual(controller.disparity_strength, 0.002)


if __name__ == "__main__":
    unittest.main()
