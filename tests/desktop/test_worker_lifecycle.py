"""Unit and integration tests for worker process isolation and cancellation."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "src" / "puregpu3d" / "runtime" / "worker.py"
FIXTURE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"


class TestWorkerLifecycle(unittest.TestCase):
    """Test worker process stdout/stderr redirection, model guards, and cancellation."""

    def test_worker_refuses_unsupported_model(self) -> None:
        """Worker must refuse unsupported model (DA3-LARGE-1.1) with explicit error protocol message and non-zero exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "cmd.json"
            out_file = Path(tmpdir) / "out.mp4"

            payload = {
                "job_id": "test_unsupported",
                "input_path": str(FIXTURE_VIDEO),
                "output_path": str(out_file),
                "model_id": "DA3-LARGE-1.1",
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

            # Check stdout: should contain JSON-lines messages including error
            stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            self.assertGreater(len(stdout_lines), 0)

            parsed = [json.loads(line) for line in stdout_lines]
            types = [m["type"] for m in parsed]
            self.assertIn("error", types)

            err_msg = next(m for m in parsed if m["type"] == "error")
            self.assertIn("not integrated", err_msg["error"].lower())

    def test_worker_strict_stdout_protocol_and_stderr_logs(self) -> None:
        """Verify all lines emitted on stdout are strictly valid JSON-lines, no raw prints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "cmd.json"
            out_file = Path(tmpdir) / "out.mp4"

            payload = {
                "job_id": "test_strict_stdout",
                "input_path": str(FIXTURE_VIDEO),
                "output_path": str(out_file),
                "model_id": "DA3-LARGE-1.1",  # Will fail fast on model guard
                "overwrite": True,
            }
            cmd_file.write_text(json.dumps(payload), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-B", str(WORKER_SCRIPT), "--command-file", str(cmd_file)],
                capture_output=True,
                text=True,
                check=False,
            )

            # Every non-empty line in stdout MUST be valid JSON with a 'type' key
            stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            self.assertGreater(len(stdout_lines), 0)
            for line in stdout_lines:
                try:
                    data = json.loads(line)
                    self.assertIsInstance(data, dict)
                    self.assertIn("type", data)
                except Exception as err:
                    self.fail(f"Corrupt line on worker stdout: {line!r}, error: {err}")

            # Diagnostics and job object notice must be on stderr
            self.assertIn("[worker-job]", proc.stderr)

    def test_worker_cancellation_cooperative(self) -> None:
        """Verify cancel file triggers cooperative exit with cancelled protocol message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "cmd.json"
            cancel_file = Path(tmpdir) / "cancel.tmp"
            out_file = Path(tmpdir) / "out.mp4"

            # Pre-create cancel file so worker cancels immediately
            cancel_file.write_text("cancel", encoding="utf-8")

            payload = {
                "job_id": "test_cancel",
                "input_path": str(FIXTURE_VIDEO),
                "output_path": str(out_file),
                "model_id": "DA3-SMALL",
                "cancel_file": str(cancel_file),
                "overwrite": True,
            }
            cmd_file.write_text(json.dumps(payload), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-B", str(WORKER_SCRIPT), "--command-file", str(cmd_file)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0)
            stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            parsed = [json.loads(line) for line in stdout_lines]
            types = [m["type"] for m in parsed]
            self.assertIn("cancelled", types)


if __name__ == "__main__":
    unittest.main()
