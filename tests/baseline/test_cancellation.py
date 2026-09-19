"""Tests for bounded job cancellation and cleanup using owned temporary jobs."""

import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from puregpu3d.config.types import Codec, EngineConfig
from puregpu3d.engine.codec.ffmpeg_backend import FfmpegSoftwareBackend
from puregpu3d.engine.core import StereoEngine, TranscodeCancelledError


class TestCancellation(unittest.TestCase):
    """Characterize cancellation response times, resource cleanup, and absence of false completed files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = FfmpegSoftwareBackend()
        if not cls.backend.is_available():
            raise unittest.SkipTest("ffmpeg/ffprobe not available in PATH")

    def _generate_synthetic_clip(
        self,
        output_path: Path,
        *,
        width: int = 320,
        height: int = 180,
        fps: int = 24,
        duration_s: float = 1.5,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            str(duration_s),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def test_cooperative_cancellation_during_run(self) -> None:
        """Job cancelled mid-stream raises TranscodeCancelledError and leaves no output or staging file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_media = tmp / "cancel_input.mp4"
            out_media = tmp / "cancel_output.mp4"

            # 1.5 seconds at 24 fps = 36 frames
            self._generate_synthetic_clip(in_media, duration_s=1.5)

            cfg = EngineConfig(
                input_path=in_media,
                output_path=out_media,
                codec=Codec.H264,
                video_encoder="libx264",
                backend_preference="ffmpeg",
                audio_passthrough=True,
            )

            engine = StereoEngine()
            frames_processed = [0]

            def on_progress(fin: int, ftot: int, fms: float, kms: float) -> None:
                frames_processed[0] = fin

            def cancel_check() -> bool:
                return frames_processed[0] >= 5

            start_t = time.perf_counter()
            with self.assertRaises(TranscodeCancelledError):
                engine.run_file(cfg, progress_callback=on_progress, cancel_requested=cancel_check)
            elapsed_s = time.perf_counter() - start_t

            # Verify bounds: cancel must be handled promptly (under 2 seconds)
            self.assertLess(elapsed_s, 2.0, f"Cancellation took too long: {elapsed_s:.2f}s")
            self.assertGreaterEqual(frames_processed[0], 5)

            # Neither final output nor intermediate staging file may exist
            self.assertFalse(out_media.exists(), "Cancelled job created a false completed output file")
            staged_file = out_media.with_suffix(".video_only.mp4")
            self.assertFalse(
                staged_file.exists(), "Cancelled job left an uncleaned intermediate staging file"
            )

    def test_out_of_band_abort_active_run(self) -> None:
        """Calling StereoEngine.abort_active_run() asynchronously stops job within 5-second bound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_media = tmp / "oob_input.mp4"
            out_media = tmp / "oob_output.mp4"

            # 2.0 seconds at 24 fps = 48 frames
            self._generate_synthetic_clip(in_media, duration_s=2.0)

            cfg = EngineConfig(
                input_path=in_media,
                output_path=out_media,
                codec=Codec.H264,
                video_encoder="libx264",
                backend_preference="ffmpeg",
                audio_passthrough=True,
            )

            engine = StereoEngine()
            caught_exceptions: list[Exception] = []
            actively_running_event = threading.Event()

            def on_progress(fin: int, ftot: int, fms: float, kms: float) -> None:
                if fin >= 3:
                    actively_running_event.set()

            def worker() -> None:
                try:
                    engine.run_file(cfg, progress_callback=on_progress)
                except Exception as exc:
                    caught_exceptions.append(exc)

            thread = threading.Thread(target=worker, name="test-abort-worker")
            thread.start()

            # Wait until the transcode loop is confirmed running
            running_confirmed = actively_running_event.wait(timeout=3.0)
            self.assertTrue(running_confirmed, "Transcode worker did not signal running within 3s")

            abort_start = time.perf_counter()
            engine.abort_active_run()

            # Must stop within 5.0 seconds per product contract
            thread.join(timeout=5.0)
            abort_elapsed = time.perf_counter() - abort_start

            self.assertFalse(thread.is_alive(), "Worker thread did not terminate within 5s of abort")
            self.assertLess(abort_elapsed, 5.0)

            self.assertEqual(len(caught_exceptions), 1)
            self.assertIsInstance(
                caught_exceptions[0],
                TranscodeCancelledError,
                f"Expected TranscodeCancelledError, got {type(caught_exceptions[0])}: {caught_exceptions[0]}",
            )

            # Check files
            self.assertFalse(out_media.exists())
            staged_file = out_media.with_suffix(".video_only.mp4")
            self.assertFalse(staged_file.exists())

    def test_cancellation_before_first_frame(self) -> None:
        """Cancellation requested immediately at job start exits cleanly before frame processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_media = tmp / "immediate_cancel.mp4"
            out_media = tmp / "immediate_output.mp4"

            self._generate_synthetic_clip(in_media, duration_s=0.5)

            cfg = EngineConfig(
                input_path=in_media,
                output_path=out_media,
                codec=Codec.H264,
                video_encoder="libx264",
                backend_preference="ffmpeg",
            )

            engine = StereoEngine()
            with self.assertRaises(TranscodeCancelledError):
                engine.run_file(cfg, cancel_requested=lambda: True)

            self.assertFalse(out_media.exists())
            self.assertFalse(out_media.with_suffix(".video_only.mp4").exists())


if __name__ == "__main__":
    unittest.main()
