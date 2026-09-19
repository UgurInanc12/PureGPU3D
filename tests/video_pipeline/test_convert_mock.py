"""Unit and safety tests for streaming video conversion using injected mock depth model."""

import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.video.convert import (
    ConversionCancelledError,
    ConversionError,
    ConversionResult,
    convert_video,
)
from puregpu3d.video.probe import find_ffmpeg, probe_video


@dataclass
class MockDepthResult:
    depth: np.ndarray


class MockDepthAdapter:
    """Fast, deterministic mock model adapter for unit safety and streaming tests."""

    def __init__(self, constant_depth: float = 1.0) -> None:
        self.constant_depth = constant_depth
        self.call_count = 0
        self.last_depth_scale: Optional[Any] = None

    def infer(
        self,
        image: np.ndarray,
        target_size: int = 504,
        depth_scale: Optional[Any] = None,
        return_original_size: bool = True,
    ) -> MockDepthResult:
        self.call_count += 1
        self.last_depth_scale = depth_scale
        h, w = image.shape[:2]
        # Create a simple horizontal ramp depth map [0.5 to 2.5]
        ramp = np.linspace(0.5, 2.5, w, dtype=np.float32)
        depth = np.tile(ramp, (h, 1))
        return MockDepthResult(depth=depth)


class TestConvertMock(unittest.TestCase):
    """Unit test suite for video conversion streaming, audio remux, and cancellation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = find_ffmpeg()

    def _create_clip(
        self,
        path: Path,
        *,
        width: int = 320,
        height: int = 180,
        fps: int = 12,
        duration: float = 0.5,
        with_audio: bool = True,
    ) -> Path:
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}",
        ]
        if with_audio:
            cmd.extend([
                "-f", "lavfi",
                "-i", "sine=frequency=1000:sample_rate=44100",
                "-c:a", "aac",
            ])
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(path),
        ])
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0)
        return path

    def test_mock_streaming_conversion_with_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=0.5, with_audio=True)

            mock_adapter = MockDepthAdapter()
            progress_records = []

            def on_progress(cur: int, tot: int) -> None:
                progress_records.append((cur, tot))

            result = convert_video(
                input_path=src,
                output_path=dst,
                model=mock_adapter,
                device="cpu",
                progress_callback=on_progress,
            )

            # Verification of conversion result
            self.assertTrue(dst.exists())
            self.assertEqual(result.input_width, 320)
            self.assertEqual(result.input_height, 180)
            self.assertEqual(result.output_width, 640)
            self.assertEqual(result.output_height, 180)
            self.assertEqual(result.total_frames_processed, 6)
            self.assertGreater(result.wall_clock_seconds, 0)
            self.assertGreater(result.effective_fps, 0)
            self.assertEqual(mock_adapter.call_count, 6)
            self.assertEqual(len(progress_records), 6)

            # Probe destination output
            probe_out = probe_video(dst, strict_sdr_cfr=False)
            self.assertEqual(probe_out.width, 640)
            self.assertEqual(probe_out.height, 180)
            self.assertTrue(probe_out.has_audio)
            self.assertEqual(probe_out.audio_streams[0].codec_name, "aac")
            self.assertAlmostEqual(probe_out.duration, 0.5, places=1)

    def test_mock_streaming_conversion_silent_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "silent_input.mp4"
            dst = Path(tmpdir) / "silent_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=0.5, with_audio=False)

            mock_adapter = MockDepthAdapter()
            result = convert_video(
                input_path=src,
                output_path=dst,
                model=mock_adapter,
                device="cpu",
            )

            self.assertTrue(dst.exists())
            self.assertEqual(result.output_width, 640)
            self.assertFalse(result.has_audio)

            probe_out = probe_video(dst, strict_sdr_cfr=False)
            self.assertFalse(probe_out.has_audio)

    def test_conversion_cancellation_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mp4"
            dst = Path(tmpdir) / "cancelled_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=1.0, with_audio=True)

            mock_adapter = MockDepthAdapter()

            # Abort after 3 frames
            def cancel_checker() -> bool:
                return mock_adapter.call_count >= 3

            with self.assertRaises(ConversionCancelledError):
                convert_video(
                    input_path=src,
                    output_path=dst,
                    model=mock_adapter,
                    device="cpu",
                    cancel_callback=cancel_checker,
                )

            # Destination must NOT exist
            self.assertFalse(dst.exists())

            # No staging files left in directory
            leftovers = [p for p in Path(tmpdir).glob(".staging_*")]
            self.assertEqual(len(leftovers), 0, f"Leftover staging files found: {leftovers}")

    def _create_clip_short_audio(
        self,
        path: Path,
        width: int = 320,
        height: int = 180,
        fps: int = 12,
        video_duration: float = 1.0,
        audio_duration: float = 0.3,
    ) -> Path:
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}",
            "-f", "lavfi",
            "-i", "sine=frequency=1000:sample_rate=44100",
            "-t", str(video_duration),
            "-filter_complex", f"[1:a]atrim=end={audio_duration}[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0)
        return path

    def _create_clip_multi_audio(
        self,
        path: Path,
        width: int = 320,
        height: int = 180,
        fps: int = 12,
        duration: float = 0.5,
    ) -> Path:
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}",
            "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=44100",
            "-f", "lavfi",
            "-i", "sine=frequency=880:sample_rate=44100",
            "-t", str(duration),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0)
        return path

    def test_short_audio_preserves_full_video(self) -> None:
        """Short audio track must not truncate video frames (no -shortest bug)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "short_audio_input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            # 1.0s video @ 12fps = 12 frames, but audio is only 0.3s
            self._create_clip_short_audio(src, width=320, height=180, fps=12, video_duration=1.0, audio_duration=0.3)

            mock_adapter = MockDepthAdapter()
            result = convert_video(
                input_path=src,
                output_path=dst,
                model=mock_adapter,
                device="cpu",
            )

            self.assertTrue(dst.exists())
            self.assertEqual(result.total_frames_processed, 12)
            self.assertEqual(mock_adapter.call_count, 12)

            probe_out = probe_video(dst, strict_sdr_cfr=False)
            self.assertEqual(probe_out.frame_count, 12)
            self.assertTrue(probe_out.has_audio)

    def test_multi_track_audio_preserved(self) -> None:
        """Multi-track audio streams must all be preserved in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "multi_audio_input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            self._create_clip_multi_audio(src, width=320, height=180, fps=12, duration=0.5)

            probe_in = probe_video(src)
            self.assertEqual(len(probe_in.audio_streams), 2)

            mock_adapter = MockDepthAdapter()
            result = convert_video(
                input_path=src,
                output_path=dst,
                model=mock_adapter,
                device="cpu",
            )

            self.assertTrue(dst.exists())
            self.assertEqual(result.total_frames_processed, 6)

            probe_out = probe_video(dst, strict_sdr_cfr=False)
            self.assertEqual(len(probe_out.audio_streams), 2)
            self.assertEqual(probe_out.audio_streams[0].codec_name, "aac")
            self.assertEqual(probe_out.audio_streams[1].codec_name, "aac")

    def test_reader_partial_frame_fails_hard(self) -> None:
        """Truncated/partial frame from reader must fail hard and never promote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=0.5, with_audio=False)

            real_popen = subprocess.Popen

            def faulty_popen(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                cmd = args[0] if args else kwargs.get("args", [])
                prog_name = Path(cmd[0]).name.lower() if cmd else ""
                # If this is the reader process (stdout piped, stdin not piped)
                if prog_name.startswith("ffmpeg") and kwargs.get("stdout") == subprocess.PIPE and kwargs.get("stdin") is None:
                    assert proc.stdout is not None
                    real_read = proc.stdout.read
                    call_count = [0]

                    def truncated_read(size=-1):
                        call_count[0] += 1
                        if call_count[0] == 2:
                            # Return partial frame (fewer bytes than required frame)
                            return b"truncated_bytes"
                        return real_read(size)

                    proc.stdout.read = truncated_read
                return proc

            mock_adapter = MockDepthAdapter()
            with patch("subprocess.Popen", side_effect=faulty_popen):
                with self.assertRaises(ConversionError) as ctx:
                    convert_video(
                        input_path=src,
                        output_path=dst,
                        model=mock_adapter,
                        device="cpu",
                    )
                self.assertIn("Truncated frame received", str(ctx.exception))

            self.assertFalse(dst.exists())
            leftovers = [p for p in Path(tmpdir).glob(".staging_*")]
            self.assertEqual(len(leftovers), 0)

    def test_reader_nonzero_exit_fails_hard(self) -> None:
        """Non-zero reader exit code must fail hard and never promote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=0.5, with_audio=False)

            real_popen = subprocess.Popen

            def failing_reader_popen(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                cmd = args[0] if args else kwargs.get("args", [])
                prog_name = Path(cmd[0]).name.lower() if cmd else ""
                if prog_name.startswith("ffmpeg") and kwargs.get("stdout") == subprocess.PIPE and kwargs.get("stdin") is None:
                    def fake_wait(timeout=None):
                        proc.poll()
                        return 42

                    proc.wait = fake_wait
                    proc.poll = lambda: 42
                return proc

            mock_adapter = MockDepthAdapter()
            with patch("subprocess.Popen", side_effect=failing_reader_popen):
                with self.assertRaises(ConversionError) as ctx:
                    convert_video(
                        input_path=src,
                        output_path=dst,
                        model=mock_adapter,
                        device="cpu",
                    )
                self.assertIn("FFmpeg reader exited with error code 42", str(ctx.exception))

            self.assertFalse(dst.exists())
            leftovers = [p for p in Path(tmpdir).glob(".staging_*")]
            self.assertEqual(len(leftovers), 0)

    def test_cancellation_during_blocking_io(self) -> None:
        """Cancellation monitor unblocks blocked IO and raises ConversionCancelledError promptly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=0.5, with_audio=False)

            real_popen = subprocess.Popen

            def slow_reader_popen(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                cmd = args[0] if args else kwargs.get("args", [])
                prog_name = Path(cmd[0]).name.lower() if cmd else ""
                if prog_name.startswith("ffmpeg") and kwargs.get("stdout") == subprocess.PIPE and kwargs.get("stdin") is None:
                    assert proc.stdout is not None
                    real_read = proc.stdout.read

                    def slow_read(size=-1):
                        # Wait until proc is killed by monitor or 2.0s
                        t0 = time.monotonic()
                        while time.monotonic() - t0 < 2.0:
                            if proc.poll() is not None:
                                break
                            time.sleep(0.01)
                        return real_read(size)

                    proc.stdout.read = slow_read
                return proc

            mock_adapter = MockDepthAdapter()
            cancel_time = time.time() + 0.1

            def cancel_check() -> bool:
                return time.time() >= cancel_time

            start_t = time.monotonic()
            with patch("subprocess.Popen", side_effect=slow_reader_popen):
                with self.assertRaises(ConversionCancelledError):
                    convert_video(
                        input_path=src,
                        output_path=dst,
                        model=mock_adapter,
                        device="cpu",
                        cancel_callback=cancel_check,
                    )
            duration = time.monotonic() - start_t
            # Must abort promptly via monitor (well under the 2.0s sleep)
            self.assertLess(duration, 1.5)
            self.assertFalse(dst.exists())
            leftovers = [p for p in Path(tmpdir).glob(".staging_*")]
            self.assertEqual(len(leftovers), 0)

    def test_no_leaked_child_processes(self) -> None:
        """All child processes must be terminated and reaped on completion, error, or cancellation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.mp4"
            dst = Path(tmpdir) / "output_sbs.mp4"
            self._create_clip(src, width=320, height=180, fps=12, duration=0.5, with_audio=True)

            spawned_procs = []
            real_popen = subprocess.Popen

            def tracking_popen(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                spawned_procs.append(proc)
                return proc

            mock_adapter = MockDepthAdapter()

            # 1. Normal execution
            with patch("subprocess.Popen", side_effect=tracking_popen):
                convert_video(
                    input_path=src,
                    output_path=dst,
                    model=mock_adapter,
                    device="cpu",
                )

            self.assertGreater(len(spawned_procs), 0)
            for p in spawned_procs:
                self.assertIsNotNone(p.poll(), f"Child process {p.pid} was leaked alive!")

            # 2. Cancelled execution
            spawned_procs.clear()
            dst_cancel = Path(tmpdir) / "cancel_sbs.mp4"
            with patch("subprocess.Popen", side_effect=tracking_popen):
                with self.assertRaises(ConversionCancelledError):
                    convert_video(
                        input_path=src,
                        output_path=dst_cancel,
                        model=mock_adapter,
                        device="cpu",
                        cancel_callback=lambda: True,
                    )

            self.assertGreater(len(spawned_procs), 0)
            for p in spawned_procs:
                self.assertIsNotNone(p.poll(), f"Cancelled child process {p.pid} was leaked alive!")


if __name__ == "__main__":
    unittest.main()
