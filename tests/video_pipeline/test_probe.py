"""Unit tests for video probing and strict vertical slice media guards."""

import fractions
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.video.probe import (
    MediaNotFoundError,
    UnsupportedMediaError,
    find_ffmpeg,
    find_ffprobe,
    probe_video,
)


class TestVideoProbe(unittest.TestCase):
    """Test suite for video metadata extraction and upfront constraint verification."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = find_ffmpeg()
        cls.ffprobe = find_ffprobe()

    def _create_test_clip(
        self,
        out_path: Path,
        *,
        width: int = 320,
        height: int = 180,
        fps: int = 12,
        duration: float = 0.5,
        with_audio: bool = True,
        audio_codec: str = "aac",
        pix_fmt: str = "yuv420p",
        extra_args: list[str] | None = None,
    ) -> Path:
        """Generate deterministic synthetic test clip using FFmpeg lavfi."""
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}",
        ]
        if with_audio:
            cmd.extend([
                "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=44100",
                "-c:a", audio_codec,
            ])
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", pix_fmt,
            "-t", str(duration),
        ])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(str(out_path))

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0, f"FFmpeg failed to create test clip: {res.stderr.decode('utf-8', 'replace')}")
        return out_path

    def test_nonexistent_file_raises_media_not_found(self) -> None:
        fake_path = Path("non_existent_puregpu3d_test_file.mp4")
        with self.assertRaises(MediaNotFoundError):
            probe_video(fake_path)

    def test_valid_cfr_sdr_probe_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip = Path(tmpdir) / "valid_clip.mp4"
            self._create_test_clip(clip, width=320, height=240, fps=12, duration=0.5, with_audio=True)

            res = probe_video(clip, strict_sdr_cfr=True)
            self.assertEqual(res.width, 320)
            self.assertEqual(res.height, 240)
            self.assertEqual(res.frame_rate, fractions.Fraction(12, 1))
            self.assertAlmostEqual(res.fps, 12.0, places=2)
            self.assertAlmostEqual(res.duration, 0.5, places=1)
            self.assertEqual(res.frame_count, 6)
            self.assertFalse(res.is_hdr)
            self.assertFalse(res.is_vfr)
            self.assertEqual(res.rotation, 0)
            self.assertTrue(res.has_audio)
            self.assertEqual(len(res.audio_streams), 1)
            self.assertEqual(res.audio_streams[0].codec_name, "aac")

    def test_silent_video_probe_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip = Path(tmpdir) / "silent_clip.mp4"
            self._create_test_clip(clip, width=320, height=240, fps=12, duration=0.5, with_audio=False)

            res = probe_video(clip, strict_sdr_cfr=True)
            self.assertFalse(res.has_audio)
            self.assertEqual(len(res.audio_streams), 0)

    def test_rejection_of_odd_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Odd width: 321x240 (libx264 supports odd with yuv444p or other format)
            # We can also test odd height with yuv444p
            clip = Path(tmpdir) / "odd_dim.mp4"
            self._create_test_clip(
                clip,
                width=321,
                height=240,
                fps=12,
                duration=0.25,
                with_audio=False,
                pix_fmt="yuv444p",
            )

            with self.assertRaises(UnsupportedMediaError) as ctx:
                probe_video(clip, strict_sdr_cfr=True)
            self.assertIn("Odd video dimensions", str(ctx.exception))

    def test_rejection_of_rotated_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip = Path(tmpdir) / "rotated.mkv"
            self._create_test_clip(
                clip,
                width=320,
                height=240,
                fps=12,
                duration=0.25,
                with_audio=False,
                extra_args=["-metadata:s:v:0", "rotate=90"],
            )

            with self.assertRaises(UnsupportedMediaError) as ctx:
                probe_video(clip, strict_sdr_cfr=True)
            self.assertIn("Rotated video metadata detected", str(ctx.exception))

    def test_rejection_of_hdr_10bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip = Path(tmpdir) / "hdr10.mp4"
            self._create_test_clip(
                clip,
                width=320,
                height=240,
                fps=12,
                duration=0.25,
                with_audio=False,
                pix_fmt="yuv420p10le",
                extra_args=[
                    "-color_trc", "smpte2084",
                    "-colorspace", "bt2020nc",
                    "-color_primaries", "bt2020",
                ],
            )

            with self.assertRaises(UnsupportedMediaError) as ctx:
                probe_video(clip, strict_sdr_cfr=True)
            self.assertIn("HDR video detected", str(ctx.exception))

    def test_rejection_of_unsupported_audio_codec_for_mp4_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # FLAC audio in MKV container
            clip = Path(tmpdir) / "unsupported_audio.mkv"
            cmd = [
                str(self.ffmpeg),
                "-y",
                "-f", "lavfi",
                "-i", "testsrc=size=320x240:rate=12",
                "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=44100",
                "-c:v", "libx264",
                "-c:a", "flac",
                "-t", "0.25",
                str(clip),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(res.returncode, 0)

            with self.assertRaises(UnsupportedMediaError) as ctx:
                probe_video(clip, strict_sdr_cfr=True)
            self.assertIn("Audio codec 'flac' cannot be directly copied", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
