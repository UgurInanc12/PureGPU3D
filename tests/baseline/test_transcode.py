"""Tests for end-to-end media transcode using FFmpeg software libx264 and audio remux."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from puregpu3d.config.types import Codec, EngineConfig
from puregpu3d.engine.codec.ffmpeg_backend import FfmpegSoftwareBackend
from puregpu3d.engine.core import StereoEngine


class TestTranscode(unittest.TestCase):
    """End-to-end characterization of FFmpeg software transcode with audio and ffprobe checks."""

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
        duration_s: float = 0.5,
        include_audio: bool = True,
    ) -> None:
        """Generate a short synthetic clip using FFmpeg lavfi generators."""
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
        ]
        if include_audio:
            cmd.extend(["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000"])

        cmd.extend(
            [
                "-t",
                str(duration_s),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if include_audio:
            cmd.extend(["-c:a", "aac", "-shortest"])
        else:
            cmd.extend(["-an"])

        cmd.append(str(output_path))
        subprocess.run(cmd, check=True, capture_output=True)

    def test_transcode_full_sbs_with_audio_and_ffprobe(self) -> None:
        """Transcode a short synthetic video with AAC audio and verify 2W x H and audio preservation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_media = tmp / "synth_input.mp4"
            out_media = tmp / "synth_output_sbs.mp4"

            # 320x180 @ 24fps for 0.5s = 12 frames
            self._generate_synthetic_clip(in_media, width=320, height=180, fps=24, duration_s=0.5)

            # Probe source input
            in_probe = self.backend.probe(in_media)
            self.assertEqual(in_probe.width, 320)
            self.assertEqual(in_probe.height, 180)
            self.assertAlmostEqual(in_probe.fps, 24.0, places=1)
            self.assertTrue(in_probe.has_audio)

            cfg = EngineConfig(
                input_path=in_media,
                output_path=out_media,
                codec=Codec.H264,
                video_encoder="libx264",
                backend_preference="ffmpeg",
                audio_passthrough=True,
                audio_codec="copy",
            )

            engine = StereoEngine()
            result = engine.run_file(cfg)

            self.assertTrue(result.ok)
            self.assertEqual(result.frames_in, 12)
            self.assertEqual(result.frames_out, 12)
            self.assertTrue(out_media.exists())
            self.assertGreater(out_media.stat().st_size, 0)

            # Intermediate .video_only.mp4 must be cleaned up
            temp_staged = out_media.with_suffix(".video_only.mp4")
            self.assertFalse(temp_staged.exists(), "Staging video was not cleaned up on success")

            # ffprobe verification of final output
            out_probe = self.backend.probe(out_media)
            self.assertEqual(out_probe.width, 640, "Output width must be exactly 2 * source width")
            self.assertEqual(out_probe.height, 180, "Output height must match source height")
            self.assertTrue(out_probe.has_audio, "Audio stream must be preserved in remux")

            # Direct stream inspection via ffprobe json
            ffprobe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(out_media),
            ]
            raw = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
            streams = json.loads(raw.stdout).get("streams", [])

            v_streams = [s for s in streams if s.get("codec_type") == "video"]
            a_streams = [s for s in streams if s.get("codec_type") == "audio"]

            self.assertEqual(len(v_streams), 1)
            self.assertEqual(len(a_streams), 1)
            self.assertEqual(int(v_streams[0]["width"]), 640)
            self.assertEqual(int(v_streams[0]["height"]), 180)
            self.assertEqual(a_streams[0]["codec_name"], "aac")

    def test_transcode_video_only_without_audio(self) -> None:
        """Transcode a video without audio stream produces valid SBS video-only output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_media = tmp / "silent_input.mp4"
            out_media = tmp / "silent_output_sbs.mp4"

            self._generate_synthetic_clip(
                in_media, width=160, height=90, fps=24, duration_s=0.25, include_audio=False
            )

            in_probe = self.backend.probe(in_media)
            self.assertFalse(in_probe.has_audio)

            cfg = EngineConfig(
                input_path=in_media,
                output_path=out_media,
                codec=Codec.H264,
                video_encoder="libx264",
                backend_preference="ffmpeg",
                audio_passthrough=True,
            )

            engine = StereoEngine()
            result = engine.run_file(cfg)

            self.assertTrue(result.ok)
            self.assertTrue(out_media.exists())

            out_probe = self.backend.probe(out_media)
            self.assertEqual(out_probe.width, 320)
            self.assertEqual(out_probe.height, 90)
            self.assertFalse(out_probe.has_audio)
            self.assertTrue(any("Source has no audio stream" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
