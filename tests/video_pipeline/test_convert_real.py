"""Integration test for real DA3 Small end-to-end 1080p video-to-Full-SBS conversion on GPU."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"
for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import DEFAULT_REVISION, DA3SmallDepthAdapter
from puregpu3d.video.convert import convert_video
from puregpu3d.video.probe import find_ffmpeg, probe_video

CHECKPOINT_DIR = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION


class TestConvertReal(unittest.TestCase):
    """Real acceptance test using Depth Anything 3 Small and PyTorch stereo renderer."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available; real GPU conversion test skipped.")
        if not (CHECKPOINT_DIR / "READY").exists():
            raise unittest.SkipTest(f"DA3 Small checkpoint not ready at {CHECKPOINT_DIR}")

        cls.ffmpeg = find_ffmpeg()
        cls.adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cuda")

    def _create_synthetic_1080p_clip(
        self,
        path: Path,
        fps: int = 12,
        duration: float = 0.5,
    ) -> Path:
        """Create synthetic moving 1920x1080 CFR clip with AAC test audio."""
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size=1920x1080:rate={fps}",
            "-f", "lavfi",
            "-i", "sine=frequency=523.25:sample_rate=44100",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-t", str(duration),
            str(path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0, f"FFmpeg failed to create 1080p fixture: {res.stderr.decode('utf-8', 'replace')}")
        return path

    def test_real_da3_small_1080p_to_full_sbs(self) -> None:
        """Verify 1920x1080 input converts to 3840x1080 Full-SBS with audio on CUDA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input_1080p.mp4"
            dst = Path(tmpdir) / "output_3840x1080_sbs.mp4"

            self._create_synthetic_1080p_clip(src, fps=12, duration=0.5)

            # Probe source
            probe_in = probe_video(src)
            self.assertEqual(probe_in.width, 1920)
            self.assertEqual(probe_in.height, 1080)
            self.assertEqual(probe_in.frame_count, 6)
            self.assertTrue(probe_in.has_audio)

            # Run conversion using loaded GPU adapter
            result = convert_video(
                input_path=src,
                output_path=dst,
                model=self.adapter,
                device="cuda",
            )

            # Check return structure
            self.assertEqual(result.input_width, 1920)
            self.assertEqual(result.input_height, 1080)
            self.assertEqual(result.output_width, 3840)
            self.assertEqual(result.output_height, 1080)
            self.assertEqual(result.total_frames_processed, 6)
            self.assertTrue(result.has_audio)
            self.assertGreater(result.wall_clock_seconds, 0)
            self.assertGreater(result.effective_fps, 0)
            self.assertGreater(result.mean_depth_ms, 0)
            self.assertGreater(result.mean_stereo_ms, 0)

            # Probe destination output
            self.assertTrue(dst.exists())
            self.assertGreater(dst.stat().st_size, 0)

            probe_out = probe_video(dst, strict_sdr_cfr=False)
            self.assertEqual(probe_out.width, 3840)
            self.assertEqual(probe_out.height, 1080)
            self.assertEqual(probe_out.frame_count, 6)
            self.assertTrue(probe_out.has_audio)
            self.assertEqual(probe_out.audio_streams[0].codec_name, "aac")
            self.assertAlmostEqual(probe_out.duration, 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
