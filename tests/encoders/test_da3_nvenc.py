"""Real DA3 Small model acceptance tests comparing NVENC and libx264 encoding on GPU."""

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
from puregpu3d.video.encoders import preflight_encoder
from puregpu3d.video.probe import probe_video

FIXTURE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"
CHECKPOINT_DIR = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION


class TestDA3NVENCRealAcceptance(unittest.TestCase):
    """End-to-end acceptance tests using real DA3 Small model on 1080p moving clip."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available; skipping DA3 real tests.")
        if not FIXTURE_VIDEO.exists():
            raise unittest.SkipTest(f"Fixture video missing: {FIXTURE_VIDEO}")
        if not (CHECKPOINT_DIR / "READY").exists():
            raise unittest.SkipTest(f"DA3 Small checkpoint not ready at {CHECKPOINT_DIR}")

        pf = preflight_encoder("hevc_nvenc", width=3840, height=1080)
        if not pf.supported:
            raise unittest.SkipTest(f"HEVC NVENC unsupported: {pf.error_message}")

        cls.adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cuda")

    def test_da3_small_hevc_nvenc_conversion(self) -> None:
        """Verify DA3 Small 1080p input produces 3840x1080 FullSBS HEVC output with audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "output_da3_hevc.mp4"
            res = convert_video(
                input_path=FIXTURE_VIDEO,
                output_path=dst,
                model=self.adapter,
                device="cuda",
                encoder="hevc_nvenc",
                cq=23,
                nvenc_preset="p4",
                overwrite=True,
            )

            # Contract verification
            self.assertEqual(res.encoder, "hevc_nvenc")
            self.assertEqual(res.input_width, 1920)
            self.assertEqual(res.input_height, 1080)
            self.assertEqual(res.output_width, 3840)
            self.assertEqual(res.output_height, 1080)
            self.assertEqual(res.total_frames_processed, 12)
            self.assertTrue(res.has_audio)
            self.assertGreater(res.wall_clock_seconds, 0)
            self.assertGreater(res.effective_fps, 0)

            # ffprobe verification
            self.assertTrue(dst.exists())
            probe = probe_video(dst)
            self.assertEqual(probe.width, 3840)
            self.assertEqual(probe.height, 1080)
            self.assertEqual(probe.frame_count, 12)
            self.assertTrue(probe.has_audio)
            self.assertEqual(len(probe.audio_streams), 1)
            self.assertEqual(probe.audio_streams[0].codec_name, "aac")

            # Check video codec is HEVC
            v_codecs = [s.get("codec_name") for s in probe.raw_info.get("streams", []) if s.get("codec_type") == "video"]
            self.assertEqual(v_codecs, ["hevc"])

    def test_da3_small_auto_encoder_resolution(self) -> None:
        """Verify auto encoder selects hevc_nvenc on NVENC-capable hardware."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "output_da3_auto.mp4"
            res = convert_video(
                input_path=FIXTURE_VIDEO,
                output_path=dst,
                model=self.adapter,
                device="cuda",
                encoder="auto",
                overwrite=True,
            )

            self.assertEqual(res.encoder, "hevc_nvenc")
            probe = probe_video(dst)
            v_codecs = [s.get("codec_name") for s in probe.raw_info.get("streams", []) if s.get("codec_type") == "video"]
            self.assertEqual(v_codecs, ["hevc"])


if __name__ == "__main__":
    unittest.main()
