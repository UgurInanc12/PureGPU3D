"""Tests for configuration validation, normalization, serialization, and profile loading."""

import tempfile
import unittest
from pathlib import Path

from puregpu3d.config.loader import load_profile
from puregpu3d.config.types import (
    Codec,
    CorruptFrameFallback,
    DepthConfig,
    EngineConfig,
    FpsMode,
    OverflowPolicy,
    SbsMode,
)


class TestConfigValidation(unittest.TestCase):
    """Characterize DepthConfig, EngineConfig validation and profile loader behavior."""

    def test_depth_config_defaults_and_validation(self) -> None:
        cfg = DepthConfig()
        cfg.validate()
        self.assertEqual(cfg.max_disparity_px, 6)
        self.assertAlmostEqual(cfg.depth_strength, 0.55)

        # Disparity bounds
        with self.assertRaises(ValueError):
            DepthConfig(max_disparity_px=-1).validate()
        with self.assertRaises(ValueError):
            DepthConfig(max_disparity_px=17).validate()

        # Strength bounds
        with self.assertRaises(ValueError):
            DepthConfig(depth_strength=-0.1).validate()
        with self.assertRaises(ValueError):
            DepthConfig(depth_strength=1.05).validate()

        # Negative weights
        with self.assertRaises(ValueError):
            DepthConfig(edge_weight=-0.01).validate()
        with self.assertRaises(ValueError):
            DepthConfig(luma_weight=-0.5).validate()
        with self.assertRaises(ValueError):
            DepthConfig(vertical_weight=-1.0).validate()

    def test_depth_config_normalized_weights(self) -> None:
        cfg = DepthConfig(edge_weight=1.0, luma_weight=2.0, vertical_weight=1.0)
        ew, lw, vw = cfg.normalized_weights()
        self.assertAlmostEqual(ew, 0.25)
        self.assertAlmostEqual(lw, 0.50)
        self.assertAlmostEqual(vw, 0.25)
        self.assertAlmostEqual(ew + lw + vw, 1.0)

        # Zero sum fallback
        cfg_zero = DepthConfig(edge_weight=0.0, luma_weight=0.0, vertical_weight=0.0)
        self.assertEqual(cfg_zero.normalized_weights(), (0.25, 0.45, 0.30))

    def test_engine_config_input_file_existence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing_input = tmp / "missing.mp4"
            out_path = tmp / "out.mp4"

            # Default requires existing input
            cfg = EngineConfig(input_path=missing_input, output_path=out_path)
            with self.assertRaises(FileNotFoundError):
                cfg.validate()

            # allow_missing_input=True permits validation before file creation
            cfg_allowed = EngineConfig(
                input_path=missing_input, output_path=out_path, allow_missing_input=True
            )
            cfg_allowed.validate()

    def test_engine_config_field_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            real_input = tmp / "real.mp4"
            real_input.write_bytes(b"dummy")
            out_path = tmp / "out.mp4"

            base = EngineConfig(input_path=real_input, output_path=out_path)
            base.validate()

            # Buffer slots >= 2
            with self.assertRaises(ValueError):
                EngineConfig(input_path=real_input, output_path=out_path, buffer_slots=1).validate()

            # Bitrate > 0 when provided
            with self.assertRaises(ValueError):
                EngineConfig(
                    input_path=real_input, output_path=out_path, video_bitrate_mbps=0.0
                ).validate()
            with self.assertRaises(ValueError):
                EngineConfig(
                    input_path=real_input, output_path=out_path, video_bitrate_mbps=-5.0
                ).validate()

            # Video encoder non-empty
            with self.assertRaises(ValueError):
                EngineConfig(
                    input_path=real_input, output_path=out_path, video_encoder="   "
                ).validate()

            # Audio codec must be copy or aac
            with self.assertRaises(ValueError):
                EngineConfig(
                    input_path=real_input, output_path=out_path, audio_codec="mp3"
                ).validate()

            # Watchdog timeout > 0
            with self.assertRaises(ValueError):
                EngineConfig(
                    input_path=real_input, output_path=out_path, watchdog_timeout_s=0.0
                ).validate()

    def test_engine_config_serialization_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_p = tmp / "in.mp4"
            in_p.touch()
            out_p = tmp / "out.mp4"

            cfg = EngineConfig(
                input_path=in_p,
                output_path=out_p,
                codec=Codec.H264,
                video_encoder="libx264",
                video_bitrate_mbps=12.5,
                fps_mode=FpsMode.OFFLINE,
                sbs_mode=SbsMode.FULL,
                buffer_slots=8,
                overflow_policy=OverflowPolicy.BLOCK,
                audio_passthrough=True,
                audio_codec="aac",
                preserve_color_metadata=True,
                backend_preference="ffmpeg",
                corrupt_frame_fallback=CorruptFrameFallback.BLACK,
                watchdog_timeout_s=15.0,
            )
            d = cfg.to_dict()
            self.assertEqual(d["codec"], "h264")
            self.assertEqual(d["fps_mode"], "offline")
            self.assertEqual(d["sbs_mode"], "sbs_full")
            self.assertEqual(d["corrupt_frame_fallback"], "black")
            self.assertEqual(d["video_bitrate_mbps"], 12.5)

            reconstructed = EngineConfig.from_mapping(d)
            self.assertEqual(reconstructed.codec, Codec.H264)
            self.assertEqual(reconstructed.video_encoder, "libx264")
            self.assertEqual(reconstructed.video_bitrate_mbps, 12.5)
            self.assertEqual(reconstructed.corrupt_frame_fallback, CorruptFrameFallback.BLACK)
            reconstructed.validate()

    def test_profile_loader_with_existing_profiles(self) -> None:
        profile_path = Path("configs/profiles/safe_low.yaml")
        self.assertTrue(profile_path.exists(), "configs/profiles/safe_low.yaml missing")

        data = load_profile(profile_path)
        self.assertIsInstance(data, dict)
        self.assertIn("depth", data)
        self.assertIn("buffer_slots", data)

        # Missing profile raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_profile(Path("configs/profiles/non_existent_profile.yaml"))


if __name__ == "__main__":
    unittest.main()
