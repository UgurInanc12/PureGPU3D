"""Tests for backend selection factory and FFmpeg software backend mechanics."""

import unittest

from puregpu3d.config.types import Codec
from puregpu3d.engine.codec.factory import create_backend
from puregpu3d.engine.codec.ffmpeg_backend import (
    FfmpegSoftwareBackend,
    _parse_bit_depth,
    _parse_rate,
)
from puregpu3d.engine.codec.mock_backend import MockBackend


class TestBackendSelection(unittest.TestCase):
    """Characterize backend resolution and codec selection behavior."""

    def test_create_backend_preferences(self) -> None:
        # Mock backend
        backend_mock, warnings_mock = create_backend("mock")
        self.assertIsInstance(backend_mock, MockBackend)
        self.assertEqual(backend_mock.name, "mock")

        # FFmpeg backend
        backend_ff, warnings_ff = create_backend("ffmpeg")
        self.assertIsInstance(backend_ff, FfmpegSoftwareBackend)
        self.assertEqual(backend_ff.name, "ffmpeg")
        self.assertTrue(backend_ff.is_available())

        # Auto preference resolves to ffmpeg in current revision
        backend_auto, warnings_auto = create_backend("auto")
        self.assertIsInstance(backend_auto, FfmpegSoftwareBackend)
        self.assertTrue(any("Auto backend preference resolves to ffmpeg" in w for w in warnings_auto))

        # NvCodec preference falls back to ffmpeg (either scaffold-only or unavailable)
        backend_nv, warnings_nv = create_backend("nvcodec")
        self.assertIsInstance(backend_nv, FfmpegSoftwareBackend)
        self.assertTrue(
            any(
                "scaffold-only in this revision; falling back to ffmpeg" in w
                or "NvCodec backend unavailable" in w
                for w in warnings_nv
            ),
            f"Unexpected warnings for nvcodec backend: {warnings_nv}",
        )

        # Unknown preference raises RuntimeError
        with self.assertRaises(RuntimeError):
            create_backend("unknown_hardware_accelerator")

    def test_ffmpeg_encoder_selection_validation(self) -> None:
        backend = FfmpegSoftwareBackend()
        self.assertTrue(backend.is_available())

        # Valid software encoders
        enc, args = backend._choose_encoder(Codec.H264, "libx264")
        self.assertEqual(enc, "libx264")
        self.assertIn("-preset", args)

        enc265, args265 = backend._choose_encoder(Codec.H265, "libx265")
        self.assertEqual(enc265, "libx265")

        # Incompatible encoder for codec raises ValueError
        with self.assertRaises(ValueError):
            backend._choose_encoder(Codec.H264, "libx265")

        with self.assertRaises(ValueError):
            backend._choose_encoder(Codec.H265, "libx264")

        # Nonexistent encoder raises ValueError (if not in valid set) or RuntimeError
        with self.assertRaises(Exception):
            backend._choose_encoder(Codec.H264, "definitely_not_a_real_encoder")

    def test_ffmpeg_resolution_compatibility(self) -> None:
        backend = FfmpegSoftwareBackend()

        # Width <= 4096 allows h264_nvenc without change
        enc, args = backend._resolve_encoder_resolution_compatibility(
            codec=Codec.H264,
            requested_encoder="h264_nvenc",
            selected_encoder="h264_nvenc",
            width=3840,
        )
        self.assertEqual(enc, "h264_nvenc")
        self.assertEqual(args, [])

        # Width > 4096 (e.g. 7680 for 4K SBS) with explicit h264_nvenc raises ValueError
        with self.assertRaises(ValueError):
            backend._resolve_encoder_resolution_compatibility(
                codec=Codec.H264,
                requested_encoder="h264_nvenc",
                selected_encoder="h264_nvenc",
                width=7680,
            )

        # Width > 4096 with auto falls back to libx264
        enc_fb, args_fb = backend._resolve_encoder_resolution_compatibility(
            codec=Codec.H264,
            requested_encoder="auto",
            selected_encoder="h264_nvenc",
            width=7680,
        )
        self.assertEqual(enc_fb, "libx264")
        self.assertTrue(len(args_fb) > 0)

    def test_rate_and_bit_depth_parsers(self) -> None:
        self.assertAlmostEqual(_parse_rate("24/1"), 24.0)
        self.assertAlmostEqual(_parse_rate("30000/1001"), 29.97002997, places=4)
        self.assertAlmostEqual(_parse_rate("0/0"), 30.0)
        self.assertAlmostEqual(_parse_rate(""), 30.0)
        self.assertAlmostEqual(_parse_rate("invalid"), 30.0)

        self.assertEqual(_parse_bit_depth("yuv420p", None), 8)
        self.assertEqual(_parse_bit_depth("nv12", 0), 8)
        self.assertEqual(_parse_bit_depth("yuv420p10le", 10), 10)
        self.assertEqual(_parse_bit_depth("p010le", None), 10)
        self.assertEqual(_parse_bit_depth("yuv420p12le", 12), 12)
        self.assertEqual(_parse_bit_depth("custom_pix_fmt", None), 8)


if __name__ == "__main__":
    unittest.main()
