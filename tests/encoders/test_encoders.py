"""Tests for NVENC encoder selection, preflight validation, rate control, and fallback."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"
for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.video.convert import ConversionError, convert_video
from puregpu3d.video.encoders import (
    DEFAULT_CQ,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_NVENC_PRESET,
    DEFAULT_X264_PRESET,
    ENCODER_AUTO,
    ENCODER_H264_NVENC,
    ENCODER_HEVC_NVENC,
    ENCODER_LIBX264,
    VALID_ENCODERS,
    EncoderPreflightError,
    InvalidEncoderError,
    PreflightResult,
    preflight_encoder,
    resolve_encoder,
    validate_encoder_name,
)
from puregpu3d.video.probe import find_ffmpeg, probe_video


def get_video_codec(probe_result) -> Optional[str]:
    """Helper to extract video stream codec_name from probe_result."""
    for s in probe_result.raw_info.get("streams", []):
        if s.get("codec_type") == "video":
            return s.get("codec_name")
    return None


class MockDepthOutput:
    def __init__(self, h: int, w: int) -> None:
        self.depth = np.ones((h, w), dtype=np.float32) * 5.0
        self.depth_raw = np.ones((h, w), dtype=np.float32) * 5.0


class MockDepthModel:
    def infer(self, image: np.ndarray, **kwargs) -> MockDepthOutput:
        h, w = image.shape[:2]
        return MockDepthOutput(h, w)


class TestEncoderSelectionUnit(unittest.TestCase):
    """Unit tests for encoder validation, rate-control contracts, and injected failures."""

    def test_valid_encoder_names(self) -> None:
        for name in ("auto", "hevc_nvenc", "h264_nvenc", "libx264"):
            self.assertEqual(validate_encoder_name(name), name)
            self.assertEqual(validate_encoder_name(name.upper()), name)
            self.assertEqual(validate_encoder_name(f"  {name}  "), name)

    def test_invalid_encoder_names_rejected(self) -> None:
        for invalid in ("prores", "vp9", "av1", "", "   ", "nvenc"):
            with self.assertRaises(InvalidEncoderError):
                validate_encoder_name(invalid)

        with self.assertRaises(InvalidEncoderError):
            validate_encoder_name(None)  # type: ignore

    def test_crf_and_cq_bounds(self) -> None:
        # Invalid CRF
        with self.assertRaises(ValueError):
            resolve_encoder("libx264", width=3840, height=1080, crf=-1)
        with self.assertRaises(ValueError):
            resolve_encoder("libx264", width=3840, height=1080, crf=52)

        # Invalid CQ
        with self.assertRaises(ValueError):
            resolve_encoder("libx264", width=3840, height=1080, cq=-1)
        with self.assertRaises(ValueError):
            resolve_encoder("libx264", width=3840, height=1080, cq=52)

    def test_libx264_contract(self) -> None:
        res = resolve_encoder(
            "libx264",
            width=3840,
            height=1080,
            crf=20,
            x264_preset="fast",
        )
        self.assertEqual(res.selected_encoder, ENCODER_LIBX264)
        self.assertEqual(res.codec_name, "h264")
        self.assertFalse(res.is_hardware)
        self.assertIsNone(res.fallback_reason)
        # Verify CRF is used, CQ is not in libx264 args
        self.assertIn("-c:v", res.ffmpeg_args)
        self.assertIn("libx264", res.ffmpeg_args)
        self.assertIn("-crf", res.ffmpeg_args)
        self.assertIn("20", res.ffmpeg_args)
        self.assertIn("-preset", res.ffmpeg_args)
        self.assertIn("fast", res.ffmpeg_args)
        self.assertNotIn("-cq", res.ffmpeg_args)
        self.assertNotIn("-rc", res.ffmpeg_args)

    def test_nvenc_rate_control_cq_contract(self) -> None:
        """Verify NVENC uses VBR CQ and preset, not CRF."""
        with patch("puregpu3d.video.encoders.preflight_encoder") as mock_pf:
            mock_pf.return_value = PreflightResult(
                encoder=ENCODER_HEVC_NVENC,
                width=3840,
                height=1080,
                pix_fmt="yuv420p",
                supported=True,
                duration_s=0.05,
            )
            res = resolve_encoder(
                "hevc_nvenc",
                width=3840,
                height=1080,
                cq=25,
                nvenc_preset="p5",
            )
            self.assertEqual(res.selected_encoder, ENCODER_HEVC_NVENC)
            self.assertEqual(res.codec_name, "hevc")
            self.assertTrue(res.is_hardware)
            # Verify CQ in VBR mode
            self.assertIn("-c:v", res.ffmpeg_args)
            self.assertIn("hevc_nvenc", res.ffmpeg_args)
            self.assertIn("-rc", res.ffmpeg_args)
            self.assertIn("vbr", res.ffmpeg_args)
            self.assertIn("-cq", res.ffmpeg_args)
            self.assertIn("25", res.ffmpeg_args)
            self.assertIn("-b:v", res.ffmpeg_args)
            self.assertIn("0", res.ffmpeg_args)
            self.assertIn("-preset", res.ffmpeg_args)
            self.assertIn("p5", res.ffmpeg_args)
            self.assertNotIn("-crf", res.ffmpeg_args)

    def test_injected_manual_nvenc_preflight_failure_rejects(self) -> None:
        """Manual selection of NVENC must fail hard if preflight fails; no silent fallback or downscale."""
        with patch("puregpu3d.video.encoders.preflight_encoder") as mock_pf:
            mock_pf.return_value = PreflightResult(
                encoder=ENCODER_HEVC_NVENC,
                width=3840,
                height=1080,
                pix_fmt="yuv420p",
                supported=False,
                error_message="NVENC initialization failed: out of memory",
            )
            with self.assertRaises(EncoderPreflightError) as ctx:
                resolve_encoder("hevc_nvenc", width=3840, height=1080)
            self.assertIn("Manual selection rejected", str(ctx.exception))
            self.assertIn("out of memory", str(ctx.exception))

    def test_injected_auto_fallback_to_h264_nvenc(self) -> None:
        """Auto mode falls back to h264_nvenc if hevc_nvenc preflight fails."""
        def mock_preflight(enc, width, height, **kwargs):
            if enc == ENCODER_HEVC_NVENC:
                return PreflightResult(
                    encoder=enc,
                    width=width,
                    height=height,
                    pix_fmt="yuv420p",
                    supported=False,
                    error_message="HEVC NVENC unsupported",
                )
            return PreflightResult(
                encoder=enc,
                width=width,
                height=height,
                pix_fmt="yuv420p",
                supported=True,
                duration_s=0.04,
            )

        with patch("puregpu3d.video.encoders.preflight_encoder", side_effect=mock_preflight):
            res = resolve_encoder("auto", width=3840, height=1080)
            self.assertEqual(res.selected_encoder, ENCODER_H264_NVENC)
            self.assertTrue(res.is_hardware)
            self.assertIsNotNone(res.fallback_reason)
            self.assertIn("fell back to h264_nvenc", str(res.fallback_reason))

    def test_injected_auto_fallback_to_libx264(self) -> None:
        """Auto mode falls back to software libx264 if all NVENC preflights fail."""
        with patch("puregpu3d.video.encoders.preflight_encoder") as mock_pf:
            mock_pf.return_value = PreflightResult(
                encoder="mock",
                width=3840,
                height=1080,
                pix_fmt="yuv420p",
                supported=False,
                error_message="No NVENC capable GPU found",
            )
            res = resolve_encoder("auto", width=3840, height=1080)
            self.assertEqual(res.selected_encoder, ENCODER_LIBX264)
            self.assertFalse(res.is_hardware)
            self.assertIsNotNone(res.fallback_reason)
            self.assertIn("fell back to software libx264", str(res.fallback_reason))


class TestEncoderRealHardware(unittest.TestCase):
    """Hardware tests executing actual NVENC preflight and conversions on GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available; skipping NVENC hardware tests.")
        cls.ffmpeg = find_ffmpeg()

        # Check if real NVENC preflight passes on this system
        pf = preflight_encoder(ENCODER_HEVC_NVENC, width=3840, height=1080, ffmpeg_path=cls.ffmpeg)
        if not pf.supported:
            raise unittest.SkipTest(f"HEVC NVENC preflight unsupported on this machine: {pf.error_message}")

    def _create_synthetic_clip(self, path: Path, fps: int = 12, frames: int = 6) -> Path:
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size=1920x1080:rate={fps}",
            "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=44100",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-frames:v", str(frames),
            str(path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0, f"FFmpeg fixture generation failed: {res.stderr.decode('utf-8', 'replace')}")
        return path

    def test_real_preflight_success(self) -> None:
        pf_hevc = preflight_encoder(ENCODER_HEVC_NVENC, width=3840, height=1080, ffmpeg_path=self.ffmpeg)
        self.assertTrue(pf_hevc.supported, f"HEVC preflight failed: {pf_hevc.error_message}")
        self.assertGreater(pf_hevc.duration_s, 0.0)

        pf_h264 = preflight_encoder(ENCODER_H264_NVENC, width=3840, height=1080, ffmpeg_path=self.ffmpeg)
        self.assertTrue(pf_h264.supported, f"H264 preflight failed: {pf_h264.error_message}")

    def test_real_preflight_excessive_dimensions_fails(self) -> None:
        """Dimensions exceeding hardware capabilities must return supported=False."""
        pf = preflight_encoder(ENCODER_HEVC_NVENC, width=32768, height=1080, ffmpeg_path=self.ffmpeg)
        self.assertFalse(pf.supported)
        self.assertIsNotNone(pf.error_message)

    def test_real_convert_hevc_nvenc(self) -> None:
        """End-to-end convert with hevc_nvenc produces verified HEVC FullSBS MP4 with audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "in.mp4"
            dst = Path(tmpdir) / "out_hevc.mp4"
            self._create_synthetic_clip(src, fps=12, frames=6)

            model = MockDepthModel()
            res = convert_video(
                input_path=src,
                output_path=dst,
                model=model,
                device="cuda",
                encoder="hevc_nvenc",
                cq=23,
                enable_temporal_stabilization=False,
            )

            self.assertEqual(res.encoder, "hevc_nvenc")
            self.assertEqual(res.output_width, 3840)
            self.assertEqual(res.output_height, 1080)
            self.assertEqual(res.total_frames_processed, 6)
            self.assertTrue(res.has_audio)

            # Probe destination with ffprobe
            probe_out = probe_video(dst)
            self.assertEqual(probe_out.width, 3840)
            self.assertEqual(probe_out.height, 1080)
            self.assertEqual(probe_out.frame_count, 6)
            self.assertEqual(get_video_codec(probe_out), "hevc")
            self.assertTrue(probe_out.has_audio)
            self.assertEqual(probe_out.audio_streams[0].codec_name, "aac")

    def test_real_convert_h264_nvenc(self) -> None:
        """End-to-end convert with h264_nvenc produces verified H264 FullSBS MP4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "in.mp4"
            dst = Path(tmpdir) / "out_h264.mp4"
            self._create_synthetic_clip(src, fps=12, frames=6)

            model = MockDepthModel()
            res = convert_video(
                input_path=src,
                output_path=dst,
                model=model,
                device="cuda",
                encoder="h264_nvenc",
                cq=23,
                enable_temporal_stabilization=False,
            )

            self.assertEqual(res.encoder, "h264_nvenc")
            self.assertEqual(res.output_width, 3840)
            self.assertEqual(res.output_height, 1080)

            probe_out = probe_video(dst)
            self.assertEqual(get_video_codec(probe_out), "h264")
            self.assertEqual(probe_out.width, 3840)
            self.assertEqual(probe_out.height, 1080)

    def test_real_convert_auto_resolves_to_hevc_nvenc(self) -> None:
        """On this RTX 3090 system, auto resolves to hevc_nvenc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "in.mp4"
            dst = Path(tmpdir) / "out_auto.mp4"
            self._create_synthetic_clip(src, fps=12, frames=6)

            model = MockDepthModel()
            res = convert_video(
                input_path=src,
                output_path=dst,
                model=model,
                device="cuda",
                encoder="auto",
                enable_temporal_stabilization=False,
            )

            self.assertEqual(res.encoder, "hevc_nvenc")
            probe_out = probe_video(dst)
            self.assertEqual(get_video_codec(probe_out), "hevc")

    def test_manual_invalid_selection_rejected_in_convert(self) -> None:
        """convert_video rejects unknown encoder upfront with InvalidEncoderError / ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "in.mp4"
            dst = Path(tmpdir) / "out.mp4"
            self._create_synthetic_clip(src, fps=12, frames=4)

            model = MockDepthModel()
            with self.assertRaises(ValueError):
                convert_video(
                    input_path=src,
                    output_path=dst,
                    model=model,
                    encoder="unsupported_codec",
                )


if __name__ == "__main__":
    unittest.main()
