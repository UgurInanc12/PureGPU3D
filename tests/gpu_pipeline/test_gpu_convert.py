"""Unit and integration tests for GPU-resident NVDEC -> DA3 -> NVENC video converter."""

from __future__ import annotations

import fractions
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"
CHECKPOINT_DIR_SMALL = REPO_ROOT / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"

from puregpu3d.jobs.output_transaction import PathCollisionError
from puregpu3d.models.da3_adapter import DA3DepthAdapter
from puregpu3d.video.convert import ConversionCancelledError
from puregpu3d.video.gpu_convert import (
    GpuPipelineUnsupportedError,
    _compute_rational_timebase_and_increment,
    check_gpu_pipeline_support,
    convert_video_gpu,
)
from puregpu3d.video.probe import UnsupportedMediaError, probe_video


class TestGpuPipeline(unittest.TestCase):
    """Test suite covering GPU-resident video pipeline contracts and end-to-end execution."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        supported, reason = check_gpu_pipeline_support()
        if not supported:
            raise unittest.SkipTest(f"GPU pipeline unsupported: {reason}")
        if not SAMPLE_VIDEO.exists():
            raise unittest.SkipTest(f"Sample video not found at {SAMPLE_VIDEO}")
        if not (CHECKPOINT_DIR_SMALL / "READY").exists():
            raise unittest.SkipTest(f"DA3 Small checkpoint not ready at {CHECKPOINT_DIR_SMALL}")

        cls.adapter = DA3DepthAdapter(
            CHECKPOINT_DIR_SMALL,
            identifier="DA3-SMALL",
            device="cuda:0",
            verify_hashes=False,
        )

    def test_01_check_gpu_pipeline_support(self) -> None:
        """Verify preflight check accurately validates CUDA and PyNvVideoCodec."""
        supported, reason = check_gpu_pipeline_support(gpu_id=0)
        self.assertTrue(supported)
        self.assertIn("GPU pipeline supported", reason)

        # Invalid GPU ID check
        bad_gpu, bad_reason = check_gpu_pipeline_support(gpu_id=999)
        self.assertFalse(bad_gpu)
        self.assertIn("Invalid GPU id", bad_reason)

    def test_02_rational_timebase_and_increment_calculation(self) -> None:
        """Verify calculation of exact integer CFR increments for rational frame rates."""
        # 12 fps
        n, d, inc = _compute_rational_timebase_and_increment(fractions.Fraction(12, 1))
        self.assertEqual(n, 1)
        self.assertEqual(d, 90000)
        self.assertEqual(inc, 7500)

        # 24 fps
        n, d, inc = _compute_rational_timebase_and_increment(fractions.Fraction(24, 1))
        self.assertEqual(inc, 3750)

        # 29.97 fps (30000/1001)
        n, d, inc = _compute_rational_timebase_and_increment(fractions.Fraction(30000, 1001))
        self.assertEqual(inc, 3003)

        # Unsupported fractional rate that cannot be represented cleanly
        with self.assertRaises(UnsupportedMediaError):
            _compute_rational_timebase_and_increment(fractions.Fraction(7777777, 100000003))

    def test_03_collision_and_overwrite_guards(self) -> None:
        """Enforce strict collision rejection and explicit overwrite gate."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_out.mp4"

            # Same source and destination must be rejected
            with self.assertRaises(PathCollisionError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=SAMPLE_VIDEO,
                    model=self.adapter,
                    overwrite=True,
                )

            # Non-overwrite refusal
            out_file.touch()
            with self.assertRaises(FileExistsError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    overwrite=False,
                )

    def test_04_dlpack_pointer_contract_and_trace(self) -> None:
        """Verify DLPack zero-copy pointer verification in conversion result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_trace.mp4"
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                overwrite=True,
                max_trace_frames=3,
            )

            self.assertGreater(len(res.pointer_traces), 0)
            for trace in res.pointer_traces:
                self.assertTrue(trace["ptrs_match"], "DLPack plane_ptr and tensor_ptr must match exactly")
                self.assertEqual(trace["device"], "cuda:0")
                self.assertEqual(trace["dtype"], "torch.uint8")
                self.assertEqual(trace["shape"], [1080, 1920, 3])

    def test_05_real_depth_stereo_difference(self) -> None:
        """Verify rendered Full SBS video has real depth disparity, not duplicated eyes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_stereo.mp4"
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                overwrite=True,
            )

            # Probe video metadata
            probe = probe_video(out_file)
            self.assertEqual(probe.width, 3840)
            self.assertEqual(probe.height, 1080)
            self.assertEqual(probe.frame_count, 12)
            self.assertEqual(probe.frame_rate, fractions.Fraction(12, 1))
            self.assertTrue(probe.has_audio)
            self.assertEqual(len(probe.audio_streams), 1)

            # Verify eye disparity difference
            cap = cv2.VideoCapture(str(out_file))
            ret, frame = cap.read()
            cap.release()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)

            left_eye = frame[:, :1920]
            right_eye = frame[:, 1920:]
            diff = np.abs(left_eye.astype(np.float32) - right_eye.astype(np.float32))
            non_identical_fraction = float(np.mean(diff > 1.0))
            self.assertGreater(
                non_identical_fraction,
                0.05,
                f"Eye difference fraction {non_identical_fraction} is too low; expected real stereoscopic disparity.",
            )

    def test_06_owned_cancellation_safety(self) -> None:
        """Verify cancellation aborts cleanly and leaves no staging artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_cancel.mp4"

            # Cancel immediately
            with self.assertRaises(ConversionCancelledError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    cancel_callback=lambda: True,
                    overwrite=True,
                )

            # Ensure destination was not created
            self.assertFalse(out_file.exists())
            # Ensure no staging files left behind
            staging_files = list(Path(tmp_dir).glob(".staging_*"))
            self.assertEqual(len(staging_files), 0)

    def test_07_device_resolution_and_support_validation(self) -> None:
        """Verify device string/index is resolved upfront before preflight check."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_bad_dev.mp4"

            # Non-existent CUDA device index must fail preflight cleanly
            with self.assertRaises(GpuPipelineUnsupportedError) as ctx_bad_cuda:
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    device="cuda:8",
                )
            self.assertIn("Invalid GPU id 8", str(ctx_bad_cuda.exception))

            # Non-CUDA device must be rejected
            with self.assertRaises(GpuPipelineUnsupportedError) as ctx_cpu:
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    device="cpu",
                )
            self.assertIn("requires a CUDA device", str(ctx_cpu.exception))

    def test_08_progress_callback_cancellation_and_error_handling(self) -> None:
        """Verify progress_callback cancellation is propagated and non-cancellation errors are non-fatal."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_cb_cancel.mp4"

            def cancelling_callback(current: int, total: int) -> None:
                if current >= 2:
                    raise ConversionCancelledError("User clicked cancel on progress bar")

            with self.assertRaises(ConversionCancelledError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    depth_scale="1/4",
                    codec="hevc",
                    progress_callback=cancelling_callback,
                    overwrite=True,
                )
            self.assertFalse(out_file.exists())
            self.assertEqual(len(list(Path(tmp_dir).glob(".staging_*"))), 0)

            # Non-cancellation exception in callback should NOT crash conversion
            out_file_ok = Path(tmp_dir) / "test_cb_ok.mp4"

            def faulty_ui_callback(current: int, total: int) -> None:
                if current == 2:
                    raise RuntimeError("UI update failed transiently")

            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file_ok,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                progress_callback=faulty_ui_callback,
                overwrite=True,
            )
            self.assertTrue(out_file_ok.exists())
            self.assertEqual(res.total_frames_processed, 12)

    def test_09_actual_rational_30000_1001_fps_pts_acceptance(self) -> None:
        """Verify real fractional CFR (30000/1001 fps) video conversion and exact packet PTS ticks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            frac_in = Path(tmp_dir) / "frac_30000_1001_in.mp4"
            frac_out = Path(tmp_dir) / "frac_30000_1001_out.mp4"

            # Create a 15-frame 29.97 fps (30000/1001) fixture with audio in temp dir
            cmd_gen = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(SAMPLE_VIDEO),
                "-r", "30000/1001",
                "-frames:v", "15",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(frac_in),
            ]
            subprocess.run(cmd_gen, check=True)

            in_probe = probe_video(frac_in, strict_sdr_cfr=True)
            self.assertEqual(in_probe.frame_rate, fractions.Fraction(30000, 1001))
            self.assertEqual(in_probe.frame_count, 15)
            self.assertTrue(in_probe.has_audio)

            res = convert_video_gpu(
                input_path=frac_in,
                output_path=frac_out,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                overwrite=True,
            )

            # Probe converted media
            out_probe = probe_video(frac_out, strict_sdr_cfr=True)
            self.assertEqual(out_probe.width, 3840)
            self.assertEqual(out_probe.height, 1080)
            self.assertEqual(out_probe.frame_count, 15)
            self.assertEqual(out_probe.frame_rate, fractions.Fraction(30000, 1001))
            self.assertTrue(out_probe.has_audio)

            # Inspect packet-level PTS timestamps using ffprobe
            ffprobe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_packets",
                "-show_entries", "packet=pts,dts,duration",
                "-of", "json",
                str(frac_out),
            ]
            ffprobe_res = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
            pkts = json.loads(ffprobe_res.stdout).get("packets", [])
            self.assertEqual(len(pkts), 15)

            pts_vals = sorted(int(p["pts"]) for p in pkts if "pts" in p)
            self.assertEqual(len(pts_vals), 15)
            # Uniform tick increment for 30000/1001 at timebase 1/90000 is 3003
            for i in range(len(pts_vals) - 1):
                delta = pts_vals[i + 1] - pts_vals[i]
                self.assertEqual(
                    delta,
                    3003,
                    f"PTS delta between frame {i} and {i+1} was {delta}, expected exactly 3003 ticks.",
                )

    def test_10_telemetry_honesty(self) -> None:
        """Verify telemetry records host dispatch honestly and reports valid non-zero timings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_telemetry.mp4"
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                overwrite=True,
            )
            self.assertGreater(res.wall_clock_seconds, 0.0)
            self.assertGreater(res.effective_fps, 0.0)
            self.assertGreater(res.mean_decode_ms, 0.0)
            self.assertGreater(res.mean_depth_ms, 0.0)
            self.assertGreater(res.mean_stereo_ms, 0.0)
            self.assertGreater(res.mean_encode_ms, 0.0)

            # Check that notes honestly declare stage telemetry dispatch vs throughput
            telemetry_notes = [n for n in res.notes if "telemetry" in n.lower() or "dispatch" in n.lower()]
            self.assertGreater(len(telemetry_notes), 0)

    def test_11_audio_remux_cancellation_safety(self) -> None:
        """Verify cancellation during audio remux cleanly removes intermediate raw video and staging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_remux_cancel.mp4"

            # Cancel specifically after frames are processed (when idx >= 11, trigger cancel)
            frames_seen = 0

            def cancel_at_end() -> bool:
                nonlocal frames_seen
                return frames_seen >= 12

            def progress_tracker(curr: int, total: int) -> None:
                nonlocal frames_seen
                frames_seen = curr

            with self.assertRaises(ConversionCancelledError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    depth_scale="1/4",
                    codec="hevc",
                    progress_callback=progress_tracker,
                    cancel_callback=cancel_at_end,
                    overwrite=True,
                )

            # Neither destination nor staging files should exist
            self.assertFalse(out_file.exists())
            self.assertEqual(len(list(Path(tmp_dir).glob("*.raw_video.mp4"))), 0)
            self.assertEqual(len(list(Path(tmp_dir).glob(".staging_*"))), 0)

    def test_12_actual_temporal_on_gpu_conversion(self) -> None:
        """Verify real GPU-resident conversion with enable_temporal_stabilization=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_temporal_on.mp4"
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                enable_temporal_stabilization=True,
                overwrite=True,
            )

            # Probe media
            probe = probe_video(out_file, strict_sdr_cfr=True)
            self.assertEqual(probe.width, 3840)
            self.assertEqual(probe.height, 1080)
            self.assertEqual(probe.frame_count, 12)
            self.assertTrue(probe.has_audio)
            self.assertEqual(probe.audio_streams[0].codec_name, "aac")

            # Metrics
            self.assertEqual(res.total_frames_processed, 12)
            self.assertGreater(res.mean_temporal_ms, 0.0, "mean_temporal_ms must be non-zero when temporal is ON")
            self.assertGreater(res.effective_fps, 0.0)
            self.assertTrue(
                any("temporal depth stabilization: enabled" in n.lower() for n in res.notes),
                f"Expected temporal enabled in notes: {res.notes}",
            )

    def test_13_device_alias_canonicalization_and_mismatch(self) -> None:
        """Verify unindexed 'cuda' device alias matches 'cuda:0', while explicit mismatch fails."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_alias.mp4"

            # An unindexed input is resolved once at the adapter boundary.
            alias_adapter = DA3DepthAdapter(
                CHECKPOINT_DIR_SMALL,
                identifier="DA3-SMALL",
                device=torch.device("cuda"),
                verify_hashes=False,
            )
            self.assertEqual(alias_adapter.device.index, torch.cuda.current_device())
            self.assertEqual(alias_adapter.device.type, "cuda")

            # Must succeed with default device ("cuda:0") without device mismatch ValueError
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=alias_adapter,
                depth_scale="1/4",
                codec="hevc",
                device="cuda:0",
                overwrite=True,
            )
            self.assertTrue(out_file.exists())
            self.assertEqual(res.total_frames_processed, 12)

            # Explicit mismatch: adapter on cuda:1 vs conversion on cuda:0 must raise ValueError
            class FakeMismatchedAdapter:
                device = torch.device("cuda:1")
                def infer_tensor(self, *args, **kwargs): pass

            with self.assertRaises(ValueError) as ctx:
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=Path(tmp_dir) / "mismatch.mp4",
                    model=FakeMismatchedAdapter(),
                    device="cuda:0",
                )
            self.assertIn("does not match conversion device", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
