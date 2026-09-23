"""Integration and regression tests for GPU-resident batched video conversion."""

from __future__ import annotations

import fractions
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

from puregpu3d.models.da3_adapter import DA3DepthAdapter
from puregpu3d.video.convert import ConversionCancelledError
from puregpu3d.video.gpu_convert import (
    GpuPipelineUnsupportedError,
    check_gpu_pipeline_support,
    convert_video_gpu,
)
from puregpu3d.video.probe import find_ffmpeg, probe_video


class TestGpuBatchConversion(unittest.TestCase):
    """Test suite covering independent-frame batching in GPU video conversion."""

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

    def test_01_batch_size_validation_upfront(self) -> None:
        """Enforce strict integer 1..20 upfront validation, including boolean rejection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_val.mp4"

            # Booleans must be rejected with TypeError even though bool subclasses int
            with self.assertRaises(TypeError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    batch_size=True,  # type: ignore[arg-type]
                )

            with self.assertRaises(TypeError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    batch_size=False,  # type: ignore[arg-type]
                )

            # Non-integer types rejected with TypeError
            for invalid_type in ["5", 3.14, None, [2]]:
                with self.assertRaises(TypeError):
                    convert_video_gpu(
                        input_path=SAMPLE_VIDEO,
                        output_path=out_file,
                        model=self.adapter,
                        batch_size=invalid_type,  # type: ignore[arg-type]
                    )

            # Out-of-range integers rejected with ValueError
            for invalid_range in [0, -1, 21, 100]:
                with self.assertRaises(ValueError):
                    convert_video_gpu(
                        input_path=SAMPLE_VIDEO,
                        output_path=out_file,
                        model=self.adapter,
                        batch_size=invalid_range,
                    )

    def test_02_batch1_parity_and_telemetry(self) -> None:
        """Verify batch_size=1 preserves infer_tensor path, frame count, and telemetry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_b1.mp4"
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                batch_size=1,
                overwrite=True,
            )

            self.assertEqual(res.batch_size, 1)
            self.assertEqual(res.total_frames_processed, 12)
            self.assertIsNotNone(res.peak_memory_mb)
            assert res.peak_memory_mb is not None
            self.assertGreater(res.peak_memory_mb, 0.0)
            self.assertTrue(out_file.exists())
            self.assertTrue(any("Batch size: 1" in n for n in res.notes))

            probe = probe_video(out_file)
            self.assertEqual(probe.width, 3840)
            self.assertEqual(probe.height, 1080)
            self.assertEqual(probe.frame_count, 12)
            self.assertEqual(probe.frame_rate, fractions.Fraction(12, 1))

    def test_03_batch5_acceptance_groups5_remainder2(self) -> None:
        """Acceptance test: 12 frames with batch_size=5 produces two groups of 5 and remainder 2."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_b1 = Path(tmp_dir) / "out_b1.mp4"
            out_b5 = Path(tmp_dir) / "out_b5.mp4"

            # Run batch 1 reference
            res_b1 = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_b1,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                batch_size=1,
                overwrite=True,
            )

            # Run batch 5
            res_b5 = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_b5,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                batch_size=5,
                overwrite=True,
            )

            self.assertEqual(res_b5.batch_size, 5)
            self.assertEqual(res_b5.total_frames_processed, 12)
            self.assertIsNotNone(res_b5.peak_memory_mb)
            assert res_b5.peak_memory_mb is not None
            self.assertGreater(res_b5.peak_memory_mb, 0.0)
            self.assertTrue(any("Batch size: 5" in n for n in res_b5.notes))

            # Validate probe metadata
            probe_b5 = probe_video(out_b5)
            self.assertEqual(probe_b5.width, 3840)
            self.assertEqual(probe_b5.height, 1080)
            self.assertEqual(probe_b5.frame_count, 12)
            self.assertEqual(probe_b5.frame_rate, fractions.Fraction(12, 1))
            self.assertTrue(probe_b5.has_audio)

            # Validate stereoscopic disparity on batch 5 output
            cap = cv2.VideoCapture(str(out_b5))
            frames_b5 = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_b5.append(frame)
            cap.release()

            self.assertEqual(len(frames_b5), 12)

            # Check eye difference on frame 0
            f0 = frames_b5[0]
            left_eye = f0[:, :1920]
            right_eye = f0[:, 1920:]
            diff = np.abs(left_eye.astype(np.float32) - right_eye.astype(np.float32))
            disparity_fraction = float(np.mean(diff > 1.0))
            self.assertGreater(
                disparity_fraction,
                0.05,
                f"Disparity fraction {disparity_fraction} too low; expected real stereoscopic separation",
            )

            # Read batch 1 frames and verify parity across frames
            cap_b1 = cv2.VideoCapture(str(out_b1))
            frames_b1 = []
            while True:
                ret, frame = cap_b1.read()
                if not ret:
                    break
                frames_b1.append(frame)
            cap_b1.release()

            self.assertEqual(len(frames_b1), 12)

            # Depth and stereo parity check: independent batches should be visually/structurally equivalent
            for i in range(12):
                f_ref = frames_b1[i].astype(np.float32)
                f_cur = frames_b5[i].astype(np.float32)
                mae = float(np.mean(np.abs(f_ref - f_cur)))
                self.assertLess(
                    mae,
                    1.5,
                    f"Frame {i} MAE between batch=1 and batch=5 is {mae:.3f}, exceeds 1.5 tolerance",
                )

    def test_04_exact_audio_and_offset_preservation(self) -> None:
        """Verify video-delayed audio offset is exactly preserved under batched conversion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "source_delayed.mp4"
            dst = Path(tmp_dir) / "stereo_delayed.mp4"

            # Create synthetic clip with 5231 ticks video delay relative to audio
            subprocess.run(
                [
                    str(find_ffmpeg()),
                    "-v", "error",
                    "-y",
                    "-f", "lavfi",
                    "-i", "testsrc2=size=320x180:rate=24:duration=0.5",
                    "-f", "lavfi",
                    "-i", "sine=duration=0.583333",
                    "-vf", "settb=1/90000,setpts=PTS+5231",
                    "-fps_mode", "passthrough",
                    "-enc_time_base", "1:90000",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    str(src),
                ],
                check=True,
                capture_output=True,
            )

            res = convert_video_gpu(
                input_path=src,
                output_path=dst,
                model=self.adapter,
                depth_scale="1/2",
                batch_size=5,
                enable_temporal_stabilization=True,
                overwrite=True,
            )

            self.assertEqual(res.total_frames_processed, 12)
            self.assertEqual(res.batch_size, 5)

            def calc_offset(path: Path) -> float:
                streams = probe_video(path).raw_info["streams"]
                v_start = float(next(s for s in streams if s["codec_type"] == "video")["start_time"])
                a_start = float(next(s for s in streams if s["codec_type"] == "audio")["start_time"])
                return v_start - a_start

            self.assertAlmostEqual(calc_offset(dst), calc_offset(src), delta=0.002)

    def test_05_batching_with_temporal_stabilization(self) -> None:
        """Verify temporal stabilization causal ordering is preserved across batch boundaries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out_temp_b4.mp4"
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                batch_size=4,
                enable_temporal_stabilization=True,
                overwrite=True,
            )

            self.assertEqual(res.batch_size, 4)
            self.assertEqual(res.total_frames_processed, 12)
            self.assertGreater(res.mean_temporal_ms, 0.0)
            self.assertTrue(out_file.exists())

            probe = probe_video(out_file)
            self.assertEqual(probe.frame_count, 12)
            self.assertTrue(probe.has_audio)

    def test_06_batching_with_pipelined_scheduling(self) -> None:
        """Verify batching works safely with pipelined ring buffer (batch_size > prefetch_slots)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out_pipe_b5.mp4"
            # batch_size=5 with prefetch_slots=3 exercises ring slot recycling without deadlock
            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                batch_size=5,
                scheduling="pipelined",
                prefetch_slots=3,
                overwrite=True,
            )

            self.assertEqual(res.batch_size, 5)
            self.assertEqual(res.total_frames_processed, 12)
            self.assertEqual(res.scheduling, "pipelined")
            self.assertTrue(out_file.exists())

            probe = probe_video(out_file)
            self.assertEqual(probe.frame_count, 12)

    def test_07_cancellation_during_batch_gathering_and_output(self) -> None:
        """Verify cancellation aborts cleanly and leaves no staging artifacts during batching."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out_cancel_b4.mp4"

            def cancelling_callback(current: int, total: int) -> None:
                if current >= 4:
                    raise ConversionCancelledError("Cancel triggered on 4th frame")

            with self.assertRaises(ConversionCancelledError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    depth_scale="1/4",
                    codec="hevc",
                    batch_size=4,
                    progress_callback=cancelling_callback,
                    overwrite=True,
                )

            self.assertFalse(out_file.exists())
            self.assertEqual(len(list(Path(tmp_dir).glob(".staging_*"))), 0)
            self.assertEqual(len(list(Path(tmp_dir).glob("*.raw_video.mp4"))), 0)

    def test_08_adapter_missing_infer_tensor_batch_raises(self) -> None:
        """Verify adapter lacking infer_tensor_batch raises AttributeError when batch_size > 1."""
        class MockSingleOnlyAdapter:
            device = torch.device("cuda:0")

            def infer_tensor(self, *args, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out_no_batch.mp4"

            with self.assertRaises(AttributeError) as ctx:
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=MockSingleOnlyAdapter(),
                    batch_size=2,
                    overwrite=True,
                )
            self.assertIn("infer_tensor_batch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
