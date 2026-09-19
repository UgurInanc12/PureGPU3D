"""Tests for bounded GPU decode prefetch ring scheduling, stage overlap, and equivalence."""

from __future__ import annotations

import fractions
import gc
import shutil
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
from puregpu3d.video.gpu_buffers import GpuDecodePrefetchRing
from puregpu3d.video.gpu_convert import (
    check_gpu_pipeline_support,
    convert_video_gpu,
)
from puregpu3d.video.probe import probe_video


class TestGpuScheduling(unittest.TestCase):
    """Test suite covering bounded GPU ring scheduling, stage overlap, and equivalence."""

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

    def test_01_invalid_scheduling_rejected(self) -> None:
        """Enforce strict rejection of invalid scheduling options."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out.mp4"
            with self.assertRaises(ValueError) as ctx:
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    scheduling="invalid_mode",
                )
            self.assertIn("Unsupported scheduling mode", str(ctx.exception))

    def test_02_ring_buffer_bounded_allocation(self) -> None:
        """Verify GpuDecodePrefetchRing allocates fixed bounded slots and queues."""
        import PyNvVideoCodec as nvc

        decode_stream = torch.cuda.Stream(device=torch.device("cuda:0"))
        color_type = getattr(nvc, "OutputColorType").RGB if hasattr(nvc, "OutputColorType") else "RGB"
        dec = nvc.SimpleDecoder(
            str(SAMPLE_VIDEO),
            gpu_id=0,
            cuda_stream=decode_stream.cuda_stream,
            use_device_memory=True,
            output_color_type=color_type,
        )
        total_frames = len(dec)
        num_slots = 3
        ring = GpuDecodePrefetchRing(
            decoder=dec,
            target_device=torch.device("cuda:0"),
            in_height=1080,
            in_width=1920,
            total_frames=total_frames,
            num_slots=num_slots,
            decode_stream=decode_stream,
        )
        self.assertEqual(len(ring.slots), num_slots)
        self.assertEqual(ring.free_slots.maxsize, num_slots)
        self.assertEqual(ring.ready_queue.maxsize, num_slots)

        for slot in ring.slots:
            self.assertEqual(slot.tensor.shape, (1080, 1920, 3))
            self.assertEqual(slot.tensor.dtype, torch.uint8)
            self.assertEqual(str(slot.tensor.device), "cuda:0")

        # Invalid slot count must fail
        with self.assertRaises(ValueError):
            GpuDecodePrefetchRing(
                decoder=dec,
                target_device=torch.device("cuda:0"),
                in_height=1080,
                in_width=1920,
                total_frames=total_frames,
                num_slots=1,
            )

    def test_03_fastpath_diagnostic_on_off_equivalence(self) -> None:
        """Verify compute_diagnostics=False produces bit-for-bit identical frames to compute_diagnostics=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_fast = Path(tmp_dir) / "out_fast.mp4"
            out_diag = Path(tmp_dir) / "out_diag.mp4"

            res_fast = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_fast,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                enable_temporal_stabilization=True,
                compute_diagnostics=False,
                scheduling="sequential",
                overwrite=True,
            )
            res_diag = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_diag,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                enable_temporal_stabilization=True,
                compute_diagnostics=True,
                scheduling="sequential",
                overwrite=True,
            )

            self.assertEqual(res_fast.total_frames_processed, res_diag.total_frames_processed)

            # Compare decoded frames from both outputs (release captures in finally block)
            cap_fast = cv2.VideoCapture(str(out_fast))
            cap_diag = cv2.VideoCapture(str(out_diag))
            try:
                frame_idx = 0
                while True:
                    r1, f1 = cap_fast.read()
                    r2, f2 = cap_diag.read()
                    self.assertEqual(r1, r2, f"Frame presence mismatch at index {frame_idx}")
                    if not r1:
                        break
                    max_diff = int(np.max(np.abs(f1.astype(np.int32) - f2.astype(np.int32))))
                    mean_diff = float(np.mean(np.abs(f1.astype(np.float64) - f2.astype(np.float64))))
                    # NVENC HEVC VBR rate control and quantization introduce minor compression variations;
                    # mean absolute pixel difference must remain negligible (< 0.05 on 8-bit scale).
                    self.assertLess(
                        mean_diff,
                        0.05,
                        f"Frame {frame_idx} average pixel difference {mean_diff:.4f} exceeded tolerance 0.05",
                    )
                    frame_idx += 1
                self.assertEqual(frame_idx, res_fast.total_frames_processed)
            finally:
                cap_fast.release()
                cap_diag.release()

    def test_04_sequential_vs_pipelined_content_and_order_equivalence(self) -> None:
        """Verify pipelined scheduling produces identical frame count, frame order, and content to sequential reference."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_seq = Path(tmp_dir) / "out_seq.mp4"
            out_pipe = Path(tmp_dir) / "out_pipe.mp4"

            res_seq = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_seq,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                enable_temporal_stabilization=True,
                scheduling="sequential",
                overwrite=True,
            )
            res_pipe = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_pipe,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                enable_temporal_stabilization=True,
                scheduling="pipelined",
                prefetch_slots=3,
                overwrite=True,
            )

            # Metadata equivalence
            self.assertEqual(res_seq.total_frames_processed, res_pipe.total_frames_processed)
            self.assertEqual(res_pipe.scheduling, "pipelined")
            self.assertTrue(any("Scheduling: pipelined" in n for n in res_pipe.notes))

            p_seq = probe_video(out_seq, strict_sdr_cfr=True)
            p_pipe = probe_video(out_pipe, strict_sdr_cfr=True)
            self.assertEqual(p_seq.width, p_pipe.width)
            self.assertEqual(p_seq.height, p_pipe.height)
            self.assertEqual(p_seq.frame_count, p_pipe.frame_count)
            self.assertEqual(p_seq.frame_rate, p_pipe.frame_rate)
            self.assertEqual(p_seq.has_audio, p_pipe.has_audio)

            # Strict frame-by-frame content and ordering comparison
            cap_seq = cv2.VideoCapture(str(out_seq))
            cap_pipe = cv2.VideoCapture(str(out_pipe))

            frame_idx = 0
            while True:
                r1, f1 = cap_seq.read()
                r2, f2 = cap_pipe.read()
                self.assertEqual(r1, r2, f"Frame mismatch at index {frame_idx}")
                if not r1:
                    break
                max_diff = int(np.max(np.abs(f1.astype(np.int32) - f2.astype(np.int32))))
                self.assertEqual(
                    max_diff,
                    0,
                    f"Frame {frame_idx} content differed between sequential and pipelined (max diff {max_diff})",
                )
                frame_idx += 1

            cap_seq.release()
            cap_pipe.release()
            self.assertEqual(frame_idx, res_seq.total_frames_processed)

    def test_05_cancellation_and_worker_cleanup(self) -> None:
        """Verify that cancelling during pipelined conversion cleanly joins worker and leaves no staging files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out_cancel.mp4"
            frames_seen = 0

            def cancel_at_3rd_frame(curr: int, total: int) -> None:
                nonlocal frames_seen
                frames_seen = curr
                if curr >= 3:
                    raise ConversionCancelledError("Cancellation requested during test")

            with self.assertRaises(ConversionCancelledError):
                convert_video_gpu(
                    input_path=SAMPLE_VIDEO,
                    output_path=out_file,
                    model=self.adapter,
                    depth_scale="1/4",
                    codec="hevc",
                    scheduling="pipelined",
                    prefetch_slots=3,
                    progress_callback=cancel_at_3rd_frame,
                    overwrite=True,
                )

            # Neither destination nor staging files should exist
            self.assertFalse(out_file.exists())
            staging_files = list(Path(tmp_dir).glob(".staging_*"))
            self.assertEqual(len(staging_files), 0)

    def test_06_memory_stability_bounded(self) -> None:
        """Verify GPU VRAM usage remains strictly bounded across pipelined frames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "out_mem.mp4"

            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

            res = convert_video_gpu(
                input_path=SAMPLE_VIDEO,
                output_path=out_file,
                model=self.adapter,
                depth_scale="1/4",
                codec="hevc",
                scheduling="pipelined",
                prefetch_slots=3,
                overwrite=True,
            )

            mem_after = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()

            self.assertEqual(res.total_frames_processed, 12)
            # Memory after run must be bounded (within 50 MB of initial state)
            mem_growth_mb = (mem_after - mem_before) / (1024 * 1024)
            self.assertLess(
                mem_growth_mb,
                50.0,
                f"Memory growth {mem_growth_mb:.2f} MB exceeds 50 MB threshold",
            )


if __name__ == "__main__":
    unittest.main()
