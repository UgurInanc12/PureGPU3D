"""Unit and integration tests for independent-frame DA3 batch tensor inference.

Verifies:
  - infer_tensor_batch API existence, signature, and return types.
  - Strict preservation of batch=1 numerical parity against infer_tensor.
  - Multi-batch GPU inference across batches 1, 2, 5, and remainder sizes (e.g. 7, 13, 20).
  - Proof of zero cross-sample leakage via neighbor permutation and neighbor modification.
  - Support for varied input layouts: List[Tensor], 4D BCHW, 4D BHWC.
  - Input validation and error handling (empty batch, device mismatch, shape mismatch, multi-view S>1 guard).
  - Clamping, scalar statistics, and geometry preservation.
  - Honest OOM behavior (no silent scale reduction).
"""

from __future__ import annotations

import gc
import sys
import unittest
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import (
    DEFAULT_REVISION,
    DA3DepthAdapter,
    DepthTensorResult,
)
from puregpu3d.models.geometry import compute_depth_geometry

CHECKPOINT_DIR_SMALL = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION
SAMPLE_IMAGE = REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"


class TestDA3BatchTensorInference(unittest.TestCase):
    """Test batch tensor inference with DA3DepthAdapter on CUDA."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for GPU batch inference tests")
        if not CHECKPOINT_DIR_SMALL.exists():
            raise unittest.SkipTest(f"DA3-SMALL weights not found at {CHECKPOINT_DIR_SMALL}")

        cls.adapter = DA3DepthAdapter(
            CHECKPOINT_DIR_SMALL,
            identifier="DA3-SMALL",
            device="cuda:0",
            verify_hashes=False,
        )

        bgr = cv2.imread(str(SAMPLE_IMAGE))
        if bgr is None:
            rgb = np.random.randint(0, 255, (252, 252, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cls.sample_rgb_hwc = torch.from_numpy(rgb).cuda()
        cls.orig_h, cls.orig_w = rgb.shape[:2]

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "adapter"):
            del cls.adapter
        if hasattr(cls, "sample_rgb_hwc"):
            del cls.sample_rgb_hwc
        gc.collect()
        torch.cuda.empty_cache()

    def test_infer_tensor_batch_single_parity(self) -> None:
        """Verify infer_tensor_batch with B=1 produces bitwise-identical output as infer_tensor."""
        single_res = self.adapter.infer_tensor(
            self.sample_rgb_hwc,
            depth_scale="1/2",
            return_original_size=True,
            autocast=True,
            timing=False,
        )

        batch_res_list = self.adapter.infer_tensor_batch(
            [self.sample_rgb_hwc],
            depth_scale="1/2",
            return_original_size=True,
            autocast=True,
            timing=False,
        )

        self.assertIsInstance(batch_res_list, list)
        self.assertEqual(len(batch_res_list), 1)
        batch_res = batch_res_list[0]
        self.assertIsInstance(batch_res, DepthTensorResult)

        # Exact bitwise parity on CUDA
        self.assertTrue(torch.equal(single_res.depth, batch_res.depth))
        self.assertTrue(torch.equal(single_res.depth_raw, batch_res.depth_raw))
        self.assertEqual(single_res.input_shape, batch_res.input_shape)
        self.assertEqual(single_res.processed_shape, batch_res.processed_shape)
        self.assertEqual(single_res.depth_units, batch_res.depth_units)
        self.assertEqual(single_res.is_metric, batch_res.is_metric)

    def test_no_cross_sample_leakage_on_neighbor_change(self) -> None:
        """Prove that changing neighbor frame B does not alter frame A depth (0.0 difference)."""
        torch.manual_seed(100)
        h, w = 140, 140
        # Synthetic distinct frames with explicit labels
        frame_a = torch.rand((h, w, 3), device=self.adapter.device, dtype=torch.float32)
        frame_b1 = torch.rand((h, w, 3), device=self.adapter.device, dtype=torch.float32)
        frame_b2 = torch.rand((h, w, 3), device=self.adapter.device, dtype=torch.float32)

        # Batch 1: [frame_a, frame_b1]
        res_ab1 = self.adapter.infer_tensor_batch(
            [frame_a, frame_b1],
            depth_scale="1/2",
            return_original_size=True,
            autocast=False,  # Float32 for exact precision check
            timing=False,
        )

        # Batch 2: [frame_a, frame_b2] (frame_b changed)
        res_ab2 = self.adapter.infer_tensor_batch(
            [frame_a, frame_b2],
            depth_scale="1/2",
            return_original_size=True,
            autocast=False,
            timing=False,
        )

        diff_depth = (res_ab1[0].depth - res_ab2[0].depth).abs().max().item()
        diff_raw = (res_ab1[0].depth_raw - res_ab2[0].depth_raw).abs().max().item()

        # Proof of zero cross-talk across independent batch samples
        self.assertEqual(diff_depth, 0.0)
        self.assertEqual(diff_raw, 0.0)

    def test_no_cross_sample_leakage_on_permutation(self) -> None:
        """Prove that permuting batch order produces identical per-frame depth (0.0 difference)."""
        torch.manual_seed(200)
        h, w = 140, 140
        frame_a = torch.rand((h, w, 3), device=self.adapter.device, dtype=torch.float32)
        frame_b = torch.rand((h, w, 3), device=self.adapter.device, dtype=torch.float32)
        frame_c = torch.rand((h, w, 3), device=self.adapter.device, dtype=torch.float32)

        # Forward order: [A, B, C]
        res_abc = self.adapter.infer_tensor_batch(
            [frame_a, frame_b, frame_c],
            depth_scale="1/2",
            return_original_size=True,
            autocast=False,
            timing=False,
        )

        # Permuted order: [C, A, B]
        res_cab = self.adapter.infer_tensor_batch(
            [frame_c, frame_a, frame_b],
            depth_scale="1/2",
            return_original_size=True,
            autocast=False,
            timing=False,
        )

        # Frame A: index 0 in abc, index 1 in cab
        diff_a = (res_abc[0].depth - res_cab[1].depth).abs().max().item()
        # Frame B: index 1 in abc, index 2 in cab
        diff_b = (res_abc[1].depth - res_cab[2].depth).abs().max().item()
        # Frame C: index 2 in abc, index 0 in cab
        diff_c = (res_abc[2].depth - res_cab[0].depth).abs().max().item()

        self.assertEqual(diff_a, 0.0)
        self.assertEqual(diff_b, 0.0)
        self.assertEqual(diff_c, 0.0)

    def test_batches_1_2_5_and_remainders(self) -> None:
        """Execute GPU inference across batches 1, 2, 5, and remainder sizes (3, 7, 13, 20)."""
        batch_sizes = [1, 2, 5, 3, 7, 13, 20]
        h, w = self.orig_h, self.orig_w

        for b in batch_sizes:
            frames = [self.sample_rgb_hwc for _ in range(b)]
            results = self.adapter.infer_tensor_batch(
                frames,
                depth_scale="1/2",
                return_original_size=True,
                autocast=True,
                timing=True,
            )

            self.assertEqual(len(results), b, f"Failed length check for batch size {b}")
            for idx, res in enumerate(results):
                self.assertTrue(res.depth.is_cuda, f"Result {idx} depth not on CUDA for batch {b}")
                self.assertTrue(res.depth_raw.is_cuda, f"Result {idx} depth_raw not on CUDA for batch {b}")
                self.assertEqual(res.depth.shape, (h, w))
                self.assertGreater(res.latency_ms, 0.0)

    def test_input_layout_variations(self) -> None:
        """Verify List[HWC], List[CHW], List[BCHW], and 4D tensor (B, 3, H, W) produce identical outputs."""
        torch.manual_seed(300)
        b, h, w = 3, 140, 140
        # Base 4D tensor (B, 3, H, W)
        t_bchw = torch.rand((b, 3, h, w), device=self.adapter.device, dtype=torch.float32)

        # 4D tensor (B, H, W, 3)
        t_bhwc = t_bchw.permute(0, 2, 3, 1).contiguous()

        # List of 3D (H, W, 3)
        list_hwc = [t_bhwc[i] for i in range(b)]

        # List of 3D (3, H, W)
        list_chw = [t_bchw[i] for i in range(b)]

        # List of single-frame 4D (1, 3, H, W)
        list_bchw = [t_bchw[i : i + 1] for i in range(b)]

        res_from_4d = self.adapter.infer_tensor_batch(t_bchw, depth_scale="1/2", autocast=False, timing=False)
        res_from_bhwc = self.adapter.infer_tensor_batch(t_bhwc, depth_scale="1/2", autocast=False, timing=False)
        res_from_list_hwc = self.adapter.infer_tensor_batch(list_hwc, depth_scale="1/2", autocast=False, timing=False)
        res_from_list_chw = self.adapter.infer_tensor_batch(list_chw, depth_scale="1/2", autocast=False, timing=False)
        res_from_list_bchw = self.adapter.infer_tensor_batch(list_bchw, depth_scale="1/2", autocast=False, timing=False)

        for i in range(b):
            self.assertTrue(torch.equal(res_from_4d[i].depth, res_from_bhwc[i].depth))
            self.assertTrue(torch.equal(res_from_4d[i].depth, res_from_list_hwc[i].depth))
            self.assertTrue(torch.equal(res_from_4d[i].depth, res_from_list_chw[i].depth))
            self.assertTrue(torch.equal(res_from_4d[i].depth, res_from_list_bchw[i].depth))

    def test_scale_and_geometry_semantics(self) -> None:
        """Verify geometry preservation and return_original_size=False across scales."""
        scales = ["1/4", "1/2", "1/1"]
        h, w = self.orig_h, self.orig_w

        for s in scales:
            geom = compute_depth_geometry(w, h, scale=s)
            results = self.adapter.infer_tensor_batch(
                [self.sample_rgb_hwc, self.sample_rgb_hwc],
                depth_scale=s,
                return_original_size=False,
                timing=False,
            )

            for res in results:
                self.assertIsNotNone(res.geometry)
                assert res.geometry is not None
                self.assertEqual(res.geometry.scale, s)
                # When return_original_size=False, depth shape is requested size
                self.assertEqual(res.depth.shape, (geom.req_height, geom.req_width))
                self.assertEqual(res.depth_raw.shape, (geom.req_height, geom.req_width))

    def test_clamping_and_stats(self) -> None:
        """Verify ensure_positive clamping and compute_stats execution."""
        results = self.adapter.infer_tensor_batch(
            [self.sample_rgb_hwc, self.sample_rgb_hwc],
            depth_scale="1/2",
            ensure_positive=True,
            compute_stats=True,
            validate_finite=True,
            timing=False,
        )

        for res in results:
            self.assertTrue((res.depth >= 1e-6).all().item())
            self.assertTrue((res.depth_raw >= 1e-6).all().item())
            self.assertIsNotNone(res.min_depth)
            self.assertIsNotNone(res.max_depth)
            self.assertIsNotNone(res.mean_depth)
            assert res.min_depth is not None
            self.assertGreaterEqual(res.min_depth, 1e-6)

    def test_input_validation_errors(self) -> None:
        """Verify strict rejection of invalid input shapes, types, and device mismatches."""
        # Empty batch
        with self.assertRaises(ValueError):
            self.adapter.infer_tensor_batch([])

        # Non-tensor item
        with self.assertRaises(TypeError):
            self.adapter.infer_tensor_batch(["not_a_tensor"])  # type: ignore[arg-type]

        # CPU tensor rejection when adapter is CUDA
        cpu_t = torch.zeros((100, 100, 3), dtype=torch.uint8, device="cpu")
        with self.assertRaises(ValueError) as ctx:
            self.adapter.infer_tensor_batch([cpu_t])
        self.assertIn("does not match adapter device", str(ctx.exception))

        # Spatial resolution mismatch
        t1 = torch.zeros((100, 100, 3), dtype=torch.uint8, device=self.adapter.device)
        t2 = torch.zeros((120, 120, 3), dtype=torch.uint8, device=self.adapter.device)
        with self.assertRaises(ValueError) as ctx:
            self.adapter.infer_tensor_batch([t1, t2])
        self.assertIn("identical spatial dimensions", str(ctx.exception))

        # Multi-view sequence guard: 5D tensor with S > 1 must be rejected to prevent cross-frame leakage
        s_multi = torch.zeros((2, 3, 3, 100, 100), device=self.adapter.device)
        with self.assertRaises(ValueError) as ctx:
            self.adapter.infer_tensor_batch(s_multi)
        self.assertIn("sequence length S", str(ctx.exception))

    def test_honest_oom_no_scale_reduction(self) -> None:
        """Verify that out-of-memory errors propagate honestly without silent scale reduction."""
        from unittest.mock import patch

        # Simulate an OutOfMemoryError from the underlying model forward pass
        with patch.object(self.adapter, "model", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                self.adapter.infer_tensor_batch(
                    [self.sample_rgb_hwc, self.sample_rgb_hwc],
                    depth_scale="1/1",
                )


if __name__ == "__main__":
    unittest.main()
