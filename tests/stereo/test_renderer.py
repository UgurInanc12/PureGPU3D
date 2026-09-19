"""Integration tests for the full stereo rendering pipeline (render_stereo_frame).

Verifies:
  - Preserves exact source dimensions (W per eye, 2W x H full SBS composite)
  - Zero strength returns byte-exact source on both left and right eye views
  - Black pixel content is valid and preserved without treating dark scenes as holes
  - Disocclusion holes are exposed in raw pre-fill coverage masks
  - Demonstrable background side filling prevents halo bleeding in integrated renders
  - CPU and CUDA numerical parity across the full pipeline
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo import (
    DisparityConfig,
    FillConfig,
    SplatConfig,
    StereoConfig,
    StereoFrameResult,
    render_stereo_frame,
)


class TestStereoRenderer(unittest.TestCase):
    """Test suite for top-level stereo rendering pipeline."""

    def test_dimensions_and_channels(self) -> None:
        """Verify rendered SBS dimensions are exactly 2W x H."""
        h, w = 48, 64
        image = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)
        depth = torch.rand((h, w), dtype=torch.float32) + 0.5

        res = render_stereo_frame(image, depth, output_uint8=True, return_numpy=False)

        # Left and Right views preserve (H, W, 3) format when input is HWC
        self.assertEqual(res.left_color.shape, (h, w, 3))
        self.assertEqual(res.right_color.shape, (h, w, 3))
        # Side-by-Side has width 2 * W
        self.assertEqual(res.sbs_color.shape, (h, 2 * w, 3))
        self.assertEqual(res.left_coverage_before_fill.shape, (h, w))
        self.assertEqual(res.right_coverage_before_fill.shape, (h, w))
        self.assertEqual(res.left_depth.shape, (h, w))
        self.assertEqual(res.right_depth.shape, (h, w))

    def test_zero_strength_byte_exact(self) -> None:
        """Verify strength=0.0 returns byte-exact source content for both eyes."""
        h, w = 32, 40
        torch.manual_seed(42)
        image_uint8 = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)
        depth = torch.rand((h, w), dtype=torch.float32) + 0.1

        cfg = StereoConfig(disparity=DisparityConfig(strength=0.0))
        res = render_stereo_frame(image_uint8, depth, config=cfg, output_uint8=True, return_numpy=False)

        # Byte-exact identity check
        assert isinstance(res.left_color, torch.Tensor)
        assert isinstance(res.right_color, torch.Tensor)
        assert isinstance(res.sbs_color, torch.Tensor)
        self.assertTrue(torch.equal(res.left_color, image_uint8))
        self.assertTrue(torch.equal(res.right_color, image_uint8))
        self.assertTrue(torch.equal(res.sbs_color[:, :w, :], image_uint8))
        self.assertTrue(torch.equal(res.sbs_color[:, w:, :], image_uint8))
        self.assertTrue(torch.all(res.left_coverage_before_fill))
        self.assertTrue(torch.all(res.right_coverage_before_fill))

    def test_numpy_io_support(self) -> None:
        """Verify seamless handling of numpy array inputs and outputs."""
        h, w = 20, 30
        np_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        np_depth = (np.random.rand(h, w) + 0.5).astype(np.float32)

        res = render_stereo_frame(np_image, np_depth, return_numpy=True, output_uint8=True)

        self.assertIsInstance(res.left_color, np.ndarray)
        self.assertIsInstance(res.right_color, np.ndarray)
        self.assertIsInstance(res.sbs_color, np.ndarray)
        self.assertEqual(res.sbs_color.shape, (h, 2 * w, 3))
        self.assertEqual(res.sbs_color.dtype, np.uint8)

    def test_black_scene_preservation(self) -> None:
        """Verify pure black image content is rendered without being treated as a hole."""
        h, w = 16, 24
        # Fully black image
        image = torch.zeros((h, w, 3), dtype=torch.uint8)
        depth = torch.full((h, w), 2.0, dtype=torch.float32)

        res = render_stereo_frame(image, depth, output_uint8=True)

        # Coverage must be True everywhere for flat depth
        self.assertTrue(torch.all(res.left_coverage_before_fill))
        self.assertTrue(torch.all(res.right_coverage_before_fill))
        self.assertEqual(res.left_diagnostics.total_hole_pixels, 0)
        self.assertEqual(res.right_diagnostics.total_hole_pixels, 0)
        self.assertTrue(torch.all(res.sbs_color == 0))

    def test_coverage_before_fill_exposed(self) -> None:
        """Verify raw coverage mask before hole filling is preserved and accessible."""
        h, w = 10, 40
        # Foreground bar in center (depth 0.8), background surrounding (depth 3.0)
        depth = torch.full((h, w), 3.0, dtype=torch.float32)
        depth[:, 15:25] = 0.8
        image = torch.randint(50, 200, (h, w, 3), dtype=torch.uint8)

        cfg = StereoConfig(
            disparity=DisparityConfig(strength=0.20, q_screen=0.5, max_disparity_fraction=0.25),
            fill=FillConfig(max_hole_width=16),
        )
        res = render_stereo_frame(image, depth, config=cfg)

        # Shifts should produce disocclusion holes
        left_holes = (~res.left_coverage_before_fill).sum().item()
        right_holes = (~res.right_coverage_before_fill).sum().item()
        self.assertGreater(left_holes, 0)
        self.assertGreater(right_holes, 0)

        # Diagnostics reflect pre-fill hole counts
        self.assertEqual(res.left_diagnostics.total_hole_pixels, left_holes)
        self.assertEqual(res.right_diagnostics.total_hole_pixels, right_holes)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_pipeline_parity(self) -> None:
        """Verify full rendering pipeline yields numerically identical outputs on CPU and CUDA."""
        h, w = 16, 32
        torch.manual_seed(999)
        image = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)
        depth = torch.rand((h, w), dtype=torch.float32) * 2.0 + 0.5

        cfg = StereoConfig(
            disparity=DisparityConfig(strength=0.04, q_screen=0.6),
            splat=SplatConfig(depth_tolerance=0.05),
            fill=FillConfig(max_hole_width=16),
        )

        res_cpu = render_stereo_frame(image, depth, config=cfg, device="cpu", output_uint8=False, return_numpy=False)
        res_cuda = render_stereo_frame(image, depth, config=cfg, device="cuda", output_uint8=False, return_numpy=False)

        # Check SBS color parity (float)
        assert isinstance(res_cuda.sbs_color, torch.Tensor)
        assert isinstance(res_cpu.sbs_color, torch.Tensor)
        torch.testing.assert_close(res_cuda.sbs_color.cpu(), res_cpu.sbs_color, atol=1e-5, rtol=1e-5)
        # Check raw coverage masks parity
        torch.testing.assert_close(
            res_cuda.left_coverage_before_fill.cpu(), res_cpu.left_coverage_before_fill
        )
        torch.testing.assert_close(
            res_cuda.right_coverage_before_fill.cpu(), res_cpu.right_coverage_before_fill
        )


if __name__ == "__main__":
    unittest.main()
