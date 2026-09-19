"""Unit tests for occlusion-aware subpixel forward splatting.

Verifies:
  - Depth ordering collision: near objects win collisions over background regardless of traversal order
  - Background is rejected before blending (no ghosting or averaging near and far)
  - Subpixel behavior: non-integer shifts partition weights linearly (1-alpha, alpha)
  - True black content: pixels with color [0, 0, 0] are valid covered content, never treated as holes
  - Zero shift short-circuit: exact tensor preservation
  - Tiny shapes: 1x1, 2x3 shapes process without boundary exceptions
  - CPU vs CUDA numerical equivalence
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.splat import (
    SplatConfig,
    SplatViewResult,
    forward_splat,
    validate_image_tensor,
)


class TestForwardSplat(unittest.TestCase):
    """Test suite for subpixel forward splatting and occlusion rejection."""

    def test_depth_ordering_collision_near_wins(self) -> None:
        """Verify near objects win collisions over background without color averaging."""
        # 1 row, 4 columns:
        # Col 0: background (depth=3.0, blue: [0, 0, 1])
        # Col 1: foreground (depth=1.0, red: [1, 0, 0])
        # Both shift so they collide at target pixel x_tgt = 2:
        # Col 0 shifts +2.0 -> lands at target 2
        # Col 1 shifts +1.0 -> lands at target 2
        h, w = 1, 4
        image = torch.zeros((3, h, w), dtype=torch.float32)
        image[:, 0, 0] = torch.tensor([0.0, 0.0, 1.0])  # Blue background
        image[:, 0, 1] = torch.tensor([1.0, 0.0, 0.0])  # Red foreground

        depth = torch.tensor([[3.0, 1.0, 2.0, 2.0]], dtype=torch.float32)
        shift = torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        cfg = SplatConfig(depth_tolerance=0.05)
        res = forward_splat(image, depth, shift, cfg)

        # Target pixel 2 received both splats. Near surface (depth 1.0) must win!
        # Background (depth 3.0) must be rejected prior to blending.
        target_color = res.color[:, 0, 2]
        expected_color = torch.tensor([1.0, 0.0, 0.0])  # Pure red

        torch.testing.assert_close(target_color, expected_color, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(res.depth[0, 2].item(), 1.0, places=4)
        self.assertTrue(res.coverage_mask[0, 2].item())

    def test_collision_order_independence(self) -> None:
        """Verify collision resolution is invariant to source column traversal order."""
        # Reverse column layout:
        # Col 2: foreground (depth=1.0, red: [1, 0, 0]), shifts -1.0 -> lands at 1
        # Col 3: background (depth=4.0, green: [0, 1, 0]), shifts -2.0 -> lands at 1
        h, w = 1, 4
        image = torch.zeros((3, h, w), dtype=torch.float32)
        image[:, 0, 2] = torch.tensor([1.0, 0.0, 0.0])  # Red foreground
        image[:, 0, 3] = torch.tensor([0.0, 1.0, 0.0])  # Green background

        depth = torch.tensor([[2.0, 2.0, 1.0, 4.0]], dtype=torch.float32)
        shift = torch.tensor([[0.0, 0.0, -1.0, -2.0]], dtype=torch.float32)

        res = forward_splat(image, depth, shift)

        # Foreground red must win target pixel 1
        target_color = res.color[:, 0, 1]
        expected_color = torch.tensor([1.0, 0.0, 0.0])
        torch.testing.assert_close(target_color, expected_color, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(res.depth[0, 1].item(), 1.0, places=4)

    def test_subpixel_weight_partition(self) -> None:
        """Verify subpixel shift partitions weights linearly into x0 and x1."""
        # 1 row, 2 columns: source pixel at x=0 shifts by +0.25
        # Col 1 shifts off-screen so target receives only contributions from x=0
        h, w = 1, 2
        image = torch.zeros((3, h, w), dtype=torch.float32)
        image[:, 0, 0] = 1.0
        depth = torch.full((h, w), 2.0, dtype=torch.float32)

        # Shift x=0 by +0.25 -> x_tgt = 0.25 -> x0=0 (w0=0.75), x1=1 (w1=0.25)
        # Shift x=1 off-screen (+10.0)
        shift = torch.tensor([[0.25, 10.0]], dtype=torch.float32)

        res = forward_splat(image, depth, shift)

        # Pixel 0 should receive weight 0.75, pixel 1 should receive weight 0.25
        self.assertAlmostEqual(res.accum_weights[0, 0].item(), 0.75, places=5)
        self.assertAlmostEqual(res.accum_weights[0, 1].item(), 0.25, places=5)

        # Normalized colors remain 1.0 at both covered locations
        self.assertAlmostEqual(res.color[0, 0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(res.color[0, 0, 1].item(), 1.0, places=5)

    def test_black_content_is_valid_not_hole(self) -> None:
        """Verify true black pixels [0, 0, 0] have valid coverage and are not marked as holes."""
        h, w = 2, 3
        # All-black image
        image = torch.zeros((3, h, w), dtype=torch.float32)
        depth = torch.full((h, w), 1.5, dtype=torch.float32)
        shift = torch.zeros((h, w), dtype=torch.float32)

        res = forward_splat(image, depth, shift)

        # Even though RGB is strictly 0, coverage_mask must be all True!
        self.assertTrue(torch.all(res.coverage_mask))
        self.assertTrue(torch.all(res.accum_weights >= 1.0))
        self.assertTrue(torch.all(res.color == 0.0))

    def test_zero_shift_exact_preservation(self) -> None:
        """Verify shift=0 produces exact tensor preservation."""
        torch.manual_seed(42)
        image = torch.rand((3, 4, 6), dtype=torch.float32)
        depth = torch.rand((4, 6), dtype=torch.float32) + 0.5
        shift = torch.zeros((4, 6), dtype=torch.float32)

        res = forward_splat(image, depth, shift)

        torch.testing.assert_close(res.color, image)
        torch.testing.assert_close(res.depth, depth)
        self.assertTrue(torch.all(res.coverage_mask))

    def test_tiny_shapes(self) -> None:
        """Verify splatting works on tiny shapes (1x1, 2x3)."""
        # 1x1
        img1 = torch.tensor([[[0.5]], [[0.3]], [[0.8]]], dtype=torch.float32)
        depth1 = torch.tensor([[1.2]], dtype=torch.float32)
        shift1 = torch.tensor([[0.0]], dtype=torch.float32)
        res1 = forward_splat(img1, depth1, shift1)
        self.assertEqual(res1.color.shape, (3, 1, 1))
        self.assertTrue(res1.coverage_mask[0, 0].item())

        # 2x3 with non-zero shift
        img2 = torch.rand((3, 2, 3), dtype=torch.float32)
        depth2 = torch.full((2, 3), 1.0, dtype=torch.float32)
        shift2 = torch.tensor([[0.5, 0.0, -0.5], [0.0, 0.2, 0.0]], dtype=torch.float32)
        res2 = forward_splat(img2, depth2, shift2)
        self.assertEqual(res2.color.shape, (3, 2, 3))
        self.assertEqual(res2.depth.shape, (2, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_splat_parity(self) -> None:
        """Verify identical splatting results between CPU and CUDA."""
        torch.manual_seed(123)
        h, w = 8, 12
        img_cpu = torch.rand((3, h, w), dtype=torch.float32)
        depth_cpu = torch.rand((h, w), dtype=torch.float32) * 3.0 + 0.5
        shift_cpu = (torch.rand((h, w), dtype=torch.float32) - 0.5) * 2.0

        img_cuda = img_cpu.cuda()
        depth_cuda = depth_cpu.cuda()
        shift_cuda = shift_cpu.cuda()

        res_cpu = forward_splat(img_cpu, depth_cpu, shift_cpu)
        res_cuda = forward_splat(img_cuda, depth_cuda, shift_cuda)

        torch.testing.assert_close(res_cuda.coverage_mask.cpu(), res_cpu.coverage_mask)
        torch.testing.assert_close(res_cuda.color.cpu(), res_cpu.color, atol=1e-5, rtol=1e-5)
        # Compare depth where covered
        cov = res_cpu.coverage_mask
        torch.testing.assert_close(res_cuda.depth.cpu()[cov], res_cpu.depth[cov], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
