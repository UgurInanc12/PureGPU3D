"""Unit tests for conservative background-aware hole filling.

Verifies:
  - Demonstrable background side preference:
      When left is background (z_L > z_R), fill strictly from left (no foreground bleed).
      When right is background (z_R > z_L), fill strictly from right (no foreground bleed).
  - Equal-depth interpolation:
      When boundary depths match within tolerance, linearly interpolate across hole.
  - Edge extension:
      Border holes fill conservatively from the available boundary.
  - Raw coverage preservation:
      coverage_before_fill accurately reflects pre-fill coverage state.
  - Large-hole diagnostics:
      Detects and counts holes exceeding max_hole_width, recording warning message.
      Supports leaving large holes unfilled when fill_large_holes=False.
  - CPU vs CUDA numerical parity.
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

from puregpu3d.stereo.fill import (
    FillConfig,
    FillResult,
    HoleDiagnostics,
    fill_holes,
)


class TestHoleFilling(unittest.TestCase):
    """Test suite for conservative hole filling logic."""

    def test_left_background_selection(self) -> None:
        """Verify hole is filled strictly from left when left is demonstrable background."""
        # 1 row, 6 columns:
        # Col 0, 1: Background (depth=3.0, blue [0, 0, 1])
        # Col 2, 3: Hole (uncovered)
        # Col 4, 5: Foreground (depth=1.0, red [1, 0, 0])
        h, w = 1, 6
        color = torch.zeros((3, h, w), dtype=torch.float32)
        color[:, 0, 0] = torch.tensor([0.0, 0.0, 1.0])
        color[:, 0, 1] = torch.tensor([0.0, 0.0, 1.0])
        color[:, 0, 4] = torch.tensor([1.0, 0.0, 0.0])
        color[:, 0, 5] = torch.tensor([1.0, 0.0, 0.0])

        depth = torch.tensor([[3.0, 3.0, float("inf"), float("inf"), 1.0, 1.0]], dtype=torch.float32)
        cov = torch.tensor([[True, True, False, False, True, True]], dtype=torch.bool)

        cfg = FillConfig(max_hole_width=8, depth_relative_tolerance=0.05)
        res = fill_holes(color, depth, cov, cfg)

        # Holes at col 2 and 3 must receive blue [0, 0, 1] from background, NOT red [1, 0, 0]
        expected_blue = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(res.color[:, 0, 2], expected_blue)
        torch.testing.assert_close(res.color[:, 0, 3], expected_blue)

        # Filled depth must match background depth (3.0)
        self.assertAlmostEqual(res.depth[0, 2].item(), 3.0, places=4)
        self.assertAlmostEqual(res.depth[0, 3].item(), 3.0, places=4)

        # Coverage before fill remains False at cols 2, 3
        self.assertFalse(res.coverage_before_fill[0, 2].item())
        self.assertFalse(res.coverage_before_fill[0, 3].item())

        # filled_mask is True at cols 2, 3
        self.assertTrue(res.filled_mask[0, 2].item())
        self.assertTrue(res.filled_mask[0, 3].item())

    def test_right_background_selection(self) -> None:
        """Verify hole is filled strictly from right when right is demonstrable background."""
        # 1 row, 6 columns:
        # Col 0, 1: Foreground (depth=1.0, red [1, 0, 0])
        # Col 2, 3: Hole (uncovered)
        # Col 4, 5: Background (depth=4.0, green [0, 1, 0])
        h, w = 1, 6
        color = torch.zeros((3, h, w), dtype=torch.float32)
        color[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        color[:, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
        color[:, 0, 4] = torch.tensor([0.0, 1.0, 0.0])
        color[:, 0, 5] = torch.tensor([0.0, 1.0, 0.0])

        depth = torch.tensor([[1.0, 1.0, float("inf"), float("inf"), 4.0, 4.0]], dtype=torch.float32)
        cov = torch.tensor([[True, True, False, False, True, True]], dtype=torch.bool)

        cfg = FillConfig(max_hole_width=8, depth_relative_tolerance=0.05)
        res = fill_holes(color, depth, cov, cfg)

        # Holes at col 2 and 3 must receive green [0, 1, 0] from background, NOT red [1, 0, 0]
        expected_green = torch.tensor([0.0, 1.0, 0.0])
        torch.testing.assert_close(res.color[:, 0, 2], expected_green)
        torch.testing.assert_close(res.color[:, 0, 3], expected_green)

        self.assertAlmostEqual(res.depth[0, 2].item(), 4.0, places=4)
        self.assertAlmostEqual(res.depth[0, 3].item(), 4.0, places=4)

    def test_equal_depth_linear_interpolation(self) -> None:
        """Verify smooth linear interpolation when boundary depths are approximately equal."""
        # 1 row, 4 columns:
        # Col 0: Black [0, 0, 0], depth 2.0
        # Col 1, 2: Hole
        # Col 3: White [1, 1, 1], depth 2.0
        h, w = 1, 4
        color = torch.zeros((3, h, w), dtype=torch.float32)
        color[:, 0, 3] = 1.0

        depth = torch.tensor([[2.0, float("inf"), float("inf"), 2.0]], dtype=torch.float32)
        cov = torch.tensor([[True, False, False, True]], dtype=torch.bool)

        cfg = FillConfig(max_hole_width=8)
        res = fill_holes(color, depth, cov, cfg)

        # Span from 0 to 3: length 3
        # At col 1: t = 1/3 ~ 0.333
        # At col 2: t = 2/3 ~ 0.667
        self.assertAlmostEqual(res.color[0, 0, 1].item(), 1.0 / 3.0, places=4)
        self.assertAlmostEqual(res.color[0, 0, 2].item(), 2.0 / 3.0, places=4)
        self.assertAlmostEqual(res.depth[0, 1].item(), 2.0, places=4)
        self.assertAlmostEqual(res.depth[0, 2].item(), 2.0, places=4)

    def test_border_hole_background_extension(self) -> None:
        """Verify holes reaching image boundary extend conservatively from the sole boundary."""
        # Left edge hole: cols 0, 1 uncovered, col 2 covered with depth 2.5, color [0.5, 0.5, 0.5]
        h, w = 1, 4
        color = torch.zeros((3, h, w), dtype=torch.float32)
        color[:, 0, 2:] = 0.5
        depth = torch.tensor([[float("inf"), float("inf"), 2.5, 2.5]], dtype=torch.float32)
        cov = torch.tensor([[False, False, True, True]], dtype=torch.bool)

        res = fill_holes(color, depth, cov)

        torch.testing.assert_close(res.color[:, 0, 0], torch.tensor([0.5, 0.5, 0.5]))
        torch.testing.assert_close(res.color[:, 0, 1], torch.tensor([0.5, 0.5, 0.5]))
        self.assertAlmostEqual(res.depth[0, 0].item(), 2.5, places=4)

    def test_large_hole_detection_and_warning(self) -> None:
        """Verify holes wider than max_hole_width trigger explicit diagnostic warnings."""
        h, w = 1, 25
        color = torch.ones((3, h, w), dtype=torch.float32)
        depth = torch.full((h, w), 2.0, dtype=torch.float32)
        cov = torch.ones((h, w), dtype=torch.bool)

        # Create a hole of width 18 pixels (cols 2 to 19 inclusive: 18 px)
        cov[0, 2:20] = False

        cfg = FillConfig(max_hole_width=10, fill_large_holes=True)
        res = fill_holes(color, depth, cov, cfg)

        self.assertEqual(res.diagnostics.max_hole_width, 18)
        self.assertEqual(res.diagnostics.large_hole_count, 1)
        self.assertIsNotNone(res.diagnostics.warning)
        assert res.diagnostics.warning is not None
        self.assertIn("18px exceeds threshold 10px", res.diagnostics.warning)

        # With fill_large_holes=False, large hole should remain in unfilled_mask
        cfg_no_fill = FillConfig(max_hole_width=10, fill_large_holes=False)
        res_no_fill = fill_holes(color, depth, cov, cfg_no_fill)

        self.assertTrue(torch.all(res_no_fill.unfilled_mask[0, 2:20]))
        self.assertFalse(torch.any(res_no_fill.filled_mask[0, 2:20]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_fill_parity(self) -> None:
        """Verify identical hole filling results on CPU and CUDA."""
        h, w = 4, 16
        color_cpu = torch.rand((3, h, w), dtype=torch.float32)
        depth_cpu = torch.rand((h, w), dtype=torch.float32) * 3.0 + 1.0
        cov_cpu = torch.ones((h, w), dtype=torch.bool)
        cov_cpu[:, 4:7] = False  # Narrow holes

        color_cuda = color_cpu.cuda()
        depth_cuda = depth_cpu.cuda()
        cov_cuda = cov_cpu.cuda()

        res_cpu = fill_holes(color_cpu, depth_cpu, cov_cpu)
        res_cuda = fill_holes(color_cuda, depth_cuda, cov_cuda)

        torch.testing.assert_close(res_cuda.color.cpu(), res_cpu.color, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(res_cuda.depth.cpu(), res_cpu.depth, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(res_cuda.filled_mask.cpu(), res_cpu.filled_mask)


if __name__ == "__main__":
    unittest.main()
