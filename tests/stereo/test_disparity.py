"""Unit tests for signed depth-to-disparity computation.

Verifies:
  - Known near/far sign convention: d = x_left - x_right, near positive (d > 0), far negative (d < 0)
  - Convergence plane: depth matching q_screen produces exactly zero disparity (x_left = x_right = x)
  - Disparity clamping against maximum/minimum limits
  - Invalid depth rejection (NaN, Inf, negative, zero, non-floating-point types)
  - Zero strength short-circuit (all disparities and shifts are exactly zero)
  - CPU vs CUDA numerical parity
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Ensure repository src is reachable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.disparity import (
    DisparityConfig,
    DisparityResult,
    compute_disparity,
    validate_depth_tensor,
)


class TestDisparity(unittest.TestCase):
    """Test suite for disparity calculation logic and invariants."""

    def test_known_near_far_sign_convention(self) -> None:
        """Verify d = x_left - x_right convention with independently derived values."""
        # 1 row, 2 columns: column 0 is far (z=4.0), column 1 is near (z=1.0)
        depth = torch.tensor([[4.0, 1.0]], dtype=torch.float32)
        cfg = DisparityConfig(
            strength=0.05,
            q_screen=0.5,
            percentile_min=0.0,
            percentile_max=100.0,
            max_disparity_fraction=0.10,
        )

        res = compute_disparity(depth, cfg)

        # Independent hand derivation:
        # Width W = 2
        # inv_z: [0.25, 1.0]
        # q_low = 0.25, q_high = 1.0, denom = 0.75
        # q: col 0 = 0.0, col 1 = 1.0
        # raw_d:
        #   col 0 (far):  0.05 * 2 * (0.0 - 0.5) = -0.05
        #   col 1 (near): 0.05 * 2 * (1.0 - 0.5) = +0.05
        expected_d_far = -0.05
        expected_d_near = +0.05

        self.assertAlmostEqual(res.disparity[0, 0].item(), expected_d_far, places=5)
        self.assertAlmostEqual(res.disparity[0, 1].item(), expected_d_near, places=5)

        # Near point has d > 0 (near positive)
        self.assertGreater(res.disparity[0, 1].item(), 0.0)
        # Far point has d < 0 (far negative)
        self.assertLess(res.disparity[0, 0].item(), 0.0)

        # Left shift = +d / 2, Right shift = -d / 2
        # So x_left - x_right = (+d/2) - (-d/2) = d
        diff = res.shift_left - res.shift_right
        torch.testing.assert_close(diff, res.disparity)

    def test_convergence_plane(self) -> None:
        """Verify depth sitting exactly at q_screen produces zero parallax."""
        # Setup depths where col 0 is far (inv_z=0.25), col 2 is near (inv_z=1.0)
        # and col 1 has inv_z = 0.625 -> z = 1 / 0.625 = 1.6
        depth = torch.tensor([[4.0, 1.6, 1.0]], dtype=torch.float32)
        cfg = DisparityConfig(
            strength=0.04,
            q_screen=0.5,
            percentile_min=0.0,
            percentile_max=100.0,
        )

        res = compute_disparity(depth, cfg)

        # Col 1 q value is (0.625 - 0.25) / 0.75 = 0.5 == q_screen
        self.assertAlmostEqual(res.q[0, 1].item(), 0.5, places=5)
        self.assertAlmostEqual(res.disparity[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(res.shift_left[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(res.shift_right[0, 1].item(), 0.0, places=6)

    def test_disparity_clamping(self) -> None:
        """Verify disparity clamps at specified max_disparity_fraction."""
        depth = torch.tensor([[100.0, 0.01]], dtype=torch.float32)
        cfg = DisparityConfig(
            strength=0.50,  # Deliberately huge strength
            q_screen=0.5,
            percentile_min=0.0,
            percentile_max=100.0,
            max_disparity_fraction=0.04,  # Cap at 4% of width
        )

        w = depth.shape[1]
        res = compute_disparity(depth, cfg)

        max_allowed = 0.04 * w
        min_allowed = -0.04 * w

        self.assertLessEqual(res.d_max, max_allowed + 1e-6)
        self.assertGreaterEqual(res.d_min, min_allowed - 1e-6)

    def test_zero_strength(self) -> None:
        """Verify strength=0.0 produces exact zero disparity and shifts."""
        depth = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        cfg = DisparityConfig(strength=0.0)

        res = compute_disparity(depth, cfg)
        self.assertEqual(res.d_min, 0.0)
        self.assertEqual(res.d_max, 0.0)
        self.assertTrue(torch.all(res.disparity == 0.0))
        self.assertTrue(torch.all(res.shift_left == 0.0))
        self.assertTrue(torch.all(res.shift_right == 0.0))

    def test_invalid_depth_rejection(self) -> None:
        """Verify defensive rejection of invalid depths (NaN, Inf, negative, zero, non-float)."""
        # Non-float dtype
        with self.assertRaises(TypeError):
            validate_depth_tensor(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))

        # NaN
        with self.assertRaises(ValueError):
            validate_depth_tensor(torch.tensor([[1.0, float("nan")]], dtype=torch.float32))

        # Inf
        with self.assertRaises(ValueError):
            validate_depth_tensor(torch.tensor([[1.0, float("inf")]], dtype=torch.float32))

        # Negative depth
        with self.assertRaises(ValueError):
            validate_depth_tensor(torch.tensor([[1.0, -0.5]], dtype=torch.float32))

        # Zero depth
        with self.assertRaises(ValueError):
            validate_depth_tensor(torch.tensor([[1.0, 0.0]], dtype=torch.float32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_disparity_parity(self) -> None:
        """Verify numerical equivalence between CPU and CUDA executions."""
        depth_cpu = torch.tensor([[2.5, 1.2, 0.8, 3.4], [1.1, 4.0, 0.5, 2.0]], dtype=torch.float32)
        depth_cuda = depth_cpu.cuda()

        cfg = DisparityConfig(strength=0.03, q_screen=0.6)

        res_cpu = compute_disparity(depth_cpu, cfg)
        res_cuda = compute_disparity(depth_cuda, cfg)

        torch.testing.assert_close(res_cuda.disparity.cpu(), res_cpu.disparity, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(res_cuda.q.cpu(), res_cpu.q, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(res_cuda.shift_left.cpu(), res_cpu.shift_left, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(res_cuda.shift_right.cpu(), res_cpu.shift_right, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
