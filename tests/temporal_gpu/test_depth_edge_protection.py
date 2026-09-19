"""Unit tests verifying edge sharpness preservation and depth discontinuity protection on GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.temporal_gpu import TemporalGPUConfig, TemporalGPUStabilizer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU temporal tests")
class TestGPUDepthDiscontinuity(unittest.TestCase):
    """Verify depth edges are not blurred and disoccluded depths do not leave trails."""

    def setUp(self) -> None:
        self.device = torch.device("cuda:0")

    def test_sharp_boundary_gradient_preservation(self) -> None:
        """Verify that step depth discontinuity retains high gradient and is not smoothed out."""
        h, w = 120, 160

        # Background: depth 5.0 (far), Foreground: depth 1.0 (near) in [30:90, 40:100]
        f1_rgb = torch.full((3, h, w), 80.0, device=self.device, dtype=torch.float32)
        f1_rgb[:, 30:90, 40:100] = 200.0

        d1 = torch.full((h, w), 5.0, device=self.device, dtype=torch.float32)
        d1[30:90, 40:100] = 1.0

        # Frame 2: foreground moves right by 4 pixels to [30:90, 44:104]
        f2_rgb = torch.full((3, h, w), 80.0, device=self.device, dtype=torch.float32)
        f2_rgb[:, 30:90, 44:104] = 200.0

        d2 = torch.full((h, w), 5.0, device=self.device, dtype=torch.float32)
        d2[30:90, 44:104] = 1.0

        cfg = TemporalGPUConfig(alpha=0.75, edge_threshold=0.15)
        stabilizer = TemporalGPUStabilizer(cfg)

        stabilizer.process_frame(f1_rgb, d1)
        r2 = stabilizer.process_frame(f2_rgb, d2)
        stab_depth = r2.depth

        self.assertEqual(stab_depth.device.type, "cuda")

        # Check disoccluded column x=41 (which was foreground in f1, but background in f2):
        # It must NOT retain foreground depth (1.0) or intermediate depth (e.g. 2.5);
        # it must equal background depth (5.0)!
        disoccluded_depth = stab_depth[50:70, 41]
        torch.testing.assert_close(
            disoccluded_depth,
            torch.full_like(disoccluded_depth, 5.0),
            rtol=0.05,
            atol=0.05,
        )

        # Check foreground interior: x=60, depth must be near 1.0
        fg_depth = stab_depth[50:70, 60]
        torch.testing.assert_close(
            fg_depth,
            torch.full_like(fg_depth, 1.0),
            rtol=0.05,
            atol=0.05,
        )

        # Measure horizontal gradient at the leading boundary (x=103 to 105):
        # Step must remain steep (drop >= 3.5 over 2 pixels)
        step_height = (stab_depth[60, 105] - stab_depth[60, 103]).item()
        self.assertGreaterEqual(step_height, 3.5, "Depth boundary must remain crisp and not smoothed")


if __name__ == "__main__":
    unittest.main()
