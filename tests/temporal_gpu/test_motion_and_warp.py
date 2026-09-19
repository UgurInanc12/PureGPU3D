"""Unit tests for GPU motion estimation, warping accuracy, consistency, and photometric rejection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.temporal_gpu import (
    TemporalGPUConfig,
    TemporalGPUStabilizer,
    _estimate_multiscale_flow_gpu,
    _warp_2d_gpu,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU temporal tests")
class TestGPUMotionAndWarp(unittest.TestCase):
    """Verify GPU motion estimation direction, warping, occlusion rejection, and photometric gating."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")

        # Create rich multi-frequency textured 2D canvas on CUDA
        h_canv, w_canv = 300, 360
        cy, cx = torch.meshgrid(
            torch.arange(h_canv, device=self.device, dtype=torch.float32),
            torch.arange(w_canv, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        pattern = (
            128.0
            + 50.0 * torch.sin(cx / 4.0) * torch.cos(cy / 4.0)
            + 40.0 * torch.sin(cx / 10.0 + cy / 8.0)
            + 30.0 * torch.cos(cx / 3.0)
        )
        self.canvas = torch.clamp(pattern, 0.0, 255.0)

    def test_flow_direction_and_warp_accuracy(self) -> None:
        """Verify that estimated backward flow has the correct sign and aligns translated textures."""
        h, w = 180, 240
        dx_gt, dy_gt = 4, 2  # object/camera translates by (+4, +2)

        # f1 is crop at (origin_y, origin_x)
        y0, x0 = 50, 50
        crop1 = self.canvas[y0 : y0 + h, x0 : x0 + w]
        # In frame 2, camera moves so window shifts by (+dy, +dx)
        crop2 = self.canvas[y0 + dy_gt : y0 + dy_gt + h, x0 + dx_gt : x0 + dx_gt + w]

        f1_rgb = crop1.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        f2_rgb = crop2.unsqueeze(0).repeat(3, 1, 1)

        d1 = torch.full((h, w), 2.5, device=self.device, dtype=torch.float32)
        d2 = torch.full((h, w), 2.5, device=self.device, dtype=torch.float32)

        cfg = TemporalGPUConfig(flow_scale=1.0, alpha=0.7)
        stabilizer = TemporalGPUStabilizer(cfg)

        r1 = stabilizer.process_frame(f1_rgb, d1)
        self.assertTrue(r1.is_cut)
        self.assertEqual(r1.depth.device.type, "cuda")

        r2 = stabilizer.process_frame(f2_rgb, d2)
        self.assertFalse(r2.is_cut)
        self.assertEqual(r2.depth.device.type, "cuda")

        # In unoccluded interior, motion compensation should be valid
        self.assertGreater(r2.valid_flow_fraction, 0.65)

        # Directly test estimated flow sign
        c1_norm = crop1.unsqueeze(0).unsqueeze(0) / 255.0
        c2_norm = crop2.unsqueeze(0).unsqueeze(0) / 255.0

        # flow_ba: maps from curr (crop2) to prev (crop1).
        # Pixel at (y, x) in crop2 equals canvas[y0+dy+y, x0+dx+x].
        # In crop1, this pixel sits at (y+dy, x+dx).
        # Therefore displacement from curr to prev is (+dx, +dy).
        flow_ba = _estimate_multiscale_flow_gpu(c2_norm, c1_norm)
        center_dx = flow_ba[0, 0, h // 2, w // 2].item()
        center_dy = flow_ba[0, 1, h // 2, w // 2].item()

        self.assertAlmostEqual(center_dx, float(dx_gt), delta=0.75)
        self.assertAlmostEqual(center_dy, float(dy_gt), delta=0.75)

        # Warp prev with flow_ba and check interior reconstruction error
        warped_c1, in_b = _warp_2d_gpu(c1_norm, flow_ba)
        interior_err = torch.abs(c2_norm[:, :, 20 : h - 20, 20 : w - 20] - warped_c1[:, :, 20 : h - 20, 20 : w - 20])
        self.assertLess(interior_err.mean().item(), 0.05)

    def test_forward_backward_consistency_rejects_occlusions(self) -> None:
        """Verify that forward-backward inconsistency rejects disoccluded boundaries."""
        h, w = 180, 240
        dx = 12

        y0, x0 = 50, 50
        crop1 = self.canvas[y0 : y0 + h, x0 : x0 + w]
        crop2 = self.canvas[y0 : y0 + h, x0 + dx : x0 + dx + w]

        f1_rgb = crop1.unsqueeze(0).repeat(3, 1, 1)
        f2_rgb = crop2.unsqueeze(0).repeat(3, 1, 1)

        d1 = torch.full((h, w), 2.0, device=self.device, dtype=torch.float32)
        # Give d2 an identifiable value
        d2 = torch.full((h, w), 4.0, device=self.device, dtype=torch.float32)

        stabilizer = TemporalGPUStabilizer(TemporalGPUConfig(flow_scale=1.0, alpha=0.8))
        stabilizer.process_frame(f1_rgb, d1)
        r2 = stabilizer.process_frame(f2_rgb, d2)

        # The trailing border pixels (w-12 to w) in crop2 did not exist in crop1 (or are at boundary)
        # They must have confidence 0 and their stabilized depth must equal d2 (4.0)
        # Check border region
        conf = r2.confidence_mask
        self.assertEqual(conf.device.type, "cuda")

        # In disoccluded border, depth must not be pulled to d1 (2.0)
        out_depth = r2.depth
        self.assertEqual(out_depth.device.type, "cuda")

        border_depth = out_depth[:, -8:]
        torch.testing.assert_close(border_depth, d2[:, -8:], rtol=1e-3, atol=1e-3)

    def test_photometric_error_rejection(self) -> None:
        """Verify sudden photometric brightness changes are rejected from temporal blending."""
        h, w = 180, 240
        y0, x0 = 50, 50
        crop1 = self.canvas[y0 : y0 + h, x0 : x0 + w]
        crop2 = crop1.clone()

        # Sudden bright flash in center of frame 2
        crop2[60:120, 80:160] = 255.0

        f1_rgb = crop1.unsqueeze(0).repeat(3, 1, 1)
        f2_rgb = crop2.unsqueeze(0).repeat(3, 1, 1)

        d1 = torch.full((h, w), 1.5, device=self.device, dtype=torch.float32)
        d2 = torch.full((h, w), 3.5, device=self.device, dtype=torch.float32)

        stabilizer = TemporalGPUStabilizer(TemporalGPUConfig(photometric_threshold=25.0, alpha=0.75))
        stabilizer.process_frame(f1_rgb, d1)
        r2 = stabilizer.process_frame(f2_rgb, d2)

        # Inside the bright flash patch, depth must NOT blend with d1 (1.5); it must equal d2 (3.5)
        patch_depth = r2.depth[70:110, 90:150]
        torch.testing.assert_close(
            patch_depth,
            torch.full_like(patch_depth, 3.5),
            rtol=1e-3,
            atol=1e-3,
        )
        patch_conf = r2.confidence_mask[70:110, 90:150]
        self.assertLess(patch_conf.max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
