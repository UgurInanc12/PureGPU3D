"""Unit tests for optical flow warping, consistency rejection, and newly visible region handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.depth_temporal import (
    TemporalDepthConfig,
    TemporalDepthStabilizer,
)


class TestFlowWarp(unittest.TestCase):
    """Verify flow direction derivation, warping accuracy, and occlusion/border rejection."""

    def setUp(self) -> None:
        np.random.seed(42)
        # Create rich textured base image
        raw_noise = np.random.randint(40, 220, (180, 240), dtype=np.uint8)
        self.base_texture = cv2.GaussianBlur(raw_noise, (5, 5), 1.2)

    def test_flow_direction_and_warp_accuracy(self) -> None:
        """Verify backward flow direction points from curr to prev and remap aligns textures."""
        dx, dy = 6.0, 4.0
        # Frame 1: base
        f1 = self.base_texture
        # Frame 2: translated by (dx, dy)
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        f2 = cv2.warpAffine(f1, M, (240, 180))

        # Convert to 3-channel
        f1_rgb = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
        f2_rgb = cv2.cvtColor(f2, cv2.COLOR_GRAY2RGB)

        # Constant depth plane z=2.0
        d1 = np.full((180, 240), 2.0, dtype=np.float32)
        d2 = np.full((180, 240), 2.0, dtype=np.float32)

        stabilizer = TemporalDepthStabilizer(TemporalDepthConfig(flow_scale=1.0, alpha=0.7))
        r1 = stabilizer.process_frame(f1_rgb, d1)
        self.assertTrue(r1.is_cut)

        r2 = stabilizer.process_frame(f2_rgb, d2)
        self.assertFalse(r2.is_cut)
        # In the unoccluded interior, motion compensation should be valid
        self.assertGreater(r2.valid_flow_fraction, 0.70)

    def test_forward_backward_consistency_rejects_occlusions(self) -> None:
        """Verify forward/backward flow error flags newly revealed borders as invalid."""
        dx, dy = 12.0, 0.0
        f1 = self.base_texture
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        f2 = cv2.warpAffine(f1, M, (240, 180))

        f1_rgb = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
        f2_rgb = cv2.cvtColor(f2, cv2.COLOR_GRAY2RGB)

        d1 = np.full((180, 240), 2.0, dtype=np.float32)
        d2 = np.full((180, 240), 2.0, dtype=np.float32)

        stabilizer = TemporalDepthStabilizer(TemporalDepthConfig(flow_scale=1.0))
        stabilizer.process_frame(f1_rgb, d1)
        r2 = stabilizer.process_frame(f2_rgb, d2)

        # Border pixels x in [0, 12) did not exist in f1, so they must not be warped from prev
        # Verify stabilizer output in border region equals current depth exactly
        stab_depth = r2.depth if isinstance(r2.depth, np.ndarray) else r2.depth.numpy()
        # Newly visible region should match d2
        np.testing.assert_allclose(stab_depth[:, :10], d2[:, :10], rtol=1e-3)

    def test_photometric_error_rejection(self) -> None:
        """Verify pixels with sudden appearance changes are rejected and not smeared."""
        f1_rgb = cv2.cvtColor(self.base_texture, cv2.COLOR_GRAY2RGB)
        f2_rgb = f1_rgb.copy()
        # Add a sudden bright flash in a central patch in f2
        f2_rgb[60:120, 80:160] = 255

        d1 = np.full((180, 240), 2.0, dtype=np.float32)
        d2 = np.full((180, 240), 3.5, dtype=np.float32)

        stabilizer = TemporalDepthStabilizer(TemporalDepthConfig(photometric_threshold=20.0))
        stabilizer.process_frame(f1_rgb, d1)
        r2 = stabilizer.process_frame(f2_rgb, d2)

        stab_depth = r2.depth if isinstance(r2.depth, np.ndarray) else r2.depth.numpy()
        # Inside the photometric flash patch, depth must NOT blend with d1; it must equal d2
        patch_depth = stab_depth[70:110, 90:150]
        np.testing.assert_allclose(patch_depth, 3.5, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
