"""Unit tests verifying edge sharpness preservation and depth discontinuity protection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.depth_temporal import (
    TemporalDepthConfig,
    TemporalDepthStabilizer,
)


class TestDepthDiscontinuity(unittest.TestCase):
    """Verify depth edges are not blurred and disoccluded depths do not leave trails."""

    def test_sharp_boundary_gradient_preservation(self) -> None:
        """Verify that step depth discontinuity retains high gradient and is not smoothed out."""
        # Create an image with a foreground object moving over background
        h, w = 120, 160
        # Background: depth 5.0 (far), Foreground: depth 1.0 (near) in [30:90, 40:100]
        f1_rgb = np.full((h, w, 3), 80, dtype=np.uint8)
        f1_rgb[30:90, 40:100] = 200

        d1 = np.full((h, w), 5.0, dtype=np.float32)
        d1[30:90, 40:100] = 1.0

        # Frame 2: foreground moves right by 4 pixels to [30:90, 44:104]
        f2_rgb = np.full((h, w, 3), 80, dtype=np.uint8)
        f2_rgb[30:90, 44:104] = 200

        d2 = np.full((h, w), 5.0, dtype=np.float32)
        d2[30:90, 44:104] = 1.0

        cfg = TemporalDepthConfig(alpha=0.75, edge_threshold=0.15)
        stabilizer = TemporalDepthStabilizer(cfg)

        stabilizer.process_frame(f1_rgb, d1)
        r2 = stabilizer.process_frame(f2_rgb, d2)
        stab_depth = r2.depth if isinstance(r2.depth, np.ndarray) else r2.depth.numpy()

        # Check disoccluded column x=41 (which was foreground in f1, but background in f2):
        # It must NOT retain foreground depth (z=1.0) or smeared intermediate depth (e.g. 2.5);
        # it must be background depth (z=5.0)!
        disoccluded_depth = stab_depth[50:70, 41]
        np.testing.assert_allclose(disoccluded_depth, 5.0, rtol=0.05)

        # Check foreground interior: x=60, depth must be near 1.0
        fg_depth = stab_depth[50:70, 60]
        np.testing.assert_allclose(fg_depth, 1.0, rtol=0.05)

        # Measure horizontal gradient at the leading boundary (x=103 to 105):
        # The step must remain steep (drop >= 3.5 over 2 pixels)
        step_height = stab_depth[60, 105] - stab_depth[60, 103]
        self.assertGreaterEqual(step_height, 3.5, "Depth boundary must remain crisp and not smoothed")


if __name__ == "__main__":
    unittest.main()
