"""Unit tests for shared-pixel inverse depth scale alignment."""

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


class TestScaleAlignment(unittest.TestCase):
    """Verify robust inverse depth scale alignment and identifiability guards."""

    def test_scale_alignment_on_identifiable_scene(self) -> None:
        """Verify intentional 10% global scale drift is detected and compensated."""
        h, w = 120, 160
        np.random.seed(42)
        tex = cv2.GaussianBlur(np.random.randint(50, 200, (h, w), dtype=np.uint8), (5, 5), 1.0)
        f_rgb = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB)

        # Depth ramp with diverse depths from 1.0 to 4.0 (identifiable)
        y, x = np.indices((h, w), dtype=np.float32)
        d1 = 1.0 + 3.0 * (x / float(w))

        # Frame 2 has static image, but depth has an artificial 1.08x scale drift in inverse space
        # inv_d2 = 1.08 * inv_d1 => d2 = d1 / 1.08
        d2 = d1 / 1.08

        cfg = TemporalDepthConfig(align_scale=True, max_scale_adjustment=0.15, alpha=0.7)
        stabilizer = TemporalDepthStabilizer(cfg)

        stabilizer.process_frame(f_rgb, d1)
        r2 = stabilizer.process_frame(f_rgb, d2)

        # Detected scale factor should be very close to 1.08
        self.assertAlmostEqual(r2.scale_factor, 1.08, delta=0.03)

    def test_scale_alignment_identifiability_guard_on_flat_scene(self) -> None:
        """Verify flat scene with zero depth diversity falls back to scale=1.0."""
        h, w = 120, 160
        tex = np.full((h, w, 3), 128, dtype=np.uint8)
        # Uniform flat depth z=2.0 everywhere
        d1 = np.full((h, w), 2.0, dtype=np.float32)
        d2 = np.full((h, w), 2.2, dtype=np.float32)

        cfg = TemporalDepthConfig(align_scale=True, max_scale_adjustment=0.15)
        stabilizer = TemporalDepthStabilizer(cfg)

        stabilizer.process_frame(tex, d1)
        r2 = stabilizer.process_frame(tex, d2)

        # Flat scene has zero IQR; identifiability check must fall back to 1.0
        self.assertEqual(r2.scale_factor, 1.0)

    def test_scale_clamping_against_runaway(self) -> None:
        """Verify extreme scale drift (e.g. 2.0x) is clamped to max_scale_adjustment."""
        h, w = 120, 160
        np.random.seed(42)
        tex = cv2.GaussianBlur(np.random.randint(50, 200, (h, w), dtype=np.uint8), (5, 5), 1.0)
        f_rgb = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB)

        y, x = np.indices((h, w), dtype=np.float32)
        d1 = 1.0 + 3.0 * (x / float(w))
        # 2x depth change
        d2 = d1 / 2.0

        cfg = TemporalDepthConfig(align_scale=True, max_scale_adjustment=0.15)
        stabilizer = TemporalDepthStabilizer(cfg)

        stabilizer.process_frame(f_rgb, d1)
        r2 = stabilizer.process_frame(f_rgb, d2)

        # Clamped to 1.0 + 0.15 = 1.15
        self.assertLessEqual(r2.scale_factor, 1.15 + 1e-5)


if __name__ == "__main__":
    unittest.main()
