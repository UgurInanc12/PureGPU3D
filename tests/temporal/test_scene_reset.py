"""Regression tests for scene-cut, resize, fade, and letterbox resets in temporal depth stabilization."""

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
from puregpu3d.video.scenes import (
    SceneCutDetector,
    SceneDetectorConfig,
    detect_letterbox,
)


class TestSceneReset(unittest.TestCase):
    """Verify reset behavior on scene cuts, dynamic resizing, fades, and letterboxes."""

    def test_hard_cut_reset_clears_temporal_history(self) -> None:
        """Verify that is_cut=True completely resets state and avoids cross-shot blending."""
        h, w = 120, 160
        # Scene A: low depth z=1.0, dark pattern
        img_a = np.full((h, w, 3), 50, dtype=np.uint8)
        depth_a = np.full((h, w), 1.0, dtype=np.float32)

        # Scene B: high depth z=8.0, bright pattern
        img_b = np.full((h, w, 3), 220, dtype=np.uint8)
        depth_b = np.full((h, w), 8.0, dtype=np.float32)

        stabilizer = TemporalDepthStabilizer(TemporalDepthConfig(alpha=0.8))

        # Frame 1: Scene A
        r1 = stabilizer.process_frame(img_a, depth_a)
        self.assertTrue(r1.is_cut)
        self.assertIsNotNone(stabilizer.prev_depth)

        # Frame 2: Scene B with explicit is_cut=True
        r2 = stabilizer.process_frame(img_b, depth_b, is_cut=True)
        self.assertTrue(r2.is_cut)
        self.assertEqual(r2.valid_flow_fraction, 0.0)

        # Depth must equal depth_b exactly without any leakage from depth_a
        np.testing.assert_allclose(r2.depth, depth_b, rtol=1e-4)

    def test_resize_reset_handles_dynamic_resolution_change(self) -> None:
        """Verify that changing input resolution automatically triggers a clean reset."""
        # Initial resolution: 120x160
        h1, w1 = 120, 160
        img1 = np.full((h1, w1, 3), 100, dtype=np.uint8)
        depth1 = np.full((h1, w1), 2.0, dtype=np.float32)

        # New resolution: 180x240
        h2, w2 = 180, 240
        img2 = np.full((h2, w2, 3), 150, dtype=np.uint8)
        depth2 = np.full((h2, w2), 3.0, dtype=np.float32)

        stabilizer = TemporalDepthStabilizer()

        # Frame 1
        r1 = stabilizer.process_frame(img1, depth1)
        self.assertEqual(r1.depth.shape, (h1, w1))

        # Frame 2 with different shape: must not raise OpenCV remap/size mismatch exception
        r2 = stabilizer.process_frame(img2, depth2)
        self.assertTrue(r2.is_cut)
        self.assertEqual(r2.valid_flow_fraction, 0.0)
        self.assertEqual(r2.depth.shape, (h2, w2))
        np.testing.assert_allclose(r2.depth, depth2, rtol=1e-4)

    def test_fade_detection_and_reset(self) -> None:
        """Verify fade-to-black and fade-from-black trigger cut signaling and clean state."""
        h, w = 120, 160
        detector = SceneCutDetector(SceneDetectorConfig(fade_threshold=10.0, cut_threshold=0.3))
        stabilizer = TemporalDepthStabilizer()

        # Frame 1: Normal bright scene
        f1 = np.full((h, w, 3), 150, dtype=np.uint8)
        d1 = np.full((h, w), 2.0, dtype=np.float32)
        cut_res1 = detector.update(f1)
        r1 = stabilizer.process_frame(f1, d1, is_cut=cut_res1.is_cut)
        self.assertFalse(cut_res1.is_fade)

        # Frame 2: Fade to black (luminance < 10)
        f_black = np.full((h, w, 3), 2, dtype=np.uint8)
        d_black = np.full((h, w), 5.0, dtype=np.float32)
        cut_res_black = detector.update(f_black)
        self.assertTrue(cut_res_black.is_fade)
        self.assertTrue(cut_res_black.is_cut)

        # Stabilizer resets on fade frame
        r_black = stabilizer.process_frame(f_black, d_black, is_cut=cut_res_black.is_cut)
        self.assertTrue(r_black.is_cut)
        self.assertEqual(r_black.valid_flow_fraction, 0.0)

        # Frame 3: Fade from black to new bright scene
        f3 = np.full((h, w, 3), 180, dtype=np.uint8)
        d3 = np.full((h, w), 4.0, dtype=np.float32)
        cut_res3 = detector.update(f3)
        self.assertTrue(cut_res3.is_cut, "Transition out of black must signal a cut")

        r3 = stabilizer.process_frame(f3, d3, is_cut=cut_res3.is_cut)
        self.assertTrue(r3.is_cut)
        np.testing.assert_allclose(r3.depth, d3, rtol=1e-4)

    def test_letterbox_detection_and_normalization_crop(self) -> None:
        """Verify letterbox detection isolates active video region and bounds calculation."""
        h, w = 120, 160
        # Image with top 20 rows and bottom 20 rows black
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[20:100, :] = 140

        box = detect_letterbox(img, threshold=12.0)
        self.assertEqual(box, (20, 100, 0, 160))

        # Depth map: active region has depth 2.0 (inv_z=0.5), black borders have depth 100.0 (inv_z=0.01)
        depth = np.full((h, w), 100.0, dtype=np.float32)
        depth[20:100, :] = 2.0

        # Stabilizer with letterbox_crop
        stabilizer = TemporalDepthStabilizer()
        r = stabilizer.process_frame(img, depth, letterbox_crop=box)

        # Normalization bounds must reflect active region depth 2.0 (inv_z=0.5), not border 100.0
        q_low, q_high = r.normalization_bounds
        self.assertAlmostEqual(q_low, 0.5, delta=0.05)
        self.assertAlmostEqual(q_high, 0.5, delta=0.05)

    def test_degenerate_letterbox_fallback(self) -> None:
        """Verify that degenerate letterbox (<20% active area) falls back to full bounds."""
        h, w = 120, 160
        # Almost completely black image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[58:62, :] = 100  # Only 4 rows active (< 4% of height)

        box = detect_letterbox(img, threshold=12.0)
        # Must fall back to full image bounds
        self.assertEqual(box, (0, h, 0, w))


if __name__ == "__main__":
    unittest.main()
