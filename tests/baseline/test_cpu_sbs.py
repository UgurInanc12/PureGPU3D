"""Tests for CPU side-by-side (SBS) stereo synthesis and depth heuristic."""

import unittest
import numpy as np

from puregpu3d.config.types import DepthConfig
from puregpu3d.engine.depth.cpu_kernel import (
    _split_nv12,
    compute_disparity_map,
    convert_nv12_to_sbs_cpu,
    make_black_nv12,
)


class TestCpuSbs(unittest.TestCase):
    """Characterize CPU SBS conversion, zero-strength behavior, and black frame handling."""

    def test_split_nv12_valid_and_invalid_shapes(self) -> None:
        width, height = 64, 32
        frame = np.full((height * 3 // 2, width), 42, dtype=np.uint8)
        y, uv = _split_nv12(frame, width, height)
        self.assertEqual(y.shape, (height, width))
        self.assertEqual(uv.shape, (height // 2, width))

        # Invalid shape mismatch raises ValueError
        bad_frame = np.zeros((height, width), dtype=np.uint8)
        with self.assertRaises(ValueError):
            _split_nv12(bad_frame, width, height)

    def test_odd_dimensions_rejected(self) -> None:
        depth_cfg = DepthConfig()
        frame_odd_w = np.zeros((30, 21), dtype=np.uint8)
        with self.assertRaises(ValueError):
            convert_nv12_to_sbs_cpu(frame_odd_w, 21, 20, depth_cfg)

        frame_odd_h = np.zeros((31, 20), dtype=np.uint8)
        with self.assertRaises(ValueError):
            convert_nv12_to_sbs_cpu(frame_odd_h, 20, 21, depth_cfg)

    def test_output_shape_invariant(self) -> None:
        depth_cfg = DepthConfig()
        test_resolutions = [(64, 32), (160, 90), (320, 180)]
        for width, height in test_resolutions:
            with self.subTest(width=width, height=height):
                in_shape = (height * 3 // 2, width)
                frame = np.random.randint(0, 256, size=in_shape, dtype=np.uint8)
                out = convert_nv12_to_sbs_cpu(frame, width, height, depth_cfg)
                expected_shape = (height * 3 // 2, width * 2)
                self.assertEqual(out.shape, expected_shape)
                self.assertEqual(out.dtype, np.uint8)

    def test_zero_strength_invariant(self) -> None:
        """With depth_strength=0.0, disparity is zero and both eyes are exact copies."""
        width, height = 128, 64
        depth_cfg = DepthConfig(depth_strength=0.0)

        # Create structured image with high contrast gradients
        in_shape = (height * 3 // 2, width)
        rng = np.random.default_rng(seed=42)
        frame = rng.integers(0, 256, size=in_shape, dtype=np.uint8)

        y_plane, uv_plane = _split_nv12(frame, width, height)
        disparity = compute_disparity_map(y_plane, depth_cfg)

        # Disparity map must be identically zero everywhere
        self.assertTrue(np.all(disparity == 0))

        out = convert_nv12_to_sbs_cpu(frame, width, height, depth_cfg)

        # Check left eye (0..width) and right eye (width..2*width)
        left_y = out[:height, :width]
        right_y = out[:height, width:]
        left_uv = out[height:, :width]
        right_uv = out[height:, width:]

        # Both eyes must be byte-for-byte identical to the input planes
        self.assertTrue(np.array_equal(left_y, y_plane))
        self.assertTrue(np.array_equal(right_y, y_plane))
        self.assertTrue(np.array_equal(left_uv, uv_plane))
        self.assertTrue(np.array_equal(right_uv, uv_plane))
        self.assertTrue(np.array_equal(left_y, right_y))
        self.assertTrue(np.array_equal(left_uv, right_uv))

    def test_black_content_invariant(self) -> None:
        """Pure black NV12 (Y=0, UV=128) must produce pure black SBS output without artifacts."""
        width, height = 128, 64
        depth_cfg = DepthConfig(depth_strength=0.55, max_disparity_px=6)

        black_frame = make_black_nv12(width, height)
        self.assertEqual(black_frame.shape, (height * 3 // 2, width))
        self.assertTrue(np.all(black_frame[:height, :] == 0))
        self.assertTrue(np.all(black_frame[height:, :] == 128))

        out = convert_nv12_to_sbs_cpu(black_frame, width, height, depth_cfg)

        # Output SBS dimensions must be height*3/2 x 2*width
        self.assertEqual(out.shape, (height * 3 // 2, width * 2))

        # Y plane across both eyes must be strictly 0 (true black)
        self.assertTrue(np.all(out[:height, :] == 0), "Black frame Y plane contains non-zero pixels")

        # UV plane across both eyes must be strictly 128 (neutral chroma)
        self.assertTrue(np.all(out[height:, :] == 128), "Black frame UV plane contains non-128 chroma")

    def test_disparity_bounds_and_stereo_separation(self) -> None:
        """Non-zero depth on non-trivial content must produce bounded disparity and eye differences."""
        width, height = 128, 64
        depth_cfg = DepthConfig(depth_strength=0.8, max_disparity_px=8)

        # Vertical bar pattern to trigger horizontal gradient / disparity
        y = np.zeros((height, width), dtype=np.uint8)
        y[:, width // 4 : 3 * width // 4] = 240
        uv = np.full((height // 2, width), 128, dtype=np.uint8)
        frame = np.vstack([y, uv])

        disparity = compute_disparity_map(y, depth_cfg)
        self.assertGreaterEqual(int(np.min(disparity)), 0)
        self.assertLessEqual(int(np.max(disparity)), 8)
        self.assertGreater(int(np.max(disparity)), 0)

        out = convert_nv12_to_sbs_cpu(frame, width, height, depth_cfg)
        left_y = out[:height, :width]
        right_y = out[:height, width:]

        # With positive disparity, left eye shifts left (x - d) and right eye shifts right (x + d)
        self.assertFalse(np.array_equal(left_y, right_y), "Stereo separation failed to produce distinct eyes")


if __name__ == "__main__":
    unittest.main()
