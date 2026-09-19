"""Deterministic known-motion geometry/depth fixture measuring motion-compensated disparity temporal jitter."""

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


def generate_synthetic_motion_sequence(
    height: int = 120,
    width: int = 160,
    num_frames: int = 8,
    dx: float = 2.0,
    dy: float = 1.0,
    noise_sigma: float = 0.06,
    cut_at_frame: int = 5,
    seed: int = 42,
):
    """Generate a deterministic synthetic scene with ground-truth geometry, texture, and known translation."""
    rng = np.random.RandomState(seed)

    # Large continuous canvas so translating doesn't create boundary discontinuities
    pad_x = int(abs(dx) * num_frames + 30)
    pad_y = int(abs(dy) * num_frames + 30)
    canvas_h = height + 2 * pad_y
    canvas_w = width + 2 * pad_x

    # Procedural texture canvas with multi-frequency contrast
    cy, cx = np.indices((canvas_h, canvas_w), dtype=np.float32)
    tex_canvas = (
        128.0
        + 50.0 * np.sin(cx / 4.0) * np.cos(cy / 4.0)
        + 40.0 * np.sin(cx / 10.0 + cy / 8.0)
        + 25.0 * np.cos(cx / 2.5)
    )
    tex_canvas = np.clip(tex_canvas, 0, 255).astype(np.uint8)

    # Smooth depth canvas between 1.2 and 3.0
    depth_canvas = 2.0 + 0.6 * np.sin(cx / 16.0) * np.cos(cy / 14.0) + 0.3 * np.sin(cx / 25.0)

    # Secondary scene for cut frame
    tex_canvas_cut = np.clip(
        128.0 + 60.0 * np.sin(cx / 7.0) * np.sin(cy / 5.0) + 40.0 * np.cos(cy / 3.0),
        0,
        255,
    ).astype(np.uint8)
    depth_canvas_cut = 4.5 + 1.2 * np.cos(cx / 20.0 + cy / 15.0)

    frames_rgb = []
    gt_depths = []
    raw_depths = []
    is_cut_flags = []

    origin_x = pad_x
    origin_y = pad_y

    for t in range(num_frames):
        if t == cut_at_frame:
            # Hard cut: completely new scene
            is_cut = True
            sub_tex = tex_canvas_cut[origin_y : origin_y + height, origin_x : origin_x + width]
            gt_d = depth_canvas_cut[origin_y : origin_y + height, origin_x : origin_x + width].copy()
        elif t > cut_at_frame:
            is_cut = False
            # Motion continues in scene 2
            curr_t = t - cut_at_frame
            cur_x = int(round(origin_x + curr_t * dx))
            cur_y = int(round(origin_y + curr_t * dy))
            sub_tex = tex_canvas_cut[cur_y : cur_y + height, cur_x : cur_x + width]
            gt_d = depth_canvas_cut[cur_y : cur_y + height, cur_x : cur_x + width].copy()
        else:
            is_cut = False
            cur_x = int(round(origin_x + t * dx))
            cur_y = int(round(origin_y + t * dy))
            sub_tex = tex_canvas[cur_y : cur_y + height, cur_x : cur_x + width]
            gt_d = depth_canvas[cur_y : cur_y + height, cur_x : cur_x + width].copy()

        # RGB 3-channel
        rgb = cv2.cvtColor(sub_tex, cv2.COLOR_GRAY2RGB)

        # Raw depth has simulated independent temporal jitter
        noise = rng.normal(0.0, noise_sigma, size=gt_d.shape).astype(np.float32)
        raw_d = np.maximum(gt_d + noise, 0.2)

        frames_rgb.append(rgb)
        gt_depths.append(gt_d)
        raw_depths.append(raw_d)
        is_cut_flags.append(is_cut)

    return {
        "frames_rgb": frames_rgb,
        "gt_depths": gt_depths,
        "raw_depths": raw_depths,
        "is_cut_flags": is_cut_flags,
        "dx": dx,
        "dy": dy,
        "cut_at_frame": cut_at_frame,
    }


def compute_motion_compensated_jitter(
    depth_sequence: list[np.ndarray],
    dx: float,
    dy: float,
    cut_at_frame: int,
) -> dict[str, float]:
    """Calculate motion-compensated temporal disparity error using known translation."""
    num_frames = len(depth_sequence)
    h, w = depth_sequence[0].shape

    # Grid for frame t
    gy, gx = np.indices((h, w), dtype=np.float32)
    # Coordinate in frame t-1 according to known forward motion dx, dy:
    # point (x, y) in frame t was at (x - dx, y - dy) in frame t-1
    prev_x = gx - float(dx)
    prev_y = gy - float(dy)

    # Valid mask where pixel existed in frame t-1 (not newly exposed border)
    valid_overlap = (prev_x >= 0.0) & (prev_x <= float(w - 1)) & (prev_y >= 0.0) & (prev_y <= float(h - 1))

    diffs_sq = []
    diffs_abs = []

    for t in range(1, num_frames):
        if t == cut_at_frame:
            # Skip cut frame for motion compensation metric as scenes are independent
            continue

        disp_curr = 1.0 / np.maximum(depth_sequence[t], 1e-4)
        disp_prev = 1.0 / np.maximum(depth_sequence[t - 1], 1e-4)

        # Warp prev disparity to curr using known ground truth transform
        warped_disp_prev = cv2.remap(
            disp_prev.astype(np.float32),
            prev_x.astype(np.float32),
            prev_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )

        err = np.abs(disp_curr - warped_disp_prev)
        valid_err = err[valid_overlap]

        diffs_sq.extend((valid_err**2).tolist())
        diffs_abs.extend(valid_err.tolist())

    rmse = float(np.sqrt(np.mean(diffs_sq)))
    mae = float(np.mean(diffs_abs))
    return {"rmse": rmse, "mae": mae}


class TestKnownMotionJitter(unittest.TestCase):
    """Verify motion-compensated disparity jitter reduction and state bounds on deterministic fixture."""

    def test_motion_compensated_jitter_reduction(self) -> None:
        """Measure motion-compensated disparity jitter before and after stabilization."""
        seq = generate_synthetic_motion_sequence(
            height=120,
            width=160,
            num_frames=8,
            dx=2.0,
            dy=1.0,
            noise_sigma=0.08,
            cut_at_frame=5,
            seed=42,
        )

        cfg = TemporalDepthConfig(
            enabled=True,
            alpha=0.70,
            flow_scale=1.0,
            consistency_threshold=1.5,
            photometric_threshold=30.0,
        )
        stabilizer = TemporalDepthStabilizer(cfg)

        stab_depths = []
        for i in range(len(seq["frames_rgb"])):
            frame_rgb = seq["frames_rgb"][i]
            raw_d = seq["raw_depths"][i]
            is_cut = seq["is_cut_flags"][i]
            res = stabilizer.process_frame(frame_rgb, raw_d, is_cut=is_cut)
            stab_depths.append(res.depth)

        # Measure motion-compensated jitter on baseline raw vs stabilized
        raw_metrics = compute_motion_compensated_jitter(
            seq["raw_depths"],
            seq["dx"],
            seq["dy"],
            seq["cut_at_frame"],
        )
        stab_metrics = compute_motion_compensated_jitter(
            stab_depths,
            seq["dx"],
            seq["dy"],
            seq["cut_at_frame"],
        )

        # Report honest numbers
        print(f"\n[Known Motion Jitter] Raw RMSE: {raw_metrics['rmse']:.6f}, Stab RMSE: {stab_metrics['rmse']:.6f}")
        print(f"[Known Motion Jitter] Raw MAE:  {raw_metrics['mae']:.6f}, Stab MAE:  {stab_metrics['mae']:.6f}")
        improvement = (raw_metrics["rmse"] - stab_metrics["rmse"]) / raw_metrics["rmse"] * 100.0
        print(f"[Known Motion Jitter] Disparity Jitter Improvement: {improvement:.2f}%")

        # Temporal stabilizer should reduce jitter compared to raw noisy input
        self.assertLess(
            stab_metrics["rmse"],
            raw_metrics["rmse"],
            "Stabilized depth should reduce motion-compensated disparity jitter",
        )

    def test_newly_exposed_region_behavior(self) -> None:
        """Verify newly exposed border pixels use current frame depth without boundary smearing."""
        seq = generate_synthetic_motion_sequence(
            height=120,
            width=160,
            num_frames=3,
            dx=8.0,  # 8 px right shift -> left columns x in [0, 8) are newly exposed
            dy=0.0,
            noise_sigma=0.0,
            seed=123,
        )

        stabilizer = TemporalDepthStabilizer(TemporalDepthConfig(flow_scale=1.0))
        stabilizer.process_frame(seq["frames_rgb"][0], seq["raw_depths"][0])
        r1 = stabilizer.process_frame(seq["frames_rgb"][1], seq["raw_depths"][1])

        # Leftmost columns (x < 5) were not visible in frame 0
        curr_d = seq["raw_depths"][1]
        np.testing.assert_allclose(r1.depth[:, :5], curr_d[:, :5], rtol=1e-3)

    def test_retained_state_array_nbytes(self) -> None:
        """Measure exact byte footprint of retained state arrays (strictly bounded O(1))."""
        h, w = 360, 640
        dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_depth = np.ones((h, w), dtype=np.float32)

        stabilizer = TemporalDepthStabilizer()
        stabilizer.process_frame(dummy_rgb, dummy_depth)

        self.assertIsNotNone(stabilizer.prev_gray)
        self.assertIsNotNone(stabilizer.prev_depth)
        assert stabilizer.prev_gray is not None
        assert stabilizer.prev_depth is not None

        prev_gray_nbytes = stabilizer.prev_gray.nbytes
        prev_depth_nbytes = stabilizer.prev_depth.nbytes
        total_nbytes = prev_gray_nbytes + prev_depth_nbytes

        # For 640x360:
        # prev_gray: 640 * 360 * 1 = 230,400 bytes
        # prev_depth: 640 * 360 * 4 = 921,600 bytes
        # total: 1,152,000 bytes (~1.15 MB, well below 10MB)
        self.assertEqual(prev_gray_nbytes, 640 * 360 * 1)
        self.assertEqual(prev_depth_nbytes, 640 * 360 * 4)
        self.assertEqual(total_nbytes, 1_152_000)
        self.assertLess(total_nbytes, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
