"""Deterministic known-motion geometry and depth jitter reduction test on CUDA GPU."""

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
)


def generate_gpu_synthetic_motion_sequence(
    device: torch.device,
    height: int = 120,
    width: int = 160,
    num_frames: int = 8,
    dx: float = 2.0,
    dy: float = 1.0,
    noise_sigma: float = 0.08,
    cut_at_frame: int = 5,
    seed: int = 42,
) -> dict:
    """Generate a deterministic synthetic scene on GPU with known translation, texture, and depth."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    pad_x = int(abs(dx) * num_frames + 30)
    pad_y = int(abs(dy) * num_frames + 30)
    canvas_h = height + 2 * pad_y
    canvas_w = width + 2 * pad_x

    cy, cx = torch.meshgrid(
        torch.arange(canvas_h, device=device, dtype=torch.float32),
        torch.arange(canvas_w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Rich multi-scale texture canvas
    tex_canvas = torch.clamp(
        128.0
        + 50.0 * torch.sin(cx / 4.0) * torch.cos(cy / 4.0)
        + 40.0 * torch.sin(cx / 10.0 + cy / 8.0)
        + 25.0 * torch.cos(cx / 2.5),
        0.0,
        255.0,
    )

    # Smooth depth canvas between 1.2 and 3.0
    depth_canvas = 2.0 + 0.6 * torch.sin(cx / 16.0) * torch.cos(cy / 14.0) + 0.3 * torch.sin(cx / 25.0)

    # Secondary scene for cut frame
    tex_canvas_cut = torch.clamp(
        128.0 + 60.0 * torch.sin(cx / 7.0) * torch.sin(cy / 5.0) + 40.0 * torch.cos(cy / 3.0),
        0.0,
        255.0,
    )
    depth_canvas_cut = 4.5 + 1.2 * torch.cos(cx / 20.0 + cy / 15.0)

    frames_rgb = []
    gt_depths = []
    raw_depths = []
    is_cut_flags = []

    origin_x = pad_x
    origin_y = pad_y

    for t in range(num_frames):
        if t == cut_at_frame:
            is_cut = True
            sub_tex = tex_canvas_cut[origin_y : origin_y + height, origin_x : origin_x + width]
            gt_d = depth_canvas_cut[origin_y : origin_y + height, origin_x : origin_x + width].clone()
        elif t > cut_at_frame:
            is_cut = False
            curr_t = t - cut_at_frame
            cur_x = int(round(origin_x + curr_t * dx))
            cur_y = int(round(origin_y + curr_t * dy))
            sub_tex = tex_canvas_cut[cur_y : cur_y + height, cur_x : cur_x + width]
            gt_d = depth_canvas_cut[cur_y : cur_y + height, cur_x : cur_x + width].clone()
        else:
            is_cut = False
            cur_x = int(round(origin_x + t * dx))
            cur_y = int(round(origin_y + t * dy))
            sub_tex = tex_canvas[cur_y : cur_y + height, cur_x : cur_x + width]
            gt_d = depth_canvas[cur_y : cur_y + height, cur_x : cur_x + width].clone()

        rgb = sub_tex.unsqueeze(0).repeat(3, 1, 1)
        noise = torch.randn(gt_d.shape, generator=g, device=device, dtype=torch.float32) * noise_sigma
        raw_d = torch.clamp(gt_d + noise, min=0.2)

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


def compute_gpu_motion_compensated_jitter(
    depth_sequence: list[torch.Tensor],
    dx: float,
    dy: float,
    cut_at_frame: int,
) -> dict[str, float]:
    """Calculate motion-compensated disparity jitter on CUDA along known motion trajectories."""
    num_frames = len(depth_sequence)
    h, w = depth_sequence[0].shape
    device = depth_sequence[0].device

    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Point (x, y) in frame t was at (x + dx, y + dy) in the source canvas coordinate frame relative to t-1
    prev_x = gx + float(dx)
    prev_y = gy + float(dy)

    valid_overlap = (prev_x >= 0.0) & (prev_x <= float(w - 1)) & (prev_y >= 0.0) & (prev_y <= float(h - 1))

    grid_x = 2.0 * prev_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * prev_y / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    sq_errors = []
    abs_errors = []

    for t in range(1, num_frames):
        if t == cut_at_frame:
            continue

        disp_curr = 1.0 / torch.clamp(depth_sequence[t], min=1e-4)
        disp_prev = 1.0 / torch.clamp(depth_sequence[t - 1], min=1e-4)

        disp_prev_4d = disp_prev.unsqueeze(0).unsqueeze(0)
        warped_disp_prev = F.grid_sample(disp_prev_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze()

        err = torch.abs(disp_curr - warped_disp_prev)
        valid_err = err[valid_overlap]

        sq_errors.append((valid_err ** 2).mean().item())
        abs_errors.append(valid_err.mean().item())

    rmse = float(torch.tensor(sq_errors).mean().sqrt().item())
    mae = float(torch.tensor(abs_errors).mean().item())
    return {"rmse": rmse, "mae": mae}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU temporal tests")
class TestGPUMotionJitter(unittest.TestCase):
    """Verify motion-compensated disparity jitter reduction on deterministic GPU sequence."""

    def setUp(self) -> None:
        self.device = torch.device("cuda:0")

    def test_motion_compensated_jitter_reduction(self) -> None:
        """Measure motion-compensated disparity jitter before and after GPU stabilization."""
        seq = generate_gpu_synthetic_motion_sequence(
            device=self.device,
            height=120,
            width=160,
            num_frames=8,
            dx=2.0,
            dy=1.0,
            noise_sigma=0.08,
            cut_at_frame=5,
            seed=42,
        )

        cfg = TemporalGPUConfig(
            enabled=True,
            alpha=0.70,
            flow_scale=1.0,
            consistency_threshold=1.5,
            photometric_threshold=30.0,
            depth_diff_threshold=0.25,
        )
        stabilizer = TemporalGPUStabilizer(cfg)

        stabilized_depths = []
        for frame_rgb, raw_d, is_cut in zip(seq["frames_rgb"], seq["raw_depths"], seq["is_cut_flags"]):
            res = stabilizer.process_frame(frame_rgb, raw_d, is_cut=is_cut)
            self.assertEqual(res.depth.device.type, "cuda")
            stabilized_depths.append(res.depth)

        raw_metrics = compute_gpu_motion_compensated_jitter(
            seq["raw_depths"],
            seq["dx"],
            seq["dy"],
            seq["cut_at_frame"],
        )
        stab_metrics = compute_gpu_motion_compensated_jitter(
            stabilized_depths,
            seq["dx"],
            seq["dy"],
            seq["cut_at_frame"],
        )

        jitter_reduction = (raw_metrics["mae"] - stab_metrics["mae"]) / raw_metrics["mae"]

        # Verification gate: stabilization must reduce temporal disparity jitter by at least 25%
        self.assertGreater(
            jitter_reduction,
            0.25,
            f"Insufficient jitter reduction: {jitter_reduction * 100:.1f}% (raw MAE={raw_metrics['mae']:.5f}, stab MAE={stab_metrics['mae']:.5f})",
        )

    def test_ineffective_stabilization_fails_threshold(self) -> None:
        """Verify that when stabilization is disabled, jitter reduction is zero and fails the improvement gate."""
        seq = generate_gpu_synthetic_motion_sequence(
            device=self.device,
            height=120,
            width=160,
            num_frames=8,
            dx=2.0,
            dy=1.0,
            noise_sigma=0.08,
            cut_at_frame=5,
            seed=42,
        )

        cfg_disabled = TemporalGPUConfig(enabled=False)
        stabilizer = TemporalGPUStabilizer(cfg_disabled)

        stabilized_depths = []
        for frame_rgb, raw_d, is_cut in zip(seq["frames_rgb"], seq["raw_depths"], seq["is_cut_flags"]):
            res = stabilizer.process_frame(frame_rgb, raw_d, is_cut=is_cut)
            stabilized_depths.append(res.depth)

        raw_metrics = compute_gpu_motion_compensated_jitter(
            seq["raw_depths"],
            seq["dx"],
            seq["dy"],
            seq["cut_at_frame"],
        )
        stab_metrics = compute_gpu_motion_compensated_jitter(
            stabilized_depths,
            seq["dx"],
            seq["dy"],
            seq["cut_at_frame"],
        )

        jitter_reduction = (raw_metrics["mae"] - stab_metrics["mae"]) / raw_metrics["mae"]
        # When disabled, jitter reduction is 0% and does not pass the 25% bar
        self.assertLess(jitter_reduction, 0.01)


if __name__ == "__main__":
    unittest.main()
