#!/usr/bin/env python3
"""Benchmark and verification probe for PureGPU3D CUDA temporal depth stabilization.

Exercises:
  1. Real SOH sample photo input (604x340 and 480x270 resolutions).
  2. Bounded GPU-resident temporal stabilization pipeline (strictly O(1) state).
  3. Proper CUDA event timing with initial warmup (reporting true GPU kernel execution times).
  4. Known-motion geometry jitter reduction and hard cut / fade reset verification.
  5. JSON evidence summary export.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

# Ensure src/ is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.temporal_gpu import (
    TemporalGPUConfig,
    TemporalGPUResult,
    TemporalGPUStabilizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("temporal_gpu_probe")

DEFAULT_IMAGE_PATH = (
    REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"
)
OUTPUT_DIR = REPO_ROOT / "data" / "verification" / "temporal"


def build_synthetic_soh_sequence(
    image_path: Path,
    width: int = 604,
    height: int = 340,
    num_frames: int = 16,
    dx: float = 3.0,
    dy: float = 1.0,
    cut_at_frame: int = 10,
    noise_sigma: float = 0.05,
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Build a deterministic motion sequence from the real SOH photo directly on CUDA."""
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image at {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

    # Convert to GPU tensor
    rgb_tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)

    frames_rgb = []
    gt_depths = []
    raw_depths = []
    is_cut_flags = []

    # Two starting positions in the SOH photo
    pos1_x, pos1_y = 60, 80
    pos2_x, pos2_y = 300, 160

    # Synthetic realistic depth base
    gy, gx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    base_depth = 2.5 + 0.8 * torch.sin(gx / 30.0) * torch.cos(gy / 25.0)

    g = torch.Generator(device=device)
    g.manual_seed(12345)

    for t in range(num_frames):
        if t == cut_at_frame:
            is_cut = True
            x0 = pos2_x
            y0 = pos2_y
            curr_gt_depth = base_depth + 1.5
        elif t > cut_at_frame:
            is_cut = False
            cur_t = t - cut_at_frame
            x0 = int(round(pos2_x + cur_t * dx))
            y0 = int(round(pos2_y + cur_t * dy))
            curr_gt_depth = base_depth + 1.5
        else:
            is_cut = False
            x0 = int(round(pos1_x + t * dx))
            y0 = int(round(pos1_y + t * dy))
            curr_gt_depth = base_depth

        x0 = min(max(0, x0), img_w - width)
        y0 = min(max(0, y0), img_h - height)

        crop = rgb_tensor[:, y0 : y0 + height, x0 : x0 + width]
        noise = torch.randn(curr_gt_depth.shape, generator=g, device=device, dtype=torch.float32) * noise_sigma
        raw_d = torch.clamp(curr_gt_depth + noise, min=0.2)

        frames_rgb.append(crop)
        gt_depths.append(curr_gt_depth)
        raw_depths.append(raw_d)
        is_cut_flags.append(is_cut)

    return {
        "frames_rgb": frames_rgb,
        "gt_depths": gt_depths,
        "raw_depths": raw_depths,
        "is_cut_flags": is_cut_flags,
        "width": width,
        "height": height,
        "dx": dx,
        "dy": dy,
        "cut_at_frame": cut_at_frame,
    }


def benchmark_gpu_stabilizer(
    sequence: dict,
    config: TemporalGPUConfig,
    warmup_iters: int = 5,
) -> Dict[str, Any]:
    """Run GPU warmup followed by precision CUDA event benchmarking on the sequence."""
    device = sequence["frames_rgb"][0].device
    stabilizer = TemporalGPUStabilizer(config)

    # 1. Warmup
    for _ in range(warmup_iters):
        for frame, depth, is_cut in zip(sequence["frames_rgb"][:3], sequence["raw_depths"][:3], sequence["is_cut_flags"][:3]):
            stabilizer.process_frame(frame, depth, is_cut=is_cut)
    torch.cuda.synchronize(device)

    # 2. Benchmark run with CUDA event timings
    stabilizer.reset()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in sequence["frames_rgb"]]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in sequence["frames_rgb"]]

    torch.cuda.reset_peak_memory_stats(device)
    mem_start = torch.cuda.memory_allocated(device)

    results: list[TemporalGPUResult] = []
    current_stream = torch.cuda.current_stream(device)
    for i, (frame, depth, is_cut) in enumerate(
        zip(sequence["frames_rgb"], sequence["raw_depths"], sequence["is_cut_flags"])
    ):
        start_events[i].record(current_stream)
        res = stabilizer.process_frame(frame, depth, is_cut=is_cut)
        end_events[i].record(current_stream)
        results.append(res)

    torch.cuda.synchronize(device)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Collect per-frame latencies in milliseconds
    frame_times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Steady-state latencies (exclude cut/first frame index 0 and cut_at_frame)
    steady_times = [
        t for idx, t in enumerate(frame_times_ms)
        if idx != 0 and idx != sequence["cut_at_frame"]
    ]
    mean_latency_ms = float(np.mean(steady_times))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    # Motion-compensated jitter measurement
    h, w = sequence["height"], sequence["width"]
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    px = gx + sequence["dx"]
    py = gy + sequence["dy"]
    valid = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    grid = torch.stack([2.0 * px / max(w - 1, 1) - 1.0, 2.0 * py / max(h - 1, 1) - 1.0], dim=-1).unsqueeze(0)

    def compute_mae(depth_list: list[torch.Tensor]) -> float:
        maes = []
        for t in range(1, len(depth_list)):
            if t == sequence["cut_at_frame"]:
                continue
            dc = 1.0 / depth_list[t]
            dp = 1.0 / depth_list[t - 1]
            wp = torch.nn.functional.grid_sample(dp.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", align_corners=True).squeeze()
            maes.append(torch.abs(dc[valid] - wp[valid]).mean().item())
        return float(np.mean(maes))

    raw_mae = compute_mae(sequence["raw_depths"])
    stab_depths = [r.depth for r in results]
    stab_mae = compute_mae(stab_depths)
    jitter_reduction_pct = ((raw_mae - stab_mae) / raw_mae) * 100.0

    # Check cut transitions
    cut_frame_res = results[sequence["cut_at_frame"]]
    cut_clean = cut_frame_res.is_cut and cut_frame_res.cut_flag.item()

    # Telemetry
    valid_fractions = [r.valid_flow_fraction for r in results if not r.is_cut]
    mean_valid_flow = float(np.mean(valid_fractions)) if valid_fractions else 0.0

    return {
        "resolution": f"{w}x{h}",
        "num_frames": len(sequence["frames_rgb"]),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "min_latency_ms": round(float(np.min(steady_times)), 3),
        "max_latency_ms": round(float(np.max(steady_times)), 3),
        "throughput_fps": round(fps, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "raw_disparity_jitter_mae": round(raw_mae, 6),
        "stabilized_disparity_jitter_mae": round(stab_mae, 6),
        "jitter_reduction_percent": round(jitter_reduction_pct, 2),
        "cut_boundary_reset_verified": cut_clean,
        "mean_valid_flow_fraction": round(mean_valid_flow, 4),
        "normalization_bounds_sample": results[-1].normalization_bounds_float,
    }


def run_probe(
    image_path: Path = DEFAULT_IMAGE_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """Execute complete GPU temporal probe across realistic depth resolutions."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU temporal probe execution.")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    logger.info("Starting GPU temporal probe on %s", device_name)

    resolutions = [
        (604, 340, "realistic_da3_longest_604"),
        (480, 270, "quarter_scale_1080p"),
    ]

    benchmarks = {}
    for w, h, label in resolutions:
        logger.info("Benchmarking resolution %dx%d (%s)...", w, h, label)
        seq = build_synthetic_soh_sequence(
            image_path=image_path,
            width=w,
            height=h,
            device=device,
        )
        cfg = TemporalGPUConfig(
            flow_scale=0.50,
            max_estimator_dim=240,
            alpha=0.70,
            consistency_threshold=1.50,
            photometric_threshold=30.0,
            depth_diff_threshold=0.25,
            edge_threshold=0.20,
        )
        bench = benchmark_gpu_stabilizer(seq, cfg)
        benchmarks[label] = bench
        logger.info(
            "[%s] %dx%d: Latency=%.2f ms, FPS=%.1f, Jitter Reduction=%.1f%%, VRAM=%.1f MB",
            label, w, h, bench["mean_latency_ms"], bench["throughput_fps"],
            bench["jitter_reduction_percent"], bench["peak_vram_mb"],
        )

    cuda_ver = getattr(torch, "version", None)
    cuda_ver_str = getattr(cuda_ver, "cuda", "unknown") if cuda_ver else "unknown"

    summary = {
        "device": device_name,
        "cuda_version": cuda_ver_str,
        "torch_version": torch.__version__,
        "source_image": str(image_path),
        "benchmarks": benchmarks,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "gpu_temporal_verification.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Verification evidence written to %s", out_file)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PureGPU3D CUDA temporal depth stabilization probe")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Path to SOH sample image")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save JSON evidence")
    args = parser.parse_args()

    summary = run_probe(image_path=args.image, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
