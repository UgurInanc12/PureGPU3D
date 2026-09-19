#!/usr/bin/env python3
"""Stereoscopic rendering probe script for PureGPU3D.

Loads Depth Anything 3 Small depth adapter, infers depth on a source image,
executes the bounded stereoscopic rendering pipeline (signed disparity,
occlusion-aware subpixel forward splatting, conservative background hole filling),
and exports:
  - Full-resolution Side-by-Side (SBS) composite PNG (2W x H)
  - Individual left and right eye views
  - Explicit pre-fill coverage masks
  - Disparity map array and visualization
  - Detailed performance and hole diagnostics JSON
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

# Ensure repository root src/ and vendor are reachable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import (
    DEFAULT_PROCESS_RES,
    DEFAULT_REVISION,
    DA3SmallDepthAdapter,
)
from puregpu3d.stereo import (
    DisparityConfig,
    FillConfig,
    SplatConfig,
    StereoConfig,
    StereoFrameResult,
    compute_disparity,
    fill_holes,
    forward_splat,
    render_stereo_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stereo_probe")

DEFAULT_IMAGE_PATH = REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "verification" / "stereo"
CHECKPOINT_DIR = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION


def colorize_disparity(disparity: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Render 2D float signed disparity array as a colormapped RGB image."""
    d_min = float(disparity.min())
    d_max = float(disparity.max())
    if abs(d_max - d_min) < 1e-7:
        norm = np.zeros_like(disparity, dtype=np.uint8)
    else:
        norm = ((disparity - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(norm, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def run_stereo_probe(
    image_path: Path,
    output_dir: Path,
    device_name: str = "cuda",
    strength: float = 0.03,
    q_screen: float = 0.6,
    max_hole_width: int = 16,
    warmup: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end stereo probe and export artifacts."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    logger.info(f"Target execution device: {device}")

    # 1. Load source image
    bgr_img = cv2.imread(str(image_path))
    if bgr_img is None:
        raise ValueError(f"cv2.imread failed to load {image_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, c = rgb_img.shape
    logger.info(f"Loaded source image: {image_path.name} ({w}x{h} px, {c} channels)")

    # 2. DA3 Small Depth Inference
    logger.info("Initializing DA3 Small depth adapter...")
    adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device=device)

    if warmup and device.type == "cuda":
        logger.info("Warming up DA3 Small adapter...")
        _ = adapter.infer(rgb_img, target_size=DEFAULT_PROCESS_RES, return_original_size=True)

    logger.info("Running DA3 Small depth inference...")
    t0 = time.perf_counter()
    depth_res = adapter.infer(
        image=rgb_img,
        target_size=DEFAULT_PROCESS_RES,
        return_original_size=True,
        autocast=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    da3_latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        f"DA3 inference complete in {da3_latency_ms:.2f} ms "
        f"(depth min={depth_res.min_depth:.4f}, max={depth_res.max_depth:.4f})"
    )

    depth_np = depth_res.depth

    # 3. Stereoscopic Rendering
    cfg = StereoConfig(
        disparity=DisparityConfig(strength=strength, q_screen=q_screen),
        splat=SplatConfig(depth_tolerance=0.05),
        fill=FillConfig(max_hole_width=max_hole_width, fill_large_holes=True),
    )

    if warmup and device.type == "cuda":
        logger.info("Warming up stereo renderer...")
        _ = render_stereo_frame(rgb_img, depth_np, config=cfg, device=device, output_uint8=True)
        torch.cuda.synchronize(device)

    logger.info(
        f"Rendering stereoscopic views (strength={strength}, q_screen={q_screen}, max_hole_w={max_hole_width})..."
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_stereo_start = time.perf_counter()

    stereo_res = render_stereo_frame(
        image=rgb_img,
        depth=depth_np,
        config=cfg,
        device=device,
        return_numpy=True,
        output_uint8=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stereo_latency_ms = (time.perf_counter() - t_stereo_start) * 1000.0

    logger.info(f"Stereo rendering complete in {stereo_latency_ms:.2f} ms")

    # 4. Save artifacts
    sbs_np = stereo_res.sbs_color
    left_np = stereo_res.left_color
    right_np = stereo_res.right_color

    assert isinstance(sbs_np, np.ndarray)
    assert isinstance(left_np, np.ndarray)
    assert isinstance(right_np, np.ndarray)

    sbs_path = output_dir / "soh_000_sbs.png"
    left_path = output_dir / "soh_000_left.png"
    right_path = output_dir / "soh_000_right.png"
    left_cov_path = output_dir / "soh_000_left_coverage.png"
    right_cov_path = output_dir / "soh_000_right_coverage.png"
    disp_npy_path = output_dir / "soh_000_disparity.npy"
    disp_vis_path = output_dir / "soh_000_disparity_vis.png"
    metrics_path = output_dir / "stereo_metrics.json"

    # Save PNGs (convert RGB to BGR for cv2.imwrite)
    cv2.imwrite(str(sbs_path), cv2.cvtColor(sbs_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(left_path), cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(right_np, cv2.COLOR_RGB2BGR))

    # Save coverage masks (255 = covered, 0 = hole)
    left_cov_np = (stereo_res.left_coverage_before_fill.cpu().numpy().astype(np.uint8)) * 255
    right_cov_np = (stereo_res.right_coverage_before_fill.cpu().numpy().astype(np.uint8)) * 255
    cv2.imwrite(str(left_cov_path), left_cov_np)
    cv2.imwrite(str(right_cov_path), right_cov_np)

    # Save disparity map
    disp_np = stereo_res.disparity.disparity.cpu().numpy()
    np.save(str(disp_npy_path), disp_np)
    disp_vis = colorize_disparity(disp_np)
    cv2.imwrite(str(disp_vis_path), cv2.cvtColor(disp_vis, cv2.COLOR_RGB2BGR))

    # Metrics dictionary
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_image": str(image_path),
        "input_resolution": {"width": w, "height": h},
        "sbs_resolution": {"width": sbs_np.shape[1], "height": sbs_np.shape[0]},
        "device": str(device),
        "da3_small": {
            "model_id": DEFAULT_REVISION,
            "latency_ms": da3_latency_ms,
            "min_depth": float(depth_np.min()),
            "max_depth": float(depth_np.max()),
            "mean_depth": float(depth_np.mean()),
        },
        "stereo_config": {
            "strength": strength,
            "q_screen": q_screen,
            "max_hole_width": max_hole_width,
            "max_disparity_fraction": cfg.disparity.max_disparity_fraction,
            "depth_tolerance": cfg.splat.depth_tolerance,
        },
        "stereo_latency_ms": stereo_latency_ms,
        "fps_stereo_only": 1000.0 / stereo_latency_ms if stereo_latency_ms > 0 else 0.0,
        "fps_end_to_end": 1000.0 / (da3_latency_ms + stereo_latency_ms) if (da3_latency_ms + stereo_latency_ms) > 0 else 0.0,
        "disparity_stats": {
            "d_min": stereo_res.disparity.d_min,
            "d_max": stereo_res.disparity.d_max,
            "mean": float(disp_np.mean()),
            "std": float(disp_np.std()),
        },
        "left_eye_diagnostics": {
            "total_hole_pixels": stereo_res.left_diagnostics.total_hole_pixels,
            "hole_fraction": stereo_res.left_diagnostics.hole_fraction,
            "max_hole_width": stereo_res.left_diagnostics.max_hole_width,
            "large_hole_count": stereo_res.left_diagnostics.large_hole_count,
            "warning": stereo_res.left_diagnostics.warning,
        },
        "right_eye_diagnostics": {
            "total_hole_pixels": stereo_res.right_diagnostics.total_hole_pixels,
            "hole_fraction": stereo_res.right_diagnostics.hole_fraction,
            "max_hole_width": stereo_res.right_diagnostics.max_hole_width,
            "large_hole_count": stereo_res.right_diagnostics.large_hole_count,
            "warning": stereo_res.right_diagnostics.warning,
        },
        "artifacts": {
            "sbs_png": str(sbs_path),
            "left_png": str(left_path),
            "right_png": str(right_path),
            "left_coverage_png": str(left_cov_path),
            "right_coverage_png": str(right_cov_path),
            "disparity_npy": str(disp_npy_path),
            "disparity_vis_png": str(disp_vis_path),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Artifacts and metrics written to {output_dir}")
    logger.info(
        f"SBS Output: {sbs_path} ({sbs_np.shape[1]}x{sbs_np.shape[0]} px, "
        f"disparity range: [{disp_np.min():.2f}, {disp_np.max():.2f}] px)"
    )
    logger.info(
        f"Left holes: {stereo_res.left_diagnostics.total_hole_pixels} ({stereo_res.left_diagnostics.hole_fraction:.2%}), "
        f"Right holes: {stereo_res.right_diagnostics.total_hole_pixels} ({stereo_res.right_diagnostics.hole_fraction:.2%})"
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo reprojection probe for PureGPU3D")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Input RGB image path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save output files")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--strength", type=float, default=0.03, help="Disparity strength multiplier")
    parser.add_argument("--q-screen", type=float, default=0.6, help="Screen zero-parallax plane in [0, 1]")
    parser.add_argument("--max-hole-width", type=int, default=16, help="Max hole width before warning")
    args = parser.parse_args()

    metrics = run_stereo_probe(
        image_path=args.image,
        output_dir=args.output_dir,
        device_name=args.device,
        strength=args.strength,
        q_screen=args.q_screen,
        max_hole_width=args.max_hole_width,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
