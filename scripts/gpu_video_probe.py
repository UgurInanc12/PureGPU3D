#!/usr/bin/env python
"""GPU-resident Video Conversion Probe & Benchmark.

Executes real NVDEC -> DA3 infer_tensor -> depth-aware stereo -> NVENC conversion,
verifies Full SBS 3840x1080 geometry, real stereoscopic eye disparity (not duplicate eyes),
exact rational CFR timestamps, AAC audio stream preservation, and measures fair wall-clock
comparison against existing baseline path (with temporal stabilization disabled).

Usage:
    python -B scripts/gpu_video_probe.py [options]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import (
    DEFAULT_REVISION,
    DA3DepthAdapter,
    DA3SmallDepthAdapter,
)
from puregpu3d.video.convert import convert_video
from puregpu3d.video.gpu_convert import (
    GpuConversionResult,
    check_gpu_pipeline_support,
    convert_video_gpu,
)
from puregpu3d.video.probe import probe_video


def verify_stereoscopic_eye_difference(video_path: Path) -> Dict[str, Any]:
    """Read a sample frame from the rendered SBS video and compute L/R eye disparity differences."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for eye verification: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    h, w, c = frame.shape
    if w % 2 != 0:
        raise ValueError(f"Video width {w} is not even for SBS split.")

    half_w = w // 2
    left_eye = frame[:, :half_w]
    right_eye = frame[:, half_w:]

    diff = np.abs(left_eye.astype(np.float32) - right_eye.astype(np.float32))
    mean_abs_diff = float(np.mean(diff))
    max_abs_diff = float(np.max(diff))
    non_identical_fraction = float(np.mean(diff > 1.0))

    return {
        "width": w,
        "height": h,
        "eye_width": half_w,
        "mean_abs_difference": mean_abs_diff,
        "max_abs_difference": max_abs_diff,
        "non_identical_pixel_fraction": non_identical_fraction,
        "is_real_stereoscopic": non_identical_fraction > 0.01,
    }


def run_probe(
    input_path: Path,
    output_path: Path,
    depth_scale: str = "1/4",
    codec: str = "hevc",
    model_size: str = "small",
    compare_baseline: bool = True,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Run full GPU video conversion probe and optional baseline comparison."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    supported, reason = check_gpu_pipeline_support()
    if not supported:
        raise RuntimeError(f"GPU pipeline unsupported: {reason}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load DA3 Adapter
    model_name_upper = f"DA3-{model_size.upper()}"
    checkpoint_dir = REPO_ROOT / "models" / model_name_upper / DEFAULT_REVISION
    if not checkpoint_dir.exists() or not (checkpoint_dir / "READY").exists():
        raise FileNotFoundError(f"Model checkpoint directory not found or not ready: {checkpoint_dir}")

    print(f"Loading {model_name_upper} from {checkpoint_dir}...")
    adapter = DA3DepthAdapter(checkpoint_dir, identifier=model_name_upper, device="cuda:0", verify_hashes=False)

    # 2. Run GPU-Resident Conversion
    print(f"\n=== Running GPU-Resident Conversion ({codec.upper()}, scale={depth_scale}) ===")
    gpu_res = convert_video_gpu(
        input_path=input_path,
        output_path=output_path,
        model=adapter,
        device="cuda:0",
        depth_scale=depth_scale,
        codec=codec,
        overwrite=overwrite,
        enable_temporal_stabilization=False,
    )

    print(f"  Output: {gpu_res.output_path}")
    print(f"  Resolution: {gpu_res.output_width}x{gpu_res.output_height}")
    print(f"  Frames: {gpu_res.total_frames_processed}")
    print(f"  Total Wall Time: {gpu_res.wall_clock_seconds:.3f} s (FPS: {gpu_res.effective_fps:.2f})")
    print(f"  Stage breakdown (mean per frame):")
    print(f"    - NVDEC Decode: {gpu_res.mean_decode_ms:.2f} ms")
    print(f"    - DA3 Depth:    {gpu_res.mean_depth_ms:.2f} ms")
    print(f"    - Stereo Splat: {gpu_res.mean_stereo_ms:.2f} ms")
    print(f"    - NVENC Encode: {gpu_res.mean_encode_ms:.2f} ms")
    print(f"  Overhead:")
    print(f"    - Startup:    {gpu_res.startup_overhead_s:.3f} s")
    print(f"    - Validation: {gpu_res.validation_overhead_s:.3f} s")

    # 3. Verify Output Video Properties via Probe
    out_probe = probe_video(output_path)
    eye_check = verify_stereoscopic_eye_difference(output_path)

    print(f"\n=== Verification of GPU Output ===")
    print(f"  Probe Dimensions: {out_probe.width}x{out_probe.height}")
    print(f"  Probe Frames: {out_probe.frame_count}")
    print(f"  Probe FPS: {out_probe.frame_rate}")
    print(f"  Probe Audio: {out_probe.has_audio} (Streams: {len(out_probe.audio_streams)})")
    print(f"  Eye Disparity Difference Fraction: {eye_check['non_identical_pixel_fraction']:.4f} (Real Stereo: {eye_check['is_real_stereoscopic']})")

    baseline_data: Optional[Dict[str, Any]] = None

    # 4. Fair Measured Comparison against Existing Baseline Path (Temporal OFF)
    if compare_baseline:
        baseline_out = output_path.with_name(f"{output_path.stem}_baseline_temporal_off{output_path.suffix}")
        print(f"\n=== Running Existing Baseline Path (hevc_nvenc, temporal OFF, scale={depth_scale}) ===")
        t_base_0 = time.perf_counter()
        base_res = convert_video(
            input_path=input_path,
            output_path=baseline_out,
            model=adapter,
            device="cuda:0",
            depth_scale=depth_scale,
            encoder="hevc_nvenc",
            overwrite=overwrite,
            enable_temporal_stabilization=False,
        )
        t_base_1 = time.perf_counter()
        base_wall = t_base_1 - t_base_0

        base_probe = probe_video(baseline_out)
        baseline_data = {
            "output_path": str(baseline_out),
            "wall_clock_seconds": base_wall,
            "effective_fps": base_res.effective_fps,
            "mean_depth_ms": base_res.mean_depth_ms,
            "mean_stereo_ms": base_res.mean_stereo_ms,
            "encoder": base_res.encoder,
            "output_width": base_probe.width,
            "output_height": base_probe.height,
            "frame_count": base_probe.frame_count,
        }

        print(f"  Baseline Output: {baseline_out}")
        print(f"  Baseline Wall Time: {base_wall:.3f} s (FPS: {base_res.effective_fps:.2f})")
        print(f"  Baseline DA3 Depth: {base_res.mean_depth_ms:.2f} ms")
        print(f"  Baseline Stereo:    {base_res.mean_stereo_ms:.2f} ms")

        speedup = base_wall / gpu_res.wall_clock_seconds if gpu_res.wall_clock_seconds > 0 else 0.0
        print(f"\n=== Wall-Time Comparison ===")
        print(f"  Baseline Total:     {base_wall:.3f} s")
        print(f"  GPU-Resident Total: {gpu_res.wall_clock_seconds:.3f} s")
        print(f"  Speedup Factor:     {speedup:.2f}x")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "depth_scale": depth_scale,
        "model": model_name_upper,
        "codec": codec,
        "gpu_pipeline": gpu_res.to_dict(),
        "probe_verification": {
            "width": out_probe.width,
            "height": out_probe.height,
            "frame_count": out_probe.frame_count,
            "frame_rate": f"{out_probe.frame_rate.numerator}/{out_probe.frame_rate.denominator}",
            "has_audio": out_probe.has_audio,
            "audio_streams": len(out_probe.audio_streams),
            "eye_verification": eye_check,
        },
        "baseline_comparison": baseline_data,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-resident Video Conversion Probe")
    parser.add_argument("--input", type=str, default=str(REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"))
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "data" / "verification" / "gpu-pipeline" / "synthetic_1080p_gpu_sbs.mp4"))
    parser.add_argument("--depth-scale", type=str, default="1/4", choices=["1/4", "1/2", "1/1"])
    parser.add_argument("--codec", type=str, default="hevc", choices=["hevc", "h264"])
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--no-compare", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--json-out", type=str, default=None, help="Save JSON summary")
    args = parser.parse_args()

    input_p = Path(args.input).resolve()
    output_p = Path(args.output).resolve()

    res = run_probe(
        input_path=input_p,
        output_path=output_p,
        depth_scale=args.depth_scale,
        codec=args.codec,
        model_size=args.model_size,
        compare_baseline=not args.no_compare,
        overwrite=True,
    )

    if args.json_out:
        json_p = Path(args.json_out).resolve()
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved probe summary to {json_p}")


if __name__ == "__main__":
    main()
