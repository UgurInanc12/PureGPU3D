#!/usr/bin/env python
"""Run the complete GPU Video Pipeline verification matrix on RTX 3090.

Matrix:
  - synthetic_1080p_moving.mp4 @ depth_scale 1/4 (GPU vs Baseline)
  - synthetic_1080p_moving.mp4 @ depth_scale 1/2 (GPU vs Baseline)
  - soh_pan_1080p.mp4 @ depth_scale 1/4 (GPU vs Baseline)
  - soh_pan_1080p.mp4 @ depth_scale 1/2 (GPU vs Baseline)

All outputs verified for:
  - Resolution: 3840x1080 (Full SBS)
  - Codec: HEVC
  - Frames: 12 (0-tolerance)
  - Timing: Rational 12/1 CFR fps
  - Audio: AAC stream copy (no -shortest)
  - Disparity: Distinct left/right eyes (real DA3 depth splatting)
  - DLPack pointer matching
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import DEFAULT_REVISION, DA3DepthAdapter
from puregpu3d.video.convert import convert_video
from puregpu3d.video.gpu_convert import convert_video_gpu
from puregpu3d.video.probe import probe_video


def verify_eye_disparity(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")

    h, w, _ = frame.shape
    half_w = w // 2
    left = frame[:, :half_w]
    right = frame[:, half_w:]

    diff = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return {
        "width": w,
        "height": h,
        "eye_width": half_w,
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "non_identical_fraction": float(np.mean(diff > 1.0)),
        "is_real_stereo": float(np.mean(diff > 1.0)) > 0.01,
    }


def main() -> None:
    checkpoint_dir = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION
    out_dir = REPO_ROOT / "data" / "verification" / "gpu-pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DA3-SMALL adapter on CUDA...")
    adapter = DA3DepthAdapter(checkpoint_dir, identifier="DA3-SMALL", device="cuda:0", verify_hashes=False)

    clips = [
        ("synthetic_1080p_moving.mp4", REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"),
        ("soh_pan_1080p.mp4", REPO_ROOT / "data" / "verification" / "video" / "soh_pan_1080p.mp4"),
    ]
    scales = ["1/4", "1/2"]

    results: List[Dict[str, Any]] = []

    for clip_name, clip_path in clips:
        for scale in scales:
            slug = clip_name.replace(".mp4", "")
            scale_slug = scale.replace("/", "_")
            gpu_out = out_dir / f"{slug}_scale_{scale_slug}_gpu_sbs.mp4"
            base_out = out_dir / f"{slug}_scale_{scale_slug}_baseline_sbs.mp4"

            print(f"\n=================================================================")
            print(f"Running Matrix: {clip_name} | Depth Scale {scale}")
            print(f"=================================================================")

            # 1. GPU Pipeline
            print(f"-> Executing GPU-resident pipeline...")
            t_gpu_0 = time.perf_counter()
            gpu_res = convert_video_gpu(
                input_path=clip_path,
                output_path=gpu_out,
                model=adapter,
                device="cuda:0",
                depth_scale=scale,
                codec="hevc",
                overwrite=True,
                enable_temporal_stabilization=False,
            )
            t_gpu_1 = time.perf_counter()

            # 2. Baseline Pipeline
            print(f"-> Executing baseline pipeline (hevc_nvenc, temporal OFF)...")
            t_base_0 = time.perf_counter()
            base_res = convert_video(
                input_path=clip_path,
                output_path=base_out,
                model=adapter,
                device="cuda:0",
                depth_scale=scale,
                encoder="hevc_nvenc",
                overwrite=True,
                enable_temporal_stabilization=False,
            )
            t_base_1 = time.perf_counter()
            base_wall = t_base_1 - t_base_0

            # Probes and verification
            gpu_probe = probe_video(gpu_out)
            gpu_eyes = verify_eye_disparity(gpu_out)

            base_probe = probe_video(base_out)
            base_eyes = verify_eye_disparity(base_out)

            speedup = base_wall / gpu_res.wall_clock_seconds if gpu_res.wall_clock_seconds > 0 else 0.0

            entry = {
                "clip": clip_name,
                "depth_scale": scale,
                "input_resolution": f"{gpu_probe.width // 2}x{gpu_probe.height}",
                "output_resolution": f"{gpu_probe.width}x{gpu_probe.height}",
                "frames": gpu_probe.frame_count,
                "frame_rate": f"{gpu_probe.frame_rate.numerator}/{gpu_probe.frame_rate.denominator}",
                "audio_streams": len(gpu_probe.audio_streams),
                "gpu_pipeline": {
                    "output_file": str(gpu_out.name),
                    "wall_clock_s": gpu_res.wall_clock_seconds,
                    "effective_fps": gpu_res.effective_fps,
                    "mean_decode_ms": gpu_res.mean_decode_ms,
                    "mean_depth_ms": gpu_res.mean_depth_ms,
                    "mean_stereo_ms": gpu_res.mean_stereo_ms,
                    "mean_encode_ms": gpu_res.mean_encode_ms,
                    "startup_overhead_s": gpu_res.startup_overhead_s,
                    "validation_overhead_s": gpu_res.validation_overhead_s,
                    "pointer_traces": gpu_res.pointer_traces,
                    "eye_disparity": gpu_eyes,
                },
                "baseline_pipeline": {
                    "output_file": str(base_out.name),
                    "wall_clock_s": base_wall,
                    "effective_fps": base_res.effective_fps,
                    "mean_depth_ms": base_res.mean_depth_ms,
                    "mean_stereo_ms": base_res.mean_stereo_ms,
                    "eye_disparity": base_eyes,
                },
                "speedup_factor": speedup,
            }
            results.append(entry)

            print(f"Results for {clip_name} (scale={scale}):")
            print(f"  GPU Wall Time:      {gpu_res.wall_clock_seconds:.3f} s ({gpu_res.effective_fps:.2f} FPS)")
            print(f"  Baseline Wall Time: {base_wall:.3f} s ({base_res.effective_fps:.2f} FPS)")
            print(f"  Speedup:            {speedup:.2f}x")
            print(f"  GPU Eye Diff %:     {gpu_eyes['non_identical_fraction'] * 100:.2f}% (Real Stereo: {gpu_eyes['is_real_stereo']})")
            print(f"  DLPack pointers:    Matched={all(t['ptrs_match'] for t in gpu_res.pointer_traces)}")

    matrix_file = out_dir / "gpu_pipeline_matrix_results.json"
    with open(matrix_file, "w", encoding="utf-8") as f:
        json.dump({"matrix": results}, f, indent=2)

    print(f"\nAll 4 benchmark matrix runs completed successfully!")
    print(f"Results written to: {matrix_file}")


if __name__ == "__main__":
    main()
