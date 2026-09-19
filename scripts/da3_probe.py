#!/usr/bin/env python3
"""DA3 Small reproducible inference probe script for PureGPU3D.

Executes Depth Anything 3 Small inference on CPU and/or CUDA, verifies shared
LayerNorm alias resolution, compares numerical consistency, and exports depth
arrays, visual PNGs, and structured performance metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Ensure repository root src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.models.da3_adapter import (
    DEFAULT_PROCESS_RES,
    DEFAULT_REVISION,
    DA3SmallDepthAdapter,
    DepthPredictionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("da3_probe")


def run_single_device(
    adapter: DA3SmallDepthAdapter,
    image_path: Path,
    device_name: str,
    output_dir: Path,
    process_res: int,
    autocast: bool,
    warmup: bool = True,
) -> DepthPredictionResult:
    """Run inference on one device with optional warmup and export results."""
    logger.info(f"Starting inference on {device_name} (process_res={process_res}, autocast={autocast})...")

    if warmup and device_name.startswith("cuda"):
        logger.info(f"Running warmup iteration on {device_name}...")
        _ = adapter.infer(
            image=image_path,
            target_size=process_res,
            return_original_size=True,
            autocast=autocast,
        )

    result = adapter.infer(
        image=image_path,
        target_size=process_res,
        return_original_size=True,
        autocast=autocast,
    )

    base_name = f"depth_{device_name.replace(':', '_')}"
    saved_files = DA3SmallDepthAdapter.save_depth_outputs(
        result=result,
        out_dir=output_dir,
        base_name=base_name,
    )
    logger.info(f"Saved {device_name} artifacts to {output_dir}:")
    for k, v in saved_files.items():
        logger.info(f"  - {k}: {v.name}")

    return result


def compare_results(cpu_res: DepthPredictionResult, cuda_res: DepthPredictionResult) -> Dict[str, Any]:
    """Compute numerical fidelity and performance comparison between CPU and CUDA."""
    diff = np.abs(cpu_res.depth - cuda_res.depth)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_diff = float(np.max(diff))
    rel_diff = float(np.mean(diff / np.clip(cpu_res.depth, 1e-6, None)))

    speedup = cpu_res.latency_ms / max(cuda_res.latency_ms, 1e-6)

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "max_absolute_difference": round(max_diff, 6),
        "mean_relative_difference": round(rel_diff, 6),
        "cpu_latency_ms": round(cpu_res.latency_ms, 2),
        "cuda_latency_ms": round(cuda_res.latency_ms, 2),
        "cuda_speedup_x": round(speedup, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DA3 Small Depth Inference Probe")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION,
        help="Path to verified checkpoint revision folder",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png",
        help="Input image path for probe inference",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Device to evaluate (default: both if CUDA available, otherwise CPU)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "verification" / "da3-small",
        help="Directory to store output depth maps and metrics",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=DEFAULT_PROCESS_RES,
        help="Longest dimension resolution multiple (default: 504)",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable float16 autocast on CUDA",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip strict SHA-256 hash checks",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    process_res = args.process_res
    autocast = not args.no_autocast
    verify_hashes = not args.skip_hash_check

    if not model_dir.is_dir():
        logger.error(f"Checkpoint directory not found: {model_dir}")
        return 1
    if not image_path.is_file():
        logger.error(f"Test image not found: {image_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    device_mode = args.device
    if device_mode == "both" and not cuda_available:
        logger.warning("CUDA is not available on this host. Falling back to device='cpu'.")
        device_mode = "cpu"

    cpu_result: Optional[DepthPredictionResult] = None
    cuda_result: Optional[DepthPredictionResult] = None

    # 1. CPU Run
    if device_mode in ("cpu", "both"):
        logger.info("=== Executing DA3 Small CPU Probe ===")
        cpu_adapter = DA3SmallDepthAdapter(
            model_dir=model_dir,
            device="cpu",
            verify_hashes=verify_hashes,
        )
        cpu_result = run_single_device(
            adapter=cpu_adapter,
            image_path=image_path,
            device_name="cpu",
            output_dir=output_dir,
            process_res=process_res,
            autocast=False,
            warmup=False,
        )
        del cpu_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. CUDA Run
    if device_mode in ("cuda", "both"):
        logger.info("=== Executing DA3 Small CUDA Probe ===")
        cuda_device = "cuda:0"
        cuda_adapter = DA3SmallDepthAdapter(
            model_dir=model_dir,
            device=cuda_device,
            verify_hashes=verify_hashes,
        )
        cuda_result = run_single_device(
            adapter=cuda_adapter,
            image_path=image_path,
            device_name="cuda",
            output_dir=output_dir,
            process_res=process_res,
            autocast=autocast,
            warmup=True,
        )
        del cuda_adapter
        torch.cuda.empty_cache()

    # 3. Cross-device Comparison
    comparison: Optional[Dict[str, Any]] = None
    if cpu_result is not None and cuda_result is not None:
        comparison = compare_results(cpu_result, cuda_result)
        comp_file = output_dir / "comparison_metrics.json"
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Cross-device comparison written to {comp_file}")

    # Summary Report
    print("\n" + "=" * 60)
    print("DA3 Small Depth Probe Execution Summary")
    print("=" * 60)
    print(f"Model Dir:    {model_dir}")
    print(f"Input Image:  {image_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Process Res:  {process_res} (patch multiple 14)")

    if cpu_result is not None:
        print(f"\n[CPU float32]")
        print(f"  Latency:    {cpu_result.latency_ms:.2f} ms")
        print(f"  Shape:      {cpu_result.depth.shape}")
        print(f"  Depth Range:[{cpu_result.min_depth:.4f}, {cpu_result.max_depth:.4f}] (mean={cpu_result.mean_depth:.4f})")

    if cuda_result is not None:
        print(f"\n[CUDA {cuda_result.dtype}]")
        print(f"  Latency:    {cuda_result.latency_ms:.2f} ms")
        print(f"  Shape:      {cuda_result.depth.shape}")
        print(f"  Depth Range:[{cuda_result.min_depth:.4f}, {cuda_result.max_depth:.4f}] (mean={cuda_result.mean_depth:.4f})")

    if comparison is not None:
        print(f"\n[CPU vs CUDA Consistency]")
        print(f"  MAE:        {comparison['mae']:.6f}")
        print(f"  RMSE:       {comparison['rmse']:.6f}")
        print(f"  Max Diff:   {comparison['max_absolute_difference']:.6f}")
        print(f"  Speedup:    {comparison['cuda_speedup_x']:.2f}x")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
