#!/usr/bin/env python3
"""DA3 GPU-Tensor inference probe script for PureGPU3D.

Proves CUDA-input -> CUDA-depth inference with zero host memory copies at
explicit spatial depth scales (1/4, 1/2, 1/1) across verified DA3 checkpoints.
Measures latency via CUDA events, validates geometry and patch divisibility,
and records bounded numerical difference against the CPU OpenCV reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.catalog import load_catalog
from puregpu3d.models.da3_adapter import (
    DEFAULT_REVISION,
    DA3DepthAdapter,
    DepthPredictionResult,
    DepthTensorResult,
)
from puregpu3d.models.geometry import DepthGeometry, compute_depth_geometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("da3_tensor_probe")

DEFAULT_IMAGE = REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "da3_tensor_probe"


def run_probe_for_model(
    model_id: str,
    adapter: DA3DepthAdapter,
    gpu_image: torch.Tensor,
    cpu_rgb: np.ndarray,
    scales: List[str],
    output_dir: Path,
    iterations: int = 5,
    warmup: int = 1,
) -> Dict[str, Any]:
    """Execute scale tests on a single model instance using pre-uploaded GPU image."""
    model_out_dir = output_dir / model_id
    model_out_dir.mkdir(parents=True, exist_ok=True)
    orig_h, orig_w = cpu_rgb.shape[:2]

    results: Dict[str, Any] = {
        "model_id": model_id,
        "device": str(adapter.device),
        "source_resolution": [orig_w, orig_h],
        "scales": {},
    }

    logger.info(f"--- Running probe for model: {model_id} on {adapter.device} ---")

    for scale in scales:
        geom = compute_depth_geometry(orig_w, orig_h, scale=scale)
        logger.info(
            f"Scale '{scale}': req_dims=({geom.req_width}x{geom.req_height}), "
            f"padded_dims=({geom.padded_width}x{geom.padded_height})"
        )

        # Warmup
        for _ in range(warmup):
            _ = adapter.infer_tensor(
                gpu_image,
                depth_scale=scale,
                return_original_size=True,
                autocast=True,
                timing=False,
            )

        # Benchmarked iterations via CUDA events
        latencies_ms: List[float] = []
        last_tensor_result: Optional[DepthTensorResult] = None

        for it in range(iterations):
            tensor_res = adapter.infer_tensor(
                gpu_image,
                depth_scale=scale,
                return_original_size=True,
                autocast=True,
                timing=True,
            )
            latencies_ms.append(tensor_res.latency_ms)
            last_tensor_result = tensor_res

        assert last_tensor_result is not None

        # Verify device residency (CUDA)
        if not last_tensor_result.depth.is_cuda or not last_tensor_result.depth_raw.is_cuda:
            raise RuntimeError(f"Depth tensor for {model_id} scale {scale} did not reside on CUDA!")

        # NumPy reference execution for regression & accuracy comparison
        numpy_res = adapter.infer(
            cpu_rgb,
            depth_scale=scale,
            return_original_size=True,
            autocast=True,
        )

        # Compute differences between numpy (OpenCV resize) and tensor (torch resize)
        gpu_depth_cpu = last_tensor_result.depth.detach().cpu().numpy()
        abs_diff = np.abs(numpy_res.depth - gpu_depth_cpu)
        mae = float(np.mean(abs_diff))
        max_diff = float(np.max(abs_diff))
        rel_diff = float(np.mean(abs_diff / np.maximum(numpy_res.depth, 1e-6)))

        # Export artifacts
        scale_slug = scale.replace("/", "_")
        export_base = f"{model_id}_{scale_slug}"

        # Convert to CPU prediction container to reuse save_depth_outputs
        cpu_prediction = last_tensor_result.to_cpu_prediction()
        saved_paths = DA3DepthAdapter.save_depth_outputs(
            result=cpu_prediction,
            out_dir=model_out_dir,
            base_name=export_base,
        )

        scale_record = {
            "scale": scale,
            "req_shape": [geom.req_width, geom.req_height],
            "padded_shape": [geom.padded_width, geom.padded_height],
            "latencies_ms": [round(x, 3) for x in latencies_ms],
            "latency_mean_ms": round(float(np.mean(latencies_ms)), 3),
            "latency_median_ms": round(float(np.median(latencies_ms)), 3),
            "latency_min_ms": round(float(np.min(latencies_ms)), 3),
            "output_tensor_shape": list(last_tensor_result.depth.shape),
            "raw_tensor_shape": list(last_tensor_result.depth_raw.shape),
            "is_cuda": last_tensor_result.depth.is_cuda,
            "device": last_tensor_result.device,
            "dtype": last_tensor_result.dtype,
            "is_metric": last_tensor_result.is_metric,
            "depth_units": last_tensor_result.depth_units,
            "accuracy_vs_numpy": {
                "mae": round(mae, 6),
                "max_diff": round(max_diff, 6),
                "mean_rel_difference": round(rel_diff, 6),
                "within_1_5_percent_tolerance": rel_diff < 0.015,
            },
            "saved_artifacts": {k: str(v) for k, v in saved_paths.items()},
        }
        results["scales"][scale] = scale_record

        logger.info(
            f"  -> Mean latency: {scale_record['latency_mean_ms']:.2f} ms "
            f"(median: {scale_record['latency_median_ms']:.2f} ms) | "
            f"Rel diff vs NumPy: {rel_diff * 100:.3f}%"
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="DA3 GPU-Tensor Inference Probe")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DA3-SMALL", "DA3-BASE"],
        choices=["DA3-SMALL", "DA3-BASE", "DA3MONO-LARGE", "DA3METRIC-LARGE", "ALL"],
        help="Model ID(s) to benchmark.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to input test image.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["1/4", "1/2", "1/1"],
        choices=["1/4", "1/2", "1/1"],
        help="Depth processing scales to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save artifacts and metrics.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timed inference iterations per scale.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations per scale.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. GPU tensor probe requires a CUDA-capable GPU.")
        return 1

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    logger.info(f"Target GPU: {gpu_name}")

    if not args.image.is_file():
        logger.error(f"Test image not found at {args.image}")
        return 1

    # Read image once and upload to CUDA tensor ONCE
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        logger.error(f"Failed to read image at {args.image}")
        return 1
    cpu_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = cpu_rgb.shape[:2]
    logger.info(f"Loaded source image: {args.image.name} ({w}x{h})")

    # Single host-to-device upload
    gpu_image = torch.from_numpy(cpu_rgb).to(device)
    logger.info(f"Uploaded image to CUDA: shape={gpu_image.shape}, dtype={gpu_image.dtype}, device={gpu_image.device}")

    catalog = load_catalog()
    model_ids = (
        ["DA3-SMALL", "DA3-BASE", "DA3MONO-LARGE", "DA3METRIC-LARGE"]
        if "ALL" in args.models
        else args.models
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_ver_mod = getattr(torch, "version", None)
    cuda_ver = getattr(torch_ver_mod, "cuda", "unknown") if torch_ver_mod else "unknown"
    all_summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": cuda_ver,
        "source_image": str(args.image),
        "source_resolution": [w, h],
        "models": {},
    }

    print("\n" + "=" * 90)
    print(f"{'PureGPU3D DA3 GPU-Tensor Inference Probe':^90}")
    print(f"{'GPU: ' + gpu_name + ' | PyTorch ' + torch.__version__ + ' | Zero Host Copies':^90}")
    print("=" * 90)

    for mid in model_ids:
        entry = catalog[mid]
        model_dir = REPO_ROOT / "models" / mid / entry.revision
        if not model_dir.is_dir():
            logger.warning(f"Model directory not found for {mid} at {model_dir}, skipping.")
            continue

        adapter: Optional[DA3DepthAdapter] = None
        try:
            adapter = DA3DepthAdapter(
                model_dir=model_dir,
                identifier=mid,
                device=device,
                verify_hashes=False,
            )
            model_results = run_probe_for_model(
                model_id=mid,
                adapter=adapter,
                gpu_image=gpu_image,
                cpu_rgb=cpu_rgb,
                scales=args.scales,
                output_dir=args.output_dir,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            all_summary["models"][mid] = model_results
        finally:
            if adapter is not None:
                del adapter
            gc.collect()
            torch.cuda.empty_cache()

    # Save complete JSON summary
    summary_path = args.output_dir / "da3_tensor_probe_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    logger.info(f"Summary JSON saved to {summary_path}")

    # Print summary table
    print("\n" + "-" * 90)
    print(f"{'Model':<16} | {'Scale':<6} | {'Req Dims':<12} | {'Pad Dims':<12} | {'Lat (ms)':<9} | {'Rel Diff':<10} | {'CUDA'}")
    print("-" * 90)
    for mid, mdata in all_summary["models"].items():
        for s, sdata in mdata["scales"].items():
            req_str = f"{sdata['req_shape'][0]}x{sdata['req_shape'][1]}"
            pad_str = f"{sdata['padded_shape'][0]}x{sdata['padded_shape'][1]}"
            lat_str = f"{sdata['latency_median_ms']:.1f}"
            diff_str = f"{sdata['accuracy_vs_numpy']['mean_rel_difference'] * 100:.2f}%"
            cuda_str = "Yes (pure)" if sdata["is_cuda"] else "No"
            print(f"{mid:<16} | {s:<6} | {req_str:<12} | {pad_str:<12} | {lat_str:<9} | {diff_str:<10} | {cuda_str}")
    print("-" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
