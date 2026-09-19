"""Comprehensive capability probe and benchmark for Depth Anything 3 models.

Probes real downloads, integrity verification, strict loading, and CUDA/CPU
depth inference for verified catalog models (Small, Base, Mono Large, Metric Large),
loading one model at a time on the GPU with VRAM tracking.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# IPv4 enforcement to prevent IPv6 DNS stall on Windows
import urllib3.util.connection
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import numpy as np
import torch

from puregpu3d.models.catalog import get_model_entry, load_catalog
from puregpu3d.models.da3_adapter import (
    DEFAULT_PROCESS_RES,
    DA3DepthAdapter,
    DepthPredictionResult,
    compute_sha256,
)
from puregpu3d.models.store import ModelStore, ModelStoreStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("models_probe")

MODELS_TO_PROBE = [
    "DA3-SMALL",
    "DA3-BASE",
    "DA3MONO-LARGE",
    "DA3METRIC-LARGE",
]

DEFAULT_SAMPLE_IMAGE = REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"
OUTPUT_BASE_DIR = REPO_ROOT / "data" / "verification" / "da3-models"


def probe_single_model(
    model_id: str,
    store: ModelStore,
    sample_image_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run full verification, download, loading, and dual-backend inference on one model."""
    catalog = load_catalog()
    entry = get_model_entry(model_id, catalog)
    model_out_dir = out_dir / model_id
    model_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info("PROBING MODEL: %s (%s)", model_id, entry.ui_name)
    logger.info("Repo: %s | Revision: %s | Params: %s", entry.repo_id, entry.revision, entry.parameters)
    logger.info("License: %s (%s)", entry.license_info.license, entry.license_info.license_type)

    # 1. App-managed download via ModelStore
    logger.info("Preparing model via ModelStore...")
    t_prep_start = time.perf_counter()
    rev_dir = store.prepare_model(identifier=model_id)
    t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0
    logger.info("Model directory ready at: %s (prep took %.1f ms)", rev_dir, t_prep_ms)

    # 2. Checkpoint integrity verification
    cfg_file = rev_dir / "config.json"
    wt_file = rev_dir / "model.safetensors"
    cfg_sha = compute_sha256(cfg_file)
    wt_sha = compute_sha256(wt_file)
    expected_cfg_sha = entry.files["config.json"].sha256
    expected_wt_sha = entry.files["model.safetensors"].sha256

    assert cfg_sha.lower() == expected_cfg_sha.lower(), f"Config SHA mismatch for {model_id}"
    assert wt_sha.lower() == expected_wt_sha.lower(), f"Weights SHA mismatch for {model_id}"
    logger.info("Verified SHA-256 integrity: config.json and model.safetensors match catalog.")

    # 3. CPU Inference Check
    logger.info("Initializing DA3DepthAdapter on CPU...")
    t_load_cpu_0 = time.perf_counter()
    cpu_adapter = DA3DepthAdapter(model_dir=rev_dir, identifier=model_id, device="cpu", verify_hashes=True)
    t_load_cpu_ms = (time.perf_counter() - t_load_cpu_0) * 1000.0

    logger.info("Running CPU inference (target_res=%d)...", DEFAULT_PROCESS_RES)
    cpu_result = cpu_adapter.infer(sample_image_path, target_size=DEFAULT_PROCESS_RES, return_original_size=True)
    logger.info("CPU inference completed in %.1f ms (min=%.4f, max=%.4f, mean=%.4f)",
                cpu_result.latency_ms, cpu_result.min_depth, cpu_result.max_depth, cpu_result.mean_depth)

    # Clean up CPU adapter
    del cpu_adapter
    gc.collect()

    # 4. CUDA Inference & VRAM Profiling
    has_cuda = torch.cuda.is_available()
    cuda_result: Optional[DepthPredictionResult] = None
    cuda_peak_vram_mb = 0.0
    cuda_alloc_vram_mb = 0.0
    t_load_cuda_ms = 0.0

    if has_cuda:
        device_str = "cuda:0"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_str)
        base_vram_bytes = torch.cuda.memory_allocated(device_str)

        logger.info("Initializing DA3DepthAdapter on CUDA (%s)...", torch.cuda.get_device_name(0))
        t_load_cuda_0 = time.perf_counter()
        cuda_adapter = DA3DepthAdapter(model_dir=rev_dir, identifier=model_id, device=device_str, verify_hashes=False)
        torch.cuda.synchronize(device_str)
        t_load_cuda_ms = (time.perf_counter() - t_load_cuda_0) * 1000.0

        # Warmup pass
        logger.info("Running CUDA warmup pass...")
        _ = cuda_adapter.infer(sample_image_path, target_size=DEFAULT_PROCESS_RES, return_original_size=True, autocast=True)

        # Timed benchmark pass
        torch.cuda.reset_peak_memory_stats(device_str)
        logger.info("Running timed CUDA inference benchmark...")
        cuda_result = cuda_adapter.infer(sample_image_path, target_size=DEFAULT_PROCESS_RES, return_original_size=True, autocast=True)

        cuda_peak_vram_mb = torch.cuda.max_memory_allocated(device_str) / (1024 * 1024)
        cuda_alloc_vram_mb = torch.cuda.memory_allocated(device_str) / (1024 * 1024)

        logger.info("CUDA inference completed: latency=%.1f ms | Peak VRAM=%.1f MB | Active VRAM=%.1f MB",
                    cuda_result.latency_ms, cuda_peak_vram_mb, cuda_alloc_vram_mb)

        # Numerical agreement between CPU and CUDA
        mae = float(np.mean(np.abs(cpu_result.depth - cuda_result.depth)))
        logger.info("CPU vs CUDA Mean Absolute Error: %.6f", mae)

        # Save artifacts using CUDA result
        saved_files = DA3DepthAdapter.save_depth_outputs(cuda_result, out_dir=model_out_dir, base_name=model_id.lower())

        # Clean up CUDA adapter
        del cuda_adapter
        gc.collect()
        torch.cuda.empty_cache()
    else:
        saved_files = DA3DepthAdapter.save_depth_outputs(cpu_result, out_dir=model_out_dir, base_name=model_id.lower())

    record: Dict[str, Any] = {
        "model_id": model_id,
        "ui_name": entry.ui_name,
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "category": entry.category,
        "role": entry.role,
        "license": entry.license_info.license,
        "license_type": entry.license_info.license_type,
        "is_metric": entry.id == "DA3METRIC-LARGE",
        "parameters": entry.parameters,
        "weight_bytes": entry.total_weight_bytes,
        "config_sha256": cfg_sha,
        "weights_sha256": wt_sha,
        "artifact_paths": {
            "model_dir": str(rev_dir),
            "npy": str(saved_files["npy"]),
            "png_u16": str(saved_files["png_u16"]),
            "png_color": str(saved_files["png_color"]),
            "metrics": str(saved_files["metrics"]),
        },
        "performance": {
            "load_time_cpu_ms": round(t_load_cpu_ms, 2),
            "load_time_cuda_ms": round(t_load_cuda_ms, 2) if has_cuda else None,
            "latency_cpu_ms": round(cpu_result.latency_ms, 2),
            "latency_cuda_ms": round(cuda_result.latency_ms, 2) if cuda_result else None,
            "peak_vram_mb": round(cuda_peak_vram_mb, 2) if has_cuda else None,
            "active_vram_mb": round(cuda_alloc_vram_mb, 2) if has_cuda else None,
        },
        "depth_metrics": {
            "input_shape": list(cpu_result.input_shape),
            "processed_shape": list(cpu_result.processed_shape),
            "min_depth": round(float(cpu_result.min_depth), 6),
            "max_depth": round(float(cpu_result.max_depth), 6),
            "mean_depth": round(float(cpu_result.mean_depth), 6),
            "finite_valid": bool(np.isfinite(cpu_result.depth).all()),
        },
    }

    record_path = model_out_dir / "summary.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    logger.info("Probe complete for %s. Artifacts written to %s", model_id, model_out_dir)
    return record


def main() -> int:
    """Run probe on all verified models and write consolidated verification report."""
    logger.info("Starting PureGPU3D Depth Anything 3 Model Suite Probe")
    logger.info("PyTorch: %s | CUDA Available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Device 0: %s (Total VRAM: %.1f MB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

    store = ModelStore()
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for model_id in MODELS_TO_PROBE:
        res = probe_single_model(
            model_id=model_id,
            store=store,
            sample_image_path=DEFAULT_SAMPLE_IMAGE,
            out_dir=OUTPUT_BASE_DIR,
        )
        results.append(res)

    # Consolidated JSON report
    consolidated_path = OUTPUT_BASE_DIR / "probe_results.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "verified_models": results,
                "unverified_models": DA3DepthAdapter.get_unverified_model_ids(),
            },
            f,
            indent=2,
        )

    logger.info("Consolidated results written to: %s", consolidated_path)

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Model ID':<18} | {'UI Name':<12} | {'Params':<7} | {'Weights':<10} | {'CUDA ms':<9} | {'Peak VRAM':<10} | {'Depth Min/Max':<18} | {'Type':<10}")
    print("-" * 105)
    for r in results:
        m_id = r["model_id"]
        ui_n = r["ui_name"]
        params = r["parameters"]
        wt_mb = f"{r['weight_bytes'] / (1024*1024):.1f}MB"
        cuda_lat = f"{r['performance']['latency_cuda_ms']}ms" if r['performance']['latency_cuda_ms'] else "N/A"
        vram = f"{r['performance']['peak_vram_mb']}MB" if r['performance']['peak_vram_mb'] else "N/A"
        d_range = f"{r['depth_metrics']['min_depth']:.2f}..{r['depth_metrics']['max_depth']:.2f}"
        m_type = "Metric" if r["is_metric"] else "Relative"
        print(f"{m_id:<18} | {ui_n:<12} | {params:<7} | {wt_mb:<10} | {cuda_lat:<9} | {vram:<10} | {d_range:<18} | {m_type:<10}")
    print("=" * 105 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
