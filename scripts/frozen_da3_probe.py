#!/usr/bin/env python3
"""DA3 Small Frozen Feasibility Probe for PureGPU3D.

Self-contained executable entrypoint to verify bundled DA3 imports,
bundled ffprobe binary, adjacent models directory, and actual CPU / CUDA
depth inference in frozen Windows environments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure puregpu3d and depth_anything_3 are resolvable
if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", EXE_DIR))
else:
    EXE_DIR = Path(__file__).resolve().parents[1]
    BUNDLE_DIR = EXE_DIR
    src_path = EXE_DIR / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    da3_path = EXE_DIR / "third_party" / "depth_anything_3" / "src"
    if str(da3_path) not in sys.path:
        sys.path.insert(0, str(da3_path))

from puregpu3d.models.da3_adapter import (
    DEFAULT_PROCESS_RES,
    DEFAULT_REVISION,
    DA3SmallDepthAdapter,
    DepthPredictionResult,
    ensure_da3_vendor_import,
)

logger = logging.getLogger("frozen_da3_probe")


def resolve_bundled_ffprobe() -> Tuple[Optional[Path], Dict[str, Any]]:
    """Discover bundled ffprobe.exe inside or adjacent to the frozen bundle."""
    candidates = [
        EXE_DIR / "bin" / "ffprobe.exe",
        BUNDLE_DIR / "bin" / "ffprobe.exe",
        EXE_DIR / "ffprobe.exe",
        BUNDLE_DIR / "ffprobe.exe",
        EXE_DIR / "_internal" / "bin" / "ffprobe.exe",
        BUNDLE_DIR / "_internal" / "bin" / "ffprobe.exe",
    ]

    found_path: Optional[Path] = None
    for cand in candidates:
        if cand.is_file():
            found_path = cand.resolve()
            break

    info: Dict[str, Any] = {
        "found": found_path is not None,
        "path": str(found_path) if found_path else None,
        "bundled": True if found_path else False,
        "version": None,
        "provenance": None,
        "license": None,
        "error": None,
    }

    if not found_path:
        info["error"] = "Bundled ffprobe.exe was not found in package bundle."
        return None, info

    try:
        proc = subprocess.run(
            [str(found_path), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            first_line = proc.stdout.strip().splitlines()[0] if proc.stdout else ""
            info["version"] = first_line
            info["provenance"] = "Gyan.dev FFmpeg build (bundled into bin/ffprobe.exe)"
            info["license"] = "GPLv3 / LGPL"
        else:
            info["error"] = f"ffprobe exited with return code {proc.returncode}: {proc.stderr.strip()}"
    except Exception as exc:
        info["error"] = f"Failed to execute bundled ffprobe: {exc}"

    return found_path, info


def resolve_model_dir(explicit_dir: Optional[Path] = None) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Locate adjacent or specified DA3 Small model directory."""
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir).resolve())

    # Adjacent models directory
    candidates.extend([
        EXE_DIR / "models" / "DA3-SMALL" / DEFAULT_REVISION,
        EXE_DIR / "models" / "DA3-SMALL",
        EXE_DIR / "models",
        EXE_DIR.parent / "models" / "DA3-SMALL" / DEFAULT_REVISION,
        EXE_DIR.parent / "models",
    ])

    found_dir: Optional[Path] = None
    for cand in candidates:
        if (cand / "config.json").is_file() and (cand / "model.safetensors").is_file():
            found_dir = cand.resolve()
            break

    info: Dict[str, Any] = {
        "found": found_dir is not None,
        "path": str(found_dir) if found_dir else None,
        "revision": DEFAULT_REVISION,
        "has_config": bool(found_dir and (found_dir / "config.json").is_file()),
        "has_weights": bool(found_dir and (found_dir / "model.safetensors").is_file()),
        "has_manifest": bool(found_dir and (found_dir / "manifest.json").is_file()),
        "error": None,
    }

    if not found_dir:
        info["error"] = "Adjacent DA3-SMALL model directory with config.json and model.safetensors not found."

    return found_dir, info


def create_synthetic_test_image(height: int = 504, width: int = 504) -> np.ndarray:
    """Generate a deterministic synthetic test pattern for self-test inference."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Vertical gradient on red channel
    y_coords = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    img[:, :, 0] = np.repeat(y_coords, width, axis=1)
    # Horizontal gradient on green channel
    x_coords = np.linspace(0, 255, width, dtype=np.uint8).reshape(1, -1)
    img[:, :, 1] = np.repeat(x_coords, height, axis=0)
    # Diagonal and geometric pattern on blue channel
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4, (0, 0, 255), -1)
    cv2.rectangle(img, (width // 8, height // 8), (width // 3, height // 3), (255, 255, 0), -1)
    return img


def run_probe_self_test(
    device_choice: str = "both",
    model_dir: Optional[Path] = None,
    image_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    window_smoke: bool = False,
) -> Dict[str, Any]:
    """Execute complete self-test of frozen probe capabilities."""
    start_time = time.time()
    report: Dict[str, Any] = {
        "status": "pending",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen": getattr(sys, "frozen", False),
        "executable": str(Path(sys.executable).resolve()),
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bundled_ffprobe": {},
        "da3_vendor_import": {},
        "model": {},
        "cpu_inference": None,
        "cuda_inference": None,
        "comparison": None,
        "window_smoke": None,
        "verification_scope": {
            "environment": "local isolated-path verification",
            "clean_machine_vm_gap": "Local hostile-PATH run outside repo; real clean-machine VM verification remains an external CI gap.",
        },
        "errors": [],
    }

    # 1. Probe bundled ffprobe
    _, ffprobe_info = resolve_bundled_ffprobe()
    report["bundled_ffprobe"] = ffprobe_info
    if not ffprobe_info["found"] or ffprobe_info["error"]:
        report["errors"].append(f"Bundled ffprobe check failed: {ffprobe_info.get('error')}")

    # 2. Probe DA3 vendor import
    try:
        vendor_path = ensure_da3_vendor_import()
        import depth_anything_3
        report["da3_vendor_import"] = {
            "success": True,
            "vendor_path": str(vendor_path),
            "module_file": getattr(depth_anything_3, "__file__", None),
        }
    except Exception as exc:
        report["da3_vendor_import"] = {"success": False, "error": str(exc)}
        report["errors"].append(f"DA3 vendor import failed: {exc}")

    # 3. Probe model directory
    resolved_model_dir, model_info = resolve_model_dir(model_dir)
    report["model"] = model_info
    if not resolved_model_dir or model_info["error"]:
        report["errors"].append(f"Model resolution failed: {model_info.get('error')}")

    if report["errors"] or resolved_model_dir is None:
        report["status"] = "failed"
        return report

    # Prepare input image
    if image_path and Path(image_path).is_file():
        image_input = Path(image_path).resolve()
        image_name = image_input.name
    else:
        image_input = create_synthetic_test_image(504, 504)
        image_name = "synthetic_504x504.png"

    if output_dir:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    # Determine devices to run
    cuda_requested = device_choice in ("cuda", "both") and torch.cuda.is_available()
    cpu_requested = device_choice in ("cpu", "both") or not torch.cuda.is_available()

    adapter: Optional[DA3SmallDepthAdapter] = None
    cpu_res: Optional[DepthPredictionResult] = None
    cuda_res: Optional[DepthPredictionResult] = None

    # 4. CPU inference
    if cpu_requested:
        try:
            logger.info("Initializing DA3SmallDepthAdapter on CPU...")
            adapter_cpu = DA3SmallDepthAdapter(
                model_dir=resolved_model_dir,
                device="cpu",
                verify_hashes=True,
            )
            logger.info("Running CPU depth inference...")
            cpu_res = adapter_cpu.infer(image=image_input, target_size=504, return_original_size=True)
            report["cpu_inference"] = cpu_res.to_dict()

            if out_path:
                saved = DA3SmallDepthAdapter.save_depth_outputs(
                    result=cpu_res,
                    out_dir=out_path,
                    base_name="depth_cpu",
                )
                report["cpu_inference"]["saved_files"] = {k: str(v) for k, v in saved.items()}
        except Exception as exc:
            report["errors"].append(f"CPU inference failed: {exc}")
            logger.exception("CPU inference failed")

    # 5. CUDA inference
    if cuda_requested:
        try:
            logger.info("Initializing DA3SmallDepthAdapter on CUDA...")
            adapter_cuda = DA3SmallDepthAdapter(
                model_dir=resolved_model_dir,
                device="cuda:0",
                verify_hashes=True,
            )
            logger.info("Running CUDA warmup...")
            _ = adapter_cuda.infer(image=image_input, target_size=504, return_original_size=True, autocast=True)
            logger.info("Running CUDA depth inference...")
            cuda_res = adapter_cuda.infer(image=image_input, target_size=504, return_original_size=True, autocast=True)
            report["cuda_inference"] = cuda_res.to_dict()

            if out_path:
                saved = DA3SmallDepthAdapter.save_depth_outputs(
                    result=cuda_res,
                    out_dir=out_path,
                    base_name="depth_cuda",
                )
                report["cuda_inference"]["saved_files"] = {k: str(v) for k, v in saved.items()}
        except Exception as exc:
            report["errors"].append(f"CUDA inference failed: {exc}")
            logger.exception("CUDA inference failed")

    # 6. Comparison if both available
    if cpu_res and cuda_res:
        diff = np.abs(cpu_res.depth - cuda_res.depth)
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_diff = float(np.max(diff))
        rel_diff = float(np.mean(diff / np.clip(cpu_res.depth, 1e-6, None)))
        speedup = cpu_res.latency_ms / max(cuda_res.latency_ms, 1e-6)

        report["comparison"] = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "max_absolute_difference": round(max_diff, 6),
            "mean_relative_difference": round(rel_diff, 6),
            "cpu_latency_ms": round(cpu_res.latency_ms, 2),
            "cuda_latency_ms": round(cuda_res.latency_ms, 2),
            "cuda_speedup_x": round(speedup, 2),
        }

    # 7. Optional window smoke
    if window_smoke and sys.platform == "win32":
        try:
            import ctypes
            # MessageBoxTimeoutW with 1 second timeout (0x00000000 = MB_OK)
            ctypes.windll.user32.MessageBoxTimeoutW(
                0,
                "PureGPU3D DA3 Feasibility Probe - Self Test Successful",
                "PureGPU3D Window Smoke",
                0,
                0,
                1000,
            )
            report["window_smoke"] = {"success": True, "type": "user32_message_box_timeout"}
        except Exception as exc:
            report["window_smoke"] = {"success": False, "error": str(exc)}

    report["elapsed_total_seconds"] = round(time.time() - start_time, 3)
    if not report["errors"]:
        report["status"] = "success"
    else:
        report["status"] = "failed"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="DA3 Small Frozen Feasibility Probe")
    parser.add_argument("--self-test", action="store_true", default=True, help="Execute full self-test suite")
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both", help="Inference device")
    parser.add_argument("--model-dir", type=Path, default=None, help="Explicit path to DA3 Small model directory")
    parser.add_argument("--image", type=Path, default=None, help="Input test image path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save depth outputs")
    parser.add_argument("--json-out", type=Path, default=None, help="File path to save JSON results")
    parser.add_argument("--window", action="store_true", default=False, help="Trigger native window smoke check")
    parser.add_argument("--quiet", action="store_true", default=False, help="Output JSON only to stdout")

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    results = run_probe_self_test(
        device_choice=args.device,
        model_dir=args.model_dir,
        image_path=args.image,
        output_dir=args.output_dir,
        window_smoke=args.window,
    )

    json_str = json.dumps(results, indent=2)
    print(json_str)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(json_str)

    return 0 if results["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
