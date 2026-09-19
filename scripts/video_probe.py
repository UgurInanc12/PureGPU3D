#!/usr/bin/env python3
"""Video probing and stereoscopic conversion CLI tool for PureGPU3D.

Exercises the video vertical slice:
  1. Inspects input video media and verifies constraints (CFR, SDR BT.709, even dims).
  2. Runs streaming conversion using Depth Anything 3 Small and the PyTorch stereo renderer.
  3. Outputs Full-SBS (2W x H) video with preserved audio.
  4. Exports structured verification evidence to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure repository root src/ and vendor are reachable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import DEFAULT_REVISION, DA3SmallDepthAdapter
from puregpu3d.stereo import DisparityConfig, FillConfig, SplatConfig, StereoConfig
from puregpu3d.video import (
    ConversionResult,
    UnsupportedMediaError,
    VideoProbeResult,
    convert_video,
    probe_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("video_probe")

DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PureGPU3D Video Probe and Stereoscopic Conversion Slice"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Full-SBS video file path (default: data/verification/video/<input_stem>_full_sbs.mp4)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to DA3 Small model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device (default: cuda)",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only probe and display input media characteristics without converting",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output destination if it exists",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.03,
        help="Disparity shift strength (default: 0.03)",
    )
    parser.add_argument(
        "--q-screen",
        type=float,
        default=0.6,
        help="Zero parallax convergence plane in [0, 1] (default: 0.6)",
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        default=None,
        help="Path to export verification evidence JSON",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        logger.info(f"Probing media: {input_path}")
        probe_res = probe_video(input_path, strict_sdr_cfr=True)
    except UnsupportedMediaError as err:
        logger.error(f"Input media rejected by vertical slice guard: {err}")
        return 2
    except Exception as err:
        logger.error(f"Failed to probe media: {err}")
        return 1

    logger.info(
        f"Probe successful: {probe_res.width}x{probe_res.height} @ {probe_res.fps:.3f} fps, "
        f"duration={probe_res.duration:.2f}s, frames={probe_res.frame_count}, "
        f"audio={'yes' if probe_res.has_audio else 'no'}, pix_fmt={probe_res.pix_fmt}"
    )

    if args.probe_only:
        print(json.dumps(probe_res.to_dict(), indent=2))
        return 0

    # Determine output path
    if args.output is not None:
        output_path = args.output.resolve()
    else:
        out_dir = REPO_ROOT / "data" / "verification" / "video"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_path.stem}_full_sbs.mp4"

    logger.info(f"Target output: {output_path}")

    # Build stereo config
    stereo_cfg = StereoConfig(
        disparity=DisparityConfig(strength=args.strength, q_screen=args.q_screen),
        splat=SplatConfig(),
        fill=FillConfig(),
    )

    # Convert video
    logger.info(f"Starting conversion using model at: {args.model_dir}")
    t0 = time.perf_counter()

    try:
        conv_res = convert_video(
            input_path=input_path,
            output_path=output_path,
            model=args.model_dir,
            device=args.device,
            stereo_config=stereo_cfg,
            overwrite=args.overwrite,
            progress_callback=lambda cur, tot: logger.info(f"Progress: {cur}/{tot} frames"),
        )
    except Exception as err:
        logger.error(f"Conversion failed: {err}")
        return 3

    t1 = time.perf_counter()
    logger.info(f"Conversion succeeded in {t1 - t0:.2f}s. Effective FPS: {conv_res.effective_fps:.2f}")

    evidence_dict: Dict[str, Any] = {
        "probe": probe_res.to_dict(),
        "conversion": conv_res.to_dict(),
        "stereo_config": {
            "strength": args.strength,
            "q_screen": args.q_screen,
        },
        "model_dir": str(args.model_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.evidence_json is not None:
        evidence_path = args.evidence_json.resolve()
    else:
        evidence_path = output_path.parent / f"{output_path.stem}_evidence.json"

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_dict, f, indent=2)
    logger.info(f"Wrote verification evidence to: {evidence_path}")

    print(json.dumps(evidence_dict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
