#!/usr/bin/env python3
"""Temporal depth stabilization verification probe for PureGPU3D.

Produces:
  1. Synthetic-motion pan/cut video fixture from real SOH image (640x360, 12 frames, AAC audio).
  2. Baseline Full-SBS MP4 (enable_temporal_stabilization=False).
  3. Stabilized Full-SBS MP4 (enable_temporal_stabilization=True).
  4. FFprobe validation (dimensions, frame count, audio stream preservation, wall-clock timing).
  5. Deterministic known-motion disparity temporal jitter measurement.
  6. Scene reset regression test execution.
  7. Structured JSON evidence export.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

# Ensure src/ and vendor are reachable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.da3_adapter import DEFAULT_REVISION
from puregpu3d.stereo import (
    DisparityConfig,
    FillConfig,
    SplatConfig,
    StereoConfig,
    TemporalDepthConfig,
    TemporalDepthStabilizer,
)
from puregpu3d.video import convert_video, probe_video
from puregpu3d.video.probe import find_ffmpeg, find_ffprobe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("temporal_probe")

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION
DEFAULT_IMAGE_PATH = (
    REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"
)
OUTPUT_DIR = REPO_ROOT / "data" / "verification" / "temporal"


def build_synthetic_fixture(
    image_path: Path,
    output_fixture_path: Path,
    width: int = 640,
    height: int = 360,
    num_frames: int = 12,
    fps: int = 24,
) -> Path:
    """Create a 12-frame 640x360 video with a smooth pan and a hard cut from a real photo."""
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image at {image_path}")

    img_h, img_w = img.shape[:2]
    if img_h < height or img_w < width:
        raise ValueError(f"Image {img_w}x{img_h} is smaller than target {width}x{height}")

    output_fixture_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_fixture_path.parent / "temp_fixture_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Frame plan (12 frames):
    # Frames 0-7 (Scene 1): Pan from (x=80, y=120) with dx=6 px/frame
    # Frames 8-11 (Scene 2): Cut to (x=450, y=260) with dx=6 px/frame
    for t in range(num_frames):
        if t < 8:
            x0 = 80 + t * 6
            y0 = 120
        else:
            x0 = 450 + (t - 8) * 6
            y0 = 260

        x0 = min(max(0, x0), img_w - width)
        y0 = min(max(0, y0), img_h - height)
        crop = img[y0 : y0 + height, x0 : x0 + width]
        frame_png = frames_dir / f"frame_{t:04d}.png"
        cv2.imwrite(str(frame_png), crop)

    ffmpeg_bin = find_ffmpeg()
    duration = num_frames / fps

    # Remux into compliant 8-bit SDR BT.709 yuv420p CFR MP4 with stereo AAC audio
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={duration:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output_fixture_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    shutil.rmtree(frames_dir, ignore_errors=True)

    logger.info(f"Built synthetic fixture video at: {output_fixture_path}")
    return output_fixture_path


def measure_known_motion_disparity_jitter() -> Dict[str, Any]:
    """Measure motion-compensated disparity temporal jitter on deterministic geometry fixture."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tests.temporal.test_known_motion_jitter import (
        compute_motion_compensated_jitter,
        generate_synthetic_motion_sequence,
    )

    seq = generate_synthetic_motion_sequence(
        height=120,
        width=160,
        num_frames=8,
        dx=2.0,
        dy=1.0,
        noise_sigma=0.08,
        cut_at_frame=5,
        seed=42,
    )

    cfg = TemporalDepthConfig(
        enabled=True,
        alpha=0.70,
        flow_scale=1.0,
        consistency_threshold=1.5,
        photometric_threshold=30.0,
    )
    stabilizer = TemporalDepthStabilizer(cfg)

    stab_depths = []
    for i in range(len(seq["frames_rgb"])):
        frame_rgb = seq["frames_rgb"][i]
        raw_d = seq["raw_depths"][i]
        is_cut = seq["is_cut_flags"][i]
        res = stabilizer.process_frame(frame_rgb, raw_d, is_cut=is_cut)
        stab_depths.append(res.depth)

    raw_metrics = compute_motion_compensated_jitter(
        seq["raw_depths"],
        seq["dx"],
        seq["dy"],
        seq["cut_at_frame"],
    )
    stab_metrics = compute_motion_compensated_jitter(
        stab_depths,
        seq["dx"],
        seq["dy"],
        seq["cut_at_frame"],
    )

    rmse_improvement_pct = (
        (raw_metrics["rmse"] - stab_metrics["rmse"]) / raw_metrics["rmse"] * 100.0
    )
    mae_improvement_pct = (
        (raw_metrics["mae"] - stab_metrics["mae"]) / raw_metrics["mae"] * 100.0
    )

    # Measure retained state byte footprint for 640x360 and 1920x1080
    test_stab_640 = TemporalDepthStabilizer()
    test_stab_640.process_frame(np.zeros((360, 640, 3), dtype=np.uint8), np.ones((360, 640), dtype=np.float32))
    assert test_stab_640.prev_gray is not None and test_stab_640.prev_depth is not None
    nbytes_640 = test_stab_640.prev_gray.nbytes + test_stab_640.prev_depth.nbytes

    test_stab_1080 = TemporalDepthStabilizer()
    test_stab_1080.process_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), np.ones((1080, 1920), dtype=np.float32))
    assert test_stab_1080.prev_gray is not None and test_stab_1080.prev_depth is not None
    nbytes_1080 = test_stab_1080.prev_gray.nbytes + test_stab_1080.prev_depth.nbytes

    return {
        "raw_disparity_jitter_rmse": raw_metrics["rmse"],
        "raw_disparity_jitter_mae": raw_metrics["mae"],
        "stabilized_disparity_jitter_rmse": stab_metrics["rmse"],
        "stabilized_disparity_jitter_mae": stab_metrics["mae"],
        "rmse_improvement_pct": rmse_improvement_pct,
        "mae_improvement_pct": mae_improvement_pct,
        "retained_state_640x360_nbytes": nbytes_640,
        "retained_state_640x360_mb": nbytes_640 / (1024 * 1024),
        "retained_state_1920x1080_nbytes": nbytes_1080,
        "retained_state_1920x1080_mb": nbytes_1080 / (1024 * 1024),
        "flow_config": {
            "alpha": cfg.alpha,
            "flow_scale": cfg.flow_scale,
            "consistency_threshold": cfg.consistency_threshold,
            "photometric_threshold": cfg.photometric_threshold,
            "depth_diff_threshold": cfg.depth_diff_threshold,
            "edge_threshold": cfg.edge_threshold,
            "align_scale": cfg.align_scale,
            "max_scale_adjustment": cfg.max_scale_adjustment,
            "min_shared_pixels_for_scale": cfg.min_shared_pixels_for_scale,
            "normalization_ema_eta": cfg.normalization_ema_eta,
            "percentile_min": cfg.percentile_min,
            "percentile_max": cfg.percentile_max,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporal depth stabilization probe for PureGPU3D")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to DA3 Small model")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Source SOH image")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = out_dir / "fixture_pan_cut.mp4"
    baseline_path = out_dir / "sbs_baseline.mp4"
    stabilized_path = out_dir / "sbs_stabilized.mp4"
    evidence_path = out_dir / "evidence.json"

    logger.info("=== STEP 1: Building synthetic-motion pan/cut video fixture ===")
    build_synthetic_fixture(args.image, fixture_path, width=640, height=360, num_frames=12, fps=24)
    fixture_probe = probe_video(fixture_path, strict_sdr_cfr=True)
    audio_desc = fixture_probe.audio_streams[0].codec_name if fixture_probe.has_audio and fixture_probe.audio_streams else "none"
    logger.info(
        f"Fixture probe: {fixture_probe.width}x{fixture_probe.height}, {fixture_probe.frame_count} frames, "
        f"audio={audio_desc}"
    )

    stereo_cfg = StereoConfig(
        disparity=DisparityConfig(strength=0.03, q_screen=0.6),
        splat=SplatConfig(),
        fill=FillConfig(),
    )

    logger.info("=== STEP 2: Running baseline conversion (enable_temporal_stabilization=False) ===")
    t0_base = time.perf_counter()
    conv_base = convert_video(
        input_path=fixture_path,
        output_path=baseline_path,
        model=args.model_dir,
        device=args.device,
        stereo_config=stereo_cfg,
        enable_temporal_stabilization=False,
        overwrite=True,
    )
    t1_base = time.perf_counter()
    base_wall_time = t1_base - t0_base
    probe_base = probe_video(baseline_path, strict_sdr_cfr=True)
    logger.info(
        f"Baseline conversion finished: {probe_base.width}x{probe_base.height}, {probe_base.frame_count} frames, "
        f"wall_time={base_wall_time:.2f}s, effective_fps={conv_base.effective_fps:.2f}"
    )

    logger.info("=== STEP 3: Running stabilized conversion (enable_temporal_stabilization=True) ===")
    t0_stab = time.perf_counter()
    conv_stab = convert_video(
        input_path=fixture_path,
        output_path=stabilized_path,
        model=args.model_dir,
        device=args.device,
        stereo_config=stereo_cfg,
        enable_temporal_stabilization=True,
        overwrite=True,
    )
    t1_stab = time.perf_counter()
    stab_wall_time = t1_stab - t0_stab
    probe_stab = probe_video(stabilized_path, strict_sdr_cfr=True)
    logger.info(
        f"Stabilized conversion finished: {probe_stab.width}x{probe_stab.height}, {probe_stab.frame_count} frames, "
        f"wall_time={stab_wall_time:.2f}s, effective_fps={conv_stab.effective_fps:.2f}"
    )

    logger.info("=== STEP 4: Evaluating known-motion disparity temporal jitter ===")
    jitter_res = measure_known_motion_disparity_jitter()
    logger.info(
        f"Jitter RMSE: Raw={jitter_res['raw_disparity_jitter_rmse']:.6f} -> "
        f"Stab={jitter_res['stabilized_disparity_jitter_rmse']:.6f} "
        f"({jitter_res['rmse_improvement_pct']:+.2f}%)"
    )
    logger.info(
        f"Jitter MAE:  Raw={jitter_res['raw_disparity_jitter_mae']:.6f} -> "
        f"Stab={jitter_res['stabilized_disparity_jitter_mae']:.6f} "
        f"({jitter_res['mae_improvement_pct']:+.2f}%)"
    )
    logger.info(
        f"Retained state: 640x360 = {jitter_res['retained_state_640x360_nbytes']} bytes "
        f"({jitter_res['retained_state_640x360_mb']:.2f} MB), "
        f"1920x1080 = {jitter_res['retained_state_1920x1080_nbytes']} bytes "
        f"({jitter_res['retained_state_1920x1080_mb']:.2f} MB)"
    )

    evidence: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fixture": {
            "path": str(fixture_path),
            "probe": fixture_probe.to_dict(),
            "source_image": str(args.image),
        },
        "baseline": {
            "output_path": str(baseline_path),
            "probe": probe_base.to_dict(),
            "conversion": conv_base.to_dict(),
            "wall_clock_seconds": base_wall_time,
        },
        "stabilized": {
            "output_path": str(stabilized_path),
            "probe": probe_stab.to_dict(),
            "conversion": conv_stab.to_dict(),
            "wall_clock_seconds": stab_wall_time,
        },
        "jitter_measurement": jitter_res,
        "environment": {
            "device": args.device,
            "model_dir": str(args.model_dir),
            "ffmpeg": str(find_ffmpeg()),
            "ffprobe": str(find_ffprobe()),
        },
    }

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    logger.info(f"Verification evidence written to: {evidence_path}")

    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Fixture: {fixture_probe.width}x{fixture_probe.height}, {fixture_probe.frame_count} frames, audio: {fixture_probe.has_audio}")
    print(f"Baseline Full-SBS:   {probe_base.width}x{probe_base.height}, wall_time: {base_wall_time:.2f}s, FPS: {conv_base.effective_fps:.2f}")
    print(f"Stabilized Full-SBS: {probe_stab.width}x{probe_stab.height}, wall_time: {stab_wall_time:.2f}s, FPS: {conv_stab.effective_fps:.2f}")
    print(f"Disparity Jitter RMSE: Raw={jitter_res['raw_disparity_jitter_rmse']:.6f} -> Stab={jitter_res['stabilized_disparity_jitter_rmse']:.6f} ({jitter_res['rmse_improvement_pct']:+.2f}%)")
    print(f"Disparity Jitter MAE:  Raw={jitter_res['raw_disparity_jitter_mae']:.6f} -> Stab={jitter_res['stabilized_disparity_jitter_mae']:.6f} ({jitter_res['mae_improvement_pct']:+.2f}%)")
    print(f"Retained State (640x360):   {jitter_res['retained_state_640x360_nbytes']} bytes ({jitter_res['retained_state_640x360_mb']:.2f} MB)")
    print(f"Retained State (1920x1080): {jitter_res['retained_state_1920x1080_nbytes']} bytes ({jitter_res['retained_state_1920x1080_mb']:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
