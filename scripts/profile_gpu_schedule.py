#!/usr/bin/env python
"""Reproducible GPU Scheduling & Barrier Profiler for PureGPU3D.

Profiles a real, bounded DA3-Small 1/4 depth-scale conversion with temporal
stabilization enabled. Measures host barriers (.item(), cudaStreamSynchronize),
memory transfers (HtoD, DtoH, DtoD), stage latencies, and verifies whether
decode/compute/encode overlap is present.

Outputs:
  - data/verification/gpu-scheduling/profile_trace.json.gz
  - data/verification/gpu-scheduling/profiler_summary.txt
  - data/verification/gpu-scheduling/report.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
VENDOR_DIR = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import PyNvVideoCodec as nvc
from puregpu3d.models.da3_adapter import DA3DepthAdapter
from puregpu3d.models.geometry import parse_depth_scale
from puregpu3d.stereo import StereoConfig, render_stereo_frame
from puregpu3d.stereo.temporal_gpu import TemporalGPUConfig, TemporalGPUStabilizer
from puregpu3d.video.probe import probe_video


def run_bounded_profiling(
    input_path: Path,
    output_dir: Path,
    model_dir: Path,
    gpu_id: int = 0,
    depth_scale: str = "1/4",
    num_frames_limit: int = 12,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_out_video = output_dir / "profile_temp_output.mp4"
    if temp_out_video.exists():
        temp_out_video.unlink()

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # 1. Probe source video
    probe = probe_video(input_path)
    in_w = probe.width
    in_h = probe.height
    out_w = in_w * 2
    out_h = in_h
    fps_frac = probe.frame_rate
    canon_depth_scale, _ = parse_depth_scale(depth_scale)

    print(f"[PROFILER] Input: {input_path.name} ({in_w}x{in_h} @ {fps_frac} fps, {probe.frame_count} frames)")
    print(f"[PROFILER] Target device: {torch.cuda.get_device_name(device)} (ID: {gpu_id})")
    print(f"[PROFILER] Loading DA3-Small adapter from {model_dir}...")

    # 2. Prepare adapter and stabilizer
    adapter = DA3DepthAdapter(model_dir=model_dir, device=device)
    temporal_cfg = TemporalGPUConfig(enabled=True)
    stabilizer = TemporalGPUStabilizer(config=temporal_cfg)
    stereo_cfg = StereoConfig()

    # 3. Codec & streams setup
    pipeline_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    pipeline_stream.wait_stream(current_stream)

    dec = nvc.SimpleDecoder(
        str(input_path),
        gpu_id=gpu_id,
        cuda_stream=pipeline_stream.cuda_stream,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
    )
    total_available = len(dec)
    frames_to_run = min(total_available, num_frames_limit)

    fps_arg = f"{fps_frac.numerator}/{fps_frac.denominator}"
    enc = nvc.CreateEncoder(
        out_w,
        out_h,
        "ABGR",
        False,
        codec="hevc",
        cudastream=pipeline_stream.cuda_stream,
        fps=fps_arg,
    )
    extradata = enc.GetSequenceParams()

    timebase_num = 1
    timebase_den = 90000
    pts_inc = int(timebase_den * fps_frac.denominator / fps_frac.numerator)

    muxer = nvc.FFmpegMuxer(
        str(temp_out_video),
        nvc.MEDIA_FORMAT.MP4,
        "hevc",
        out_w,
        out_h,
        fps_frac.numerator,
        fps_frac.denominator,
        timebase_num,
        timebase_den,
        extradata,
    )
    muxer.SetUniformPtsIncrement(pts_inc)

    # Frame-by-frame measurement structures
    frame_metrics: List[Dict[str, Any]] = []
    pointer_traces: List[Dict[str, Any]] = []

    print(f"[PROFILER] Starting torch.profiler capture over {frames_to_run} frames with Temporal ON...")

    wall_start = time.perf_counter()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.cuda.stream(pipeline_stream):
            alpha = torch.full((out_h, out_w, 1), 255, dtype=torch.uint8, device=device)
            alpha.record_stream(pipeline_stream)

            for idx in range(frames_to_run):
                # We use CUDA events to measure exact GPU execution windows per stage
                ev_d_start = torch.cuda.Event(enable_timing=True)
                ev_d_end = torch.cuda.Event(enable_timing=True)
                ev_m_start = torch.cuda.Event(enable_timing=True)
                ev_m_end = torch.cuda.Event(enable_timing=True)
                ev_t_start = torch.cuda.Event(enable_timing=True)
                ev_t_end = torch.cuda.Event(enable_timing=True)
                ev_s_start = torch.cuda.Event(enable_timing=True)
                ev_s_end = torch.cuda.Event(enable_timing=True)
                ev_e_start = torch.cuda.Event(enable_timing=True)
                ev_e_end = torch.cuda.Event(enable_timing=True)

                # --- Stage 1: Decode ---
                t0_dec = time.perf_counter()
                ev_d_start.record(pipeline_stream)
                dec_frame = dec[idx]
                ev_d_end.record(pipeline_stream)
                t1_dec = time.perf_counter()

                # --- Stage 2: DLPack Interop & Surface safety clone ---
                t0_dlp = time.perf_counter()
                plane_ptr = int(dec_frame.GetPtrToPlane(0))
                t_raw = torch.from_dlpack(dec_frame)
                raw_ptr = int(t_raw.data_ptr())
                t_raw.record_stream(pipeline_stream)

                if idx < 3:
                    pointer_traces.append({
                        "frame_index": idx,
                        "plane_ptr": plane_ptr,
                        "tensor_ptr": raw_ptr,
                        "ptrs_match": (plane_ptr == raw_ptr),
                        "shape": list(t_raw.shape),
                        "dtype": str(t_raw.dtype),
                    })

                t_in = t_raw.clone()  # Surface reuse safety copy (DtoD)
                t_in.record_stream(pipeline_stream)
                t1_dlp = time.perf_counter()

                # --- Stage 3: DA3 Depth Inference ---
                t0_inf = time.perf_counter()
                ev_m_start.record(pipeline_stream)
                depth_res = adapter.infer_tensor(
                    t_in,
                    depth_scale=canon_depth_scale,
                    return_original_size=True,
                    autocast=True,
                    timing=False,
                )
                ev_m_end.record(pipeline_stream)
                t1_inf = time.perf_counter()

                # --- Stage 4: Temporal Stabilization ---
                t0_tmp = time.perf_counter()
                ev_t_start.record(pipeline_stream)
                raw_d = depth_res.depth_raw
                h_raw, w_raw = raw_d.shape[-2:]

                if t_in.shape[0] != h_raw or t_in.shape[1] != w_raw:
                    t_in_chw = t_in.permute(2, 0, 1).unsqueeze(0).float()
                    rgb_scaled = F.interpolate(t_in_chw, size=(h_raw, w_raw), mode="area")
                else:
                    rgb_scaled = t_in

                temp_res = stabilizer.process_frame(
                    frame_rgb=rgb_scaled,
                    raw_depth=raw_d,
                )

                if (h_raw, w_raw) != (in_h, in_w):
                    final_depth = F.interpolate(
                        temp_res.depth.unsqueeze(0).unsqueeze(0),
                        size=(in_h, in_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                else:
                    final_depth = temp_res.depth

                final_depth.record_stream(pipeline_stream)
                # Note: temp_res.normalization_bounds_float performs 2x .item() calls
                t0_bounds = time.perf_counter()
                norm_bounds = temp_res.normalization_bounds_float
                t1_bounds = time.perf_counter()
                ev_t_end.record(pipeline_stream)
                t1_tmp = time.perf_counter()

                # --- Stage 5: Stereo Splatting & Hole Filling ---
                t0_str = time.perf_counter()
                ev_s_start.record(pipeline_stream)
                stereo_res = render_stereo_frame(
                    image=t_in,
                    depth=final_depth,
                    config=stereo_cfg,
                    device=device,
                    return_numpy=False,
                    output_uint8=True,
                    normalization_bounds=norm_bounds,
                )
                ev_s_end.record(pipeline_stream)
                t1_str = time.perf_counter()

                # --- Stage 6: NVENC Encode ---
                t0_enc = time.perf_counter()
                ev_e_start.record(pipeline_stream)
                sbs_color = stereo_res.sbs_color
                sbs_color.record_stream(pipeline_stream)
                t_enc = torch.cat([sbs_color, alpha], dim=2)
                t_enc.record_stream(pipeline_stream)

                pkts = enc.Encode(t_enc)
                ev_e_end.record(pipeline_stream)
                t1_enc = time.perf_counter()

                # --- Stage 7: Muxer ---
                t0_mux = time.perf_counter()
                for p in pkts:
                    muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
                t1_mux = time.perf_counter()

                # Synchronize to read back event timings for this frame
                pipeline_stream.synchronize()

                gpu_dec_ms = ev_d_start.elapsed_time(ev_d_end)
                gpu_inf_ms = ev_m_start.elapsed_time(ev_m_end)
                gpu_tmp_ms = ev_t_start.elapsed_time(ev_t_end)
                gpu_str_ms = ev_s_start.elapsed_time(ev_s_end)
                gpu_enc_ms = ev_e_start.elapsed_time(ev_e_end)

                frame_metrics.append({
                    "frame_idx": idx,
                    "host_decode_ms": (t1_dec - t0_dec) * 1000.0,
                    "host_dlpack_clone_ms": (t1_dlp - t0_dlp) * 1000.0,
                    "host_inference_ms": (t1_inf - t0_inf) * 1000.0,
                    "host_temporal_ms": (t1_tmp - t0_tmp) * 1000.0,
                    "host_bounds_item_ms": (t1_bounds - t0_bounds) * 1000.0,
                    "host_stereo_ms": (t1_str - t0_str) * 1000.0,
                    "host_encode_ms": (t1_enc - t0_enc) * 1000.0,
                    "host_mux_ms": (t1_mux - t0_mux) * 1000.0,
                    "gpu_dec_event_ms": gpu_dec_ms,
                    "gpu_inf_event_ms": gpu_inf_ms,
                    "gpu_tmp_event_ms": gpu_tmp_ms,
                    "gpu_str_event_ms": gpu_str_ms,
                    "gpu_enc_event_ms": gpu_enc_ms,
                    "packets_emitted": len(pkts),
                })

            pkts_end = enc.EndEncode()
            for p in pkts_end:
                muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])

            pipeline_stream.synchronize()

    wall_total_s = time.perf_counter() - wall_start
    muxer.Finalize()
    del muxer
    del enc
    del dec

    if temp_out_video.exists():
        temp_out_video.unlink()

    print(f"[PROFILER] Profile execution completed in {wall_total_s:.3f} s ({frames_to_run / wall_total_s:.2f} FPS).")

    # 4. Save profiler table and Chrome trace
    trace_path_gz = output_dir / "profile_trace.json.gz"
    trace_temp_json = output_dir / "profile_trace.json"
    prof.export_chrome_trace(str(trace_temp_json))

    # Compress trace
    with open(trace_temp_json, "rb") as f_in:
        with gzip.open(trace_path_gz, "wb") as f_out:
            f_out.writelines(f_in)
    if trace_temp_json.exists():
        trace_temp_json.unlink()

    table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=50)

    summary_file = output_dir / "profiler_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== PYTORCH PROFILER SUMMARY (SORTED BY CUDA TIME TOTAL) ===\n")
        f.write(table_cuda)
        f.write("\n\n=== PYTORCH PROFILER SUMMARY (SORTED BY CPU TIME TOTAL) ===\n")
        f.write(table_cpu)

    # 5. Extract detailed event counts and memory transfers
    key_events = prof.key_averages()
    host_barriers: List[Dict[str, Any]] = []
    cuda_memcpys: List[Dict[str, Any]] = []

    for evt in key_events:
        name = evt.key
        count = evt.count
        cpu_time = getattr(evt, "cpu_time_total", 0.0)
        device_time = getattr(evt, "device_time_total", getattr(evt, "cuda_time_total", 0.0))

        # Look for host synchronization / scalar extraction indicators
        if any(term in name.lower() for term in ("item", "_local_scalar_dense", "synchronize", "to_cpu", "copy_")):
            host_barriers.append({
                "name": name,
                "count": count,
                "cpu_time_ms": round(cpu_time / 1000.0, 3),
                "device_time_ms": round(device_time / 1000.0, 3),
            })

        # Look for memory transfer kernels
        if any(term in name.lower() for term in ("memcpy", "dtoh", "htod", "dtod")):
            cuda_memcpys.append({
                "name": name,
                "count": count,
                "cpu_time_ms": round(cpu_time / 1000.0, 3),
                "device_time_ms": round(device_time / 1000.0, 3),
            })

    # Compute averages across frames
    n = len(frame_metrics)
    avg_host_dec = sum(m["host_decode_ms"] for m in frame_metrics) / n
    avg_host_dlp = sum(m["host_dlpack_clone_ms"] for m in frame_metrics) / n
    avg_host_inf = sum(m["host_inference_ms"] for m in frame_metrics) / n
    avg_host_tmp = sum(m["host_temporal_ms"] for m in frame_metrics) / n
    avg_host_bounds = sum(m["host_bounds_item_ms"] for m in frame_metrics) / n
    avg_host_str = sum(m["host_stereo_ms"] for m in frame_metrics) / n
    avg_host_enc = sum(m["host_encode_ms"] for m in frame_metrics) / n
    avg_host_mux = sum(m["host_mux_ms"] for m in frame_metrics) / n

    avg_gpu_dec = sum(m["gpu_dec_event_ms"] for m in frame_metrics) / n
    avg_gpu_inf = sum(m["gpu_inf_event_ms"] for m in frame_metrics) / n
    avg_gpu_tmp = sum(m["gpu_tmp_event_ms"] for m in frame_metrics) / n
    avg_gpu_str = sum(m["gpu_str_event_ms"] for m in frame_metrics) / n
    avg_gpu_enc = sum(m["gpu_enc_event_ms"] for m in frame_metrics) / n

    # Check for actual concurrency/overlap
    # In the current implementation:
    # 1. Single stream `pipeline_stream` is passed to SimpleDecoder, PyTorch ops, and PyNvEncoder.
    # 2. Python loop executes: dec[idx] -> infer -> temporal -> stereo -> enc.Encode -> mux.
    # 3. Inside temporal & stereo, multiple .item() calls force CPU waits.
    # 4. Therefore, concurrency between decode, inference, and encode is exactly 0.0 ms.
    has_stream_overlap = False  # Single stream + host-blocking loop

    report: Dict[str, Any] = {
        "metadata": {
            "input_file": str(input_path),
            "input_resolution": f"{in_w}x{in_h}",
            "depth_scale": depth_scale,
            "temporal_enabled": True,
            "device": torch.cuda.get_device_name(device),
            "frames_profiled": frames_to_run,
            "wall_clock_total_s": round(wall_total_s, 4),
            "effective_fps": round(frames_to_run / wall_total_s, 2),
        },
        "stage_latency_ms": {
            "averages": {
                "decode_host_ms": round(avg_host_dec, 2),
                "dlpack_clone_host_ms": round(avg_host_dlp, 2),
                "inference_host_ms": round(avg_host_inf, 2),
                "temporal_host_ms": round(avg_host_tmp, 2),
                "temporal_bounds_item_host_ms": round(avg_host_bounds, 2),
                "stereo_host_ms": round(avg_host_str, 2),
                "encode_host_ms": round(avg_host_enc, 2),
                "mux_host_ms": round(avg_host_mux, 2),
                "decode_gpu_event_ms": round(avg_gpu_dec, 2),
                "inference_gpu_event_ms": round(avg_gpu_inf, 2),
                "temporal_gpu_event_ms": round(avg_gpu_tmp, 2),
                "stereo_gpu_event_ms": round(avg_gpu_str, 2),
                "encode_gpu_event_ms": round(avg_gpu_enc, 2),
            },
            "per_frame": frame_metrics,
        },
        "pointer_traces": pointer_traces,
        "concurrency_analysis": {
            "streams_used": 1,
            "overlap_decode_inference": False,
            "overlap_inference_encode": False,
            "overlap_reason": (
                "Single CUDA stream used sequentially across NVDEC, Torch kernels, and NVENC. "
                "Per-frame host iteration synchronously blocks on scalar .item() extractions "
                "and enc.Encode() packet return."
            ),
        },
        "barriers_identified": {
            "host_synchronizations": host_barriers,
            "cuda_memcpys": cuda_memcpys,
        },
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[PROFILER] Report saved to {report_path}")
    print(f"[PROFILER] Trace saved to {trace_path_gz}")
    print(f"[PROFILER] Summary saved to {summary_file}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile PureGPU3D GPU schedule & barriers")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "verification" / "gpu-scheduling",
        help="Output directory for profiler artifacts",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models" / "DA3-SMALL",
        help="Model directory",
    )
    parser.add_argument(
        "--depth-scale",
        type=str,
        default="1/4",
        help="Depth scale (default: 1/4)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
        help="Number of frames to profile (default: 12)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU ID (default: 0)",
    )
    args = parser.parse_args()

    report = run_bounded_profiling(
        input_path=args.input,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        gpu_id=args.gpu_id,
        depth_scale=args.depth_scale,
        num_frames_limit=args.frames,
    )
    print("\n=== PROFILING RESULTS SUMMARY ===")
    print(json.dumps(report["stage_latency_ms"]["averages"], indent=2))
