#!/usr/bin/env python
"""
PyNvVideoCodec Interop Probe (Phase P3 Feasibility).

Demonstrates GPU-resident video decoding (NVDEC) -> PyTorch CUDA tensor (DLPack)
-> GPU processing -> NVENC hardware encoding -> MP4 muxing entirely in GPU VRAM
on Windows without CPU host round-trips.

Usage:
    python -B scripts/nvcodec_probe.py [options]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
try:
    import PyNvVideoCodec as nvc
except ImportError as err:
    print(f"ERROR: PyNvVideoCodec not found in environment: {err}", file=sys.stderr)
    print("Install with: uv pip install pynvvideocodec", file=sys.stderr)
    sys.exit(1)


def probe_ffprobe(file_path: Path) -> Dict[str, Any]:
    """Inspect media container using ffprobe and return parsed stream info."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_name,codec_type,width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def remux_audio_with_ffmpeg(video_only_path: Path, source_with_audio_path: Path, output_path: Path) -> None:
    """Remux video stream from video_only_path and audio stream from source_with_audio_path."""
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(video_only_path),
        "-i", str(source_with_audio_path),
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def run_probe(
    input_path: Path,
    output_path: Path,
    codec: str = "hevc",
    mode: str = "sbs",
    gpu_id: int = 0,
    safe_clone: bool = True,
    remux_audio: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute NVDEC -> PyTorch -> NVENC probe.

    Returns dict with benchmark metrics, pointer verification, and output metadata.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_out_path = output_path
    if remux_audio:
        video_out_path = output_path.with_suffix(".tmp_video.mp4")

    if video_out_path.exists():
        video_out_path.unlink()

    device = torch.device(f"cuda:{gpu_id}")
    stream = torch.cuda.Stream(device=device)

    # 1. Initialize NVDEC SimpleDecoder
    t_start = time.perf_counter()
    dec = nvc.SimpleDecoder(
        str(input_path),
        gpu_id=gpu_id,
        cuda_stream=stream.cuda_stream,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
    )
    meta = dec.get_stream_metadata()
    total_frames = len(dec)
    fps_val = meta.average_fps if meta.average_fps > 0 else 24.0
    fps_int = int(round(fps_val))

    src_w, src_h = meta.width, meta.height
    if mode == "sbs":
        out_w, out_h = src_w * 2, src_h
    elif mode == "passthrough":
        out_w, out_h = src_w, src_h
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if verbose:
        print(f"[NVDEC] Input: {input_path.name} ({src_w}x{src_h}, {total_frames} frames, {fps_val:.2f} fps)")
        print(f"[NVENC] Target: {mode.upper()} {out_w}x{out_h} {codec.upper()} on {torch.cuda.get_device_name(device)}")

    # 2. Initialize NVENC Encoder
    # Format 'ABGR' maps to PyTorch RGBA memory layout (little endian uint32 0xAABBGGRR)
    enc = nvc.CreateEncoder(
        out_w,
        out_h,
        "ABGR",
        False,
        codec=codec,
        cudastream=stream.cuda_stream,
        fps=str(fps_int),
    )
    extradata = enc.GetSequenceParams()

    # 3. Initialize MP4 Muxer
    timebase_den = 90000
    muxer = nvc.FFmpegMuxer(
        str(video_out_path),
        nvc.MEDIA_FORMAT.MP4,
        codec,
        out_w,
        out_h,
        fps_int,
        1,
        1,
        timebase_den,
        extradata,
    )
    muxer.SetUniformPtsIncrement(timebase_den // fps_int)

    # 4. Processing Loop with GPU timing events
    pointer_info = []
    decode_events_start = []
    decode_events_end = []
    compute_events_start = []
    compute_events_end = []
    encode_events_start = []
    encode_events_end = []

    t_loop_start = time.perf_counter()

    with torch.cuda.stream(stream):
        for idx in range(total_frames):
            # Decode frame
            ev_d0 = torch.cuda.Event(enable_timing=True)
            ev_d1 = torch.cuda.Event(enable_timing=True)
            ev_d0.record(stream)
            dec_frame = dec[idx]
            ev_d1.record(stream)
            decode_events_start.append(ev_d0)
            decode_events_end.append(ev_d1)

            # Interop: zero-copy DLPack view into NVDEC device buffer
            plane_ptr = dec_frame.GetPtrToPlane(0)
            t_raw = torch.from_dlpack(dec_frame)
            raw_ptr = t_raw.data_ptr()

            # Record pointer transfer details for first few frames
            if idx < 3:
                pointer_info.append({
                    "frame": idx,
                    "plane_ptr": plane_ptr,
                    "tensor_ptr": raw_ptr,
                    "ptrs_match": (plane_ptr == raw_ptr),
                    "shape": list(t_raw.shape),
                    "dtype": str(t_raw.dtype),
                    "device": str(t_raw.device),
                })

            # GPU Operation:
            ev_c0 = torch.cuda.Event(enable_timing=True)
            ev_c1 = torch.cuda.Event(enable_timing=True)
            ev_c0.record(stream)

            # To avoid the surface reuse hazard (dec[i+1] overwriting dec[i]),
            # we either clone into dedicated GPU tensor memory or execute
            # kernel immediately on stream.
            t_in = t_raw.clone() if safe_clone else t_raw

            if mode == "sbs":
                # Create Full SBS: Left Eye original, Right Eye horizontal disparity shift
                right_eye = torch.roll(t_in, shifts=16, dims=1)
                sbs_rgb = torch.cat([t_in, right_eye], dim=1)  # [H, 2W, 3]
                alpha = torch.full((out_h, out_w, 1), 255, dtype=torch.uint8, device=device)
                t_enc = torch.cat([sbs_rgb, alpha], dim=2)      # [H, 2W, 4] RGBA/ABGR
            else:
                alpha = torch.full((out_h, out_w, 1), 255, dtype=torch.uint8, device=device)
                t_enc = torch.cat([t_in, alpha], dim=2)

            ev_c1.record(stream)
            compute_events_start.append(ev_c0)
            compute_events_end.append(ev_c1)

            # NVENC Hardware Encode
            ev_e0 = torch.cuda.Event(enable_timing=True)
            ev_e1 = torch.cuda.Event(enable_timing=True)
            ev_e0.record(stream)
            pkts = enc.Encode(t_enc)
            for p in pkts:
                muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
            ev_e1.record(stream)
            encode_events_start.append(ev_e0)
            encode_events_end.append(ev_e1)

        # Flush encoder
        pkts_end = enc.EndEncode()
        for p in pkts_end:
            muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])

    stream.synchronize()
    t_loop_end = time.perf_counter()

    # Finalize Muxer
    muxer.Finalize()
    del muxer

    # Audio remux if requested
    if remux_audio:
        if verbose:
            print("[FFmpeg] Remuxing source audio into final container...")
        remux_audio_with_ffmpeg(video_out_path, input_path, output_path)
        if video_out_path.exists():
            video_out_path.unlink()
        final_video_path = output_path
    else:
        final_video_path = video_out_path

    t_total_end = time.perf_counter()

    # Calculate GPU event timings
    gpu_decode_ms = sum(s.elapsed_time(e) for s, e in zip(decode_events_start, decode_events_end))
    gpu_compute_ms = sum(s.elapsed_time(e) for s, e in zip(compute_events_start, compute_events_end))
    gpu_encode_ms = sum(s.elapsed_time(e) for s, e in zip(encode_events_start, encode_events_end))

    wall_loop_s = t_loop_end - t_loop_start
    total_wall_s = t_total_end - t_start
    fps = total_frames / wall_loop_s if wall_loop_s > 0 else 0.0

    # Probe output file
    probe_data = probe_ffprobe(final_video_path)
    video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in probe_data.get("streams", []))

    result = {
        "status": "success",
        "input_video": str(input_path),
        "output_video": str(final_video_path),
        "file_size_bytes": final_video_path.stat().st_size,
        "codec": codec,
        "mode": mode,
        "total_frames": total_frames,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "reported_nb_frames": int(video_stream.get("nb_frames", 0)),
        "has_audio": has_audio,
        "safe_clone": safe_clone,
        "pointer_transfers": pointer_info,
        "timings": {
            "total_wall_time_s": round(total_wall_s, 4),
            "loop_wall_time_s": round(wall_loop_s, 4),
            "sustained_fps": round(fps, 2),
            "gpu_decode_ms": round(gpu_decode_ms, 2),
            "gpu_compute_ms": round(gpu_compute_ms, 2),
            "gpu_encode_ms": round(gpu_encode_ms, 2),
        },
    }

    if verbose:
        print("\n=== Probe Execution Summary ===")
        print(f"Output File: {final_video_path} ({result['file_size_bytes']} bytes)")
        print(f"Dimensions: {result['width']}x{result['height']} (Expected: {out_w}x{out_h})")
        print(f"Frames: {result['reported_nb_frames']} (Expected: {total_frames})")
        print(f"Audio Stream Present: {has_audio}")
        print(f"Pointer Matches: {all(p['ptrs_match'] for p in pointer_info)}")
        print(f"Loop Wall Time: {result['timings']['loop_wall_time_s']:.3f} s ({result['timings']['sustained_fps']:.1f} FPS)")
        print(f"GPU Decode Time: {result['timings']['gpu_decode_ms']:.2f} ms")
        print(f"GPU Compute Time: {result['timings']['gpu_compute_ms']:.2f} ms")
        print(f"GPU Encode Time: {result['timings']['gpu_encode_ms']:.2f} ms")
        print(f"Total Wall Time (inc. init/mux): {result['timings']['total_wall_time_s']:.3f} s")

    return result


def main():
    parser = argparse.ArgumentParser(description="PyNvVideoCodec NVDEC -> PyTorch -> NVENC Interop Probe")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/verification/video/synthetic_1080p_moving.mp4"),
        help="Input video file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/verification/nvcodec/probe_sbs_3840x1080_hevc.mp4"),
        help="Output video file",
    )
    parser.add_argument("--codec", choices=["hevc", "h264"], default="hevc", help="Target video codec")
    parser.add_argument("--mode", choices=["sbs", "passthrough"], default="sbs", help="Output mode")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU Device ID")
    parser.add_argument("--no-clone", action="store_true", help="Do not clone tensor (use raw view)")
    parser.add_argument("--remux-audio", action="store_true", help="Remux audio from source video")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    args = parser.parse_args()

    try:
        res = run_probe(
            input_path=args.input,
            output_path=args.output,
            codec=args.codec,
            mode=args.mode,
            gpu_id=args.gpu_id,
            safe_clone=(not args.no_clone),
            remux_audio=args.remux_audio,
            verbose=(not args.json),
        )
        if args.json:
            print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
