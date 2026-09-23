"""Experimental real GPU-resident video conversion pipeline for PureGPU3D.

Architecture:
  NVDEC decode into device surfaces
  -> DLPack zero-copy tensor view (GPU VRAM)
  -> DA3 infer_tensor (GPU VRAM, CUDA autocast fp16, timing=False)
  -> Depth-aware stereo forward splatting + hole filling (GPU VRAM)
  -> NVENC hardware encoding (GPU VRAM, ABGR)
  -> MP4 muxing + FFmpeg multi-track audio remuxing (CPU container staging)
  -> OutputTransaction preflight validation, process locking, and atomic promotion.

Constraints & Safety Contracts:
  - Strict zero full-frame host transfers (.cpu() or .numpy() full frames are forbidden).
  - Surface reuse safety: clone decoder frame tensor before subsequent dec[i] reuse.
  - Rational CFR timestamps: exact rational frame rate preserved from ffprobe Fraction;
    reject upfront if exact integer timebase increment cannot be preserved.
  - Audio preservation: remux all eligible audio tracks without -shortest.
  - Safe staging & atomic promotion: strict source/destination collision check, process lock,
    comprehensive pre-promotion validation (0-tolerance frame count, decodability check).
  - Bounded memory: no unbounded per-frame timing or event lists.
  - Owned resource cancellation: immediate cleanup of dec/enc/muxer and staging files.
  - Temporal depth stabilization: explicitly disabled with warning (GPU optical flow not yet ready).
"""

from __future__ import annotations

import collections
import fractions
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import PyNvVideoCodec as nvc_module
    nvc: Any = nvc_module
    _NVC_AVAILABLE = True
    _NVC_ERROR = ""
except ImportError as err:
    nvc = None
    _NVC_AVAILABLE = False
    _NVC_ERROR = str(err)

from puregpu3d.jobs.output_transaction import (
    OutputTransaction,
    PathCollisionError,
    TransactionCancelledError,
    TransactionError,
    ValidationError,
)
from puregpu3d.models.geometry import parse_depth_scale
from puregpu3d.stereo import StereoConfig, render_stereo_frame
from puregpu3d.stereo.temporal_gpu import TemporalGPUConfig, TemporalGPUStabilizer
from puregpu3d.video.convert import (
    ConversionCancelledError,
    ConversionError,
    ConversionResult,
)
from puregpu3d.video.gpu_buffers import GpuDecodePrefetchRing
from puregpu3d.video.probe import (
    UnsupportedMediaError,
    VideoProbeResult,
    find_ffmpeg,
    find_ffprobe,
    probe_video,
)

logger = logging.getLogger(__name__)


class GpuPipelineUnsupportedError(ConversionError):
    """Raised when GPU video hardware pipeline cannot run on current system."""
    pass


@dataclass(frozen=True)
class InteropPointerTrace:
    """Zero-copy pointer trace information for an individual decoded frame."""
    frame_index: int
    plane_ptr: int
    tensor_ptr: int
    ptrs_match: bool
    shape: List[int]
    dtype: str
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GpuConversionResult:
    """Detailed telemetry and verified output metadata from GPU-resident conversion.

    Note on stage timings:
        mean_decode_ms, mean_depth_ms, mean_stereo_ms, mean_encode_ms represent host-side
        pipeline dispatch/enqueue and synchronization intervals (measured via host perf_counter),
        not isolated GPU hardware kernel compute times. This avoids intrusive per-frame full
        CUDA flushes. wall_clock_seconds and effective_fps represent true end-to-end throughput.
    """
    input_path: Path
    output_path: Path
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    frame_rate_str: str
    total_frames_processed: int
    wall_clock_seconds: float
    effective_fps: float
    mean_depth_ms: float
    mean_stereo_ms: float
    mean_decode_ms: float
    mean_encode_ms: float
    has_audio: bool
    audio_stream_count: int
    device: str
    encoder: str
    notes: List[str]
    depth_scale: Optional[str] = None
    pointer_traces: List[Dict[str, Any]] = field(default_factory=list)
    startup_overhead_s: float = 0.0
    validation_overhead_s: float = 0.0
    mean_temporal_ms: float = 0.0
    scheduling: str = "sequential"
    batch_size: int = 1
    peak_memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_path"] = str(self.input_path)
        d["output_path"] = str(self.output_path)
        return d


def _canonicalize_cuda_device(dev: Union[str, torch.device]) -> torch.device:
    """Canonicalize CUDA device specification, expanding unindexed 'cuda' to active GPU index."""
    d = torch.device(dev)
    if d.type == "cuda" and d.index is None:
        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(f"cuda:{idx}")
    return d


def check_gpu_pipeline_support(gpu_id: int = 0) -> Tuple[bool, str]:
    """Verify system driver, PyNvVideoCodec, and CUDA capability for GPU conversion."""
    if not _NVC_AVAILABLE:
        return False, f"PyNvVideoCodec not available: {_NVC_ERROR}"

    if not torch.cuda.is_available():
        return False, "CUDA is not available in PyTorch."

    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        return False, f"Invalid GPU id {gpu_id}; only {device_count} CUDA devices present."

    try:
        if hasattr(nvc, "supportedNvEncVersion") and hasattr(nvc, "NVENC_VER_12"):
            if nvc.supportedNvEncVersion < nvc.NVENC_VER_12:
                return False, f"Installed driver NVENC version ({nvc.supportedNvEncVersion}) < 12.0."
    except Exception as err:
        return False, f"Error inspecting PyNvVideoCodec capabilities: {err}"

    return True, "GPU pipeline supported."


def _compute_rational_timebase_and_increment(
    fps_frac: fractions.Fraction,
) -> Tuple[int, int, int]:
    """Compute exact integer timebase numerator, denominator, and tick increment.

    Returns:
        Tuple of (timebase_num, timebase_den, pts_inc)

    Raises:
        UnsupportedMediaError: If no standard timebase yields an exact integer tick increment.
    """
    fps_num = fps_frac.numerator
    fps_den = fps_frac.denominator

    candidate_bases = [90000, 180000, 360000, 720000, 1000000]
    for base in candidate_bases:
        total_ticks = base * fps_den
        if total_ticks % fps_num == 0:
            pts_inc = total_ticks // fps_num
            return 1, base, pts_inc

    lcm_base = (90000 * fps_num) // math.gcd(90000, fps_num)
    if (lcm_base * fps_den) % fps_num == 0 and lcm_base <= 10000000:
        return 1, lcm_base, (lcm_base * fps_den) // fps_num

    raise UnsupportedMediaError(
        f"Cannot preserve exact rational CFR timestamps ({fps_num}/{fps_den}) "
        f"in experimental GPU video pipeline without non-integer timestamp rounding."
    )


def convert_video_gpu(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    model: Any,
    device: Optional[Union[str, torch.device]] = None,
    stereo_config: Optional[StereoConfig] = None,
    depth_scale: Optional[Union[str, float]] = None,
    codec: str = "hevc",
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    ffmpeg_path: Optional[Union[str, Path]] = None,
    ffprobe_path: Optional[Union[str, Path]] = None,
    enable_temporal_stabilization: bool = False,
    temporal_config: Optional[Any] = None,
    scene_config: Optional[Any] = None,
    safe_clone: bool = True,
    gpu_id: int = 0,
    max_trace_frames: int = 3,
    compute_diagnostics: bool = False,
    scheduling: str = "sequential",
    prefetch_slots: int = 3,
    batch_size: int = 1,
) -> GpuConversionResult:
    """Execute real GPU-resident NVDEC -> DA3 infer_tensor -> depth-aware stereo -> NVENC conversion.

    All frame pixels remain strictly in GPU memory. Left and right stereoscopic views
    are generated from true DA3 depth maps via depth-aware forward splatting.
    """
    wall_start = time.perf_counter()

    # 0. Upfront batch_size validation
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            f"batch_size must be an integer, got {type(batch_size).__name__} ({batch_size!r})"
        )
    if batch_size < 1 or batch_size > 20:
        raise ValueError(f"batch_size must be an integer between 1 and 20, got {batch_size}")

    resolved_input = Path(input_path).resolve()
    resolved_output = Path(output_path).resolve()

    # 1. Resolve target GPU device and index upfront
    if device is None:
        target_device = torch.device(f"cuda:{gpu_id}")
    else:
        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise GpuPipelineUnsupportedError(
                f"GPU-resident pipeline requires a CUDA device, got '{target_device}'."
            )
        if target_device.index is not None:
            gpu_id = target_device.index
        else:
            gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else gpu_id
            target_device = torch.device(f"cuda:{gpu_id}")

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(gpu_id)
        except Exception:
            pass

    # 2. System support preflight for target GPU
    supported, reason = check_gpu_pipeline_support(gpu_id=gpu_id)
    if not supported:
        raise GpuPipelineUnsupportedError(f"Experimental GPU pipeline unsupported on this host: {reason}")

    # 3. Upfront depth scale parsing
    canon_depth_scale: Optional[str] = None
    if depth_scale is not None:
        canon_depth_scale, _ = parse_depth_scale(depth_scale)

    ffmpeg_bin = find_ffmpeg(ffmpeg_path)
    ffprobe_bin = find_ffprobe(ffprobe_path)

    # 4. Probe input media with strict upfront validation
    probe = probe_video(resolved_input, ffprobe_path=ffprobe_bin, strict_sdr_cfr=True)

    in_w = probe.width
    in_h = probe.height
    out_w = in_w * 2
    out_h = in_h
    fps_frac = probe.frame_rate

    # 5. Exact rational CFR timestamps verification
    timebase_num, timebase_den, pts_inc = _compute_rational_timebase_and_increment(fps_frac)

    # 6. Model adapter preparation
    adapter = model
    if not hasattr(adapter, "infer_tensor"):
        if isinstance(model, (str, Path)):
            from puregpu3d.models.da3_adapter import DA3DepthAdapter
            adapter = DA3DepthAdapter(model_dir=Path(model), device=target_device)
        else:
            raise TypeError(
                f"Model adapter must provide infer_tensor() for GPU-resident pipeline, got {type(model)}."
            )

    if batch_size > 1 and not hasattr(adapter, "infer_tensor_batch"):
        raise AttributeError(
            f"Model adapter does not provide infer_tensor_batch() for GPU-resident batching, got {type(adapter)}."
        )

    if hasattr(adapter, "device"):
        adapter_dev = _canonicalize_cuda_device(adapter.device)
        target_dev = _canonicalize_cuda_device(target_device)
        if adapter_dev != target_dev:
            raise ValueError(
                f"Model adapter device '{adapter.device}' does not match conversion device '{target_device}'."
            )

    # 7. Stereo configuration
    if stereo_config is None:
        stereo_config = StereoConfig()

    notes: List[str] = []
    stabilizer: Optional[TemporalGPUStabilizer] = None
    if enable_temporal_stabilization:
        if temporal_config is None:
            t_cfg = TemporalGPUConfig(enabled=True)
        elif isinstance(temporal_config, dict):
            t_cfg = TemporalGPUConfig(**temporal_config)
        elif isinstance(temporal_config, TemporalGPUConfig):
            t_cfg = temporal_config
        else:
            t_cfg = TemporalGPUConfig(enabled=True)

        if scene_config is not None and hasattr(scene_config, "cut_threshold"):
            cfg_dict = asdict(t_cfg) if hasattr(t_cfg, "__dataclass_fields__") else {}
            cfg_dict["cut_threshold"] = scene_config.cut_threshold
            t_cfg = TemporalGPUConfig(**cfg_dict)

        stabilizer = TemporalGPUStabilizer(config=t_cfg)
        notes.append(
            "Temporal depth stabilization: enabled (GPU-resident causal motion-compensated filtering with online normalization)."
        )
    else:
        notes.append("Temporal depth stabilization: disabled (baseline per-frame processing).")

    notes.append(f"GPU pipeline: NVDEC -> PyTorch CUDA -> NVENC ({codec.upper()}) on {torch.cuda.get_device_name(target_device)}.")
    notes.append(f"Output geometry: Full SBS {out_w}x{out_h} at rational {fps_frac.numerator}/{fps_frac.denominator} fps.")
    if batch_size > 1:
        notes.append(f"Batch size: {batch_size} (independent-frame batching).")
    else:
        notes.append("Batch size: 1 (single-frame processing).")
    valid_schedulers = {"sequential", "pipelined", "prefetch", "overlap"}
    if scheduling not in valid_schedulers:
        raise ValueError(
            f"Unsupported scheduling mode '{scheduling}'. "
            f"Supported options are: {sorted(valid_schedulers)}."
        )
    is_pipelined = (scheduling in {"pipelined", "prefetch", "overlap"})
    notes.append(f"Scheduling: {scheduling} (prefetch_slots={prefetch_slots if is_pipelined else 0}).")
    notes.append(
        "Stage telemetry: mean_decode_ms, mean_depth_ms, mean_temporal_ms, mean_stereo_ms, mean_encode_ms "
        "measure host-side dispatch/enqueue intervals without per-frame GPU stalls; "
        "effective_fps and wall_clock_seconds measure real end-to-end throughput."
    )

    # 8. OutputTransaction setup for atomic staging & process locking
    txn = OutputTransaction(
        source_path=resolved_input,
        destination_path=resolved_output,
        overwrite=overwrite,
        expected_width=out_w,
        expected_height=out_h,
        expected_frames=probe.frame_count if probe.frame_count > 0 else None,
        expect_audio=probe.has_audio,
        expected_audio_streams=len(probe.audio_streams),
        expected_duration=probe.duration if probe.duration > 0 else None,
        expected_frame_rate=fps_frac,
        ffprobe_path=ffprobe_bin,
        ffmpeg_path=ffmpeg_bin,
    )

    t_pipeline_init = time.perf_counter()
    startup_overhead_s = t_pipeline_init - wall_start

    # Bounded metrics accumulators (no growing lists)
    total_decode_time_s: float = 0.0
    total_depth_time_s: float = 0.0
    total_temporal_time_s: float = 0.0
    total_stereo_time_s: float = 0.0
    total_encode_time_s: float = 0.0
    processed_count: int = 0
    validation_overhead_s: float = 0.0
    pointer_traces: List[Dict[str, Any]] = []

    dec = None
    enc = None
    muxer = None
    ring: Optional[GpuDecodePrefetchRing] = None
    video_staging_path: Optional[Path] = None

    stream: Any = torch.cuda.Stream(device=target_device)
    # Explicit stream wait: synchronize custom pipeline stream with ambient current stream
    # so model weight transfers, preallocations, or prior operations are fully ordered before pipeline work.
    current_stream = torch.cuda.current_stream(device=target_device)
    stream.wait_stream(current_stream)

    try:
        with txn:
            staging_file = txn.staging_path

            # Determine video output destination during encoding
            if probe.has_audio:
                video_staging_path = staging_file.with_name(f"{staging_file.name}.raw_video.mp4")
                if video_staging_path.exists():
                    video_staging_path.unlink()
                active_video_target = video_staging_path
            else:
                active_video_target = staging_file

            try:
                decode_stream: Optional[Any] = None
                if is_pipelined:
                    decode_stream = torch.cuda.Stream(device=target_device)
                    decode_stream.wait_stream(current_stream)
                    dec_stream_arg = decode_stream.cuda_stream
                else:
                    dec_stream_arg = stream.cuda_stream

                # Initialize NVDEC SimpleDecoder
                dec = nvc.SimpleDecoder(
                    str(resolved_input),
                    gpu_id=gpu_id,
                    cuda_stream=dec_stream_arg,
                    use_device_memory=True,
                    output_color_type=nvc.OutputColorType.RGB,
                )
                total_frames = len(dec)
                if total_frames <= 0:
                    raise ConversionError(f"NVDEC reported 0 decodable frames in '{resolved_input}'.")

                # Initialize NVENC Encoder
                fps_arg = f"{fps_frac.numerator}/{fps_frac.denominator}"
                enc = nvc.CreateEncoder(
                    out_w,
                    out_h,
                    "ABGR",
                    False,
                    codec=codec,
                    cudastream=stream.cuda_stream,
                    fps=fps_arg,
                )
                extradata = enc.GetSequenceParams()

                # Initialize FFmpegMuxer
                muxer = nvc.FFmpegMuxer(
                    str(active_video_target),
                    nvc.MEDIA_FORMAT.MP4,
                    codec,
                    out_w,
                    out_h,
                    fps_frac.numerator,
                    fps_frac.denominator,
                    timebase_num,
                    timebase_den,
                    extradata,
                )
                muxer.SetUniformPtsIncrement(pts_inc)

                ring: Optional[GpuDecodePrefetchRing] = None
                if is_pipelined:
                    ring = GpuDecodePrefetchRing(
                        decoder=dec,
                        target_device=target_device,
                        in_height=in_h,
                        in_width=in_w,
                        total_frames=total_frames,
                        num_slots=prefetch_slots,
                        decode_stream=decode_stream,
                        cancel_callback=cancel_callback,
                    )
                    ring.start()

                with torch.cuda.stream(stream):
                    # Preallocate Alpha plane for RGBA/ABGR NVENC compatibility on pipeline stream
                    alpha = torch.full((out_h, out_w, 1), 255, dtype=torch.uint8, device=target_device)
                    alpha.record_stream(stream)

                    if batch_size == 1:
                        for idx in range(total_frames):
                            if cancel_callback is not None and cancel_callback():
                                raise ConversionCancelledError("Conversion aborted: cancellation requested by caller.")

                            slot_idx: Optional[int] = None
                            if is_pipelined and ring is not None:
                                frame_item = ring.acquire_next_frame()
                                if frame_item is None:
                                    raise ConversionError(
                                        f"Premature end of decode prefetch ring at frame {idx}/{total_frames}."
                                    )
                                f_idx, slot_idx, plane_ptr, raw_ptr, ptrs_match, dec_wall_s = frame_item
                                total_decode_time_s += dec_wall_s
                                slot = ring.slots[slot_idx]
                                stream.wait_event(slot.ready_event)
                                slot.tensor.record_stream(stream)
                                t_in = slot.tensor

                                if len(pointer_traces) < max_trace_frames:
                                    pointer_traces.append(
                                        InteropPointerTrace(
                                            frame_index=f_idx,
                                            plane_ptr=plane_ptr,
                                            tensor_ptr=raw_ptr,
                                            ptrs_match=ptrs_match,
                                            shape=list(t_in.shape),
                                            dtype=str(t_in.dtype),
                                            device=str(t_in.device),
                                        ).to_dict()
                                    )
                            else:
                                # Stage 1: NVDEC Decode
                                t_d0 = time.perf_counter()
                                dec_frame: Any = dec[idx]
                                t_d1 = time.perf_counter()
                                total_decode_time_s += (t_d1 - t_d0)

                                # Stage 2: DLPack Zero-Copy Interop & Pointer Verification
                                plane_ptr = int(dec_frame.GetPtrToPlane(0))
                                t_raw = torch.from_dlpack(dec_frame)
                                raw_ptr = int(t_raw.data_ptr())
                                t_raw.record_stream(stream)

                                if len(pointer_traces) < max_trace_frames:
                                    pointer_traces.append(
                                        InteropPointerTrace(
                                            frame_index=idx,
                                            plane_ptr=plane_ptr,
                                            tensor_ptr=raw_ptr,
                                            ptrs_match=(plane_ptr == raw_ptr),
                                            shape=list(t_raw.shape),
                                            dtype=str(t_raw.dtype),
                                            device=str(t_raw.device),
                                        ).to_dict()
                                    )

                                # Surface reuse safety: clone tensor memory so decoder surface can be reused
                                t_in = t_raw.clone() if safe_clone else t_raw
                                t_in.record_stream(stream)

                            # Stage 3: Real DA3 Depth Inference (GPU VRAM)
                            t_m0 = time.perf_counter()
                            depth_res = adapter.infer_tensor(
                                t_in,
                                depth_scale=canon_depth_scale,
                                return_original_size=True,
                                autocast=True,
                                timing=False,  # timing=False avoids host synchronizations
                            )
                            t_m1 = time.perf_counter()
                            total_depth_time_s += (t_m1 - t_m0)

                            # Stage 3b: GPU Temporal Depth Stabilization & Online Normalization
                            final_depth = depth_res.depth
                            norm_bounds: Optional[Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor], Any]] = None
                            if stabilizer is not None:
                                t_t0 = time.perf_counter()
                                raw_d = depth_res.depth_raw
                                h_raw, w_raw = raw_d.shape[-2:]

                                # Scale RGB on GPU to match depth_raw shape if necessary
                                if t_in.shape[0] != h_raw or t_in.shape[1] != w_raw:
                                    t_in_chw = t_in.permute(2, 0, 1).unsqueeze(0).float()
                                    rgb_scaled = F.interpolate(t_in_chw, size=(h_raw, w_raw), mode="area")
                                else:
                                    rgb_scaled = t_in

                                temp_res = stabilizer.process_frame(
                                    frame_rgb=rgb_scaled,
                                    raw_depth=raw_d,
                                    compute_diagnostics=compute_diagnostics,
                                )

                                # Upscale stabilized depth back onto full-resolution geometry
                                if (h_raw, w_raw) != (in_h, in_w):
                                    final_depth = F.interpolate(
                                        temp_res.depth.unsqueeze(0).unsqueeze(0),
                                        size=(in_h, in_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).squeeze(0).squeeze(0)
                                else:
                                    final_depth = temp_res.depth

                                final_depth.record_stream(stream)
                                norm_bounds = temp_res.normalization_bounds
                                t_t1 = time.perf_counter()
                                total_temporal_time_s += (t_t1 - t_t0)

                            # Stage 4: Depth-Aware Stereoscopic Rendering (GPU VRAM)
                            t_s0 = time.perf_counter()
                            stereo_res = render_stereo_frame(
                                image=t_in,
                                depth=final_depth,
                                config=stereo_config,
                                device=target_device,
                                return_numpy=False,
                                output_uint8=True,
                                normalization_bounds=norm_bounds,
                                compute_diagnostics=compute_diagnostics,
                            )
                            t_s1 = time.perf_counter()
                            total_stereo_time_s += (t_s1 - t_s0)

                            if is_pipelined and ring is not None and slot_idx is not None:
                                ring.release_slot(slot_idx, stream)

                            # Stage 5: Full SBS 4-channel composition and NVENC Encode
                            t_e0 = time.perf_counter()
                            sbs_color = stereo_res.sbs_color
                            if not isinstance(sbs_color, torch.Tensor):
                                raise TypeError(f"Expected sbs_color to be torch.Tensor, got {type(sbs_color)}")
                            sbs_color.record_stream(stream)
                            t_enc = torch.cat([sbs_color, alpha], dim=2)  # [H, 2W, 4] ABGR
                            t_enc.record_stream(stream)

                            pkts = enc.Encode(t_enc)
                            for p in pkts:
                                muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
                            t_e1 = time.perf_counter()
                            total_encode_time_s += (t_e1 - t_e0)

                            processed_count += 1
                            if progress_callback is not None:
                                try:
                                    progress_callback(processed_count, total_frames)
                                except (ConversionCancelledError, TransactionCancelledError, InterruptedError):
                                    raise
                                except Exception as cb_err:
                                    if "cancel" in type(cb_err).__name__.lower() or "cancel" in str(cb_err).lower():
                                        raise ConversionCancelledError(f"Conversion cancelled by progress callback: {cb_err}") from cb_err
                                    logger.warning("Non-fatal exception in progress_callback ignored: %s", cb_err)
                    else:
                        # Batched path (batch_size > 1)
                        for batch_start in range(0, total_frames, batch_size):
                            batch_end = min(batch_start + batch_size, total_frames)
                            group_size = batch_end - batch_start
                            batch_tensors: List[torch.Tensor] = []

                            for frame_idx in range(batch_start, batch_end):
                                if cancel_callback is not None and cancel_callback():
                                    raise ConversionCancelledError("Conversion aborted: cancellation requested by caller.")

                                if is_pipelined and ring is not None:
                                    frame_item = ring.acquire_next_frame()
                                    if frame_item is None:
                                        raise ConversionError(
                                            f"Premature end of decode prefetch ring at frame {frame_idx}/{total_frames}."
                                        )
                                    f_idx, slot_idx, plane_ptr, raw_ptr, ptrs_match, dec_wall_s = frame_item
                                    total_decode_time_s += dec_wall_s
                                    slot = ring.slots[slot_idx]
                                    stream.wait_event(slot.ready_event)
                                    slot.tensor.record_stream(stream)

                                    if len(pointer_traces) < max_trace_frames:
                                        pointer_traces.append(
                                            InteropPointerTrace(
                                                frame_index=f_idx,
                                                plane_ptr=plane_ptr,
                                                tensor_ptr=raw_ptr,
                                                ptrs_match=ptrs_match,
                                                shape=list(slot.tensor.shape),
                                                dtype=str(slot.tensor.dtype),
                                                device=str(slot.tensor.device),
                                            ).to_dict()
                                        )

                                    # Surface reuse safety: clone tensor memory into owned buffer on compute stream
                                    # and immediately release slot back to the decode ring
                                    t_in = slot.tensor.clone()
                                    t_in.record_stream(stream)
                                    ring.release_slot(slot_idx, stream)
                                    batch_tensors.append(t_in)
                                else:
                                    # Stage 1: NVDEC Decode
                                    t_d0 = time.perf_counter()
                                    dec_frame: Any = dec[frame_idx]
                                    t_d1 = time.perf_counter()
                                    total_decode_time_s += (t_d1 - t_d0)

                                    # Stage 2: DLPack Zero-Copy Interop & Pointer Verification
                                    plane_ptr = int(dec_frame.GetPtrToPlane(0))
                                    t_raw = torch.from_dlpack(dec_frame)
                                    raw_ptr = int(t_raw.data_ptr())
                                    t_raw.record_stream(stream)

                                    if len(pointer_traces) < max_trace_frames:
                                        pointer_traces.append(
                                            InteropPointerTrace(
                                                frame_index=frame_idx,
                                                plane_ptr=plane_ptr,
                                                tensor_ptr=raw_ptr,
                                                ptrs_match=(plane_ptr == raw_ptr),
                                                shape=list(t_raw.shape),
                                                dtype=str(t_raw.dtype),
                                                device=str(t_raw.device),
                                            ).to_dict()
                                        )

                                    # Surface reuse safety: must be OWNED clone before next NVDEC decode
                                    t_in = t_raw.clone()
                                    t_in.record_stream(stream)
                                    batch_tensors.append(t_in)

                            # Check cancellation before launching batch inference
                            if cancel_callback is not None and cancel_callback():
                                raise ConversionCancelledError("Conversion aborted: cancellation requested by caller.")

                            # Stage 3: Real DA3 Depth Inference (GPU VRAM) for the batch
                            t_m0 = time.perf_counter()
                            depth_results = adapter.infer_tensor_batch(
                                batch_tensors,
                                depth_scale=canon_depth_scale,
                                return_original_size=True,
                                autocast=True,
                                timing=False,
                            )
                            t_m1 = time.perf_counter()
                            total_depth_time_s += (t_m1 - t_m0)

                            if len(depth_results) != group_size:
                                raise ConversionError(
                                    f"infer_tensor_batch returned {len(depth_results)} results, expected {group_size}."
                                )

                            # Chronological downstream processing for each frame in the batch
                            for i, t_in in enumerate(batch_tensors):
                                if cancel_callback is not None and cancel_callback():
                                    raise ConversionCancelledError("Conversion aborted: cancellation requested by caller.")

                                depth_res = depth_results[i]

                                # Stage 3b: GPU Temporal Depth Stabilization & Online Normalization
                                final_depth = depth_res.depth
                                norm_bounds: Optional[Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor], Any]] = None
                                if stabilizer is not None:
                                    t_t0 = time.perf_counter()
                                    raw_d = depth_res.depth_raw
                                    h_raw, w_raw = raw_d.shape[-2:]

                                    # Scale RGB on GPU to match depth_raw shape if necessary
                                    if t_in.shape[0] != h_raw or t_in.shape[1] != w_raw:
                                        t_in_chw = t_in.permute(2, 0, 1).unsqueeze(0).float()
                                        rgb_scaled = F.interpolate(t_in_chw, size=(h_raw, w_raw), mode="area")
                                    else:
                                        rgb_scaled = t_in

                                    temp_res = stabilizer.process_frame(
                                        frame_rgb=rgb_scaled,
                                        raw_depth=raw_d,
                                        compute_diagnostics=compute_diagnostics,
                                    )

                                    # Upscale stabilized depth back onto full-resolution geometry
                                    if (h_raw, w_raw) != (in_h, in_w):
                                        final_depth = F.interpolate(
                                            temp_res.depth.unsqueeze(0).unsqueeze(0),
                                            size=(in_h, in_w),
                                            mode="bilinear",
                                            align_corners=False,
                                        ).squeeze(0).squeeze(0)
                                    else:
                                        final_depth = temp_res.depth

                                    final_depth.record_stream(stream)
                                    norm_bounds = temp_res.normalization_bounds
                                    t_t1 = time.perf_counter()
                                    total_temporal_time_s += (t_t1 - t_t0)

                                # Stage 4: Depth-Aware Stereoscopic Rendering (GPU VRAM)
                                t_s0 = time.perf_counter()
                                stereo_res = render_stereo_frame(
                                    image=t_in,
                                    depth=final_depth,
                                    config=stereo_config,
                                    device=target_device,
                                    return_numpy=False,
                                    output_uint8=True,
                                    normalization_bounds=norm_bounds,
                                    compute_diagnostics=compute_diagnostics,
                                )
                                t_s1 = time.perf_counter()
                                total_stereo_time_s += (t_s1 - t_s0)

                                # Stage 5: Full SBS 4-channel composition and NVENC Encode
                                t_e0 = time.perf_counter()
                                sbs_color = stereo_res.sbs_color
                                if not isinstance(sbs_color, torch.Tensor):
                                    raise TypeError(f"Expected sbs_color to be torch.Tensor, got {type(sbs_color)}")
                                sbs_color.record_stream(stream)
                                t_enc = torch.cat([sbs_color, alpha], dim=2)  # [H, 2W, 4] ABGR
                                t_enc.record_stream(stream)

                                pkts = enc.Encode(t_enc)
                                for p in pkts:
                                    muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
                                t_e1 = time.perf_counter()
                                total_encode_time_s += (t_e1 - t_e0)

                                processed_count += 1
                                if progress_callback is not None:
                                    try:
                                        progress_callback(processed_count, total_frames)
                                    except (ConversionCancelledError, TransactionCancelledError, InterruptedError):
                                        raise
                                    except Exception as cb_err:
                                        if "cancel" in type(cb_err).__name__.lower() or "cancel" in str(cb_err).lower():
                                            raise ConversionCancelledError(f"Conversion cancelled by progress callback: {cb_err}") from cb_err
                                        logger.warning("Non-fatal exception in progress_callback ignored: %s", cb_err)

                    # Flush remaining frames from encoder
                    t_flush_0 = time.perf_counter()
                    pkts_end = enc.EndEncode()
                    for p in pkts_end:
                        muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
                    t_flush_1 = time.perf_counter()
                    total_encode_time_s += (t_flush_1 - t_flush_0)

                stream.synchronize()

                # Finalize muxer and release handles
                if muxer is not None:
                    muxer.Finalize()
                    del muxer
                    muxer = None
                if enc is not None:
                    del enc
                    enc = None
                if dec is not None:
                    del dec
                    dec = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stage 6: Remux all audio tracks without -shortest
                if probe.has_audio:
                    if cancel_callback is not None and cancel_callback():
                        raise ConversionCancelledError("Conversion cancelled prior to audio remuxing.")

                    source_video = next(s for s in probe.raw_info["streams"] if s.get("codec_type") == "video")
                    video_start = source_video.get("start_time", "0")
                    if video_start in (None, "N/A"):
                        video_start = "0"
                    audio_remux_cmd = [
                        str(ffmpeg_bin),
                        "-y",
                        "-v", "error",
                        "-copyts",
                        "-itsoffset", str(video_start),
                        "-i", str(video_staging_path),
                        "-i", str(resolved_input),
                        "-c:v", "copy",
                        "-c:a", "copy",
                        "-map", "0:v:0",
                    ]
                    for idx in range(len(probe.audio_streams)):
                        audio_remux_cmd.extend(["-map", f"1:a:{idx}"])
                    audio_remux_cmd.append(str(staging_file))

                    remux_proc = subprocess.Popen(
                        audio_remux_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    while True:
                        ret = remux_proc.poll()
                        if ret is not None:
                            break
                        if cancel_callback is not None and cancel_callback():
                            try:
                                remux_proc.terminate()
                                try:
                                    remux_proc.wait(timeout=1.0)
                                except subprocess.TimeoutExpired:
                                    remux_proc.kill()
                                    remux_proc.wait(timeout=1.0)
                            except Exception as proc_err:
                                logger.warning("Error terminating audio remux process: %s", proc_err)
                            raise ConversionCancelledError("Conversion cancelled during audio remuxing.")
                        time.sleep(0.05)

                    stdout_data, stderr_data = remux_proc.communicate()
                    if remux_proc.returncode != 0:
                        raise ConversionError(
                            f"FFmpeg audio remux failed (code {remux_proc.returncode}): {stderr_data.strip()}"
                        )

                    # Remove intermediate raw video staging file
                    if video_staging_path and video_staging_path.exists():
                        try:
                            video_staging_path.unlink()
                        except Exception:
                            pass

                # Stage 7: Validate rendered staging file and atomically promote to destination
                t_val_start = time.perf_counter()
                txn.validate_and_promote(cancel_callback=cancel_callback)
                t_val_end = time.perf_counter()
                validation_overhead_s = t_val_end - t_val_start

            finally:
                # Ensure child handles and intermediate files are released even on error
                if ring is not None:
                    try:
                        ring.close()
                    except Exception:
                        pass
                    ring = None
                if muxer is not None:
                    try:
                        del muxer
                    except Exception:
                        pass
                    muxer = None
                if enc is not None:
                    try:
                        del enc
                    except Exception:
                        pass
                    enc = None
                if dec is not None:
                    try:
                        del dec
                    except Exception:
                        pass
                    dec = None
                gc.collect()

                if video_staging_path and video_staging_path.exists():
                    try:
                        video_staging_path.unlink()
                    except Exception:
                        pass

    except (TransactionCancelledError, ConversionCancelledError) as err:
        logger.info(f"GPU conversion cancelled: {err}")
        raise ConversionCancelledError(f"Conversion cancelled: {err}") from err
    except Exception as err:
        logger.error(f"GPU conversion failed: {err}")
        raise

    wall_end = time.perf_counter()
    wall_total = wall_end - wall_start
    effective_fps = processed_count / wall_total if wall_total > 0 else 0.0

    mean_dec = (total_decode_time_s / processed_count * 1000.0) if processed_count > 0 else 0.0
    mean_dep = (total_depth_time_s / processed_count * 1000.0) if processed_count > 0 else 0.0
    mean_tem = (total_temporal_time_s / processed_count * 1000.0) if processed_count > 0 else 0.0
    mean_ste = (total_stereo_time_s / processed_count * 1000.0) if processed_count > 0 else 0.0
    mean_enc = (total_encode_time_s / processed_count * 1000.0) if processed_count > 0 else 0.0

    peak_memory_mb: Optional[float] = None
    if torch.cuda.is_available():
        try:
            peak_bytes = torch.cuda.max_memory_allocated(gpu_id)
            peak_memory_mb = round(peak_bytes / (1024 * 1024), 2)
        except Exception:
            peak_memory_mb = None

    return GpuConversionResult(
        input_path=resolved_input,
        output_path=resolved_output,
        input_width=in_w,
        input_height=in_h,
        output_width=out_w,
        output_height=out_h,
        frame_rate_str=f"{fps_frac.numerator}/{fps_frac.denominator}",
        total_frames_processed=processed_count,
        wall_clock_seconds=wall_total,
        effective_fps=effective_fps,
        mean_depth_ms=mean_dep,
        mean_stereo_ms=mean_ste,
        mean_decode_ms=mean_dec,
        mean_encode_ms=mean_enc,
        has_audio=probe.has_audio,
        audio_stream_count=len(probe.audio_streams),
        device=str(target_device),
        encoder=f"nvenc_{codec.lower()}",
        notes=notes,
        depth_scale=canon_depth_scale,
        pointer_traces=pointer_traces,
        startup_overhead_s=startup_overhead_s,
        validation_overhead_s=validation_overhead_s,
        mean_temporal_ms=mean_tem,
        scheduling=scheduling,
        batch_size=batch_size,
        peak_memory_mb=peak_memory_mb,
    )
