"""Video conversion pipeline for PureGPU3D.

Executes streaming conversion of monocular SDR CFR video into full-resolution
Side-by-Side (Full-SBS, 2W x H) stereoscopic video with preserved audio using
Depth Anything 3 (DA3) depth inference and the bounded PyTorch stereo renderer.

Architecture & Safety Guarantees:
  - Bounded Streaming: Exactly 1 frame in flight across reader -> depth -> stereo -> writer.
    Never loads entire video or large batches into system memory or GPU VRAM.
  - Rational Frame Rate: Preserves exact rational CFR frame timing (e.g. 24/1, 30000/1001).
  - Pipe Deadlock Immunity: Bounded concurrent background stderr draining prevents OS pipe
    buffer saturation on Windows.
  - Safe Child Lifecycle: Reader and writer child processes are terminated/killed FIRST
    before pipe closure to prevent stderr drainer lockups and buffer flushing deadlocks.
  - Cancellation Monitor: Bounded owner cancellation monitor polls cancel callbacks and
    terminates ONLY owned child handles, unblocking stuck IO and converting failures
    into Cancellation errors.
  - Fail-Hard Reader: Truncated frames and non-zero reader exit codes fail immediately
    and never promote staging outputs.
  - Multi-track Audio Preservation: Retains all copy-compatible audio streams without
    -shortest truncation.
  - O(1) Memory Metrics: Uses online cumulative sums and counts for performance statistics.
  - Transactional Export: Output is written to a unique same-volume staging file and validated
    (dimensions, decodability, duration, exact frame count, audio stream count & properties)
    under per-destination file locking before atomic promotion.
  - Wall-Clock Timing: Measures end-to-end wall-clock throughput, not isolated kernel sums.

First Slice Limitations:
  - Temporal Stability: Depth Anything 3 Small operates per-frame, and the stereo renderer
    normalizes depth per-frame (percentiles q_p1, q_p99). This initial vertical slice validates
    bounded streaming video I/O, audio remux, and GPU synthesis, but does NOT yet include
    cross-frame temporal smoothing or multi-frame window consistency.
"""

from __future__ import annotations

import collections
import logging
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import numpy as np
import cv2
import torch

from puregpu3d.jobs.output_transaction import (
    OutputTransaction,
    TransactionCancelledError,
    TransactionError,
    ValidationError,
)
from puregpu3d.stereo import (
    StereoConfig,
    TemporalDepthConfig,
    TemporalDepthStabilizer,
    render_stereo_frame,
)
from puregpu3d.video.encoders import (
    DEFAULT_CQ,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_NVENC_PRESET,
    DEFAULT_X264_PRESET,
    EncoderError,
    EncoderPreflightError,
    InvalidEncoderError,
    resolve_encoder,
)
from puregpu3d.video.probe import (
    SUPPORTED_COPY_AUDIO_CODECS,
    UnsupportedMediaError,
    VideoProbeResult,
    find_ffmpeg,
    find_ffprobe,
    probe_video,
)
from puregpu3d.video.scenes import SceneCutDetector, SceneDetectorConfig

logger = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    """Base error raised when video conversion fails."""
    pass


class ConversionCancelledError(ConversionError):
    """Raised when conversion is cancelled via callback."""
    pass


@dataclass(frozen=True)
class ConversionResult:
    """Detailed results and performance metrics from video conversion."""

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
    has_audio: bool
    device: str
    encoder: str
    notes: List[str]
    depth_scale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_path"] = str(self.input_path)
        d["output_path"] = str(self.output_path)
        return d


def _drain_pipe_bounded(pipe, deque_buf: Deque[str], max_lines: int = 100) -> None:
    """Continuously drain process pipe in a background thread to prevent buffer deadlock."""
    try:
        for line in iter(pipe.readline, b""):
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    deque_buf.append(decoded)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            if pipe and not pipe.closed:
                pipe.close()
        except Exception:
            pass


def _terminate_process_safely(proc: Optional[subprocess.Popen], name: str = "child") -> None:
    """Terminate and reap child process safely without leaking resources or blocking pipes.

    Terminates/kills the process FIRST, then closes open pipes to avoid flushing deadlocks
    or blocking background drainer threads.
    """
    if proc is None:
        return

    # 1. Terminate or kill first while pipes remain accessible to OS kernel
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except OSError:
                pass
        proc.poll()
        logger.debug(f"Process {name} (PID {proc.pid}) terminated.")

    # 2. Close pipes safely after process is terminated/reaped
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe and not pipe.closed:
            try:
                pipe.close()
            except Exception:
                pass

    try:
        proc.wait(timeout=0.2)
    except Exception:
        pass


class CancellationMonitor:
    """Bounded background monitor that polls cancel_callback and terminates ONLY owned child handles."""

    def __init__(
        self,
        cancel_callback: Optional[Callable[[], bool]],
        poll_interval: float = 0.05,
    ) -> None:
        self.cancel_callback = cancel_callback
        self.poll_interval = poll_interval
        self._cancelled = threading.Event()
        self._stop_event = threading.Event()
        self._procs: List[subprocess.Popen] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def register_proc(self, proc: Optional[subprocess.Popen]) -> None:
        if proc is None:
            return
        with self._lock:
            if proc not in self._procs:
                self._procs.append(proc)
            if self._cancelled.is_set():
                _terminate_process_safely(proc, "cancelled_child")

    def unregister_proc(self, proc: Optional[subprocess.Popen]) -> None:
        if proc is None:
            return
        with self._lock:
            if proc in self._procs:
                self._procs.remove(proc)

    def is_cancelled(self) -> bool:
        if self._cancelled.is_set():
            return True
        if self.cancel_callback is not None:
            try:
                if self.cancel_callback():
                    self._cancelled.set()
                    return True
            except Exception:
                pass
        return False

    def start(self) -> None:
        if self.cancel_callback is None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="puregpu3d-cancellation-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.cancel_callback and self.cancel_callback():
                    self._cancelled.set()
                    with self._lock:
                        for p in list(self._procs):
                            _terminate_process_safely(p, "cancelled_child")
                    break
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


def convert_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    model: Any,
    device: Optional[Union[str, torch.device]] = None,
    stereo_config: Optional[StereoConfig] = None,
    depth_scale: Optional[Union[str, float]] = None,
    overwrite: bool = False,
    ffmpeg_path: Optional[Union[str, Path]] = None,
    ffprobe_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    encoder: str = DEFAULT_ENCODER,
    x264_preset: str = DEFAULT_X264_PRESET,
    crf: int = DEFAULT_CRF,
    nvenc_preset: str = DEFAULT_NVENC_PRESET,
    cq: int = DEFAULT_CQ,
    enable_temporal_stabilization: bool = True,
    temporal_config: Optional[TemporalDepthConfig] = None,
    scene_config: Optional[SceneDetectorConfig] = None,
) -> ConversionResult:
    """Convert input SDR CFR video into full-resolution Side-by-Side (Full-SBS) video.

    Args:
        input_path: Source media file path.
        output_path: Destination stereoscopic MP4 path.
        model: DA3 model adapter instance, mock model instance with .infer(),
            or Path to local model checkpoint folder.
        device: PyTorch device ('cuda' or 'cpu'). Defaults to CUDA if available.
        stereo_config: Stereoscopic rendering configuration.
        depth_scale: Optional depth processing scale ('auto', '1/4', '1/2', '1/1' or float).
        overwrite: If True, allows replacing an existing destination file upon success.
        ffmpeg_path: Optional path to ffmpeg binary.
        ffprobe_path: Optional path to ffprobe binary.
        progress_callback: Optional callback receiving (processed_frames, total_frames).
        cancel_callback: Optional callback returning True if conversion should abort.
        encoder: Video encoder selection ('auto', 'hevc_nvenc', 'h264_nvenc', 'libx264').
            Defaults to 'libx264' for baseline stability.
        x264_preset: x264 encoder speed preset (default 'medium').
        crf: x264 constant rate factor (default 18, not used by NVENC).
        nvenc_preset: NVENC encoder preset (default 'p4').
        cq: NVENC target constant quality level in VBR mode (default 23, not used by libx264).
        enable_temporal_stabilization: Whether to apply causal optical-flow depth stabilization.
        temporal_config: Optional TemporalDepthConfig parameters.
        scene_config: Optional SceneDetectorConfig parameters.

    Returns:
        ConversionResult containing execution metrics and verified output info.
    """
    resolved_input = Path(input_path).resolve()
    resolved_output = Path(output_path).resolve()

    # Upfront depth scale validation before any subprocess or model invocation
    canon_depth_scale: Optional[str] = None
    if depth_scale is not None:
        from puregpu3d.models.geometry import parse_depth_scale
        canon_depth_scale, _ = parse_depth_scale(depth_scale)

    ffmpeg_bin = find_ffmpeg(ffmpeg_path)
    ffprobe_bin = find_ffprobe(ffprobe_path)

    # Resolve target device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 1. Probe input media with strict upfront validation
    probe = probe_video(resolved_input, ffprobe_path=ffprobe_bin, strict_sdr_cfr=True)

    in_w = probe.width
    in_h = probe.height
    out_w = 2 * in_w
    out_h = in_h
    fps_frac = probe.frame_rate
    total_frames = probe.frame_count

    # Upfront validation of all audio streams for multi-track support
    if probe.has_audio:
        for audio_info in probe.audio_streams:
            if audio_info.codec_name not in SUPPORTED_COPY_AUDIO_CODECS:
                raise UnsupportedMediaError(
                    f"Unsupported media: Audio track #{audio_info.index} has codec '{audio_info.codec_name}' "
                    f"which cannot be copied into MP4 container. Supported stream copy codecs are: "
                    f"{sorted(SUPPORTED_COPY_AUDIO_CODECS)}."
                )

    # Resolve and validate video encoder upfront with FullSBS dimensions
    try:
        resolved_encoder = resolve_encoder(
            encoder=encoder,
            width=out_w,
            height=out_h,
            crf=crf,
            cq=cq,
            x264_preset=x264_preset,
            nvenc_preset=nvenc_preset,
            ffmpeg_path=ffmpeg_bin,
        )
    except InvalidEncoderError:
        raise
    except EncoderError as err:
        raise ConversionError(f"Encoder configuration or preflight failed: {err}") from err

    # Resolve model adapter instance
    if isinstance(model, (str, Path)):
        from puregpu3d.models.da3_adapter import DA3SmallDepthAdapter
        logger.info(f"Instantiating DA3SmallDepthAdapter from '{model}' on device {target_device}...")
        adapter = DA3SmallDepthAdapter(model_dir=model, device=target_device)
    else:
        adapter = model

    # Calculate frame byte boundaries (RGB24)
    in_frame_bytes = in_w * in_h * 3
    out_frame_bytes = out_w * out_h * 3

    # Initialize safe transactional staging on destination volume
    txn = OutputTransaction(
        source_path=resolved_input,
        destination_path=resolved_output,
        overwrite=overwrite,
        expected_width=out_w,
        expected_height=out_h,
        expected_frames=total_frames if total_frames > 0 else None,
        expect_audio=probe.has_audio,
        expected_audio_streams=len(probe.audio_streams),
        expected_duration=probe.duration if probe.duration > 0 else None,
        expected_frame_rate=fps_frac,
        ffprobe_path=ffprobe_bin,
        ffmpeg_path=ffmpeg_bin,
    )

    encoder_note = (
        f"Video encoder: {resolved_encoder.selected_encoder} "
        f"({'hardware NVENC' if resolved_encoder.is_hardware else 'software libx264'})."
    )
    notes = [
        "Temporal depth stabilization: motion-compensated Farneback flow with shot cut reset and online normalization."
        if enable_temporal_stabilization
        else "Temporal depth stabilization: disabled (baseline per-frame quantile normalization).",
        encoder_note,
    ]
    if resolved_encoder.fallback_reason:
        notes.append(f"Hardware encoder fallback: {resolved_encoder.fallback_reason}")

    # Initialize temporal stabilizer and scene cut detector if enabled
    if enable_temporal_stabilization:
        t_cfg = temporal_config if temporal_config is not None else TemporalDepthConfig(enabled=True)
        temporal_stabilizer = TemporalDepthStabilizer(config=t_cfg)
        scene_detector = SceneCutDetector(config=scene_config)
    else:
        temporal_stabilizer = None
        scene_detector = None

    reader_proc: Optional[subprocess.Popen] = None
    writer_proc: Optional[subprocess.Popen] = None
    reader_stderr_lines: Deque[str] = collections.deque(maxlen=100)
    writer_stderr_lines: Deque[str] = collections.deque(maxlen=100)
    reader_thread: Optional[threading.Thread] = None
    writer_thread: Optional[threading.Thread] = None

    # O(1) online metrics tracking
    depth_time_total_ms: float = 0.0
    depth_count: int = 0
    stereo_time_total_ms: float = 0.0
    stereo_count: int = 0
    processed_count = 0

    monitor = CancellationMonitor(cancel_callback)
    monitor.start()

    wall_start = time.perf_counter()

    try:
        with txn:
            staging_file = txn.staging_path

            # Inner try-finally ensures child processes are reaped BEFORE txn.__exit__ unlinks staging
            try:
                # 2. Construct FFmpeg Reader Command
                reader_cmd = [
                    str(ffmpeg_bin),
                    "-v", "warning",
                    "-i", str(resolved_input),
                    "-map", "0:v:0",
                    "-fps_mode", "passthrough",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-",
                ]

                # 3. Construct FFmpeg Writer Command
                # Input 0: Raw SBS RGB24 stream from stdin
                source_video = next(s for s in probe.raw_info["streams"] if s.get("codec_type") == "video")
                video_start = source_video.get("start_time", "0")
                if video_start in (None, "N/A"):
                    video_start = "0"
                writer_cmd = [
                    str(ffmpeg_bin),
                    "-v", "warning",
                    "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-s", f"{out_w}x{out_h}",
                    "-framerate", f"{fps_frac.numerator}/{fps_frac.denominator}",
                    "-i", "-",
                ]

                if probe.has_audio:
                    writer_cmd.append("-copyts")
                    # Input 1: Original media for direct audio stream copies
                    writer_cmd.extend([
                        "-i", str(resolved_input),
                        "-map", "0:v:0",
                    ])
                    # Map every audio stream explicitly and remove -shortest to prevent short audio truncating video
                    for i in range(len(probe.audio_streams)):
                        writer_cmd.extend(["-map", f"1:a:{i}"])
                    writer_cmd.extend(["-c:a", "copy"])
                else:
                    writer_cmd.extend([
                        "-map", "0:v:0",
                    ])

                writer_cmd.extend(resolved_encoder.ffmpeg_args)
                writer_cmd.extend([
                    "-vf", f"settb=1/90000,setpts=PTS+{float(video_start) if probe.has_audio else 0}/TB",
                    "-enc_time_base", "1:90000",
                    "-fps_mode", "passthrough",
                    "-movflags", "+faststart",
                    str(staging_file),
                ])

                logger.info(f"Spawning FFmpeg reader: {' '.join(reader_cmd)}")
                reader_proc = subprocess.Popen(
                    reader_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=in_frame_bytes * 2,
                )
                monitor.register_proc(reader_proc)

                logger.info(f"Spawning FFmpeg writer: {' '.join(writer_cmd)}")
                writer_proc = subprocess.Popen(
                    writer_cmd,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=out_frame_bytes * 2,
                )
                monitor.register_proc(writer_proc)

                # Start bounded stderr drainers
                reader_thread = threading.Thread(
                    target=_drain_pipe_bounded,
                    args=(reader_proc.stderr, reader_stderr_lines),
                    daemon=True,
                )
                writer_thread = threading.Thread(
                    target=_drain_pipe_bounded,
                    args=(writer_proc.stderr, writer_stderr_lines),
                    daemon=True,
                )
                reader_thread.start()
                writer_thread.start()

                # 4. Streaming conversion loop
                assert reader_proc.stdout is not None
                assert writer_proc.stdin is not None

                while True:
                    # Check cancellation callback
                    if monitor.is_cancelled():
                        raise ConversionCancelledError("Conversion cancelled by user request.")

                    # Read 1 frame of raw RGB24
                    try:
                        raw_bytes = reader_proc.stdout.read(in_frame_bytes)
                    except (OSError, ValueError) as err:
                        if monitor.is_cancelled():
                            raise ConversionCancelledError("Conversion cancelled by user request.") from err
                        raise ConversionError(f"Error reading frame from FFmpeg reader: {err}") from err

                    if not raw_bytes:
                        break  # End of stream reached cleanly

                    # Fail hard on partial/truncated frames; never promote partial data
                    if len(raw_bytes) < in_frame_bytes:
                        raise ConversionError(
                            f"Truncated frame received from FFmpeg reader at frame {processed_count}: "
                            f"received {len(raw_bytes)} bytes, expected full frame of {in_frame_bytes} bytes."
                        )

                    # Frame to RGB array: (H, W, 3)
                    frame_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).copy().reshape((in_h, in_w, 3))

                    # Step 1: Infer depth
                    t_d0 = time.perf_counter()
                    if canon_depth_scale is not None:
                        depth_result = adapter.infer(
                            frame_rgb,
                            depth_scale=canon_depth_scale,
                            return_original_size=True,
                        )
                    else:
                        depth_result = adapter.infer(
                            frame_rgb,
                            return_original_size=True,
                        )
                    t_d1 = time.perf_counter()
                    depth_time_total_ms += (t_d1 - t_d0) * 1000.0
                    depth_count += 1

                    # Step 1b: Causal temporal depth stabilization & online shot normalization
                    norm_bounds = None
                    effective_depth = depth_result.depth
                    if temporal_stabilizer is not None and scene_detector is not None:
                        # Stabilize the model-resolution depth, not a full-HD
                        # upsample. Preserve source-resolution depth edges by
                        # upsampling only the temporal correction.
                        raw_low = getattr(depth_result, "depth_raw", depth_result.depth)
                        low_h, low_w = raw_low.shape
                        frame_low = cv2.resize(frame_rgb, (low_w, low_h), interpolation=cv2.INTER_AREA)
                        cut_res = scene_detector.update(frame_low)
                        stab_res = temporal_stabilizer.process_frame(
                            frame_rgb=frame_low,
                            raw_depth=raw_low,
                            is_cut=cut_res.is_cut,
                            letterbox_crop=cut_res.letterbox_crop,
                        )
                        correction = stab_res.depth - raw_low
                        effective_depth = np.maximum(
                            depth_result.depth + cv2.resize(correction, (in_w, in_h), interpolation=cv2.INTER_LINEAR),
                            np.finfo(np.float32).eps,
                        )
                        norm_bounds = stab_res.normalization_bounds

                    # Step 2: Render stereoscopic frame
                    t_s0 = time.perf_counter()
                    stereo_result = render_stereo_frame(
                        frame_rgb,
                        effective_depth,
                        config=stereo_config,
                        device=target_device,
                        return_numpy=True,
                        output_uint8=True,
                        normalization_bounds=norm_bounds,
                    )
                    t_s1 = time.perf_counter()
                    stereo_time_total_ms += (t_s1 - t_s0) * 1000.0
                    stereo_count += 1

                    sbs_frame = stereo_result.sbs_color
                    if isinstance(sbs_frame, np.ndarray):
                        sbs_bytes = sbs_frame.tobytes()
                        sbs_shape = sbs_frame.shape
                    else:
                        sbs_np = sbs_frame.detach().cpu().numpy()
                        sbs_bytes = sbs_np.tobytes()
                        sbs_shape = sbs_np.shape

                    assert sbs_shape == (out_h, out_w, 3), f"Invalid SBS shape: {sbs_shape}"

                    # Step 3: Write SBS frame to encoder
                    try:
                        if writer_proc.stdin.closed:
                            if monitor.is_cancelled():
                                raise ConversionCancelledError("Conversion cancelled by user request.")
                            raise ConversionError("FFmpeg encoder stdin unexpectedly closed.")
                        writer_proc.stdin.write(sbs_bytes)
                    except (BrokenPipeError, OSError, ValueError) as err:
                        if monitor.is_cancelled():
                            raise ConversionCancelledError("Conversion cancelled by user request.") from err
                        writer_err = "\n".join(list(writer_stderr_lines)[-20:])
                        raise ConversionError(
                            f"FFmpeg encoder pipe broke during frame {processed_count}: {err}\n"
                            f"Encoder stderr tail:\n{writer_err}"
                        ) from err

                    processed_count += 1
                    if progress_callback is not None:
                        progress_callback(processed_count, total_frames)

                # Close writer stdin to signal EOF to FFmpeg encoder
                try:
                    writer_proc.stdin.flush()
                    writer_proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

                # Check if cancelled before waiting
                if monitor.is_cancelled():
                    raise ConversionCancelledError("Conversion cancelled by user request.")

                # Wait for reader process to exit cleanly
                try:
                    r_code = reader_proc.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    _terminate_process_safely(reader_proc, "reader")
                    r_code = reader_proc.poll()

                # Wait for writer process to exit cleanly
                try:
                    w_code = writer_proc.wait(timeout=60.0)
                except subprocess.TimeoutExpired:
                    _terminate_process_safely(writer_proc, "writer")
                    w_code = writer_proc.poll()

                if monitor.is_cancelled():
                    raise ConversionCancelledError("Conversion cancelled by user request.")

                # Reader nonzero fails hard - never promote corrupted or truncated reader outputs
                if r_code != 0 and r_code is not None:
                    err_text = "\n".join(list(reader_stderr_lines)[-30:])
                    raise ConversionError(
                        f"FFmpeg reader exited with error code {r_code}.\nStderr:\n{err_text}"
                    )

                # Writer nonzero fails hard
                if w_code != 0 and w_code is not None:
                    err_text = "\n".join(list(writer_stderr_lines)[-30:])
                    raise ConversionError(
                        f"FFmpeg encoder exited with error code {w_code}.\nStderr:\n{err_text}"
                    )

                # Validate processed frame count against probed input frame count
                if total_frames > 0 and processed_count != total_frames:
                    raise ConversionError(
                        f"Processed frame count mismatch: processed {processed_count} frames, "
                        f"expected {total_frames} frames from source."
                    )

                # 5. Validate staging output and atomically promote to final destination
                logger.info("Validating rendered staging file and promoting to destination...")
                try:
                    txn.validate_and_promote(cancel_callback=monitor.is_cancelled)
                except TransactionCancelledError as err:
                    raise ConversionCancelledError(f"Conversion cancelled during validation: {err}") from err

            finally:
                # Ensure children are completely terminated and flushed before staging cleanup
                _terminate_process_safely(reader_proc, "reader")
                _terminate_process_safely(writer_proc, "writer")

    except Exception:
        _terminate_process_safely(reader_proc, "reader")
        _terminate_process_safely(writer_proc, "writer")
        raise
    finally:
        monitor.stop()
        _terminate_process_safely(reader_proc, "reader")
        _terminate_process_safely(writer_proc, "writer")
        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        if writer_thread and writer_thread.is_alive():
            writer_thread.join(timeout=1.0)

    wall_end = time.perf_counter()
    wall_duration = wall_end - wall_start
    effective_fps = processed_count / wall_duration if wall_duration > 0 else 0.0

    mean_depth = (depth_time_total_ms / depth_count) if depth_count > 0 else 0.0
    mean_stereo = (stereo_time_total_ms / stereo_count) if stereo_count > 0 else 0.0

    logger.info(
        f"Conversion complete: {processed_count} frames processed in {wall_duration:.2f}s "
        f"({effective_fps:.2f} FPS). Output: {resolved_output}"
    )

    return ConversionResult(
        input_path=resolved_input,
        output_path=resolved_output,
        input_width=in_w,
        input_height=in_h,
        output_width=out_w,
        output_height=out_h,
        frame_rate_str=f"{fps_frac.numerator}/{fps_frac.denominator}",
        total_frames_processed=processed_count,
        wall_clock_seconds=wall_duration,
        effective_fps=effective_fps,
        mean_depth_ms=mean_depth,
        mean_stereo_ms=mean_stereo,
        has_audio=probe.has_audio,
        device=str(target_device),
        encoder=resolved_encoder.selected_encoder,
        notes=notes,
        depth_scale=canon_depth_scale,
    )
