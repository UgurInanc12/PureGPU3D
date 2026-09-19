"""Dedicated worker process for PureGPU3D model preparation and video conversion.

Runs isolated from the desktop UI process to:
  1. Prevent heavy PyTorch/CUDA and model import overhead in the UI.
  2. Shield UI responsiveness from memory and GIL contention during deep inference.
  3. Enforce strict JSON-lines protocol on stdout while redirecting upstream logs
     and library prints (like DA3's '[INFO ] using MLP layer as FFN') to stderr.
  4. Ensure bounded child process lifecycle via Windows Job Object, preventing
     orphaned FFmpeg processes on sudden termination or cancellation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Early stdout isolation: Redirect standard stdout (fd 1) to stderr (fd 2).
# Protocol messages are written exclusively to a preserved protocol descriptor.
# ---------------------------------------------------------------------------
_protocol_stream: Any = None


def _init_protocol_stream() -> Any:
    global _protocol_stream
    if _protocol_stream is not None:
        return _protocol_stream

    try:
        real_stdout_fd = os.dup(1)
        # Redirect C-level and Python-level fd 1 to fd 2 (stderr)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        _protocol_stream = open(real_stdout_fd, "w", encoding="utf-8", buffering=1)
    except Exception as err:
        # Fallback in environments where fd duplication is constrained
        sys.stderr.write(f"[worker-init] Warning: fd redirection fallback: {err}\n")
        _protocol_stream = sys.__stdout__
    return _protocol_stream


def emit_protocol(msg: Dict[str, Any]) -> None:
    """Emit a single JSON-lines protocol message to the isolated protocol stream."""
    stream = _init_protocol_stream()
    from puregpu3d.runtime.protocol import serialize_protocol_line
    line = serialize_protocol_line(msg)
    try:
        stream.write(line + "\n")
        stream.flush()
    except Exception as err:
        sys.stderr.write(f"[worker-protocol] Failed to emit protocol line: {err}\n")


# ---------------------------------------------------------------------------
# Windows Job Object: Auto-kill child processes (FFmpeg reader/writer) on exit.
# ---------------------------------------------------------------------------
_job_handle: Optional[int] = None


def _setup_windows_job_object() -> None:
    """Attach current worker process to a Windows Job Object with KILL_ON_JOB_CLOSE."""
    global _job_handle
    if sys.platform != "win32":
        return

    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryLimit", ctypes.c_size_t),
                ("PeakJobMemoryLimit", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            sys.stderr.write(f"[worker-job] Warning: CreateJobObjectW failed: {ctypes.get_last_error()}\n")
            return

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        res = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not res:
            sys.stderr.write(f"[worker-job] Warning: SetInformationJobObject failed: {ctypes.get_last_error()}\n")
            return

        current_process = kernel32.GetCurrentProcess()
        res_assign = kernel32.AssignProcessToJobObject(job, current_process)
        if not res_assign:
            sys.stderr.write(f"[worker-job] Warning: AssignProcessToJobObject failed: {ctypes.get_last_error()}\n")
            return

        _job_handle = job
        sys.stderr.write("[worker-job] Windows Job Object active with KILL_ON_JOB_CLOSE.\n")
    except Exception as err:
        sys.stderr.write(f"[worker-job] Warning: Job object setup failed: {err}\n")


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------
def run_worker(command: Any) -> int:
    """Execute the conversion job according to WorkerCommand specification."""
    from puregpu3d.models.catalog import get_model_entry, load_catalog
    from puregpu3d.models.store import (
        DownloadCancelledError,
        LicenseAcknowledgmentRequiredError,
        ModelStore,
        ModelStoreStatus,
    )
    from puregpu3d.runtime.paths import resolve_app_root
    from puregpu3d.runtime.protocol import (
        Stage,
        make_cancelled_msg,
        make_completed_msg,
        make_conversion_progress_msg,
        make_download_progress_msg,
        make_error_msg,
        make_status_msg,
    )
    from puregpu3d.stereo import DisparityConfig, StereoConfig
    from puregpu3d.video.convert import (
        ConversionCancelledError,
        convert_video,
    )
    from puregpu3d.video.probe import find_ffmpeg, find_ffprobe, probe_video

    job_id = command.job_id
    current_stage = Stage.STARTING
    emit_protocol(make_status_msg(job_id, Stage.STARTING, "Worker initialized"))

    cancel_file_path = Path(command.cancel_file).resolve() if command.cancel_file else None

    def check_cancelled() -> bool:
        if cancel_file_path and cancel_file_path.is_file():
            return True
        return False

    try:
        # 1. Resolve media binaries
        current_stage = Stage.VALIDATING
        emit_protocol(make_status_msg(job_id, Stage.VALIDATING, "Validating media binaries and paths"))

        if getattr(sys, "frozen", False):
            app_root = resolve_app_root()
            ffmpeg_bin = app_root / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
            ffprobe_bin = app_root / "bin" / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
            if not ffmpeg_bin.is_file() or not ffprobe_bin.is_file():
                raise FileNotFoundError(
                    f"Frozen executable requires bundled media binaries at {app_root / 'bin'}, host fallback forbidden."
                )
        else:
            ffmpeg_bin = find_ffmpeg(command.ffmpeg_path)
            ffprobe_bin = find_ffprobe(command.ffprobe_path)

        input_path = Path(command.input_path).resolve()
        output_path = Path(command.output_path).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path == output_path:
            raise ValueError(f"Input and output paths cannot be identical: {input_path}")

        if output_path.exists() and not command.overwrite:
            raise FileExistsError(f"Destination output file already exists: {output_path}")

        # Pre-probe input to report format
        probe = probe_video(input_path, ffprobe_path=ffprobe_bin, strict_sdr_cfr=True)
        sys.stderr.write(
            f"[worker] Probed: {probe.width}x{probe.height} @ {probe.fps:.2f}fps, "
            f"frames={probe.frame_count}, audio={probe.has_audio}\n"
        )

        if check_cancelled():
            raise ConversionCancelledError("Job cancelled before model loading.")

        # 2. Check model catalog and supported slice
        catalog = load_catalog()
        model_entry = get_model_entry(command.model_id, catalog)

        from puregpu3d.models.da3_adapter import DA3DepthAdapter

        # Wire verified models: Small, Base, Mono Large, Metric Large
        verified_ids = set(DA3DepthAdapter.get_verified_model_ids())
        if model_entry.id not in verified_ids:
            err_msg = (
                f"Model '{model_entry.ui_name}' ({model_entry.id}) is unverified or not integrated for inference "
                f"in this version. Supported models: {', '.join(sorted(verified_ids))}."
            )
            emit_protocol(make_error_msg(job_id, err_msg, stage=Stage.FAILED, detail="Unsupported model selection"))
            return 1

        # 3. Model preparation via ModelStore
        store = ModelStore(catalog=catalog)
        current_status = store.get_status(model_entry.id)

        if current_status == ModelStoreStatus.AWAITING_ACKNOWLEDGMENT and not command.acknowledge_license:
            emit_protocol(
                make_status_msg(
                    job_id,
                    Stage.AWAITING_ACKNOWLEDGMENT,
                    f"Model '{model_entry.ui_name}' requires explicit license acknowledgment.",
                )
            )
            emit_protocol(
                make_error_msg(
                    job_id,
                    f"License acknowledgment required for {model_entry.ui_name}: {model_entry.license_info.license}",
                    stage=Stage.AWAITING_ACKNOWLEDGMENT,
                )
            )
            return 1

        # Determine target device
        import torch
        if command.device:
            target_device = torch.device(command.device)
        else:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if target_device.type == "cuda" and target_device.index is None:
            idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
            target_device = torch.device(f"cuda:{idx}")

        # Upfront GPU pipeline preflight and route resolution
        from puregpu3d.runtime.protocol import PipelineRoute
        from puregpu3d.video.gpu_convert import check_gpu_pipeline_support, convert_video_gpu

        requested_route = getattr(command, "pipeline_route", PipelineRoute.AUTO) or PipelineRoute.AUTO
        enable_temporal = getattr(command, "enable_temporal_stabilization", True)

        gpu_supported = False
        gpu_reason = "Non-CUDA device"
        if target_device.type == "cuda":
            gpu_id = target_device.index if target_device.index is not None else 0
            gpu_supported, gpu_reason = check_gpu_pipeline_support(gpu_id=gpu_id)

        if requested_route == PipelineRoute.GPU:
            if not gpu_supported:
                err_msg = f"GPU pipeline explicitly requested but unsupported on this host: {gpu_reason}"
                emit_protocol(make_error_msg(job_id, err_msg, stage=Stage.VALIDATING, detail=gpu_reason))
                return 1
            chosen_route = "gpu"
        elif requested_route == PipelineRoute.COMPATIBLE:
            chosen_route = "compatible"
        else:  # PipelineRoute.AUTO
            if target_device.type == "cuda" and gpu_supported:
                chosen_route = "gpu"
            else:
                chosen_route = "compatible"
                if target_device.type == "cuda" and not gpu_supported:
                    emit_protocol(
                        make_status_msg(
                            job_id,
                            Stage.STARTING,
                            f"GPU pipeline unsupported ({gpu_reason}); falling back upfront to compatible pipeline.",
                        )
                    )

        emit_protocol(
            make_status_msg(
                job_id,
                Stage.STARTING,
                f"Resolved route: {chosen_route.upper()} (device={target_device}, temporal={'ON' if enable_temporal else 'OFF'}).",
            )
        )

        def load_verifier(rev_dir: Path) -> None:
            sys.stderr.write(f"[worker] Verifying model load at {rev_dir} on {target_device}...\n")
            DA3DepthAdapter(model_dir=rev_dir, identifier=model_entry.id, device=target_device)

        def dl_progress(p: Any) -> None:
            if check_cancelled():
                raise DownloadCancelledError("Download cancelled by user.")
            emit_protocol(
                make_download_progress_msg(
                    job_id=job_id,
                    downloaded_bytes=p.downloaded_bytes,
                    total_bytes=p.total_bytes,
                    percent=p.fraction * 100.0,
                    filename=p.file_name,
                )
            )

        if current_status != ModelStoreStatus.READY:
            current_stage = Stage.DOWNLOADING
            emit_protocol(
                make_status_msg(
                    job_id,
                    Stage.DOWNLOADING,
                    f"Preparing {model_entry.ui_name} ({model_entry.revision[:8]})...",
                )
            )
            rev_dir = store.prepare_model(
                identifier=model_entry.id,
                progress_callback=dl_progress,
                is_cancelled=check_cancelled,
                load_verifier=load_verifier,
                acknowledge_license=command.acknowledge_license,
            )
        else:
            rev_dir = store.get_revision_dir(model_entry.id)

        if check_cancelled():
            raise ConversionCancelledError("Job cancelled after model preparation.")

        # 4. Load adapter
        current_stage = Stage.LOADING
        emit_protocol(make_status_msg(job_id, Stage.LOADING, f"Loading {model_entry.ui_name} adapter onto {target_device}..."))
        adapter = DA3DepthAdapter(model_dir=rev_dir, identifier=model_entry.id, device=target_device)

        if check_cancelled():
            raise ConversionCancelledError("Job cancelled before conversion start.")

        # 5. Conversion
        current_stage = Stage.CONVERTING
        if chosen_route == "gpu":
            emit_protocol(
                make_status_msg(
                    job_id,
                    Stage.CONVERTING,
                    f"Converting to Full-SBS via GPU-resident pipeline (scale={command.depth_scale}, "
                    f"temporal={'ON' if enable_temporal else 'OFF'}, strength={command.disparity_strength:.3f})...",
                )
            )
        else:
            emit_protocol(
                make_status_msg(
                    job_id,
                    Stage.CONVERTING,
                    f"Converting to Full-SBS via Compatible pipeline (scale={command.depth_scale}, "
                    f"temporal={'ON' if enable_temporal else 'OFF'}, strength={command.disparity_strength:.3f})...",
                )
            )

        stereo_config = StereoConfig(
            disparity=DisparityConfig(
                strength=command.disparity_strength,
                q_screen=command.q_screen,
            )
        )

        start_time = time.monotonic()
        last_progress_time = start_time
        last_frame_count = 0

        def conversion_progress_cb(current_frame: int, total_frames: int) -> None:
            nonlocal last_progress_time, last_frame_count
            now = time.monotonic()
            delta_t = now - last_progress_time
            delta_f = current_frame - last_frame_count

            fps = (delta_f / delta_t) if delta_t > 0.05 else 0.0
            if delta_t >= 0.25 or current_frame == total_frames:
                last_progress_time = now
                last_frame_count = current_frame

            percent = (current_frame / total_frames * 100.0) if total_frames > 0 else 0.0
            rem_frames = total_frames - current_frame
            eta = (rem_frames / fps) if (fps > 0 and rem_frames > 0) else None

            emit_protocol(
                make_conversion_progress_msg(
                    job_id=job_id,
                    current_frame=current_frame,
                    total_frames=total_frames,
                    percent=percent,
                    fps=fps,
                    eta_seconds=eta,
                )
            )

        if chosen_route == "gpu":
            res = convert_video_gpu(
                input_path=input_path,
                output_path=output_path,
                model=adapter,
                device=target_device,
                stereo_config=stereo_config,
                depth_scale=command.depth_scale,
                codec="hevc",
                enable_temporal_stabilization=enable_temporal,
                overwrite=command.overwrite,
                ffmpeg_path=ffmpeg_bin,
                ffprobe_path=ffprobe_bin,
                progress_callback=conversion_progress_cb,
                cancel_callback=check_cancelled,
            )
        else:
            res = convert_video(
                input_path=input_path,
                output_path=output_path,
                model=adapter,
                device=target_device,
                stereo_config=stereo_config,
                depth_scale=command.depth_scale,
                encoder="auto",
                enable_temporal_stabilization=enable_temporal,
                overwrite=command.overwrite,
                ffmpeg_path=ffmpeg_bin,
                ffprobe_path=ffprobe_bin,
                progress_callback=conversion_progress_cb,
                cancel_callback=check_cancelled,
            )

        current_stage = Stage.FINALIZING
        emit_protocol(make_status_msg(job_id, Stage.FINALIZING, "Finalizing transaction and verifying output..."))

        # Verify output exists and is readable
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Conversion reported success but output file is missing or empty: {output_path}")

        current_stage = Stage.COMPLETED
        emit_protocol(make_status_msg(job_id, Stage.COMPLETED, "Conversion completed successfully."))

        res_dict = res.to_dict()
        res_dict["pipeline_route"] = chosen_route
        res_dict["backend"] = "gpu_resident" if chosen_route == "gpu" else "ffmpeg_compatible"
        res_dict["resolved_backend"] = (
            "GPU-Resident (NVDEC -> CUDA -> NVENC)"
            if chosen_route == "gpu"
            else "Compatible (FFmpeg -> PyTorch -> NVENC/x264)"
        )
        emit_protocol(make_completed_msg(job_id, res_dict))
        return 0

    except (ConversionCancelledError, DownloadCancelledError):
        emit_protocol(make_status_msg(job_id, Stage.CANCELLED, "Conversion cancelled by user."))
        emit_protocol(make_cancelled_msg(job_id, stage=current_stage, message="Operation cancelled."))
        return 0
    except Exception as err:
        tb_str = traceback.format_exc()
        sys.stderr.write(f"[worker-error] {tb_str}\n")
        emit_protocol(
            make_error_msg(
                job_id=job_id,
                error=str(err),
                stage=current_stage,
                detail=tb_str,
            )
        )
        return 1


def main() -> int:
    _init_protocol_stream()
    _setup_windows_job_object()

    parser = argparse.ArgumentParser(description="PureGPU3D video conversion worker process.")
    parser.add_argument("--worker", action="store_true", help="Worker mode flag (used by frozen binary).")
    parser.add_argument("--command-file", type=str, help="Path to JSON file containing WorkerCommand.")
    parser.add_argument("--command-json", type=str, help="Direct JSON string containing WorkerCommand.")
    parser.add_argument("--stdin", action="store_true", help="Read JSON command from stdin.")

    args = parser.parse_args()

    from puregpu3d.runtime.protocol import WorkerCommand

    command_raw: Optional[str] = None
    if args.command_file:
        with open(args.command_file, "r", encoding="utf-8") as f:
            command_raw = f.read()
    elif args.command_json:
        command_raw = args.command_json
    elif args.stdin:
        command_raw = sys.stdin.read()
    else:
        sys.stderr.write("[worker] Error: No command specified via --command-file, --command-json, or --stdin.\n")
        return 2

    if command_raw is None:
        sys.stderr.write("[worker] Error: Command data is empty.\n")
        return 2

    try:
        command = WorkerCommand.from_json(command_raw)
    except Exception as err:
        sys.stderr.write(f"[worker] Failed to parse WorkerCommand JSON: {err}\n")
        return 2

    return run_worker(command)


if __name__ == "__main__":
    sys.exit(main())
