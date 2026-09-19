"""Desktop application controller for PureGPU3D.

Orchestrates UI state, video probing, model catalog inspection, and child conversion worker
lifecycle over strict JSON-lines QProcess IPC. Does not import PyTorch or heavy model libraries
in the desktop process.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from puregpu3d.models.catalog import (
    ModelCatalogEntry,
    get_model_entry,
    load_catalog,
)
from puregpu3d.models.store import ModelStore, ModelStoreStatus
from puregpu3d.runtime.paths import resolve_app_root
from puregpu3d.runtime.protocol import (
    MessageType,
    SUPPORTED_MODEL_IDS,
    Stage,
    UNSUPPORTED_MODEL_IDS,
    WorkerCommand,
    parse_protocol_line,
)
from puregpu3d.video.probe import (
    VideoProbeResult,
    find_ffmpeg,
    find_ffprobe,
    probe_video,
)

logger = logging.getLogger(__name__)


class DesktopState(str, enum.Enum):
    IDLE = "idle"
    VALIDATING = "validating"
    PREPARING_MODEL = "preparing_model"
    CONVERTING = "converting"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DesktopController(QObject):
    """Controller managing desktop state and worker QProcess execution."""

    # Signals
    state_changed = Signal(str)  # DesktopState value
    input_probed = Signal(object)  # VideoProbeResult or None
    model_changed = Signal(str, dict)  # model_id, info dict
    depth_scale_changed = Signal(str)  # scale string, e.g. "1/2"
    pipeline_route_changed = Signal(str)  # "auto", "gpu", "compatible"
    status_updated = Signal(str, str)  # stage, message
    download_progress = Signal(int, int, float, str)  # downloaded_bytes, total_bytes, percent, filename
    conversion_progress = Signal(int, int, float, float, object)  # frame, total, percent, fps, eta
    conversion_completed = Signal(dict)  # result dict
    conversion_failed = Signal(str, str)  # error, stage
    conversion_cancelled = Signal()
    log_received = Signal(str)  # worker stderr line

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: DesktopState = DesktopState.IDLE
        self._catalog: Dict[str, ModelCatalogEntry] = load_catalog()
        self._store = ModelStore(catalog=self._catalog)

        self._input_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._input_probe: Optional[VideoProbeResult] = None
        self._selected_model_id: str = "DA3-SMALL"
        self._disparity_strength: float = 0.03
        self._q_screen: float = 0.6
        self._depth_scale: str = "1/2"
        self._pipeline_route: str = "auto"
        self._enable_temporal_stabilization: bool = True
        self._overwrite: bool = False
        self._license_acknowledged: bool = False

        self._worker_process: Optional[QProcess] = None
        self._stdout_buffer: str = ""
        self._cancel_file: Optional[Path] = None
        self._command_file: Optional[Path] = None
        self._active_job_id: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None

        self._cancel_timer: Optional[QTimer] = None

    # Properties
    @property
    def state(self) -> DesktopState:
        return self._state

    @property
    def input_path(self) -> Optional[Path]:
        return self._input_path

    @property
    def output_path(self) -> Optional[Path]:
        return self._output_path

    @property
    def input_probe(self) -> Optional[VideoProbeResult]:
        return self._input_probe

    @property
    def selected_model_id(self) -> str:
        return self._selected_model_id

    @property
    def disparity_strength(self) -> float:
        return self._disparity_strength

    @property
    def q_screen(self) -> float:
        return self._q_screen

    @property
    def depth_scale(self) -> str:
        return self._depth_scale

    @property
    def pipeline_route(self) -> str:
        return self._pipeline_route

    @property
    def enable_temporal_stabilization(self) -> bool:
        return self._enable_temporal_stabilization

    @property
    def catalog(self) -> Dict[str, ModelCatalogEntry]:
        return self._catalog

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _set_state(self, new_state: DesktopState) -> None:
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(self._state.value)

    def get_catalog_entries(self) -> List[ModelCatalogEntry]:
        return list(self._catalog.values())

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Return comprehensive metadata and current availability for a catalog model."""
        entry = get_model_entry(model_id, self._catalog)
        status = self._store.get_status(entry.id)
        is_supported = (entry.id in SUPPORTED_MODEL_IDS)
        is_ack_needed = entry.license_info.noncommercial_ack_required or entry.license_info.license_conflict
        is_acked = self._store.is_license_acknowledged(entry)

        return {
            "id": entry.id,
            "repo_id": entry.repo_id,
            "ui_name": entry.ui_name,
            "revision": entry.revision,
            "parameters": entry.parameters,
            "role": entry.role,
            "category": entry.category,
            "license": entry.license_info.license,
            "license_type": entry.license_info.license_type,
            "license_conflict": entry.license_info.license_conflict,
            "conflict_details": entry.license_info.conflict_details,
            "ack_needed": is_ack_needed,
            "ack_recorded": is_acked,
            "weight_bytes": entry.total_weight_bytes,
            "total_bytes": entry.total_bytes,
            "status": status.value,
            "is_supported": is_supported,
            "support_note": (
                f"Verified Depth Anything 3 {entry.ui_name} inference ready."
                if is_supported
                else f"Not yet integrated: {entry.ui_name} model is not supported for conversion in this version."
            ),
        }

    def set_selected_model(self, model_id: str) -> None:
        """Select a model from the catalog."""
        entry = get_model_entry(model_id, self._catalog)
        self._selected_model_id = entry.id
        info = self.get_model_info(entry.id)
        self.model_changed.emit(entry.id, info)

    def set_disparity_strength(self, strength: float) -> None:
        """Update disparity strength factor."""
        self._disparity_strength = max(0.0, float(strength))

    def set_q_screen(self, q_screen: float) -> None:
        """Update zero parallax screen depth."""
        self._q_screen = min(1.0, max(0.0, float(q_screen)))

    def set_depth_scale(self, scale: str) -> None:
        """Update depth processing scale ('1/4', '1/2', '1/1')."""
        from puregpu3d.models.geometry import parse_depth_scale
        canon, _ = parse_depth_scale(scale)
        if self._depth_scale != canon:
            self._depth_scale = canon
            self.depth_scale_changed.emit(self._depth_scale)

    def set_pipeline_route(self, route: str) -> None:
        """Update requested pipeline route ('auto', 'gpu', 'compatible')."""
        from puregpu3d.runtime.protocol import PipelineRoute
        norm = str(route).strip().lower()
        if norm not in (PipelineRoute.AUTO, PipelineRoute.GPU, PipelineRoute.COMPATIBLE):
            raise ValueError(f"Invalid pipeline route '{route}'. Expected 'auto', 'gpu', or 'compatible'.")
        if self._pipeline_route != norm:
            self._pipeline_route = norm
            self.pipeline_route_changed.emit(self._pipeline_route)

    def set_enable_temporal_stabilization(self, enabled: bool) -> None:
        """Update temporal depth stabilization setting."""
        self._enable_temporal_stabilization = bool(enabled)

    def record_license_acknowledgment(self, model_id: str) -> None:
        """Record explicit user acknowledgment for license terms."""
        self._store.record_license_acknowledgment(model_id)
        self._license_acknowledged = True
        info = self.get_model_info(model_id)
        self.model_changed.emit(model_id, info)

    def set_input_path(self, path: Union[str, Path]) -> Tuple[bool, str]:
        """Set and probe input video file. Returns (success, error_or_warning_message)."""
        resolved = Path(path).resolve()
        if not resolved.is_file():
            self._input_path = None
            self._input_probe = None
            self.input_probed.emit(None)
            return False, f"File does not exist: {resolved}"

        self._input_path = resolved

        # Suggest default output path if not already customized
        if self._output_path is None or self._output_path.parent == resolved.parent:
            out_name = f"{resolved.stem}_FullSBS_LR.mp4"
            self._output_path = resolved.with_name(out_name)

        # Probe file properties using ffprobe
        try:
            ffprobe_bin = self._resolve_ffprobe()
            probe = probe_video(resolved, ffprobe_path=ffprobe_bin, strict_sdr_cfr=False)
            self._input_probe = probe
            self.input_probed.emit(probe)

            # Check unsupported limitations
            warnings = []
            if probe.is_hdr:
                warnings.append("HDR color detected: HDR is not supported in the current vertical slice.")
            if probe.is_vfr:
                warnings.append("Variable Frame Rate (VFR) detected: CFR required for reliable synchronization.")
            if probe.rotation != 0:
                warnings.append(f"Non-zero rotation metadata ({probe.rotation}°): unsupported geometry.")
            if probe.width % 2 != 0 or probe.height % 2 != 0:
                warnings.append("Odd dimensions detected: even dimensions required for 4:2:0 Full-SBS encoding.")

            msg = "\n".join(warnings) if warnings else "Input video probed successfully."
            return True, msg
        except Exception as err:
            self._input_probe = None
            self.input_probed.emit(None)
            return False, f"Failed to probe video: {err}"

    def set_output_path(self, path: Union[str, Path]) -> Tuple[bool, str]:
        """Set destination stereoscopic video path."""
        resolved = Path(path).resolve()
        if self._input_path and resolved == self._input_path:
            return False, "Destination output path cannot be identical to the source video."

        self._output_path = resolved
        return True, ""

    def validate_for_conversion(self) -> Tuple[bool, str]:
        """Check if all conditions are met to launch conversion. Returns (ready, error_message)."""
        if self._worker_process is not None and self._worker_process.state() != QProcess.ProcessState.NotRunning:
            return False, "A conversion job is already running."

        if not self._input_path or not self._input_path.is_file():
            return False, "Please select an existing input video file."

        if not self._output_path:
            return False, "Please choose an output file location."

        if self._input_path == self._output_path:
            return False, "Input and output files cannot be identical."

        if self._input_probe is None:
            return False, "Input video has not been probed or probe failed."

        # Model support check
        if self._selected_model_id not in SUPPORTED_MODEL_IDS:
            entry = get_model_entry(self._selected_model_id, self._catalog)
            return (
                False,
                f"Model '{entry.ui_name}' ({entry.id}) is not yet integrated for inference in this version. "
                f"Supported models: {', '.join(SUPPORTED_MODEL_IDS)}.",
            )

        # License acknowledgment check
        entry = get_model_entry(self._selected_model_id, self._catalog)
        if (entry.license_info.noncommercial_ack_required or entry.license_info.license_conflict) and not self._store.is_license_acknowledged(entry):
            return (
                False,
                f"Model '{entry.ui_name}' requires explicit license terms acknowledgment before conversion.",
            )

        # Media limitations check
        if self._input_probe.is_hdr:
            return False, "HDR input is not supported in the current vertical slice."
        if self._input_probe.is_vfr:
            return False, "Variable Frame Rate (VFR) is not supported; CFR required."
        if self._input_probe.rotation != 0:
            return False, f"Rotated video ({self._input_probe.rotation}°) is not supported."
        if self._input_probe.width % 2 != 0 or self._input_probe.height % 2 != 0:
            return False, "Odd frame dimensions are not supported for Full-SBS 4:2:0 export."

        return True, ""

    def start_conversion(self, overwrite_confirmed: bool = False) -> Tuple[bool, str]:
        """Launch worker child process to perform model download/conversion."""
        valid, msg = self.validate_for_conversion()
        if not valid:
            return False, msg

        assert self._input_path is not None
        assert self._output_path is not None

        if self._output_path.exists() and not overwrite_confirmed:
            return False, f"Destination file already exists: {self._output_path}. Overwrite confirmation required."

        self._overwrite = overwrite_confirmed
        self._active_job_id = uuid.uuid4().hex[:12]
        self._last_result = None
        self._last_error = None
        self._stdout_buffer = ""

        # Cancel any stale cancellation timer from a previous run
        if self._cancel_timer and self._cancel_timer.isActive():
            self._cancel_timer.stop()
            self._cancel_timer = None

        # Setup cancellation file and command file in system temp
        temp_dir = Path(tempfile.gettempdir())
        self._cancel_file = temp_dir / f"puregpu3d_cancel_{self._active_job_id}.tmp"
        if self._cancel_file.exists():
            try:
                self._cancel_file.unlink()
            except OSError:
                pass

        self._command_file = temp_dir / f"puregpu3d_cmd_{self._active_job_id}.json"

        ffmpeg_bin = self._resolve_ffmpeg()
        ffprobe_bin = self._resolve_ffprobe()

        command = WorkerCommand(
            job_id=self._active_job_id,
            input_path=str(self._input_path),
            output_path=str(self._output_path),
            model_id=self._selected_model_id,
            disparity_strength=self._disparity_strength,
            q_screen=self._q_screen,
            depth_scale=self._depth_scale,
            pipeline_route=self._pipeline_route,
            enable_temporal_stabilization=self._enable_temporal_stabilization,
            overwrite=self._overwrite,
            acknowledge_license=self._store.is_license_acknowledged(
                get_model_entry(self._selected_model_id, self._catalog)
            ),
            cancel_file=str(self._cancel_file),
            ffmpeg_path=str(ffmpeg_bin),
            ffprobe_path=str(ffprobe_bin),
        )

        with open(self._command_file, "w", encoding="utf-8") as f:
            f.write(command.to_json())

        # Initialize QProcess
        self._worker_process = QProcess(self)
        self._worker_process.readyReadStandardOutput.connect(self._on_worker_stdout)
        self._worker_process.readyReadStandardError.connect(self._on_worker_stderr)
        self._worker_process.finished.connect(self._on_worker_finished)

        # Launch program & arguments
        program, arguments = self._get_worker_launch_spec()

        self._set_state(DesktopState.VALIDATING)
        self.status_updated.emit(Stage.STARTING, "Launching conversion worker process...")

        self._worker_process.start(program, arguments)
        if not self._worker_process.waitForStarted(3000):
            err = self._worker_process.errorString()
            self._last_error = f"Failed to start worker process: {err}"
            self._set_state(DesktopState.FAILED)
            self.conversion_failed.emit(self._last_error, Stage.STARTING)
            self._cleanup_temp_files()
            try:
                self._worker_process.deleteLater()
            except Exception:
                pass
            self._worker_process = None
            return False, self._last_error

        return True, ""

    def _get_worker_launch_spec(self) -> Tuple[str, List[str]]:
        """Resolve worker executable/script and arguments."""
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).resolve().parent
            worker_name = "PureGPU3D-worker.exe" if sys.platform == "win32" else "PureGPU3D-worker"
            worker_bin = app_dir / worker_name
            if worker_bin.is_file():
                return str(worker_bin), ["--command-file", str(self._command_file)]
            return sys.executable, ["--worker", "--command-file", str(self._command_file)]
        worker_script = Path(__file__).resolve().parent.parent / "runtime" / "worker.py"
        return sys.executable, ["-B", str(worker_script), "--command-file", str(self._command_file)]

    def cancel_conversion(self) -> None:
        """Request cooperative cancellation of the running conversion worker."""
        target_proc = self._worker_process
        target_job_id = self._active_job_id
        if target_proc is None or target_proc.state() == QProcess.ProcessState.NotRunning:
            return

        self._set_state(DesktopState.CANCELLING)
        self.status_updated.emit(Stage.CANCELLED, "Cancelling conversion...")

        # Write cooperative cancel file
        if self._cancel_file:
            try:
                self._cancel_file.write_text("cancel", encoding="utf-8")
            except OSError:
                pass

        # Schedule bounded escalation safely bound to this specific process and job id
        if self._cancel_timer and self._cancel_timer.isActive():
            self._cancel_timer.stop()

        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.timeout.connect(lambda: self._escalate_cancellation(target_proc, target_job_id))
        self._cancel_timer.start(4000)

    def _escalate_cancellation(
        self,
        target_proc: Optional[QProcess] = None,
        target_job_id: Optional[str] = None,
    ) -> None:
        """Force terminate worker process if cooperative cancellation timed out."""
        proc = target_proc or self._worker_process
        if target_job_id is not None and self._active_job_id != target_job_id:
            # Active job changed; do not touch new job
            return
        if proc and proc.state() != QProcess.ProcessState.NotRunning:
            logger.warning("Worker did not exit cooperatively within 4s; terminating process...")
            proc.terminate()
            # If still alive after 1 second, kill
            QTimer.singleShot(
                1000,
                lambda: proc.kill() if proc and proc.state() != QProcess.ProcessState.NotRunning else None,
            )

    def _on_worker_stdout(self) -> None:
        """Read and parse strictly formatted JSON-lines protocol messages from worker stdout."""
        if not self._worker_process:
            return

        raw_bytes = bytes(self._worker_process.readAllStandardOutput().data())
        text = raw_bytes.decode("utf-8", errors="replace")
        self._stdout_buffer += text

        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            msg = parse_protocol_line(line)
            if msg:
                self._dispatch_protocol_message(msg)

    def _on_worker_stderr(self) -> None:
        """Read diagnostic log lines from worker stderr."""
        if not self._worker_process:
            return

        raw_bytes = bytes(self._worker_process.readAllStandardError().data())
        text = raw_bytes.decode("utf-8", errors="replace")
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                self.log_received.emit(cleaned)

    def _dispatch_protocol_message(self, msg: Dict[str, Any]) -> None:
        """Route parsed protocol event to controller state and signals."""
        mtype = msg.get("type")

        if mtype == MessageType.STATUS:
            stage = msg.get("stage", "")
            message = msg.get("message", "")
            if stage in (Stage.DOWNLOADING, Stage.VERIFYING):
                self._set_state(DesktopState.PREPARING_MODEL)
            elif stage in (Stage.LOADING, Stage.CONVERTING, Stage.FINALIZING):
                self._set_state(DesktopState.CONVERTING)
            self.status_updated.emit(stage, message)

        elif mtype == MessageType.DOWNLOAD_PROGRESS:
            self._set_state(DesktopState.PREPARING_MODEL)
            self.download_progress.emit(
                msg.get("downloaded_bytes", 0),
                msg.get("total_bytes", 0),
                msg.get("percent", 0.0),
                msg.get("filename", ""),
            )

        elif mtype == MessageType.CONVERSION_PROGRESS:
            self._set_state(DesktopState.CONVERTING)
            self.conversion_progress.emit(
                msg.get("current_frame", 0),
                msg.get("total_frames", 0),
                msg.get("percent", 0.0),
                msg.get("fps", 0.0),
                msg.get("eta_seconds"),
            )

        elif mtype == MessageType.COMPLETED:
            self._set_state(DesktopState.COMPLETED)
            self._last_result = msg.get("result", {})
            self.conversion_completed.emit(self._last_result)

        elif mtype == MessageType.ERROR:
            self._set_state(DesktopState.FAILED)
            err = msg.get("error", "Unknown error")
            stage = msg.get("stage", Stage.FAILED)
            self._last_error = err
            self.conversion_failed.emit(err, stage)

        elif mtype == MessageType.CANCELLED:
            self._set_state(DesktopState.CANCELLED)
            self.conversion_cancelled.emit()

    def _on_worker_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Handle worker process termination."""
        if self._cancel_timer and self._cancel_timer.isActive():
            self._cancel_timer.stop()
            self._cancel_timer = None

        # Flush any remaining buffer
        if self._stdout_buffer.strip():
            msg = parse_protocol_line(self._stdout_buffer)
            if msg:
                self._dispatch_protocol_message(msg)
            self._stdout_buffer = ""

        if self._state == DesktopState.CANCELLING:
            self._set_state(DesktopState.CANCELLED)
            self.conversion_cancelled.emit()
        elif exit_code != 0:
            if self._state not in (DesktopState.FAILED, DesktopState.CANCELLED):
                self._set_state(DesktopState.FAILED)
                err = self._last_error or f"Worker process exited with code {exit_code}"
                self._last_error = err
                self.conversion_failed.emit(err, Stage.FAILED)
        else:
            # Clean exit code 0: check if it exited without terminal completion/error message
            if self._state in (DesktopState.VALIDATING, DesktopState.PREPARING_MODEL, DesktopState.CONVERTING):
                self._set_state(DesktopState.FAILED)
                err = "Worker process exited prematurely without completion."
                self._last_error = err
                self.conversion_failed.emit(err, Stage.FAILED)

        self._cleanup_temp_files()
        self._worker_process = None
        # Notify state refresh so UI re-evaluates button readiness once process is fully dead
        self.state_changed.emit(self._state.value)

    def _cleanup_temp_files(self) -> None:
        if self._cancel_file and self._cancel_file.is_file():
            try:
                self._cancel_file.unlink()
            except OSError:
                pass
            self._cancel_file = None

        if self._command_file and self._command_file.is_file():
            try:
                self._command_file.unlink()
            except OSError:
                pass
            self._command_file = None

    def _resolve_ffmpeg(self) -> Path:
        if getattr(sys, "frozen", False):
            root = resolve_app_root()
            bin_path = root / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
            if bin_path.is_file():
                return bin_path
            raise FileNotFoundError(f"Frozen app requires bundled ffmpeg at {bin_path}, host fallback forbidden.")
        return find_ffmpeg()

    def _resolve_ffprobe(self) -> Path:
        if getattr(sys, "frozen", False):
            root = resolve_app_root()
            bin_path = root / "bin" / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
            if bin_path.is_file():
                return bin_path
            raise FileNotFoundError(f"Frozen app requires bundled ffprobe at {bin_path}, host fallback forbidden.")
        return find_ffprobe()
