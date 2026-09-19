"""Inter-process communication protocol between PureGPU3D Desktop UI and Conversion Worker.

Defines versioned JSON-lines messages exchanged over child process standard I/O:
  - UI Controller -> Worker: WorkerCommand (via command-file or stdin)
  - Worker -> UI Controller: JSON-lines events on stdout (status, progress, completed, error, cancelled)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

PROTOCOL_VERSION = "1.0"

SUPPORTED_MODEL_IDS: tuple[str, ...] = (
    "DA3-SMALL",
    "DA3-BASE",
    "DA3MONO-LARGE",
    "DA3METRIC-LARGE",
)

UNSUPPORTED_MODEL_IDS: tuple[str, ...] = (
    "DA3-LARGE-1.1",
    "DA3-GIANT-1.1",
    "DA3NESTED-GIANT-LARGE-1.1",
)


class PipelineRoute:
    AUTO = "auto"
    GPU = "gpu"
    COMPATIBLE = "compatible"


class MessageType:
    STATUS = "status"
    DOWNLOAD_PROGRESS = "download_progress"
    CONVERSION_PROGRESS = "conversion_progress"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class Stage:
    IDLE = "idle"
    STARTING = "starting"
    VALIDATING = "validating"
    AWAITING_ACKNOWLEDGMENT = "awaiting_acknowledgment"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    LOADING = "loading"
    CONVERTING = "converting"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkerCommand:
    """Command payload sent to the conversion worker."""

    job_id: str
    input_path: str
    output_path: str
    model_id: str = "DA3-SMALL"
    device: Optional[str] = None
    disparity_strength: float = 0.03
    q_screen: float = 0.6
    depth_scale: str = "1/2"
    pipeline_route: str = PipelineRoute.AUTO
    enable_temporal_stabilization: bool = True
    overwrite: bool = False
    acknowledge_license: bool = False
    cancel_file: Optional[str] = None
    ffmpeg_path: Optional[str] = None
    ffprobe_path: Optional[str] = None
    protocol_version: str = PROTOCOL_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerCommand:
        return cls(
            job_id=str(data["job_id"]),
            input_path=str(data["input_path"]),
            output_path=str(data["output_path"]),
            model_id=str(data.get("model_id", "DA3-SMALL")),
            device=data.get("device"),
            disparity_strength=float(data.get("disparity_strength", 0.03)),
            q_screen=float(data.get("q_screen", 0.6)),
            depth_scale=str(data.get("depth_scale", "1/2")),
            pipeline_route=str(data.get("pipeline_route", PipelineRoute.AUTO)),
            enable_temporal_stabilization=bool(data.get("enable_temporal_stabilization", True)),
            overwrite=bool(data.get("overwrite", False)),
            acknowledge_license=bool(data.get("acknowledge_license", False)),
            cancel_file=data.get("cancel_file"),
            ffmpeg_path=data.get("ffmpeg_path"),
            ffprobe_path=data.get("ffprobe_path"),
            protocol_version=str(data.get("protocol_version", PROTOCOL_VERSION)),
        )

    @classmethod
    def from_json(cls, text: str) -> WorkerCommand:
        return cls.from_dict(json.loads(text))


def make_status_msg(job_id: str, stage: str, message: str) -> Dict[str, Any]:
    return {
        "type": MessageType.STATUS,
        "job_id": job_id,
        "stage": stage,
        "message": message,
        "timestamp": time.time(),
    }


def make_download_progress_msg(
    job_id: str,
    downloaded_bytes: int,
    total_bytes: int,
    percent: float,
    filename: str = "",
) -> Dict[str, Any]:
    return {
        "type": MessageType.DOWNLOAD_PROGRESS,
        "job_id": job_id,
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": total_bytes,
        "percent": round(percent, 2),
        "filename": filename,
        "timestamp": time.time(),
    }


def make_conversion_progress_msg(
    job_id: str,
    current_frame: int,
    total_frames: int,
    percent: float,
    fps: float = 0.0,
    eta_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "type": MessageType.CONVERSION_PROGRESS,
        "job_id": job_id,
        "current_frame": current_frame,
        "total_frames": total_frames,
        "percent": round(percent, 2),
        "fps": round(fps, 2),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "timestamp": time.time(),
    }


def make_completed_msg(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": MessageType.COMPLETED,
        "job_id": job_id,
        "result": result,
        "timestamp": time.time(),
    }


def make_error_msg(job_id: str, error: str, stage: str = Stage.FAILED, detail: str = "") -> Dict[str, Any]:
    return {
        "type": MessageType.ERROR,
        "job_id": job_id,
        "error": error,
        "stage": stage,
        "detail": detail,
        "timestamp": time.time(),
    }


def make_cancelled_msg(job_id: str, stage: str = Stage.CANCELLED, message: str = "Conversion cancelled") -> Dict[str, Any]:
    return {
        "type": MessageType.CANCELLED,
        "job_id": job_id,
        "stage": stage,
        "message": message,
        "timestamp": time.time(),
    }


def serialize_protocol_line(msg: Dict[str, Any]) -> str:
    """Serialize message dict to a single JSON line without internal newlines."""
    return json.dumps(msg, ensure_ascii=False)


def parse_protocol_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSON-lines protocol message, returning None if line is empty or invalid JSON."""
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
        if isinstance(data, dict) and "type" in data:
            return data
    except Exception:
        pass
    return None
