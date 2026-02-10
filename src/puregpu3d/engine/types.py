from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool
    video_codec: str = "unknown"
    pix_fmt: str = "nv12"
    bit_depth: int = 8
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    duration_s: float | None = None


@dataclass(slots=True)
class FrameMetric:
    frame_index: int
    frame_ms: float
    kernel_ms: float
    queue_depth: int
    block_wait_ms: float
    decode_wait_ms: float = 0.0
    encode_wait_ms: float = 0.0
    h2d_ms: float = 0.0
    kernel_compute_ms: float = 0.0
    d2h_ms: float = 0.0


@dataclass(slots=True)
class TranscodeResult:
    ok: bool
    elapsed_s: float
    frames_in: int
    frames_out: int
    avg_fps: float
    p95_kernel_ms: float
    output_path: Path
    backend_name: str
    warnings: list[str] = field(default_factory=list)
