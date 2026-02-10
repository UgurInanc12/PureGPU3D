from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Codec(str, Enum):
    H264 = "h264"
    H265 = "h265"


class FpsMode(str, Enum):
    REALTIME = "realtime"
    OFFLINE = "offline"


class SbsMode(str, Enum):
    FULL = "sbs_full"


class OverflowPolicy(str, Enum):
    BLOCK = "block"


class CorruptFrameFallback(str, Enum):
    LAST_GOOD = "last_good"
    BLACK = "black"


@dataclass(slots=True)
class DepthConfig:
    max_disparity_px: int = 6
    depth_strength: float = 0.55
    edge_weight: float = 0.25
    luma_weight: float = 0.45
    vertical_weight: float = 0.30

    def validate(self) -> None:
        if not 0 <= self.max_disparity_px <= 16:
            raise ValueError("max_disparity_px must be in [0, 16]")
        if not 0.0 <= self.depth_strength <= 1.0:
            raise ValueError("depth_strength must be in [0.0, 1.0]")
        for name, value in (
            ("edge_weight", self.edge_weight),
            ("luma_weight", self.luma_weight),
            ("vertical_weight", self.vertical_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")

    def normalized_weights(self) -> tuple[float, float, float]:
        total = self.edge_weight + self.luma_weight + self.vertical_weight
        if total <= 0:
            return (0.25, 0.45, 0.30)
        return (
            self.edge_weight / total,
            self.luma_weight / total,
            self.vertical_weight / total,
        )


@dataclass(slots=True)
class EngineConfig:
    input_path: Path
    output_path: Path
    codec: Codec = Codec.H265
    video_encoder: str = "auto"
    video_bitrate_mbps: float | None = None
    fps_mode: FpsMode = FpsMode.OFFLINE
    sbs_mode: SbsMode = SbsMode.FULL
    depth_profile: str = "safe_low"
    depth: DepthConfig = field(default_factory=DepthConfig)
    buffer_slots: int = 6
    overflow_policy: OverflowPolicy = OverflowPolicy.BLOCK
    audio_passthrough: bool = True
    audio_codec: str = "copy"
    preserve_color_metadata: bool = True
    backend_preference: str = "ffmpeg"
    corrupt_frame_fallback: CorruptFrameFallback = CorruptFrameFallback.LAST_GOOD
    allow_missing_input: bool = False
    watchdog_timeout_s: float = 10.0
    log_level: str = "info"

    def validate(self) -> None:
        if not self.allow_missing_input and not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if self.buffer_slots < 2:
            raise ValueError("buffer_slots must be >= 2")
        if self.video_bitrate_mbps is not None and self.video_bitrate_mbps <= 0:
            raise ValueError("video_bitrate_mbps must be > 0 when provided")
        encoder = self.video_encoder.strip().lower()
        if not encoder:
            raise ValueError("video_encoder must not be empty")
        audio_codec = self.audio_codec.strip().lower()
        if audio_codec not in {"copy", "aac"}:
            raise ValueError("audio_codec must be one of: copy, aac")
        if self.watchdog_timeout_s <= 0:
            raise ValueError("watchdog_timeout_s must be > 0")
        self.depth.validate()

    def ensure_output_parent(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["codec"] = self.codec.value
        payload["fps_mode"] = self.fps_mode.value
        payload["sbs_mode"] = self.sbs_mode.value
        payload["overflow_policy"] = self.overflow_policy.value
        payload["corrupt_frame_fallback"] = self.corrupt_frame_fallback.value
        payload["input_path"] = str(self.input_path)
        payload["output_path"] = str(self.output_path)
        return payload

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "EngineConfig":
        depth_raw = mapping.get("depth", {}) or {}
        bitrate_raw = mapping.get("video_bitrate_mbps")
        bitrate_mbps: float | None
        if bitrate_raw in (None, "", 0, 0.0, "0", "0.0"):
            bitrate_mbps = None
        else:
            bitrate_mbps = float(bitrate_raw)
        depth = DepthConfig(
            max_disparity_px=int(depth_raw.get("max_disparity_px", 6)),
            depth_strength=float(depth_raw.get("depth_strength", 0.55)),
            edge_weight=float(depth_raw.get("edge_weight", 0.25)),
            luma_weight=float(depth_raw.get("luma_weight", 0.45)),
            vertical_weight=float(depth_raw.get("vertical_weight", 0.30)),
        )
        return cls(
            input_path=Path(mapping["input_path"]),
            output_path=Path(mapping["output_path"]),
            codec=Codec(mapping.get("codec", Codec.H265.value)),
            video_encoder=str(mapping.get("video_encoder", "auto")),
            video_bitrate_mbps=bitrate_mbps,
            fps_mode=FpsMode(mapping.get("fps_mode", FpsMode.OFFLINE.value)),
            sbs_mode=SbsMode(mapping.get("sbs_mode", SbsMode.FULL.value)),
            depth_profile=str(mapping.get("depth_profile", "safe_low")),
            depth=depth,
            buffer_slots=int(mapping.get("buffer_slots", 6)),
            overflow_policy=OverflowPolicy(
                mapping.get("overflow_policy", OverflowPolicy.BLOCK.value)
            ),
            audio_passthrough=bool(mapping.get("audio_passthrough", True)),
            audio_codec=str(mapping.get("audio_codec", "copy")),
            preserve_color_metadata=bool(mapping.get("preserve_color_metadata", True)),
            backend_preference=str(mapping.get("backend_preference", "ffmpeg")),
            corrupt_frame_fallback=CorruptFrameFallback(
                mapping.get(
                    "corrupt_frame_fallback",
                    CorruptFrameFallback.LAST_GOOD.value,
                )
            ),
            allow_missing_input=bool(mapping.get("allow_missing_input", False)),
            watchdog_timeout_s=float(mapping.get("watchdog_timeout_s", 10.0)),
            log_level=str(mapping.get("log_level", "info")),
        )
