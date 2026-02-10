from __future__ import annotations

from .base import CodecBackend
from .ffmpeg_backend import FfmpegSoftwareBackend
from .mock_backend import MockBackend
from .nvcodec_backend import NvCodecBackend


def create_backend(preference: str = "auto") -> tuple[CodecBackend, list[str]]:
    warnings: list[str] = []
    pref = preference.lower().strip()

    if pref == "nvcodec":
        try:
            _ = NvCodecBackend()
            warnings.append(
                "NvCodec backend detected but still scaffold-only in this revision; falling back to ffmpeg."
            )
        except Exception as exc:
            warnings.append(f"NvCodec backend unavailable: {exc}")
    elif pref == "auto":
        warnings.append("Auto backend preference resolves to ffmpeg in this revision.")

    if pref in {"auto", "ffmpeg", "nvcodec"}:
        ffmpeg_backend = FfmpegSoftwareBackend()
        if ffmpeg_backend.is_available():
            return ffmpeg_backend, warnings
        warnings.append("FFmpeg backend unavailable.")

    if pref == "mock":
        return MockBackend(), warnings

    raise RuntimeError(
        "No usable codec backend found. Warnings: " + " | ".join(warnings)
    )
