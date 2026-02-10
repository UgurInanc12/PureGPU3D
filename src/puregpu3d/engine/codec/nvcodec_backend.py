from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ...config.types import Codec
from ..types import VideoProbe
from .base import BackendNotImplementedError, BackendUnavailableError, CodecBackend, Nv12Writer


def _import_codec_module() -> object:
    for module_name in ("PyNvVideoCodec", "PyNvCodec"):
        try:
            return __import__(module_name)
        except Exception:
            continue
    raise BackendUnavailableError(
        "Neither PyNvVideoCodec nor PyNvCodec is importable in current environment"
    )


class NvCodecBackend(CodecBackend):
    name = "nvcodec"

    def __init__(self) -> None:
        self._module = _import_codec_module()

    def is_available(self) -> bool:
        return self._module is not None

    def probe(self, input_path: Path) -> VideoProbe:
        raise BackendNotImplementedError(
            "NvCodec backend scaffold is present but full V1 implementation is pending. "
            "Use backend_preference='ffmpeg' for now."
        )

    def decode_iter(self, input_path: Path, probe: VideoProbe) -> Iterator[np.ndarray]:
        raise BackendNotImplementedError("NvCodec decode path not implemented in this revision")

    def open_writer(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: Codec,
        video_encoder: str = "auto",
        video_bitrate_mbps: float | None = None,
        source_probe: VideoProbe | None = None,
    ) -> Nv12Writer:
        raise BackendNotImplementedError("NvCodec encode path not implemented in this revision")

    def remux_with_audio(
        self,
        *,
        encoded_video_path: Path,
        source_input_path: Path,
        output_path: Path,
        codec: Codec,
        audio_passthrough: bool,
        audio_codec: str = "copy",
    ) -> list[str]:
        raise BackendNotImplementedError("NvCodec remux path not implemented in this revision")

    def request_abort(self) -> None:
        return None
