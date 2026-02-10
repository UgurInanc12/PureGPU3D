from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np

from ...config.types import Codec
from ..types import VideoProbe


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be used in current environment."""


class BackendNotImplementedError(RuntimeError):
    """Raised when backend is scaffolded but not completed yet."""


class Nv12Writer(ABC):
    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class CodecBackend(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def probe(self, input_path: Path) -> VideoProbe:
        raise NotImplementedError

    @abstractmethod
    def decode_iter(self, input_path: Path, probe: VideoProbe) -> Iterator[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def request_abort(self) -> None:
        """Best-effort immediate abort for active decode/encode subprocesses."""
        raise NotImplementedError
