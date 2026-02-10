from __future__ import annotations

import shutil
from pathlib import Path
from threading import Event
from typing import Iterator

import numpy as np

from ...config.types import Codec
from ..types import VideoProbe
from .base import CodecBackend, Nv12Writer


class _MockWriter(Nv12Writer):
    def __init__(self, output_path: Path, expected_shape: tuple[int, int]) -> None:
        self._output_path = output_path
        self._expected_shape = expected_shape
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._output_path.open("wb")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self._expected_shape:
            raise ValueError(f"Expected frame shape {self._expected_shape}, got {frame.shape}")
        self._fh.write(frame.tobytes(order="C"))

    def close(self) -> None:
        self._fh.close()


class MockBackend(CodecBackend):
    name = "mock"

    def __init__(self, width: int = 320, height: int = 180, fps: float = 30.0, frame_count: int = 24) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_count = frame_count
        self._abort_event = Event()

    def is_available(self) -> bool:
        return True

    def probe(self, input_path: Path) -> VideoProbe:
        return VideoProbe(
            width=self._width,
            height=self._height,
            fps=self._fps,
            frame_count=self._frame_count,
            has_audio=False,
            video_codec="mock",
            pix_fmt="nv12",
            bit_depth=8,
            duration_s=self._frame_count / self._fps,
        )

    def decode_iter(self, input_path: Path, probe: VideoProbe) -> Iterator[np.ndarray]:
        self._abort_event.clear()
        h = probe.height
        w = probe.width
        for frame_idx in range(self._frame_count):
            if self._abort_event.is_set():
                break
            y = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
            y = (y + frame_idx * 3).astype(np.uint8)
            uv = np.full((h // 2, w), 128, dtype=np.uint8)
            frame = np.empty((h * 3 // 2, w), dtype=np.uint8)
            frame[:h, :] = y
            frame[h:, :] = uv
            yield frame

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
        return _MockWriter(output_path=output_path, expected_shape=(height * 3 // 2, width))

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
        shutil.copyfile(encoded_video_path, output_path)
        return []

    def request_abort(self) -> None:
        self._abort_event.set()
