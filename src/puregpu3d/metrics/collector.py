from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from ..engine.types import FrameMetric, TranscodeResult


class MetricsCollector:
    def __init__(self) -> None:
        self._started = perf_counter()
        self.frame_metrics: list[FrameMetric] = []
        self.warnings: list[str] = []

    def record_frame(
        self,
        *,
        frame_index: int,
        frame_ms: float,
        kernel_ms: float,
        queue_depth: int,
        block_wait_ms: float,
        decode_wait_ms: float = 0.0,
        encode_wait_ms: float = 0.0,
        h2d_ms: float = 0.0,
        kernel_compute_ms: float = 0.0,
        d2h_ms: float = 0.0,
    ) -> None:
        self.frame_metrics.append(
            FrameMetric(
                frame_index=frame_index,
                frame_ms=frame_ms,
                kernel_ms=kernel_ms,
                queue_depth=queue_depth,
                block_wait_ms=block_wait_ms,
                decode_wait_ms=decode_wait_ms,
                encode_wait_ms=encode_wait_ms,
                h2d_ms=h2d_ms,
                kernel_compute_ms=kernel_compute_ms,
                d2h_ms=d2h_ms,
            )
        )

    def add_warning(self, text: str) -> None:
        self.warnings.append(text)

    def build_result(
        self,
        *,
        ok: bool,
        output_path,
        backend_name: str,
        frames_in: int,
        frames_out: int,
        extra_warnings: list[str] | None = None,
    ) -> TranscodeResult:
        elapsed_s = perf_counter() - self._started
        kernel = np.array([m.kernel_ms for m in self.frame_metrics], dtype=np.float64)
        p95_kernel_ms = float(np.percentile(kernel, 95)) if kernel.size else 0.0
        avg_fps = float(frames_out / elapsed_s) if elapsed_s > 0 else 0.0
        warnings = [*self.warnings]
        if extra_warnings:
            warnings.extend(extra_warnings)
        return TranscodeResult(
            ok=ok,
            elapsed_s=elapsed_s,
            frames_in=frames_in,
            frames_out=frames_out,
            avg_fps=avg_fps,
            p95_kernel_ms=p95_kernel_ms,
            output_path=output_path,
            backend_name=backend_name,
            warnings=warnings,
        )

    def to_json(self, result: TranscodeResult) -> dict:
        payload = asdict(result)
        payload["output_path"] = str(result.output_path)
        payload["frames"] = [asdict(m) for m in self.frame_metrics]
        return payload
