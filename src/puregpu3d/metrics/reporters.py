from __future__ import annotations

import csv
import json
from pathlib import Path

from ..engine.types import TranscodeResult
from .collector import MetricsCollector


def write_metrics_json(path: Path, collector: MetricsCollector, result: TranscodeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collector.to_json(result), indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, collector: MetricsCollector) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame_index",
                "frame_ms",
                "kernel_ms",
                "decode_wait_ms",
                "encode_wait_ms",
                "h2d_ms",
                "kernel_compute_ms",
                "d2h_ms",
                "queue_depth",
                "block_wait_ms",
            ]
        )
        for row in collector.frame_metrics:
            writer.writerow(
                [
                    row.frame_index,
                    f"{row.frame_ms:.4f}",
                    f"{row.kernel_ms:.4f}",
                    f"{row.decode_wait_ms:.4f}",
                    f"{row.encode_wait_ms:.4f}",
                    f"{row.h2d_ms:.4f}",
                    f"{row.kernel_compute_ms:.4f}",
                    f"{row.d2h_ms:.4f}",
                    row.queue_depth,
                    f"{row.block_wait_ms:.4f}",
                ]
            )


def write_benchmark_markdown(path: Path, result: TranscodeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    warning_lines = ["- none"] if not result.warnings else [f"- {w}" for w in result.warnings]
    content = "\n".join(
        [
            "# Benchmark Report",
            "",
            f"- Backend: `{result.backend_name}`",
            f"- Output: `{result.output_path}`",
            f"- Frames in: `{result.frames_in}`",
            f"- Frames out: `{result.frames_out}`",
            f"- Elapsed s: `{result.elapsed_s:.4f}`",
            f"- Avg FPS: `{result.avg_fps:.4f}`",
            f"- p95 kernel ms: `{result.p95_kernel_ms:.4f}`",
            "",
            "## Warnings",
            *warning_lines,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
