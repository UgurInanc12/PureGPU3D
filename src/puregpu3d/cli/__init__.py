from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from ..config.loader import load_profile
from ..config.types import EngineConfig
from ..engine.core import StereoEngine
from ..metrics.reporters import (
    write_benchmark_markdown,
    write_metrics_csv,
    write_metrics_json,
)
from ..ui.gradio_app import launch_gradio

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_profile() -> Path:
    return _repo_root() / "configs" / "profiles" / "safe_low.yaml"


def _build_config(
    *,
    input_path: Path,
    output_path: Path,
    profile_path: Path,
    codec: str | None,
    video_encoder: str | None,
    video_bitrate_mbps: float | None,
    fps_mode: str | None,
    backend: str | None,
    depth_strength: float | None,
    max_disparity: int | None,
    buffer_slots: int | None,
    audio_passthrough: bool | None,
    audio_codec: str | None,
    preserve_color_metadata: bool | None,
) -> EngineConfig:
    payload = load_profile(profile_path)

    depth_payload: dict[str, Any] = dict(payload.get("depth", {}) or {})
    if depth_strength is not None:
        depth_payload["depth_strength"] = depth_strength
    if max_disparity is not None:
        depth_payload["max_disparity_px"] = max_disparity
    if depth_payload:
        payload["depth"] = depth_payload

    if codec is not None:
        payload["codec"] = codec
    if video_encoder is not None:
        payload["video_encoder"] = video_encoder
    if video_bitrate_mbps is not None:
        payload["video_bitrate_mbps"] = video_bitrate_mbps
    if fps_mode is not None:
        payload["fps_mode"] = fps_mode
    if backend is not None:
        payload["backend_preference"] = backend
    if buffer_slots is not None:
        payload["buffer_slots"] = buffer_slots
    if audio_passthrough is not None:
        payload["audio_passthrough"] = audio_passthrough
    if audio_codec is not None:
        payload["audio_codec"] = audio_codec
    if preserve_color_metadata is not None:
        payload["preserve_color_metadata"] = preserve_color_metadata

    payload["input_path"] = str(input_path)
    payload["output_path"] = str(output_path)

    config = EngineConfig.from_mapping(payload)
    config.validate()
    return config


def _save_metrics(engine: StereoEngine, result, metrics_dir: Path) -> None:
    if engine.last_metrics is None:
        return
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_metrics_json(metrics_dir / f"metrics_{stamp}.json", engine.last_metrics, result)
    write_metrics_csv(metrics_dir / f"frames_{stamp}.csv", engine.last_metrics)


@app.command()
def transcode(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output"),
    profile: Path = typer.Option(_default_profile(), "--profile", exists=True, dir_okay=False),
    codec: str | None = typer.Option(None, "--codec", help="h264 or h265"),
    video_encoder: str | None = typer.Option(
        None,
        "--video-encoder",
        help="auto, h264_nvenc, hevc_nvenc, libx264, libx265",
    ),
    video_bitrate_mbps: float | None = typer.Option(None, "--video-bitrate-mbps"),
    fps_mode: str | None = typer.Option(None, "--fps-mode", help="realtime or offline"),
    backend: str | None = typer.Option(None, "--backend", help="auto, ffmpeg, mock, nvcodec"),
    depth_strength: float | None = typer.Option(None, "--depth-strength"),
    max_disparity: int | None = typer.Option(None, "--max-disparity"),
    buffer_slots: int | None = typer.Option(None, "--buffer-slots"),
    audio_passthrough: bool | None = typer.Option(None, "--audio-passthrough/--no-audio-passthrough"),
    audio_codec: str | None = typer.Option(None, "--audio-codec", help="copy or aac"),
    preserve_color_metadata: bool | None = typer.Option(
        None,
        "--preserve-color-metadata/--no-preserve-color-metadata",
    ),
    metrics_dir: Path = typer.Option(Path("outputs") / "metrics", "--metrics-dir"),
) -> None:
    config = _build_config(
        input_path=input_path,
        output_path=output_path,
        profile_path=profile,
        codec=codec,
        video_encoder=video_encoder,
        video_bitrate_mbps=video_bitrate_mbps,
        fps_mode=fps_mode,
        backend=backend,
        depth_strength=depth_strength,
        max_disparity=max_disparity,
        buffer_slots=buffer_slots,
        audio_passthrough=audio_passthrough,
        audio_codec=audio_codec,
        preserve_color_metadata=preserve_color_metadata,
    )

    engine = StereoEngine()
    result = engine.run_file(config)
    _save_metrics(engine, result, metrics_dir)

    rprint(json.dumps({
        "ok": result.ok,
        "backend": result.backend_name,
        "output": str(result.output_path),
        "frames_in": result.frames_in,
        "frames_out": result.frames_out,
        "elapsed_s": result.elapsed_s,
        "avg_fps": result.avg_fps,
        "p95_kernel_ms": result.p95_kernel_ms,
        "warnings": result.warnings,
    }, indent=2))


@app.command()
def batch(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("outputs") / "batch", "--output-dir"),
    pattern: str = typer.Option("*.mp4", "--pattern"),
    profile: Path = typer.Option(_default_profile(), "--profile", exists=True, dir_okay=False),
    codec: str | None = typer.Option(None, "--codec"),
    fps_mode: str | None = typer.Option(None, "--fps-mode"),
    backend: str | None = typer.Option(None, "--backend"),
) -> None:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise typer.BadParameter(f"No files matching pattern '{pattern}' in {input_dir}")

    first_output = output_dir / f"{files[0].stem}_sbs_full.mp4"
    base_config = _build_config(
        input_path=files[0],
        output_path=first_output,
        profile_path=profile,
        codec=codec,
        video_encoder=None,
        video_bitrate_mbps=None,
        fps_mode=fps_mode,
        backend=backend,
        depth_strength=None,
        max_disparity=None,
        buffer_slots=None,
        audio_passthrough=None,
        audio_codec=None,
        preserve_color_metadata=None,
    )

    engine = StereoEngine()
    results = engine.run_batch(inputs=files, output_dir=output_dir, base_config=base_config)

    summary = {
        str(path): {
            "ok": res.ok,
            "output": str(res.output_path),
            "avg_fps": res.avg_fps,
            "warnings": res.warnings,
        }
        for path, res in results.items()
    }
    rprint(json.dumps(summary, indent=2))


@app.command()
def benchmark(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output_path: Path = typer.Option(Path("outputs") / "benchmark_sbs.mp4", "--output"),
    profile: Path = typer.Option(_default_profile(), "--profile", exists=True, dir_okay=False),
    backend: str | None = typer.Option(None, "--backend"),
    report_dir: Path = typer.Option(Path("docs") / "benchmarks", "--report-dir"),
) -> None:
    config = _build_config(
        input_path=input_path,
        output_path=output_path,
        profile_path=profile,
        codec=None,
        video_encoder=None,
        video_bitrate_mbps=None,
        fps_mode=None,
        backend=backend,
        depth_strength=None,
        max_disparity=None,
        buffer_slots=None,
        audio_passthrough=None,
        audio_codec=None,
        preserve_color_metadata=None,
    )

    engine = StereoEngine()
    result = engine.run_file(config)

    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if engine.last_metrics is not None:
        write_metrics_json(report_dir / f"benchmark_{stamp}.json", engine.last_metrics, result)
        write_metrics_csv(report_dir / f"benchmark_{stamp}.csv", engine.last_metrics)
    write_benchmark_markdown(report_dir / "benchmark_report.md", result)

    rprint(json.dumps({
        "ok": result.ok,
        "backend": result.backend_name,
        "output": str(result.output_path),
        "avg_fps": result.avg_fps,
        "p95_kernel_ms": result.p95_kernel_ms,
    }, indent=2))


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
) -> None:
    launch_gradio(host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
