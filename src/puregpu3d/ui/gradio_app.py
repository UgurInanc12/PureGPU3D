from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Lock

from ..config.types import Codec, DepthConfig, EngineConfig, FpsMode
from ..engine.codec.factory import create_backend
from ..engine.core import StereoEngine, TranscodeCancelledError
from ..engine.types import VideoProbe

SUPPORTED_CONTAINERS = ["mp4", "mkv", "mov"]


def _result_to_json(result) -> str:
    payload = {
        "ok": result.ok,
        "elapsed_s": result.elapsed_s,
        "frames_in": result.frames_in,
        "frames_out": result.frames_out,
        "avg_fps": result.avg_fps,
        "p95_kernel_ms": result.p95_kernel_ms,
        "output_path": str(result.output_path),
        "backend": result.backend_name,
        "warnings": result.warnings,
    }
    return json.dumps(payload, indent=2)


def _default_output_path(input_path: str, container: str) -> str:
    if not input_path:
        return ""
    in_path = Path(input_path)
    suffix = container.lower().strip()
    if suffix not in SUPPORTED_CONTAINERS:
        suffix = "mp4"
    return str(in_path.with_name(f"{in_path.stem}_out_SBS.{suffix}"))


def _coerce_output_extension(path_text: str, container: str) -> str:
    if not path_text:
        return path_text
    out_path = Path(path_text)
    suffix = container.lower().strip()
    if suffix not in SUPPORTED_CONTAINERS:
        suffix = "mp4"
    return str(out_path.with_suffix(f".{suffix}"))


def _pick_input_with_windows_dialog(current_path: str, container: str, backend: str) -> tuple[str, str, str, str]:
    selected_path = current_path
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        initial_dir = str(Path(current_path).parent) if current_path else str(Path.home())
        selected = filedialog.askopenfilename(
            title="Select input video",
            initialdir=initial_dir,
            filetypes=[
                ("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mts *.ts"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        if selected:
            selected_path = selected
    except Exception as exc:
        return current_path, _default_output_path(current_path, container), "", f"Input picker failed: {exc}"

    if not selected_path:
        return "", "", "No input selected.", "No input selected."

    auto_out = _default_output_path(selected_path, container)
    info = _build_input_info(selected_path, backend)
    return selected_path, auto_out, info, "Input selected."


def _pick_output_with_windows_dialog(input_path: str, container: str, current_output: str) -> str:
    suggested = current_output or _default_output_path(input_path, container)
    suggested = _coerce_output_extension(suggested, container)
    if not suggested:
        suggested = str((Path.cwd() / f"output_out_SBS.{container}").resolve())

    suggested_path = Path(suggested)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.asksaveasfilename(
            title="Select output video",
            initialdir=str(suggested_path.parent),
            initialfile=suggested_path.name,
            defaultextension=f".{container}",
            filetypes=[
                (f"{container.upper()} Files", f"*.{container}"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        if selected:
            return _coerce_output_extension(selected, container)
    except Exception:
        return suggested

    return suggested


def _probe_input(path_text: str, backend: str) -> tuple[VideoProbe | None, str]:
    input_path = Path(path_text)
    if not path_text:
        return None, "No input selected."
    if not input_path.exists():
        return None, f"Input not found: {input_path}"

    try:
        probe_backend, _ = create_backend("ffmpeg")
        probe = probe_backend.probe(input_path)
        return probe, ""
    except Exception as exc:
        return None, f"Probe failed: {exc}"


def _build_input_info(path_text: str, backend: str) -> str:
    probe, err = _probe_input(path_text, backend)
    if probe is None:
        return err

    frame_count = probe.frame_count
    if frame_count <= 0 and probe.duration_s and probe.fps > 0:
        frame_count = int(round(probe.duration_s * probe.fps))

    duration_text = f"{probe.duration_s:.3f} s" if probe.duration_s is not None else "unknown"
    frame_count_text = str(frame_count) if frame_count > 0 else "unknown"
    output_resolution = f"{probe.width * 2}x{probe.height}"

    return "\n".join(
        [
            "### Input Video Info",
            f"- Input resolution: `{probe.width}x{probe.height}`",
            f"- FPS: `{probe.fps:.3f}`",
            f"- Total frames: `{frame_count_text}`",
            f"- Duration: `{duration_text}`",
            f"- Video codec: `{probe.video_codec}`",
            f"- Pixel format: `{probe.pix_fmt}`",
            f"- Bit depth: `{probe.bit_depth}`",
            f"- Color range: `{probe.color_range or 'unknown'}`",
            f"- Color space: `{probe.color_space or 'unknown'}`",
            f"- Color transfer: `{probe.color_transfer or 'unknown'}`",
            f"- Color primaries: `{probe.color_primaries or 'unknown'}`",
            f"- Audio stream: `{'yes' if probe.has_audio else 'no'}`",
            f"- Output resolution (SBS Full): `{output_resolution}`",
        ]
    )


def _on_input_changed(input_path: str, container: str, backend: str) -> tuple[str, str]:
    if not input_path:
        return "", "No input selected."
    return _default_output_path(input_path, container), _build_input_info(input_path, backend)


def _on_container_changed(input_path: str, current_output: str, container: str) -> str:
    if input_path:
        return _default_output_path(input_path, container)
    return _coerce_output_extension(current_output, container)


def launch_gradio(host: str = "127.0.0.1", port: int = 7860) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is not installed in current environment") from exc

    engine = StereoEngine()
    run_lock = Lock()
    run_state: dict[str, object] = {
        "is_running": False,
        "run_id": 0,
        "cancel_event": None,
    }

    def _encoder_choices_for_codec(codec_name: str) -> list[str]:
        if codec_name == "h264":
            return ["auto", "h264_nvenc", "libx264"]
        return ["auto", "hevc_nvenc", "libx265"]

    def _run(
        input_path: str,
        output_path: str,
        output_container: str,
        codec: str,
        video_encoder: str,
        video_bitrate_mbps: float,
        fps_mode: str,
        audio_mode: str,
        preserve_color_metadata: bool,
        depth_strength: float,
        max_disparity: int,
        buffer_slots: int,
        backend: str,
        progress=gr.Progress(),
    ):
        with run_lock:
            active_cancel_event = run_state["cancel_event"]
            is_running = bool(run_state["is_running"])
            if is_running and isinstance(active_cancel_event, Event):
                active_cancel_event.set()
                engine.abort_active_run()
                yield gr.skip(), "Cancel requested. Stopping current transcode...", gr.update(value="Cancel")
                return
            cancel_event = Event()
            run_state["cancel_event"] = cancel_event
            run_state["is_running"] = True
            run_state["run_id"] = int(run_state["run_id"]) + 1
            run_id = int(run_state["run_id"])

        yield gr.skip(), "Starting transcode...", gr.update(value="Cancel")

        if not input_path:
            with run_lock:
                if run_state["run_id"] == run_id:
                    run_state["cancel_event"] = None
                    run_state["is_running"] = False
            yield (
                json.dumps({"ok": False, "error": "Input video path is required."}, indent=2),
                "Input required.",
                gr.update(value="Run Transcode"),
            )
            return

        resolved_output = output_path.strip() if output_path else _default_output_path(input_path, output_container)
        resolved_output = _coerce_output_extension(resolved_output, output_container)
        normalized_audio_mode = audio_mode.strip().lower()
        audio_passthrough = normalized_audio_mode != "none"
        audio_codec = "aac" if normalized_audio_mode == "aac" else "copy"
        bitrate_mbps = float(video_bitrate_mbps) if video_bitrate_mbps and video_bitrate_mbps > 0 else None

        cfg = EngineConfig(
            input_path=Path(input_path),
            output_path=Path(resolved_output),
            codec=Codec(codec),
            video_encoder=video_encoder,
            video_bitrate_mbps=bitrate_mbps,
            fps_mode=FpsMode(fps_mode),
            depth=DepthConfig(
                max_disparity_px=max_disparity,
                depth_strength=depth_strength,
            ),
            buffer_slots=buffer_slots,
            audio_passthrough=audio_passthrough,
            audio_codec=audio_codec,
            preserve_color_metadata=preserve_color_metadata,
            backend_preference=backend,
        )

        progress(0, desc="Starting transcode...")

        def _progress_cb(frame_idx: int, total_frames: int, frame_ms: float, kernel_ms: float) -> None:
            if cancel_event.is_set():
                raise TranscodeCancelledError("Transcode cancelled by user.")
            if total_frames > 0:
                progress(
                    (frame_idx, total_frames),
                    desc=f"Frame {frame_idx}/{total_frames} | frame {frame_ms:.2f} ms | kernel {kernel_ms:.2f} ms",
                )
            else:
                progress(None, desc=f"Frame {frame_idx} | frame {frame_ms:.2f} ms | kernel {kernel_ms:.2f} ms")

        try:
            result = engine.run_file(
                cfg,
                progress_callback=_progress_cb,
                cancel_requested=cancel_event.is_set,
            )
            progress(1.0, desc=f"Completed. Frames out: {result.frames_out}")
            status = (
                f"Done. Frames: {result.frames_out}, "
                f"Avg FPS: {result.avg_fps:.2f}, "
                f"p95 kernel: {result.p95_kernel_ms:.2f} ms"
            )
            yield _result_to_json(result), status, gr.update(value="Run Transcode")
        except TranscodeCancelledError as exc:
            engine.abort_active_run()
            progress(0, desc="Cancelled.")
            yield (
                json.dumps({"ok": False, "cancelled": True, "error": str(exc)}, indent=2),
                "Cancelled.",
                gr.update(value="Run Transcode"),
            )
        except Exception as exc:
            yield (
                json.dumps({"ok": False, "error": str(exc)}, indent=2),
                f"Failed: {exc}",
                gr.update(value="Run Transcode"),
            )
        finally:
            with run_lock:
                if run_state["run_id"] == run_id:
                    run_state["cancel_event"] = None
                    run_state["is_running"] = False

    def _on_codec_changed(codec_name: str, current_encoder: str):
        choices = _encoder_choices_for_codec(codec_name)
        value = current_encoder if current_encoder in choices else "auto"
        return gr.update(choices=choices, value=value)

    with gr.Blocks(title="PureGPU3D") as app:
        gr.Markdown("# PureGPU3D SBS Engine")

        with gr.Row():
            input_path = gr.Textbox(label="Input video path", scale=8)
            browse_input_btn = gr.Button("Select Input Video", scale=2)

        with gr.Row():
            output_path = gr.Textbox(label="Output video path", scale=8)
            browse_output_btn = gr.Button("Select Output Location", scale=2)

        with gr.Row():
            output_container = gr.Dropdown(
                label="Output container",
                choices=SUPPORTED_CONTAINERS,
                value="mp4",
            )
            codec = gr.Dropdown(label="Codec", choices=["h264", "h265"], value="h265")
            video_encoder = gr.Dropdown(
                label="Video encoder",
                choices=_encoder_choices_for_codec("h265"),
                value="auto",
            )
            video_bitrate_mbps = gr.Slider(
                label="Video bitrate (Mbps, 0=auto)",
                minimum=0.0,
                maximum=120.0,
                value=0.0,
                step=0.5,
            )

        with gr.Row():
            fps_mode = gr.Dropdown(label="Mode", choices=["realtime", "offline"], value="offline")
            audio_mode = gr.Dropdown(label="Audio", choices=["copy", "aac", "none"], value="copy")
            preserve_color_metadata = gr.Checkbox(label="Preserve color metadata", value=True)
            backend = gr.Dropdown(label="Backend", choices=["ffmpeg", "auto", "mock", "nvcodec"], value="ffmpeg")

        with gr.Row():
            depth_strength = gr.Slider(label="Depth strength", minimum=0.0, maximum=1.0, value=0.55, step=0.01)
            max_disparity = gr.Slider(label="Max disparity", minimum=0, maximum=16, value=6, step=1)
            buffer_slots = gr.Slider(label="Buffer slots", minimum=2, maximum=16, value=6, step=1)

        input_info = gr.Markdown("No input selected.")

        run_btn = gr.Button("Run Transcode", variant="primary")
        run_status = gr.Textbox(label="Run status", value="Idle", interactive=False)
        output_json = gr.Code(label="Result", language="json")

        browse_input_btn.click(
            fn=_pick_input_with_windows_dialog,
            inputs=[input_path, output_container, backend],
            outputs=[input_path, output_path, input_info, run_status],
        )

        input_path.change(
            fn=_on_input_changed,
            inputs=[input_path, output_container, backend],
            outputs=[output_path, input_info],
        )

        output_container.change(
            fn=_on_container_changed,
            inputs=[input_path, output_path, output_container],
            outputs=[output_path],
        )

        codec.change(
            fn=_on_codec_changed,
            inputs=[codec, video_encoder],
            outputs=[video_encoder],
        )

        backend.change(
            fn=lambda path_text, backend_name: _build_input_info(path_text, backend_name),
            inputs=[input_path, backend],
            outputs=[input_info],
        )

        browse_output_btn.click(
            fn=_pick_output_with_windows_dialog,
            inputs=[input_path, output_container, output_path],
            outputs=[output_path],
        )

        run_btn.click(
            fn=_run,
            inputs=[
                input_path,
                output_path,
                output_container,
                codec,
                video_encoder,
                video_bitrate_mbps,
                fps_mode,
                audio_mode,
                preserve_color_metadata,
                depth_strength,
                max_disparity,
                buffer_slots,
                backend,
            ],
            outputs=[output_json, run_status, run_btn],
            trigger_mode="multiple",
        )

    app.queue(default_concurrency_limit=8)
    app.launch(server_name=host, server_port=port)
