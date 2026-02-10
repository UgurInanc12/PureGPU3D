from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator

import numpy as np

from ...config.types import Codec
from ..types import VideoProbe
from .base import BackendUnavailableError, CodecBackend, Nv12Writer


def _terminate_process(proc: subprocess.Popen[bytes], *, grace_timeout_s: float = 0.4) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=grace_timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except Exception:
        return
    try:
        proc.wait(timeout=grace_timeout_s)
    except Exception:
        return


class _FfmpegWriter(Nv12Writer):
    def __init__(
        self,
        *,
        command: list[str],
        width: int,
        height: int,
        on_closed: Callable[["_FfmpegWriter"], None] | None = None,
    ) -> None:
        self._expected_shape = (height * 3 // 2, width)
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._on_closed = on_closed
        self._lock = Lock()
        self._aborted = False
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self._expected_shape:
            raise ValueError(f"Expected frame shape {self._expected_shape}, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError("Frame dtype must be uint8")

        with self._lock:
            if self._closed:
                raise RuntimeError("FFmpeg writer is closed")
            if self._aborted:
                raise RuntimeError("FFmpeg writer was aborted")
            stdin = self._proc.stdin

        if stdin is None:
            raise RuntimeError("FFmpeg writer stdin is unavailable")

        try:
            stdin.write(frame.tobytes(order="C"))
        except Exception as exc:
            if self._aborted:
                raise RuntimeError("FFmpeg writer was aborted") from exc
            raise RuntimeError(f"FFmpeg encode pipe write failed: {exc}") from exc

    def abort(self) -> None:
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
            proc = self._proc
            stdin = proc.stdin

        if stdin is not None:
            try:
                stdin.close()
            except Exception:
                pass
        _terminate_process(proc)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
            was_aborted = self._aborted

        stderr_data = b""
        close_error: Exception | None = None
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.stderr is not None:
                try:
                    stderr_data = proc.stderr.read()
                except Exception:
                    stderr_data = b""
            rc = proc.wait()
            if rc != 0 and not was_aborted:
                close_error = RuntimeError(
                    f"FFmpeg encode failed ({rc}): {stderr_data.decode('utf-8', errors='ignore')}"
                )
        finally:
            if self._on_closed is not None:
                self._on_closed(self)

        if close_error is not None:
            raise close_error


def _parse_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 30.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0:
                return 30.0
            return num_f / den_f
        except ValueError:
            return 30.0
    try:
        return float(rate)
    except ValueError:
        return 30.0


def _parse_int(raw: object, default: int) -> int:
    try:
        return int(str(raw))
    except Exception:
        return default


def _parse_bit_depth(pix_fmt: str, bits_per_raw_sample: object) -> int:
    bits = _parse_int(bits_per_raw_sample, default=0)
    if bits > 0:
        return bits
    lowered = pix_fmt.lower()
    if lowered in {"nv12", "yuv420p", "yuvj420p"}:
        return 8
    if lowered in {"p010", "p010le", "yuv420p10le", "yuv420p10be"}:
        return 10
    if lowered in {"p012", "p012le", "yuv420p12le", "yuv420p12be"}:
        return 12
    if lowered in {"p016", "p016le", "yuv420p16le", "yuv420p16be"}:
        return 16
    match = re.search(r"(\d+)(?:le|be)?$", lowered)
    if match is None:
        return 8
    parsed = _parse_int(match.group(1), default=8)
    return parsed if parsed > 0 else 8


def _clean_ffmpeg_color_value(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if value in {"unknown", "unspecified", "reserved", "n/a"}:
        return None
    return value


class FfmpegSoftwareBackend(CodecBackend):
    name = "ffmpeg"

    def __init__(self) -> None:
        self._encoders_cache: set[str] | None = None
        self._hwaccels_cache: set[str] | None = None
        self._active_lock = Lock()
        self._active_decode_proc: subprocess.Popen[bytes] | None = None
        self._active_writer: _FfmpegWriter | None = None
        self.last_selected_encoder: str | None = None
        self.last_selected_bitrate_mbps: float | None = None
        self.last_decode_mode: str | None = None
        self.runtime_warnings: list[str] = []

    def _set_active_decode_proc(self, proc: subprocess.Popen[bytes]) -> None:
        with self._active_lock:
            self._active_decode_proc = proc

    def _clear_active_decode_proc(self, proc: subprocess.Popen[bytes]) -> None:
        with self._active_lock:
            if self._active_decode_proc is proc:
                self._active_decode_proc = None

    def _set_active_writer(self, writer: _FfmpegWriter) -> None:
        with self._active_lock:
            self._active_writer = writer

    def _clear_active_writer(self, writer: _FfmpegWriter) -> None:
        with self._active_lock:
            if self._active_writer is writer:
                self._active_writer = None

    def is_available(self) -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise BackendUnavailableError("ffmpeg/ffprobe not found in PATH")

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=True, capture_output=True, text=True)

    def _load_encoders(self) -> set[str]:
        if self._encoders_cache is not None:
            return self._encoders_cache
        proc = self._run(["ffmpeg", "-hide_banner", "-encoders"])
        encoders: set[str] = set()
        for line in proc.stdout.splitlines():
            if not line.startswith(" "):
                continue
            tokens = line.split()
            if len(tokens) >= 2:
                encoders.add(tokens[1])
        self._encoders_cache = encoders
        return encoders

    def _load_hwaccels(self) -> set[str]:
        if self._hwaccels_cache is not None:
            return self._hwaccels_cache
        proc = self._run(["ffmpeg", "-hide_banner", "-hwaccels"])
        hwaccels: set[str] = set()
        for line in proc.stdout.splitlines():
            token = line.strip().lower()
            if not token or token.endswith(":"):
                continue
            if " " in token:
                continue
            hwaccels.add(token)
        self._hwaccels_cache = hwaccels
        return hwaccels

    def _cuda_hwaccel_available(self) -> bool:
        try:
            return "cuda" in self._load_hwaccels()
        except Exception:
            return False

    def _build_decode_cmd(self, input_path: Path, *, use_cuda: bool) -> list[str]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if use_cuda:
            cmd.extend(
                [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_output_format",
                    "cuda",
                    "-extra_hw_frames",
                    "16",
                ]
            )
        else:
            cmd.extend(["-threads", "0"])
        cmd.extend(
            [
                "-i",
                str(input_path),
                "-an",
                "-sn",
                "-dn",
            ]
        )
        if use_cuda:
            cmd.extend(["-vf", "hwdownload,format=nv12"])
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "nv12",
                "-",
            ]
        )
        return cmd

    def probe(self, input_path: Path) -> VideoProbe:
        self._ensure_available()
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ]
        result = self._run(cmd)
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video_stream is None:
            raise RuntimeError(f"No video stream found in {input_path}")

        width = int(video_stream.get("width"))
        height = int(video_stream.get("height"))
        fps = _parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
        frame_count = int(video_stream.get("nb_frames") or 0)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        pix_fmt = str(video_stream.get("pix_fmt", "unknown"))
        bit_depth = _parse_bit_depth(
            pix_fmt=pix_fmt,
            bits_per_raw_sample=video_stream.get("bits_per_raw_sample"),
        )
        duration_raw = payload.get("format", {}).get("duration")
        duration_s = float(duration_raw) if duration_raw else None

        return VideoProbe(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            has_audio=has_audio,
            video_codec=str(video_stream.get("codec_name", "unknown")),
            pix_fmt=pix_fmt,
            bit_depth=bit_depth,
            color_range=_clean_ffmpeg_color_value(video_stream.get("color_range")),
            color_space=_clean_ffmpeg_color_value(video_stream.get("color_space")),
            color_transfer=_clean_ffmpeg_color_value(video_stream.get("color_transfer")),
            color_primaries=_clean_ffmpeg_color_value(video_stream.get("color_primaries")),
            duration_s=duration_s,
        )

    def decode_iter(self, input_path: Path, probe: VideoProbe) -> Iterator[np.ndarray]:
        self._ensure_available()
        self.last_decode_mode = None
        frame_size = probe.width * probe.height * 3 // 2
        candidate_modes: list[str] = ["software"]
        if self._cuda_hwaccel_available():
            candidate_modes = ["cuda", "software"]

        for idx, mode in enumerate(candidate_modes):
            cmd = self._build_decode_cmd(input_path, use_cuda=(mode == "cuda"))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._set_active_decode_proc(proc)
            assert proc.stdout is not None
            assert proc.stderr is not None
            yielded_any = False
            decode_error: Exception | None = None
            try:
                while True:
                    chunk = proc.stdout.read(frame_size)
                    if not chunk:
                        break
                    if len(chunk) != frame_size:
                        raise RuntimeError(
                            f"Short frame read from ffmpeg. Expected {frame_size}, got {len(chunk)}"
                        )
                    yielded_any = True
                    frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                        (probe.height * 3 // 2, probe.width)
                    ).copy()
                    yield frame
            except Exception as exc:
                decode_error = exc
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                stderr_text = proc.stderr.read().decode("utf-8", errors="ignore")
                rc = proc.wait()
                self._clear_active_decode_proc(proc)
                if decode_error is None and rc != 0:
                    decode_error = RuntimeError(f"FFmpeg decode failed ({rc}): {stderr_text}")

            if decode_error is None:
                self.last_decode_mode = mode
                if mode == "cuda":
                    self.runtime_warnings.append("Decode mode: cuda_hwaccel")
                else:
                    self.runtime_warnings.append("Decode mode: software")
                return

            has_fallback = idx + 1 < len(candidate_modes)
            if mode == "cuda" and not yielded_any and has_fallback:
                self.runtime_warnings.append(
                    f"CUDA decode unavailable, fell back to software decode: {decode_error}"
                )
                continue
            raise decode_error

    def _choose_encoder(self, codec: Codec, requested: str) -> tuple[str, list[str]]:
        encoders = self._load_encoders()
        normalized = requested.strip().lower() if requested else "auto"
        if normalized in {"", "auto"}:
            if codec == Codec.H264:
                if "h264_nvenc" in encoders:
                    return "h264_nvenc", ["-preset", "p4", "-tune", "ll", "-bf", "0", "-g", "60"]
                return "libx264", ["-preset", "veryfast", "-tune", "zerolatency", "-bf", "0", "-g", "60"]
            if "hevc_nvenc" in encoders:
                return "hevc_nvenc", ["-preset", "p4", "-tune", "ll", "-bf", "0", "-g", "60"]
            return "libx265", ["-preset", "fast", "-x265-params", "keyint=60:min-keyint=60"]

        valid_h264 = {"h264_nvenc", "libx264"}
        valid_h265 = {"hevc_nvenc", "libx265"}
        if codec == Codec.H264 and normalized not in valid_h264:
            raise ValueError(
                f"Encoder '{normalized}' is incompatible with codec '{codec.value}'. "
                f"Allowed: {sorted(valid_h264)}"
            )
        if codec == Codec.H265 and normalized not in valid_h265:
            raise ValueError(
                f"Encoder '{normalized}' is incompatible with codec '{codec.value}'. "
                f"Allowed: {sorted(valid_h265)}"
            )
        if normalized not in encoders:
            raise RuntimeError(
                f"Requested encoder '{normalized}' is unavailable in this ffmpeg build."
            )

        if normalized == "h264_nvenc":
            return normalized, ["-preset", "p4", "-tune", "ll", "-bf", "0", "-g", "60"]
        if normalized == "hevc_nvenc":
            return normalized, ["-preset", "p4", "-tune", "ll", "-bf", "0", "-g", "60"]
        if normalized == "libx264":
            return normalized, ["-preset", "veryfast", "-tune", "zerolatency", "-bf", "0", "-g", "60"]
        return normalized, ["-preset", "fast", "-x265-params", "keyint=60:min-keyint=60"]

    def _format_bitrate_mbps(self, bitrate_mbps: float) -> str:
        text = f"{bitrate_mbps:.3f}".rstrip("0").rstrip(".")
        return f"{text}M"

    def _build_color_metadata_args(self, source_probe: VideoProbe | None) -> list[str]:
        if source_probe is None:
            return []
        args: list[str] = []
        if source_probe.color_range:
            args.extend(["-color_range", source_probe.color_range])
        if source_probe.color_space:
            args.extend(["-colorspace", source_probe.color_space])
        if source_probe.color_transfer:
            args.extend(["-color_trc", source_probe.color_transfer])
        if source_probe.color_primaries:
            args.extend(["-color_primaries", source_probe.color_primaries])
        return args

    def _resolve_encoder_resolution_compatibility(
        self,
        *,
        codec: Codec,
        requested_encoder: str,
        selected_encoder: str,
        width: int,
    ) -> tuple[str, list[str]]:
        if selected_encoder != "h264_nvenc" or width <= 4096:
            return selected_encoder, []

        requested = requested_encoder.strip().lower() if requested_encoder else "auto"
        if requested in {"", "auto"}:
            encoders = self._load_encoders()
            if "libx264" in encoders and codec == Codec.H264:
                self.runtime_warnings.append(
                    f"Output width {width} exceeds h264_nvenc limits; fell back to libx264."
                )
                return "libx264", ["-preset", "veryfast", "-tune", "zerolatency", "-bf", "0", "-g", "60"]
            raise RuntimeError(
                f"Output width {width} exceeds h264_nvenc limits and no compatible fallback encoder is available."
            )

        raise ValueError(
            f"Encoder '{selected_encoder}' is incompatible with output width {width}. "
            "Choose 'libx264' or switch codec/encoder."
        )

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
        self._ensure_available()
        self.runtime_warnings = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoder, encoder_args = self._choose_encoder(codec, video_encoder)
        encoder, encoder_args = self._resolve_encoder_resolution_compatibility(
            codec=codec,
            requested_encoder=video_encoder,
            selected_encoder=encoder,
            width=width,
        )
        self.last_selected_encoder = encoder
        self.last_selected_bitrate_mbps = video_bitrate_mbps
        bitrate_args: list[str] = []
        if video_bitrate_mbps is not None:
            bitrate = self._format_bitrate_mbps(video_bitrate_mbps)
            bitrate_args = [
                "-b:v",
                bitrate,
                "-maxrate",
                bitrate,
                "-bufsize",
                self._format_bitrate_mbps(video_bitrate_mbps * 2.0),
            ]
        output_pix_fmt = "nv12" if encoder.endswith("_nvenc") else "yuv420p"

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            encoder,
            "-pix_fmt",
            output_pix_fmt,
        ]
        cmd.extend(encoder_args)
        cmd.extend(bitrate_args)
        cmd.extend(self._build_color_metadata_args(source_probe))
        if codec == Codec.H265:
            cmd.extend(["-tag:v", "hvc1"])
        cmd.append(str(output_path))

        writer = _FfmpegWriter(
            command=cmd,
            width=width,
            height=height,
            on_closed=self._clear_active_writer,
        )
        self._set_active_writer(writer)
        return writer

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
        self._ensure_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        normalized_audio_codec = audio_codec.strip().lower()
        if normalized_audio_codec not in {"copy", "aac"}:
            raise ValueError(f"Unsupported audio codec mode: {audio_codec}")

        def _build_remux_cmd(selected_audio_codec: str) -> list[str]:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encoded_video_path),
                "-i",
                str(source_input_path),
                "-map",
                "0:v:0",
            ]

            if audio_passthrough:
                cmd.extend(["-map", "1:a?"])

            cmd.extend(["-c:v", "copy"])

            if audio_passthrough:
                if selected_audio_codec == "copy":
                    cmd.extend(["-c:a", "copy"])
                else:
                    cmd.extend(["-c:a", "aac", "-b:a", "192k"])

            output_suffix = output_path.suffix.lower()
            if output_suffix in {".mp4", ".mov", ".m4v"}:
                if codec == Codec.H265:
                    cmd.extend(["-tag:v", "hvc1"])
                cmd.extend(["-movflags", "+faststart"])

            cmd.append(str(output_path))
            return cmd

        try:
            self._run(_build_remux_cmd(normalized_audio_codec))
        except subprocess.CalledProcessError as exc:
            if audio_passthrough and normalized_audio_codec == "copy":
                warnings.append("Audio copy failed; audio was transcoded to AAC for compatibility.")
                self._run(_build_remux_cmd("aac"))
            else:
                stderr_text = exc.stderr or exc.stdout or str(exc)
                raise RuntimeError(f"FFmpeg remux failed: {stderr_text}") from exc
        return warnings

    def request_abort(self) -> None:
        with self._active_lock:
            decode_proc = self._active_decode_proc
            writer = self._active_writer

        if writer is not None:
            try:
                writer.abort()
            except Exception:
                pass
        if decode_proc is not None:
            _terminate_process(decode_proc)
