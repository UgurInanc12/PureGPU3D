"""Encoder resolution, preflight probing, and rate control configuration for PureGPU3D.

Supports explicit hardware-accelerated NVENC encoding (HEVC and H.264) and software
libx264 encoding with bounded preflight probing, safe software fallback for 'auto',
and strict rejection of manual invalid hardware selections.

CQ vs CRF Rate Control Contract:
    - libx264 uses Constant Rate Factor (CRF, default 18) and standard x264 presets
      (e.g. 'medium'). CRF dynamically modulates macroblock quantization via multi-frame
      macroblock tree (mbtree) and psycho-visual lookaheads in software.
    - NVENC (hevc_nvenc, h264_nvenc) uses Constant Quality (CQ, default 23) in Variable
      Bitrate mode (-rc vbr -cq <cq> -b:v 0) and NVENC presets (e.g. 'p4' medium).
      NVENC executes on the NVIDIA GPU ASIC without CPU lookahead.
    - CQ and CRF are NOT interchangeable: numerically equal numbers do NOT yield
      equivalent bitrates or visual transparent fidelity, and NVENC flags differ from
      libx264 flags. They are configured through dedicated, explicit parameters.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from puregpu3d.video.probe import find_ffmpeg

logger = logging.getLogger(__name__)

# Canonical supported encoder identifiers
ENCODER_AUTO = "auto"
ENCODER_HEVC_NVENC = "hevc_nvenc"
ENCODER_H264_NVENC = "h264_nvenc"
ENCODER_LIBX264 = "libx264"

VALID_ENCODERS: tuple[str, ...] = (
    ENCODER_AUTO,
    ENCODER_HEVC_NVENC,
    ENCODER_H264_NVENC,
    ENCODER_LIBX264,
)

NVENC_HARDWARE_ENCODERS: tuple[str, ...] = (
    ENCODER_HEVC_NVENC,
    ENCODER_H264_NVENC,
)

# Quality defaults
DEFAULT_ENCODER = ENCODER_LIBX264
DEFAULT_CRF = 18
DEFAULT_CQ = 23
DEFAULT_X264_PRESET = "medium"
DEFAULT_NVENC_PRESET = "p4"
DEFAULT_PIX_FMT = "yuv420p"


class EncoderError(RuntimeError):
    """Base error raised for encoder selection, validation, or preflight failures."""
    pass


class InvalidEncoderError(EncoderError, ValueError):
    """Raised when an unrecognized or malformed encoder string is specified."""
    pass


class EncoderPreflightError(EncoderError):
    """Raised when an encoder fails runtime hardware preflight validation."""
    pass


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of actual short encode preflight at requested dimensions and pixel format."""

    encoder: str
    width: int
    height: int
    pix_fmt: str
    supported: bool
    error_message: Optional[str] = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class ResolvedEncoder:
    """Resolved and validated encoder configuration ready for FFmpeg writer construction."""

    requested_encoder: str
    selected_encoder: str
    codec_name: str
    ffmpeg_args: List[str]
    fallback_reason: Optional[str] = None
    is_hardware: bool = False
    preflight: Optional[PreflightResult] = None


def validate_encoder_name(encoder: str) -> str:
    """Validate and normalize an encoder identifier string.

    Args:
        encoder: User-provided encoder string (e.g. 'auto', 'hevc_nvenc', 'h264_nvenc', 'libx264').

    Returns:
        Normalized lower-case encoder string.

    Raises:
        InvalidEncoderError: If the encoder name is not in VALID_ENCODERS.
    """
    if not isinstance(encoder, str):
        raise InvalidEncoderError(
            f"Encoder must be a string, got {type(encoder).__name__}: {encoder!r}"
        )
    normalized = encoder.strip().lower()
    if normalized not in VALID_ENCODERS:
        valid_list = ", ".join(repr(e) for e in VALID_ENCODERS)
        raise InvalidEncoderError(
            f"Invalid encoder {encoder!r}. Supported encoders are: {valid_list}"
        )
    return normalized


def preflight_encoder(
    encoder: str,
    width: int,
    height: int,
    *,
    pix_fmt: str = DEFAULT_PIX_FMT,
    preset: Optional[str] = None,
    ffmpeg_path: Optional[Union[str, Path]] = None,
    timeout: float = 8.0,
) -> PreflightResult:
    """Execute an actual short encode preflight at the exact requested dimensions and pixel format.

    Does NOT rely on static capability lists or driver version heuristics alone. Spawns
    FFmpeg with a minimal 2-frame lavfi test input at the target FullSBS dimensions to
    verify GPU ASIC session allocation, dimension boundary limits, and driver compatibility.

    Args:
        encoder: Target encoder name (e.g. 'hevc_nvenc', 'h264_nvenc', 'libx264').
        width: Exact export width (e.g. 3840 for FullSBS of 1080p).
        height: Exact export height (e.g. 1080).
        pix_fmt: Target pixel format (default 'yuv420p').
        preset: Optional encoder speed/quality preset.
        ffmpeg_path: Optional path to ffmpeg executable.
        timeout: Maximum seconds to wait for preflight test to complete.

    Returns:
        PreflightResult detailing whether preflight succeeded and any captured stderr error.
    """
    if width <= 0 or height <= 0:
        return PreflightResult(
            encoder=encoder,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            supported=False,
            error_message=f"Invalid preflight dimensions: {width}x{height}",
        )

    # Software libx264 is always locally available when FFmpeg is present
    if encoder == ENCODER_LIBX264:
        return PreflightResult(
            encoder=encoder,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            supported=True,
            error_message=None,
            duration_s=0.0,
        )

    ffmpeg_bin = find_ffmpeg(ffmpeg_path)

    # Build minimal test encode command: 2 frames of black color filter to null muxer
    cmd = [
        str(ffmpeg_bin),
        "-v", "error",
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r=24",
        "-frames:v", "2",
        "-c:v", encoder,
    ]
    if preset:
        cmd.extend(["-preset", preset])
    cmd.extend([
        "-pix_fmt", pix_fmt,
        "-f", "null",
        "-",
    ])

    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.perf_counter() - t0
        if res.returncode == 0:
            return PreflightResult(
                encoder=encoder,
                width=width,
                height=height,
                pix_fmt=pix_fmt,
                supported=True,
                error_message=None,
                duration_s=duration,
            )
        else:
            stderr_cleaned = res.stderr.strip() or f"FFmpeg exited with code {res.returncode}"
            logger.warning(f"NVENC preflight failed for {encoder} at {width}x{height}: {stderr_cleaned}")
            return PreflightResult(
                encoder=encoder,
                width=width,
                height=height,
                pix_fmt=pix_fmt,
                supported=False,
                error_message=stderr_cleaned,
                duration_s=duration,
            )
    except subprocess.TimeoutExpired:
        logger.warning(f"NVENC preflight timed out after {timeout}s for {encoder} at {width}x{height}")
        return PreflightResult(
            encoder=encoder,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            supported=False,
            error_message=f"Preflight timed out after {timeout}s",
            duration_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.warning(f"NVENC preflight exception for {encoder}: {exc}")
        return PreflightResult(
            encoder=encoder,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            supported=False,
            error_message=str(exc),
            duration_s=time.perf_counter() - t0,
        )


def resolve_encoder(
    encoder: str = DEFAULT_ENCODER,
    *,
    width: int,
    height: int,
    crf: int = DEFAULT_CRF,
    cq: int = DEFAULT_CQ,
    x264_preset: str = DEFAULT_X264_PRESET,
    nvenc_preset: str = DEFAULT_NVENC_PRESET,
    pix_fmt: str = DEFAULT_PIX_FMT,
    ffmpeg_path: Optional[Union[str, Path]] = None,
    preflight_timeout: float = 8.0,
) -> ResolvedEncoder:
    """Resolve, validate, and preflight the video encoder for FullSBS encoding.

    Rules:
        1. 'auto':
           - Preferred wide SBS choice is 'hevc_nvenc' (HEVC naturally accommodates 4K/8K wide SBS).
           - Executes actual short preflight at target width and height.
           - If 'hevc_nvenc' succeeds -> selects 'hevc_nvenc'.
           - If 'hevc_nvenc' fails -> probes 'h264_nvenc'. If that succeeds -> selects 'h264_nvenc'
             with recorded fallback reason.
           - If all NVENC probes fail -> falls back to 'libx264' with full recorded reason.
           - NEVER silently downscales the image.
        2. Explicit manual 'hevc_nvenc' or 'h264_nvenc':
           - Executes actual preflight at requested dimensions.
           - If preflight fails, strictly REJECTS the selection with EncoderPreflightError.
           - NEVER falls back silently or downscales on manual hardware selection.
        3. Explicit manual 'libx264':
           - Uses CPU software encoding with requested crf and x264_preset.
        4. Unknown encoder strings:
           - Immediately rejected with InvalidEncoderError.

    Rate Control Contract:
        - libx264 uses '-crf <crf>' and '-preset <x264_preset>'.
        - NVENC uses '-rc vbr -cq <cq> -b:v 0' and '-preset <nvenc_preset>'.
        - CQ and CRF values are kept distinct and never confused.

    Args:
        encoder: One of 'auto', 'hevc_nvenc', 'h264_nvenc', 'libx264'.
        width: Target output width (e.g. 2W for FullSBS).
        height: Target output height (e.g. H).
        crf: Constant Rate Factor for libx264 (default 18).
        cq: Constant Quality target for NVENC (default 23).
        x264_preset: x264 preset (default 'medium').
        nvenc_preset: NVENC preset (default 'p4').
        pix_fmt: Output pixel format (default 'yuv420p').
        ffmpeg_path: Optional path to ffmpeg executable.
        preflight_timeout: Timeout in seconds for preflight test.

    Returns:
        ResolvedEncoder containing the selected encoder, arguments, and fallback notes.

    Raises:
        InvalidEncoderError: If encoder is not in VALID_ENCODERS.
        EncoderPreflightError: If a manual hardware encoder selection fails preflight.
    """
    canon_encoder = validate_encoder_name(encoder)

    # Validate rate control bounds
    if not (0 <= crf <= 51):
        raise ValueError(f"libx264 CRF must be between 0 and 51, got {crf}")
    if not (0 <= cq <= 51):
        raise ValueError(f"NVENC CQ must be between 0 and 51, got {cq}")

    if canon_encoder == ENCODER_AUTO:
        # Step 1: Probe default wide-SBS encoder: hevc_nvenc
        pf_hevc = preflight_encoder(
            ENCODER_HEVC_NVENC,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            preset=nvenc_preset,
            ffmpeg_path=ffmpeg_path,
            timeout=preflight_timeout,
        )
        if pf_hevc.supported:
            logger.info(f"Auto encoder resolved to {ENCODER_HEVC_NVENC} (preflight passed in {pf_hevc.duration_s:.3f}s)")
            return ResolvedEncoder(
                requested_encoder=ENCODER_AUTO,
                selected_encoder=ENCODER_HEVC_NVENC,
                codec_name="hevc",
                ffmpeg_args=[
                    "-c:v", ENCODER_HEVC_NVENC,
                    "-preset", nvenc_preset,
                    "-rc", "vbr",
                    "-cq", str(cq),
                    "-b:v", "0",
                    "-pix_fmt", pix_fmt,
                ],
                fallback_reason=None,
                is_hardware=True,
                preflight=pf_hevc,
            )

        # Step 2: HEVC NVENC failed; probe h264_nvenc
        hevc_err = pf_hevc.error_message or "Unknown HEVC NVENC preflight error"
        logger.info(f"Auto encoder: {ENCODER_HEVC_NVENC} preflight failed ({hevc_err}); trying {ENCODER_H264_NVENC}...")

        pf_h264 = preflight_encoder(
            ENCODER_H264_NVENC,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            preset=nvenc_preset,
            ffmpeg_path=ffmpeg_path,
            timeout=preflight_timeout,
        )
        if pf_h264.supported:
            fallback_msg = f"hevc_nvenc preflight failed ({hevc_err}); fell back to h264_nvenc"
            logger.info(f"Auto encoder resolved to {ENCODER_H264_NVENC} ({fallback_msg})")
            return ResolvedEncoder(
                requested_encoder=ENCODER_AUTO,
                selected_encoder=ENCODER_H264_NVENC,
                codec_name="h264",
                ffmpeg_args=[
                    "-c:v", ENCODER_H264_NVENC,
                    "-preset", nvenc_preset,
                    "-rc", "vbr",
                    "-cq", str(cq),
                    "-b:v", "0",
                    "-pix_fmt", pix_fmt,
                ],
                fallback_reason=fallback_msg,
                is_hardware=True,
                preflight=pf_h264,
            )

        # Step 3: All hardware NVENC options failed; safe fallback to libx264 software encoder
        h264_err = pf_h264.error_message or "Unknown H.264 NVENC preflight error"
        fallback_msg = (
            f"Hardware NVENC preflight failed (hevc_nvenc: {hevc_err}; "
            f"h264_nvenc: {h264_err}); fell back to software {ENCODER_LIBX264}"
        )
        logger.warning(f"Auto encoder: {fallback_msg}")
        return ResolvedEncoder(
            requested_encoder=ENCODER_AUTO,
            selected_encoder=ENCODER_LIBX264,
            codec_name="h264",
            ffmpeg_args=[
                "-c:v", ENCODER_LIBX264,
                "-preset", x264_preset,
                "-crf", str(crf),
                "-pix_fmt", pix_fmt,
            ],
            fallback_reason=fallback_msg,
            is_hardware=False,
            preflight=pf_hevc,
        )

    elif canon_encoder in NVENC_HARDWARE_ENCODERS:
        codec_name = "hevc" if canon_encoder == ENCODER_HEVC_NVENC else "h264"
        pf = preflight_encoder(
            canon_encoder,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            preset=nvenc_preset,
            ffmpeg_path=ffmpeg_path,
            timeout=preflight_timeout,
        )
        if not pf.supported:
            err_msg = pf.error_message or "Unknown hardware preflight error"
            raise EncoderPreflightError(
                f"Manual hardware encoder selection '{canon_encoder}' failed preflight at "
                f"{width}x{height} ({pix_fmt}): {err_msg}. Manual selection rejected; "
                f"no silent downscale or fallback applied."
            )
        return ResolvedEncoder(
            requested_encoder=canon_encoder,
            selected_encoder=canon_encoder,
            codec_name=codec_name,
            ffmpeg_args=[
                "-c:v", canon_encoder,
                "-preset", nvenc_preset,
                "-rc", "vbr",
                "-cq", str(cq),
                "-b:v", "0",
                "-pix_fmt", pix_fmt,
            ],
            fallback_reason=None,
            is_hardware=True,
            preflight=pf,
        )

    else:  # ENCODER_LIBX264
        return ResolvedEncoder(
            requested_encoder=ENCODER_LIBX264,
            selected_encoder=ENCODER_LIBX264,
            codec_name="h264",
            ffmpeg_args=[
                "-c:v", ENCODER_LIBX264,
                "-preset", x264_preset,
                "-crf", str(crf),
                "-pix_fmt", pix_fmt,
            ],
            fallback_reason=None,
            is_hardware=False,
            preflight=None,
        )
