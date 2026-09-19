"""Video media inspection, format verification, and early refusal guards for PureGPU3D.

Wraps FFprobe to inspect container and stream properties, enforcing strict
upfront validation for the current vertical slice:
  - Supported: 8-bit SDR BT.709, Constant Frame Rate (CFR), even dimensions,
    1:1 square pixel aspect ratio, unrotated geometry, and copy-compatible
    audio (AAC, MP3, AC-3, E-AC-3) or silent media.
  - Explicit upfront refusal: High Dynamic Range (HDR), Variable Frame Rate (VFR),
    odd dimensions, rotation metadata, anamorphic pixel aspect ratios, or
    unsupported audio formats.
"""

from __future__ import annotations

import fractions
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Supported audio codecs for direct MP4 stream copying (-c:a copy)
SUPPORTED_COPY_AUDIO_CODECS: Set[str] = {
    "aac",
    "mp3",
    "ac3",
    "eac3",
}

# Known HDR transfer characteristics
HDR_COLOR_TRANSFERS: Set[str] = {
    "smpte2084",      # PQ (HDR10, Dolby Vision)
    "arib-std-b67",   # HLG
    "linear",
}

# Known HDR color spaces and primaries
HDR_COLOR_SPACES: Set[str] = {
    "bt2020nc",
    "bt2020c",
    "bt2020",
}


class ProbeError(RuntimeError):
    """Base error raised when media probing fails."""
    pass


class MediaNotFoundError(ProbeError, FileNotFoundError):
    """Raised when the specified media file does not exist."""
    pass


class UnsupportedMediaError(ProbeError, ValueError):
    """Raised when media violates vertical slice constraints (HDR, VFR, rotation, etc.)."""
    pass


def find_ffprobe(explicit_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve ffprobe executable path.

    Checks:
      1. Injected explicit path.
      2. Environment variable PUREGPU3D_FFPROBE_PATH.
      3. System PATH via shutil.which('ffprobe').

    Raises:
        FileNotFoundError: If ffprobe binary cannot be located.
    """
    if explicit_path is not None:
        p = Path(explicit_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit ffprobe binary not found at '{explicit_path}'")

    if "PUREGPU3D_FFPROBE_PATH" in os.environ:
        p = Path(os.environ["PUREGPU3D_FFPROBE_PATH"]).resolve()
        if p.is_file():
            return p

    which_path = shutil.which("ffprobe")
    if which_path:
        return Path(which_path).resolve()

    raise FileNotFoundError(
        "ffprobe executable not found. Ensure ffprobe is on PATH or set PUREGPU3D_FFPROBE_PATH."
    )


def find_ffmpeg(explicit_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve ffmpeg executable path.

    Checks:
      1. Injected explicit path.
      2. Environment variable PUREGPU3D_FFMPEG_PATH.
      3. System PATH via shutil.which('ffmpeg').

    Raises:
        FileNotFoundError: If ffmpeg binary cannot be located.
    """
    if explicit_path is not None:
        p = Path(explicit_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit ffmpeg binary not found at '{explicit_path}'")

    if "PUREGPU3D_FFMPEG_PATH" in os.environ:
        p = Path(os.environ["PUREGPU3D_FFMPEG_PATH"]).resolve()
        if p.is_file():
            return p

    which_path = shutil.which("ffmpeg")
    if which_path:
        return Path(which_path).resolve()

    raise FileNotFoundError(
        "ffmpeg executable not found. Ensure ffmpeg is on PATH or set PUREGPU3D_FFMPEG_PATH."
    )


@dataclass(frozen=True)
class AudioStreamInfo:
    """Information for a single audio stream."""

    index: int
    codec_name: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None
    duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoProbeResult:
    """Complete probed metadata for an input media file."""

    path: Path
    width: int
    height: int
    frame_rate: fractions.Fraction
    fps: float
    duration: float
    frame_count: int
    pix_fmt: str
    bit_depth: int
    color_space: Optional[str]
    color_transfer: Optional[str]
    color_primaries: Optional[str]
    is_hdr: bool
    is_vfr: bool
    rotation: int
    sar: str
    has_audio: bool
    audio_streams: List[AudioStreamInfo]
    raw_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        d["frame_rate"] = f"{self.frame_rate.numerator}/{self.frame_rate.denominator}"
        d["audio_streams"] = [a.to_dict() for a in self.audio_streams]
        return d


def parse_fraction(val: Optional[str]) -> Optional[fractions.Fraction]:
    """Parse string representation of a fraction like '24/1' or '30000/1001'."""
    if not val or val == "0/0" or val == "N/A":
        return None
    try:
        parts = val.strip().split("/")
        if len(parts) == 2:
            num = int(parts[0])
            den = int(parts[1])
            if den == 0:
                return None
            return fractions.Fraction(num, den)
        return fractions.Fraction(float(val)).limit_denominator(10000)
    except Exception:
        return None


def probe_video(
    video_path: Union[str, Path],
    *,
    ffprobe_path: Optional[Union[str, Path]] = None,
    strict_sdr_cfr: bool = True,
) -> VideoProbeResult:
    """Probe input media file using FFprobe and validate vertical slice constraints.

    Args:
        video_path: Path to media file.
        ffprobe_path: Optional explicit path to ffprobe binary.
        strict_sdr_cfr: If True, enforce upfront rejection for HDR, VFR,
            odd dimensions, rotation, anamorphic aspect ratio, and incompatible audio.

    Returns:
        VideoProbeResult containing parsed media metadata.

    Raises:
        MediaNotFoundError: If input file does not exist.
        ProbeError: If ffprobe execution or JSON parsing fails.
        UnsupportedMediaError: If strict checks fail with clear actionable descriptions.
    """
    resolved_path = Path(video_path).resolve()
    if not resolved_path.is_file():
        raise MediaNotFoundError(f"Input media file does not exist: '{resolved_path}'")

    ffprobe = find_ffprobe(ffprobe_path)

    cmd = [
        str(ffprobe),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_error",
        str(resolved_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except Exception as err:
        raise ProbeError(f"Failed to execute ffprobe on '{resolved_path}': {err}") from err

    if result.returncode != 0:
        raise ProbeError(
            f"ffprobe returned non-zero exit code {result.returncode} for '{resolved_path}'. "
            f"Stderr: {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise ProbeError(f"Failed to parse ffprobe JSON output for '{resolved_path}': {err}") from err

    if "error" in data:
        raise ProbeError(f"ffprobe reported error for '{resolved_path}': {data['error']}")

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream: Optional[Dict[str, Any]] = None
    audio_streams: List[AudioStreamInfo] = []

    for s in streams:
        c_type = s.get("codec_type")
        if c_type == "video" and video_stream is None:
            # First video stream is primary
            video_stream = s
        elif c_type == "audio":
            idx = int(s.get("index", len(audio_streams)))
            c_name = str(s.get("codec_name", "unknown")).lower()
            s_rate = int(s.get("sample_rate", 44100))
            channels = int(s.get("channels", 2))
            bit_rate_raw = s.get("bit_rate")
            bit_rate = int(bit_rate_raw) if bit_rate_raw and str(bit_rate_raw).isdigit() else None
            dur_raw = s.get("duration")
            dur = float(dur_raw) if dur_raw and dur_raw != "N/A" else None
            audio_streams.append(
                AudioStreamInfo(
                    index=idx,
                    codec_name=c_name,
                    sample_rate=s_rate,
                    channels=channels,
                    bit_rate=bit_rate,
                    duration=dur,
                )
            )

    if video_stream is None:
        raise UnsupportedMediaError(f"No video stream found in media file '{resolved_path}'")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise UnsupportedMediaError(f"Invalid video dimensions: {width}x{height}")

    # Precise rational frame rate
    r_frame_rate_str = video_stream.get("r_frame_rate", "0/0")
    avg_frame_rate_str = video_stream.get("avg_frame_rate", "0/0")
    r_frac = parse_fraction(r_frame_rate_str)
    avg_frac = parse_fraction(avg_frame_rate_str)

    if r_frac is None and avg_frac is None:
        raise UnsupportedMediaError(f"Could not determine frame rate from stream: r='{r_frame_rate_str}', avg='{avg_frame_rate_str}'")

    primary_frac = r_frac if r_frac is not None else avg_frac
    assert primary_frac is not None
    fps = float(primary_frac)

    # Detect VFR
    is_vfr = False
    if r_frac is not None and avg_frac is not None and r_frac.denominator > 0 and avg_frac.denominator > 0:
        if abs(float(r_frac) - float(avg_frac)) > 0.005:
            is_vfr = True

    # Duration and frame count
    dur_str = video_stream.get("duration") or format_info.get("duration")
    duration = float(dur_str) if dur_str and dur_str != "N/A" else 0.0

    nb_frames_str = video_stream.get("nb_frames")
    if nb_frames_str and nb_frames_str != "N/A" and nb_frames_str.isdigit():
        frame_count = int(nb_frames_str)
    elif duration > 0.0 and fps > 0.0:
        frame_count = max(1, int(round(duration * fps)))
    else:
        frame_count = 0

    # Color & HDR detection
    pix_fmt = str(video_stream.get("pix_fmt", "")).lower()
    color_space = video_stream.get("color_space")
    color_transfer = video_stream.get("color_transfer")
    color_primaries = video_stream.get("color_primaries")

    # Bit depth determination
    bits_raw = video_stream.get("bits_per_raw_sample")
    if bits_raw and str(bits_raw).isdigit():
        bit_depth = int(bits_raw)
    elif "10" in pix_fmt or "p10" in pix_fmt:
        bit_depth = 10
    elif "12" in pix_fmt:
        bit_depth = 12
    elif "16" in pix_fmt:
        bit_depth = 16
    else:
        bit_depth = 8

    is_hdr = (
        bit_depth > 8
        or (color_transfer is not None and color_transfer.lower() in HDR_COLOR_TRANSFERS)
        or (color_space is not None and color_space.lower() in HDR_COLOR_SPACES)
        or (color_primaries is not None and color_primaries.lower() in HDR_COLOR_SPACES)
    )

    # Rotation detection (tags.rotate or displaymatrix)
    rotation = 0
    tags = video_stream.get("tags", {})
    rotate_tag = None
    for k, v in tags.items():
        if k.lower() in ("rotate", "rotation"):
            rotate_tag = v
            break

    if rotate_tag:
        try:
            rotation = int(float(rotate_tag)) % 360
        except ValueError:
            rotation = 0

    for side_data in video_stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                rotation = int(float(side_data["rotation"])) % 360
            except ValueError:
                pass

    # Aspect ratio detection
    sar = str(video_stream.get("sample_aspect_ratio", "1:1"))

    has_audio = len(audio_streams) > 0

    # Strict upfront validation
    if strict_sdr_cfr:
        if is_hdr:
            raise UnsupportedMediaError(
                f"Unsupported media: HDR video detected (color_transfer='{color_transfer}', "
                f"color_space='{color_space}', bit_depth={bit_depth}). This vertical slice "
                f"strictly requires standard dynamic range (SDR) BT.709 video. HDR preservation "
                f"is planned for subsequent development phases."
            )

        if is_vfr:
            raise UnsupportedMediaError(
                f"Unsupported media: Variable frame rate (VFR) detected (r_frame_rate='{r_frame_rate_str}', "
                f"avg_frame_rate='{avg_frame_rate_str}'). This vertical slice strictly requires "
                f"constant frame rate (CFR) media."
            )

        if width % 2 != 0 or height % 2 != 0:
            raise UnsupportedMediaError(
                f"Unsupported media: Odd video dimensions {width}x{height}. Both width and height "
                f"must be even numbers for standard YUV420p video encoding."
            )

        if rotation != 0:
            raise UnsupportedMediaError(
                f"Unsupported media: Rotated video metadata detected ({rotation} degrees). "
                f"This vertical slice requires unrotated (0 degree) video orientation."
            )

        # Check SAR (must be 1:1 or 1 or 0:1)
        if sar and sar not in ("1:1", "1", "0:1", "N/A"):
            sar_frac = parse_fraction(sar)
            if sar_frac is not None and sar_frac != fractions.Fraction(1, 1):
                raise UnsupportedMediaError(
                    f"Unsupported media: Anamorphic non-square pixels detected (sample_aspect_ratio='{sar}'). "
                    f"This slice requires 1:1 square pixel aspect ratio."
                )

        # Check audio codec compatibility for MP4 stream copy
        if has_audio:
            primary_audio = audio_streams[0]
            if primary_audio.codec_name not in SUPPORTED_COPY_AUDIO_CODECS:
                raise UnsupportedMediaError(
                    f"Unsupported media: Audio codec '{primary_audio.codec_name}' cannot be directly copied "
                    f"into standard MP4 container. Supported stream copy codecs are: "
                    f"{sorted(SUPPORTED_COPY_AUDIO_CODECS)}."
                )

    return VideoProbeResult(
        path=resolved_path,
        width=width,
        height=height,
        frame_rate=primary_frac,
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
        color_space=color_space,
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        is_hdr=is_hdr,
        is_vfr=is_vfr,
        rotation=rotation,
        sar=sar,
        has_audio=has_audio,
        audio_streams=audio_streams,
        raw_info=data,
    )
