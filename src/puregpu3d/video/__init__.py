"""Video I/O, format inspection, and stereoscopic conversion pipeline for PureGPU3D.

Provides:
  - probe_video: Media property extraction and upfront refusal guards (HDR, VFR, rotation, odd dims).
  - convert_video: Bounded streaming monocular video to Full-SBS conversion with audio remux.
  - VideoProbeResult, ConversionResult, and associated exceptions.
"""

from puregpu3d.video.convert import (
    ConversionCancelledError,
    ConversionError,
    ConversionResult,
    convert_video,
)
from puregpu3d.video.probe import (
    AudioStreamInfo,
    MediaNotFoundError,
    ProbeError,
    UnsupportedMediaError,
    VideoProbeResult,
    find_ffmpeg,
    find_ffprobe,
    probe_video,
)
from puregpu3d.video.scenes import (
    SceneCutDetector,
    SceneCutResult,
    SceneDetectorConfig,
    detect_letterbox,
)

__all__ = [
    "AudioStreamInfo",
    "ConversionCancelledError",
    "ConversionError",
    "ConversionResult",
    "MediaNotFoundError",
    "ProbeError",
    "UnsupportedMediaError",
    "VideoProbeResult",
    "convert_video",
    "find_ffmpeg",
    "find_ffprobe",
    "probe_video",
    "SceneCutDetector",
    "SceneCutResult",
    "SceneDetectorConfig",
    "detect_letterbox",
]
