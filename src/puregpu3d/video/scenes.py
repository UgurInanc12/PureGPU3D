"""Scene cut detection and letterbox analysis for PureGPU3D video pipeline.

Provides:
  - Fast, bounded O(1) shot cut detection via color histogram and pixel delta analysis.
  - Letterbox (black border) detection and active image area isolation.
  - Fade-to-black and fade-from-black protection.
  - Reset signaling for downstream temporal filters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class SceneDetectorConfig:
    """Configuration for scene cut and transition detection.

    Attributes:
        cut_threshold: Combined histogram and pixel delta threshold in [0, 1] to trigger a hard cut.
        fade_threshold: Mean luminance below which a frame is classified as a fade/black frame.
        min_scene_frames: Minimum frame count between cuts to prevent false flutter.
        detect_letterbox: Whether to detect and isolate black letterbox margins.
        letterbox_threshold: Pixel intensity threshold to consider border rows/cols as letterbox.
        thumbnail_size: (width, height) for fast thumbnail analysis.
    """

    cut_threshold: float = 0.35
    fade_threshold: float = 8.0
    min_scene_frames: int = 1
    detect_letterbox: bool = True
    letterbox_threshold: float = 12.0
    thumbnail_size: Tuple[int, int] = (128, 72)

    def __post_init__(self) -> None:
        if not (0.0 < self.cut_threshold <= 1.0):
            raise ValueError(f"cut_threshold must be in (0, 1], got {self.cut_threshold}")
        if self.fade_threshold < 0.0:
            raise ValueError(f"fade_threshold must be non-negative, got {self.fade_threshold}")
        if self.min_scene_frames < 1:
            raise ValueError(f"min_scene_frames must be >= 1, got {self.min_scene_frames}")
        if self.letterbox_threshold < 0.0:
            raise ValueError(f"letterbox_threshold must be >= 0, got {self.letterbox_threshold}")


@dataclass(frozen=True)
class SceneCutResult:
    """Detection results for a single frame.

    Attributes:
        is_cut: True if this frame represents a hard cut, scene boundary, or reset.
        score: Scene change score in [0, 1].
        is_fade: True if frame is a fade/black frame.
        letterbox_crop: (ymin, ymax, xmin, xmax) bounding coordinates of active content.
        frame_index: 0-based frame index within the stream.
    """

    is_cut: bool
    score: float
    is_fade: bool
    letterbox_crop: Optional[Tuple[int, int, int, int]]
    frame_index: int


def detect_letterbox(
    frame_rgb: np.ndarray,
    threshold: float = 12.0,
) -> Tuple[int, int, int, int]:
    """Detect black letterbox borders and return active bounding box (ymin, ymax, xmin, xmax).

    If no letterbox is found or active region is degenerate (< 20% of area),
    returns full image bounds (0, H, 0, W).
    """
    h, w = frame_rgb.shape[:2]
    if h < 4 or w < 4:
        return (0, h, 0, w)

    # Average luminance per row and col
    # Approximate luminance: 0.299*R + 0.587*G + 0.114*B or mean across channels
    row_means = frame_rgb.mean(axis=(1, 2))
    col_means = frame_rgb.mean(axis=(0, 2))

    ymin = 0
    while ymin < h and row_means[ymin] < threshold:
        ymin += 1

    ymax = h
    while ymax > ymin and row_means[ymax - 1] < threshold:
        ymax -= 1

    xmin = 0
    while xmin < w and col_means[xmin] < threshold:
        xmin += 1

    xmax = w
    while xmax > xmin and col_means[xmax - 1] < threshold:
        xmax -= 1

    # Safety: if active region is too small (< 20% of dimension), don't crop
    if (ymax - ymin < int(h * 0.2)) or (xmax - xmin < int(w * 0.2)):
        return (0, h, 0, w)

    return (ymin, ymax, xmin, xmax)


class SceneCutDetector:
    """Bounded, causal scene cut detector for video streams.

    Retains strictly O(1) memory: stores only one downscaled thumbnail and color
    histogram of the previous frame.
    """

    def __init__(self, config: Optional[SceneDetectorConfig] = None) -> None:
        self.config = config if config is not None else SceneDetectorConfig()
        self._prev_thumb_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._frame_index: int = 0
        self._last_cut_frame: int = 0
        self._prev_shape: Optional[Tuple[int, int]] = None
        self._prev_was_fade: bool = False

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def reset(self) -> None:
        """Reset internal state for a new video stream or after an external reset."""
        self._prev_thumb_gray = None
        self._prev_hist = None
        self._frame_index = 0
        self._last_cut_frame = 0
        self._prev_shape = None
        self._prev_was_fade = False

    def update(self, frame_rgb: Union[np.ndarray, torch.Tensor]) -> SceneCutResult:
        """Process one incoming RGB frame and determine if it triggers a shot cut.

        Args:
            frame_rgb: (H, W, 3) uint8 image.

        Returns:
            SceneCutResult with cut decision, difference score, fade status, and letterbox crop.
        """
        if isinstance(frame_rgb, torch.Tensor):
            frame_np = frame_rgb.detach().cpu().numpy()
        else:
            frame_np = frame_rgb

        h, w = frame_np.shape[:2]
        idx = self._frame_index
        self._frame_index += 1

        # Check resolution change
        if self._prev_shape is not None and self._prev_shape != (h, w):
            self.reset()
            self._frame_index = idx + 1
            self._prev_shape = (h, w)
            return SceneCutResult(
                is_cut=True,
                score=1.0,
                is_fade=False,
                letterbox_crop=(0, h, 0, w),
                frame_index=idx,
            )
        self._prev_shape = (h, w)

        # 1. Letterbox detection
        letterbox_crop = (
            detect_letterbox(frame_np, self.config.letterbox_threshold)
            if self.config.detect_letterbox
            else (0, h, 0, w)
        )
        ymin, ymax, xmin, xmax = letterbox_crop
        active_crop = frame_np[ymin:ymax, xmin:xmax]

        # 2. Fade detection
        mean_lum = float(active_crop.mean())
        is_fade = mean_lum < self.config.fade_threshold

        # First frame is always a scene boundary
        if self._prev_thumb_gray is None:
            self._update_reference(active_crop)
            self._prev_was_fade = is_fade
            self._last_cut_frame = idx
            return SceneCutResult(
                is_cut=True,
                score=1.0,
                is_fade=is_fade,
                letterbox_crop=letterbox_crop,
                frame_index=idx,
            )

        # Transition into or out of fade triggers a reset
        if is_fade != self._prev_was_fade:
            self._prev_was_fade = is_fade
            self._update_reference(active_crop)
            self._last_cut_frame = idx
            return SceneCutResult(
                is_cut=True,
                score=1.0,
                is_fade=is_fade,
                letterbox_crop=letterbox_crop,
                frame_index=idx,
            )
        self._prev_was_fade = is_fade

        if is_fade:
            # During continuous black/fade, keep cut = False (or do not smooth across)
            self._update_reference(active_crop)
            return SceneCutResult(
                is_cut=False,
                score=0.0,
                is_fade=True,
                letterbox_crop=letterbox_crop,
                frame_index=idx,
            )

        # 3. Content comparison on active thumbnail
        tw, th = self.config.thumbnail_size
        thumb_rgb = cv2.resize(active_crop, (tw, th), interpolation=cv2.INTER_AREA)
        thumb_gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)

        # Color histogram (16 bins per channel)
        hist_r = cv2.calcHist([thumb_rgb], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([thumb_rgb], [1], None, [16], [0, 256])
        hist_b = cv2.calcHist([thumb_rgb], [2], None, [16], [0, 256])
        hist = np.concatenate([hist_r, hist_g, hist_b]).flatten().astype(np.float32)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

        assert self._prev_hist is not None and self._prev_thumb_gray is not None

        # Compare histograms: correlation in [-1, 1], convert to distance [0, 1]
        hist_corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
        hist_dist = np.clip((1.0 - hist_corr) / 2.0, 0.0, 1.0)

        # Pixel delta on grayscale thumbnail
        pix_delta = np.mean(np.abs(thumb_gray.astype(np.float32) - self._prev_thumb_gray.astype(np.float32))) / 255.0

        # Combined metric
        score = float(0.6 * hist_dist + 0.4 * pix_delta)

        frames_since_cut = idx - self._last_cut_frame
        is_cut = (score >= self.config.cut_threshold) and (frames_since_cut >= self.config.min_scene_frames)

        if is_cut:
            self._last_cut_frame = idx

        # Update bounded reference state
        self._prev_thumb_gray = thumb_gray
        self._prev_hist = hist

        return SceneCutResult(
            is_cut=is_cut,
            score=score,
            is_fade=is_fade,
            letterbox_crop=letterbox_crop,
            frame_index=idx,
        )

    def _update_reference(self, active_crop: np.ndarray) -> None:
        tw, th = self.config.thumbnail_size
        thumb_rgb = cv2.resize(active_crop, (tw, th), interpolation=cv2.INTER_AREA)
        self._prev_thumb_gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
        hist_r = cv2.calcHist([thumb_rgb], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([thumb_rgb], [1], None, [16], [0, 256])
        hist_b = cv2.calcHist([thumb_rgb], [2], None, [16], [0, 256])
        hist = np.concatenate([hist_r, hist_g, hist_b]).flatten().astype(np.float32)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        self._prev_hist = hist
