"""Geometry helpers for aspect-preserving depth scale and patch padding.

PureGPU3D P1 depth processing scale architecture:
- Explicit scale selector: 1/4, 1/2, 1/1 (referring to spatial dimensions).
- Preserves aspect ratio by scaling each spatial dimension independently
  using an explicit half-up rounding contract, then padding upward to
  multiples of PATCH_SIZE (14) instead of stretching.
- Depth predictions are unpadded back to requested dimensions before
  being mapped to original full-resolution video frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import numpy as np

PATCH_SIZE: int = 14

SUPPORTED_DEPTH_SCALES: tuple[str, ...] = ("1/4", "1/2", "1/1")

SCALE_FACTORS: dict[str, float] = {
    "1/4": 0.25,
    "1/2": 0.5,
    "1/1": 1.0,
}


@dataclass(frozen=True)
class DepthGeometry:
    """Canonical geometric specification for depth scale, padding, and unpadding."""

    orig_width: int
    orig_height: int
    scale: str
    scale_factor: float
    req_width: int
    req_height: int
    padded_width: int
    padded_height: int
    pad_right: int
    pad_bottom: int
    patch_size: int = PATCH_SIZE

    @property
    def orig_shape(self) -> Tuple[int, int]:
        """(height, width) of original frame."""
        return (self.orig_height, self.orig_width)

    @property
    def req_shape(self) -> Tuple[int, int]:
        """(height, width) of requested unpadded depth map."""
        return (self.req_height, self.req_width)

    @property
    def padded_shape(self) -> Tuple[int, int]:
        """(height, width) of padded tensor fed to ViT model."""
        return (self.padded_height, self.padded_width)


def parse_depth_scale(scale: Union[str, float]) -> Tuple[str, float]:
    """Parse and validate depth scale string or float factor.

    Supported inputs:
      - String: '1/4', '1/2', '1/1'
      - Float: 0.25, 0.5, 1.0 (with 1e-4 tolerance)

    Returns:
      (canonical_str, factor_float)

    Raises:
      ValueError if the scale is unsupported.
    """
    if isinstance(scale, str):
        cleaned = scale.strip()
        if cleaned in SCALE_FACTORS:
            return cleaned, SCALE_FACTORS[cleaned]
        raise ValueError(
            f"Unsupported depth scale string '{scale}'. Supported: {list(SUPPORTED_DEPTH_SCALES)}"
        )

    if isinstance(scale, (int, float)):
        factor = float(scale)
        for canon, val in SCALE_FACTORS.items():
            if math.isclose(factor, val, abs_tol=1e-4):
                return canon, val
        raise ValueError(
            f"Unsupported depth scale factor {scale}. Supported: {list(SCALE_FACTORS.values())}"
        )

    raise TypeError(f"Depth scale must be str or float, got {type(scale).__name__}")


def compute_requested_dimension(dim: int, factor: float) -> int:
    """Compute requested content dimension with explicit rounding contract.

    Contract:
      1. Validation: dim must be a strictly positive integer (> 0).
      2. Factor scaling: dim * factor.
      3. Explicit rounding: standard arithmetic half-up rounding:
         math.floor(dim * factor + 0.5).
      4. Lower bound: max(1, rounded).
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"Dimension must be a positive integer (> 0), got {dim!r}")
    if factor <= 0:
        raise ValueError(f"Scale factor must be positive (> 0), got {factor!r}")
    return max(1, int(math.floor(dim * factor + 0.5)))


def compute_padded_dimension(dim: int, patch_size: int = PATCH_SIZE) -> int:
    """Compute upward-padded dimension to the next multiple of patch_size.

    Contract:
      padded = ((dim + patch_size - 1) // patch_size) * patch_size
      padded >= dim and padded % patch_size == 0
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"Dimension must be a positive integer (> 0), got {dim!r}")
    if patch_size <= 0:
        raise ValueError(f"Patch size must be positive (> 0), got {patch_size!r}")
    return ((dim + patch_size - 1) // patch_size) * patch_size


def compute_depth_geometry(
    width: int,
    height: int,
    scale: Union[str, float] = "1/2",
    patch_size: int = PATCH_SIZE,
) -> DepthGeometry:
    """Calculate complete DepthGeometry for an image dimension and depth scale.

    Preserves aspect ratio by scaling width and height independently with
    the explicit rounding contract, then pads right and bottom to patch_size multiples.
    """
    canon_scale, factor = parse_depth_scale(scale)
    req_w = compute_requested_dimension(width, factor)
    req_h = compute_requested_dimension(height, factor)

    padded_w = compute_padded_dimension(req_w, patch_size=patch_size)
    padded_h = compute_padded_dimension(req_h, patch_size=patch_size)

    pad_right = padded_w - req_w
    pad_bottom = padded_h - req_h

    return DepthGeometry(
        orig_width=width,
        orig_height=height,
        scale=canon_scale,
        scale_factor=factor,
        req_width=req_w,
        req_height=req_h,
        padded_width=padded_w,
        padded_height=padded_h,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        patch_size=patch_size,
    )


def pad_image_for_depth(
    rgb: np.ndarray,
    geometry: DepthGeometry,
) -> np.ndarray:
    """Resize image to requested dimensions and pad upward to patch_size multiples.

    Resizes rgb image (H, W, 3) to (req_h, req_w) without stretching.
    If padding is required (pad_right > 0 or pad_bottom > 0), pads the bottom and
    right edges using BORDER_REFLECT_101 (or BORDER_REPLICATE for degenerate 1px inputs).

    Returns:
      Padded uint8 image of shape (padded_h, padded_w, 3).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {rgb.shape}")
    orig_h, orig_w = rgb.shape[:2]
    if (orig_w, orig_h) != (geometry.orig_width, geometry.orig_height):
        raise ValueError(
            f"Image dimensions ({orig_w}x{orig_h}) do not match geometry "
            f"({geometry.orig_width}x{geometry.orig_height})"
        )

    # 1. Resize to requested dimensions (aspect-preserving, no stretch)
    if (orig_w, orig_h) == (geometry.req_width, geometry.req_height):
        resized = rgb.copy()
    else:
        interp = cv2.INTER_CUBIC if geometry.scale_factor > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(rgb, (geometry.req_width, geometry.req_height), interpolation=interp)

    # 2. Pad right and bottom upward to patch multiples
    if geometry.pad_right == 0 and geometry.pad_bottom == 0:
        return resized

    border_mode = (
        cv2.BORDER_REFLECT_101
        if (geometry.req_width > 1 and geometry.req_height > 1)
        else cv2.BORDER_REPLICATE
    )
    padded = cv2.copyMakeBorder(
        resized,
        top=0,
        bottom=geometry.pad_bottom,
        left=0,
        right=geometry.pad_right,
        borderType=border_mode,
    )
    return padded


def unpad_depth_map(
    raw_depth: np.ndarray,
    geometry: DepthGeometry,
) -> np.ndarray:
    """Unpad predicted model depth map back to requested content dimensions.

    Slices out raw_depth[:req_h, :req_w], stripping the right and bottom padding.

    Returns:
      Depth array of shape (req_h, req_w).
    """
    if raw_depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {raw_depth.shape}")
    h, w = raw_depth.shape
    if (w, h) != (geometry.padded_width, geometry.padded_height):
        raise ValueError(
            f"Depth map shape ({w}x{h}) does not match padded geometry "
            f"({geometry.padded_width}x{geometry.padded_height})"
        )

    return raw_depth[: geometry.req_height, : geometry.req_width].copy()
