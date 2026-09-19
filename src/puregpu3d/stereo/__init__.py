"""Stereoscopic reprojection and rendering slice for PureGPU3D.

Integrates:
  - Signed depth-to-disparity with configurable convergence plane and limits.
  - Occlusion-aware subpixel forward splatting with z-buffer depth rejection.
  - Conservative background-aware disocclusion hole filling with explicit diagnostics.
  - Full-resolution Side-by-Side (SBS) stereoscopic output (2W x H).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from puregpu3d.stereo.depth_temporal import (
    TemporalDepthConfig,
    TemporalDepthStabilizer,
    TemporalStabilizationResult,
)
from puregpu3d.stereo.disparity import (
    DisparityConfig,
    DisparityResult,
    compute_disparity,
    validate_depth_tensor,
)
from puregpu3d.stereo.fill import (
    FillConfig,
    FillResult,
    HoleDiagnostics,
    fill_holes,
)
from puregpu3d.stereo.splat import (
    SplatConfig,
    SplatViewResult,
    forward_splat,
    validate_image_tensor,
)


@dataclass
class StereoConfig:
    """Consolidated configuration for stereo reprojection and rendering.

    Attributes:
        disparity: Disparity computation settings.
        splat: Subpixel splatting and occlusion settings.
        fill: Hole filling and diagnostic settings.
    """

    disparity: DisparityConfig = field(default_factory=DisparityConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)
    fill: FillConfig = field(default_factory=FillConfig)


@dataclass
class StereoFrameResult:
    """Complete outputs from stereoscopic rendering of a single frame.

    Attributes:
        left_color: Rendered left-eye color (3, H, W) or (H, W, 3).
        right_color: Rendered right-eye color (3, H, W) or (H, W, 3).
        sbs_color: Full-resolution Side-by-Side color (3, H, 2W) or (H, 2W, 3).
        left_coverage_before_fill: Boolean mask of left-eye coverage prior to filling.
        right_coverage_before_fill: Boolean mask of right-eye coverage prior to filling.
        left_depth: Depth buffer for left view (H, W).
        right_depth: Depth buffer for right view (H, W).
        disparity: Detailed DisparityResult instance.
        left_diagnostics: Hole diagnostics for left view.
        right_diagnostics: Hole diagnostics for right view.
        is_uint8: True if returned color arrays are uint8 [0, 255].
    """

    left_color: Union[torch.Tensor, np.ndarray]
    right_color: Union[torch.Tensor, np.ndarray]
    sbs_color: Union[torch.Tensor, np.ndarray]
    left_coverage_before_fill: torch.Tensor
    right_coverage_before_fill: torch.Tensor
    left_depth: torch.Tensor
    right_depth: torch.Tensor
    disparity: DisparityResult
    left_diagnostics: HoleDiagnostics
    right_diagnostics: HoleDiagnostics
    is_uint8: bool


def render_stereo_frame(
    image: Union[torch.Tensor, np.ndarray],
    depth: Union[torch.Tensor, np.ndarray],
    config: Optional[StereoConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_numpy: bool = False,
    output_uint8: bool = True,
    normalization_bounds: Optional[Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor], Any]] = None,
    compute_diagnostics: bool = True,
) -> StereoFrameResult:
    """Render a stereoscopic left/right pair and Side-by-Side (SBS) composite frame.

    Args:
        image: Source RGB image as (H, W, 3) or (3, H, W), uint8 [0, 255] or float [0, 1].
        depth: Source depth map as (H, W) or (1, H, W), float32/64, positive finite values.
        config: StereoConfig parameters. If None, default configuration is used.
        device: Target execution device. If None, inferred from inputs or defaults to CUDA if available.
        return_numpy: If True, returns image arrays as numpy ndarrays (H, W, 3) and (H, 2W, 3).
        output_uint8: If True, returns color representations scaled to uint8 [0, 255].
        normalization_bounds: Optional external (q_low, q_high) normalization bounds in inverse depth space.
                              Overrides per-frame quantiles to prevent temporal depth pumping.
        compute_diagnostics: If True, computes detailed disparity and hole diagnostics synchronously.
                             If False, defers diagnostic scalar synchronization to avoid host stalls.

    Returns:
        StereoFrameResult containing left, right, and SBS views with coverage diagnostics.
    """
    if config is None:
        config = StereoConfig()

    # Determine execution device
    if device is None:
        if isinstance(image, torch.Tensor):
            target_device = image.device
        elif isinstance(depth, torch.Tensor):
            target_device = depth.device
        else:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # Convert image to torch.Tensor if numpy
    orig_is_uint8 = False
    orig_is_hwc = False
    if isinstance(image, np.ndarray):
        orig_is_hwc = (image.ndim == 3 and image.shape[2] == 3)
        orig_is_uint8 = (image.dtype == np.uint8)
        img_tensor = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        orig_is_hwc = (image.ndim == 3 and image.shape[2] == 3)
        orig_is_uint8 = (image.dtype == torch.uint8)
        img_tensor = image
    else:
        raise TypeError(f"Image must be a torch.Tensor or numpy.ndarray, got {type(image).__name__}")

    # Convert depth to torch.Tensor if numpy
    if isinstance(depth, np.ndarray):
        depth_tensor = torch.from_numpy(depth)
    elif isinstance(depth, torch.Tensor):
        depth_tensor = depth
    else:
        raise TypeError(f"Depth must be a torch.Tensor or numpy.ndarray, got {type(depth).__name__}")

    img_tensor = img_tensor.to(target_device)
    depth_tensor = depth_tensor.to(target_device)

    # Validate inputs
    img_float, was_uint8 = validate_image_tensor(img_tensor)
    depth_2d = validate_depth_tensor(depth_tensor)

    c, h, w = img_float.shape
    if depth_2d.shape != (h, w):
        raise ValueError(f"Depth shape {depth_2d.shape} does not match image shape ({h}, {w})")

    # -------------------------------------------------------------
    # Zero Strength Short-Circuit: Byte-exact source identity
    # -------------------------------------------------------------
    if config.disparity.strength == 0.0:
        disp_res = compute_disparity(
            depth_2d,
            config.disparity,
            normalization_bounds=normalization_bounds,
            compute_diagnostics=compute_diagnostics,
        )
        ones_mask = torch.ones((h, w), dtype=torch.bool, device=target_device)
        empty_diag = HoleDiagnostics(
            total_hole_pixels=0 if compute_diagnostics else None,
            hole_fraction=0.0 if compute_diagnostics else None,
            max_hole_width=0 if compute_diagnostics else None,
            large_hole_count=0 if compute_diagnostics else None,
            warning=None,
            deferred=not compute_diagnostics,
        )

        if output_uint8:
            if was_uint8 and isinstance(image, torch.Tensor) and image.device == target_device and not orig_is_hwc:
                left_out = image.clone()
                right_out = image.clone()
            elif was_uint8 and isinstance(image, np.ndarray) and not orig_is_hwc:
                left_out = torch.from_numpy(image).to(target_device)
                right_out = left_out.clone()
            else:
                uint8_color = (img_float * 255.0).round().clamp(0, 255).to(torch.uint8)
                left_out = uint8_color.clone()
                right_out = uint8_color.clone()
            sbs_out = torch.cat([left_out, right_out], dim=2)
        else:
            left_out = img_float.clone()
            right_out = img_float.clone()
            sbs_out = torch.cat([left_out, right_out], dim=2)

        if return_numpy:
            # Convert (3, H, W) to (H, W, 3) and (3, H, 2W) to (H, 2W, 3)
            left_final = left_out.permute(1, 2, 0).detach().cpu().numpy()
            right_final = right_out.permute(1, 2, 0).detach().cpu().numpy()
            sbs_final = sbs_out.permute(1, 2, 0).detach().cpu().numpy()
        else:
            if orig_is_hwc:
                left_final = left_out.permute(1, 2, 0)
                right_final = right_out.permute(1, 2, 0)
                sbs_final = sbs_out.permute(1, 2, 0)
            else:
                left_final = left_out
                right_final = right_out
                sbs_final = sbs_out

        return StereoFrameResult(
            left_color=left_final,
            right_color=right_final,
            sbs_color=sbs_final,
            left_coverage_before_fill=ones_mask,
            right_coverage_before_fill=ones_mask.clone(),
            left_depth=depth_2d.clone(),
            right_depth=depth_2d.clone(),
            disparity=disp_res,
            left_diagnostics=empty_diag,
            right_diagnostics=empty_diag,
            is_uint8=output_uint8,
        )

    # -------------------------------------------------------------
    # 1. Disparity computation
    # -------------------------------------------------------------
    disp_res = compute_disparity(
        depth_2d,
        config.disparity,
        normalization_bounds=normalization_bounds,
        compute_diagnostics=compute_diagnostics,
    )

    # -------------------------------------------------------------
    # 2. Forward Splatting (left and right views)
    # -------------------------------------------------------------
    left_splat = forward_splat(img_float, depth_2d, disp_res.shift_left, config.splat)
    right_splat = forward_splat(img_float, depth_2d, disp_res.shift_right, config.splat)

    # -------------------------------------------------------------
    # 3. Conservative Background-Aware Hole Filling
    # -------------------------------------------------------------
    left_fill = fill_holes(
        left_splat.color,
        left_splat.depth,
        left_splat.coverage_mask,
        config.fill,
        compute_diagnostics=compute_diagnostics,
    )
    right_fill = fill_holes(
        right_splat.color,
        right_splat.depth,
        right_splat.coverage_mask,
        config.fill,
        compute_diagnostics=compute_diagnostics,
    )

    # -------------------------------------------------------------
    # 4. SBS Composition & Output Formatting
    # -------------------------------------------------------------
    left_color = left_fill.color
    right_color = right_fill.color

    # Concatenate horizontally: (3, H, W) + (3, H, W) -> (3, H, 2W)
    sbs_tensor = torch.cat([left_color, right_color], dim=2)

    if output_uint8:
        left_out = (left_color * 255.0).round().clamp(0, 255).to(torch.uint8)
        right_out = (right_color * 255.0).round().clamp(0, 255).to(torch.uint8)
        sbs_out = (sbs_tensor * 255.0).round().clamp(0, 255).to(torch.uint8)
    else:
        left_out = left_color
        right_out = right_color
        sbs_out = sbs_tensor

    if return_numpy:
        left_final = left_out.permute(1, 2, 0).detach().cpu().numpy()
        right_final = right_out.permute(1, 2, 0).detach().cpu().numpy()
        sbs_final = sbs_out.permute(1, 2, 0).detach().cpu().numpy()
    else:
        if orig_is_hwc:
            left_final = left_out.permute(1, 2, 0)
            right_final = right_out.permute(1, 2, 0)
            sbs_final = sbs_out.permute(1, 2, 0)
        else:
            left_final = left_out
            right_final = right_out
            sbs_final = sbs_out

    return StereoFrameResult(
        left_color=left_final,
        right_color=right_final,
        sbs_color=sbs_final,
        left_coverage_before_fill=left_splat.coverage_mask,
        right_coverage_before_fill=right_splat.coverage_mask,
        left_depth=left_fill.depth,
        right_depth=right_fill.depth,
        disparity=disp_res,
        left_diagnostics=left_fill.diagnostics,
        right_diagnostics=right_fill.diagnostics,
        is_uint8=output_uint8,
    )


__all__ = [
    "DisparityConfig",
    "DisparityResult",
    "compute_disparity",
    "validate_depth_tensor",
    "SplatConfig",
    "SplatViewResult",
    "forward_splat",
    "validate_image_tensor",
    "FillConfig",
    "FillResult",
    "HoleDiagnostics",
    "fill_holes",
    "StereoConfig",
    "StereoFrameResult",
    "render_stereo_frame",
    "TemporalDepthConfig",
    "TemporalDepthStabilizer",
    "TemporalStabilizationResult",
]
