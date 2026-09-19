"""Occlusion-aware subpixel forward splatting for PureGPU3D stereo renderer.

Implements pure PyTorch vectorized forward splatting:
  - 1D horizontal subpixel splatting (y coordinate unchanged).
  - Two-pass depth-rejection:
      Pass 1: Computes minimum depth (z-buffer) per target pixel via scatter_reduce amin.
      Pass 2: Rejects occluded background contributions exceeding (z_min + tolerance)
              prior to subpixel weighted blending, ensuring near objects win collisions
              without background bleeding or edge ghosting.
  - Explicit boolean coverage masks (independent of color value, validating true black).
  - Preserves exact source dimensions (W per eye).
  - Exact identity short-circuit for zero shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from puregpu3d.stereo.disparity import validate_depth_tensor


@dataclass(frozen=True)
class SplatConfig:
    """Configuration parameters for forward splatting.

    Attributes:
        depth_tolerance: Relative depth margin for occlusion rejection (e.g. 0.05 = 5%).
                         Splats with depth > z_min * (1 + depth_tolerance) + abs_tol are rejected.
        abs_depth_tolerance: Absolute depth margin to prevent precision cutoff on flat surfaces.
        weight_threshold: Minimum accumulated weight to mark a pixel as covered.
    """

    depth_tolerance: float = 0.05
    abs_depth_tolerance: float = 1e-4
    weight_threshold: float = 1e-4

    def __post_init__(self) -> None:
        if self.depth_tolerance < 0.0:
            raise ValueError(f"depth_tolerance must be non-negative, got {self.depth_tolerance}")
        if self.abs_depth_tolerance < 0.0:
            raise ValueError(f"abs_depth_tolerance must be non-negative, got {self.abs_depth_tolerance}")
        if self.weight_threshold <= 0.0:
            raise ValueError(f"weight_threshold must be positive, got {self.weight_threshold}")


@dataclass
class SplatViewResult:
    """Output of forward splatting for one eye view.

    Attributes:
        color: Splatted color image (3, H, W) in float32 [0, 1].
        depth: Splatted depth buffer (H, W) in float32. Holes contain float('inf').
        coverage_mask: Boolean mask (H, W), True where valid splats covered the pixel.
        accum_weights: Raw accumulated weights tensor (H, W).
    """

    color: torch.Tensor
    depth: torch.Tensor
    coverage_mask: torch.Tensor
    accum_weights: torch.Tensor


def validate_image_tensor(image: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Validate image tensor and return (canonical float32 (3, H, W), is_uint8_orig).

    Accepts:
      - (3, H, W) or (H, W, 3)
      - uint8 [0, 255] or floating point [0, 1]
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Image must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"Image must be a 3D tensor, got shape {image.shape}")

    is_uint8 = image.dtype == torch.uint8

    if image.shape[0] == 3:
        # Already (3, H, W)
        img_chw = image
    elif image.shape[2] == 3:
        # (H, W, 3) -> (3, H, W)
        img_chw = image.permute(2, 0, 1)
    else:
        raise ValueError(f"Image must have 3 color channels (either dim 0 or dim 2), got shape {image.shape}")

    if is_uint8:
        img_float = img_chw.to(dtype=torch.float32) / 255.0
    elif image.is_floating_point():
        img_float = img_chw.to(dtype=torch.float32)
        if not torch.all(torch.isfinite(img_float)):
            raise ValueError("Image tensor contains non-finite values (NaN or Inf)")
    else:
        raise TypeError(f"Unsupported image dtype {image.dtype}; expected uint8 or floating-point")

    return img_float, is_uint8


def forward_splat(
    image: torch.Tensor,
    depth: torch.Tensor,
    shift: torch.Tensor,
    config: Optional[SplatConfig] = None,
) -> SplatViewResult:
    """Forward splat source image and depth using horizontal shifts with depth rejection.

    Args:
        image: Color tensor (3, H, W) or (H, W, 3), float32 [0, 1] or uint8 [0, 255].
        depth: Canonical depth tensor (H, W) or (1, H, W), float32/64, positive finite.
        shift: Horizontal pixel shift map (H, W) or (1, H, W), float.
        config: SplatConfig parameters. If None, default config is used.

    Returns:
        SplatViewResult containing splatted color, depth, coverage mask, and accumulated weights.
    """
    if config is None:
        config = SplatConfig()

    img_float, _ = validate_image_tensor(image)
    depth_2d = validate_depth_tensor(depth).to(dtype=torch.float32)

    if not isinstance(shift, torch.Tensor):
        raise TypeError(f"Shift must be a torch.Tensor, got {type(shift).__name__}")
    if not shift.is_floating_point():
        raise TypeError(f"Shift tensor must be floating-point, got {shift.dtype}")

    if shift.ndim == 3 and shift.shape[0] == 1:
        shift_2d = shift.squeeze(0)
    elif shift.ndim == 2:
        shift_2d = shift
    else:
        raise ValueError(f"Shift tensor must have shape (H, W) or (1, H, W), got {shift.shape}")

    c, h, w = img_float.shape
    if depth_2d.shape != (h, w):
        raise ValueError(f"Depth shape {depth_2d.shape} does not match image shape ({h}, {w})")
    if shift_2d.shape != (h, w):
        raise ValueError(f"Shift shape {shift_2d.shape} does not match image shape ({h}, {w})")

    if img_float.device != depth_2d.device or img_float.device != shift_2d.device:
        raise ValueError(
            f"Device mismatch: image on {img_float.device}, depth on {depth_2d.device}, shift on {shift_2d.device}"
        )

    device = img_float.device

    # Zero shift fast-path: exact source identity
    if torch.all(shift_2d == 0.0):
        cov = torch.ones((h, w), dtype=torch.bool, device=device)
        weights = torch.ones((h, w), dtype=torch.float32, device=device)
        return SplatViewResult(
            color=img_float.clone(),
            depth=depth_2d.clone(),
            coverage_mask=cov,
            accum_weights=weights,
        )

    # 1D horizontal subpixel decomposition
    x_coords = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).expand(h, w)
    x_tgt = x_coords + shift_2d

    x0 = torch.floor(x_tgt).long()
    x1 = x0 + 1
    alpha = x_tgt - x0.float()
    w0 = 1.0 - alpha
    w1 = alpha

    # Target validity masks: in-bounds and above weight threshold
    m0 = (x0 >= 0) & (x0 < w) & (w0 > config.weight_threshold)
    m1 = (x1 >= 0) & (x1 < w) & (w1 > config.weight_threshold)

    # Flat indices for 2D scatter operations
    row_idx = torch.arange(h, device=device, dtype=torch.long).unsqueeze(1).expand(h, w)
    idx0_flat = (row_idx * w + x0.clamp(0, w - 1)).view(-1)
    idx1_flat = (row_idx * w + x1.clamp(0, w - 1)).view(-1)

    m0_flat = m0.view(-1)
    m1_flat = m1.view(-1)
    z_flat = depth_2d.view(-1)

    # -------------------------------------------------------------
    # Pass 1: Compute z-buffer (minimum depth) at each target pixel
    # -------------------------------------------------------------
    z_buf_flat = torch.full((h * w,), float("inf"), device=device, dtype=torch.float32)
    if m0_flat.any():
        z_buf_flat.scatter_reduce_(0, idx0_flat[m0_flat], z_flat[m0_flat], reduce="amin", include_self=True)
    if m1_flat.any():
        z_buf_flat.scatter_reduce_(0, idx1_flat[m1_flat], z_flat[m1_flat], reduce="amin", include_self=True)

    z_buf = z_buf_flat.view(h, w)

    # -------------------------------------------------------------
    # Pass 2: Occlusion rejection & Subpixel Weighted Blending
    # -------------------------------------------------------------
    # Candidate splats must satisfy depth <= z_target + tau
    # where tau = depth_tolerance * z_target + abs_depth_tolerance
    tau = config.depth_tolerance * z_buf + config.abs_depth_tolerance

    # Query target minimum depth and threshold for landing 0
    z_tgt0 = z_buf.gather(1, x0.clamp(0, w - 1))
    tau0 = tau.gather(1, x0.clamp(0, w - 1))
    acc0 = m0 & (depth_2d <= (z_tgt0 + tau0))

    # Query target minimum depth and threshold for landing 1
    z_tgt1 = z_buf.gather(1, x1.clamp(0, w - 1))
    tau1 = tau.gather(1, x1.clamp(0, w - 1))
    acc1 = m1 & (depth_2d <= (z_tgt1 + tau1))

    acc0_flat = acc0.view(-1)
    acc1_flat = acc1.view(-1)

    accum_weights = torch.zeros(h * w, device=device, dtype=torch.float32)
    accum_depths = torch.zeros(h * w, device=device, dtype=torch.float32)
    accum_colors = torch.zeros(3, h * w, device=device, dtype=torch.float32)

    # Accumulate landing 0
    if acc0_flat.any():
        w0_acc = (w0 * acc0).view(-1)
        zw0_acc = (depth_2d * w0 * acc0).view(-1)
        accum_weights.scatter_add_(0, idx0_flat[acc0_flat], w0_acc[acc0_flat])
        accum_depths.scatter_add_(0, idx0_flat[acc0_flat], zw0_acc[acc0_flat])
        for ch in range(3):
            cw0_acc = (img_float[ch] * w0 * acc0).view(-1)
            accum_colors[ch].scatter_add_(0, idx0_flat[acc0_flat], cw0_acc[acc0_flat])

    # Accumulate landing 1
    if acc1_flat.any():
        w1_acc = (w1 * acc1).view(-1)
        zw1_acc = (depth_2d * w1 * acc1).view(-1)
        accum_weights.scatter_add_(0, idx1_flat[acc1_flat], w1_acc[acc1_flat])
        accum_depths.scatter_add_(0, idx1_flat[acc1_flat], zw1_acc[acc1_flat])
        for ch in range(3):
            cw1_acc = (img_float[ch] * w1 * acc1).view(-1)
            accum_colors[ch].scatter_add_(0, idx1_flat[acc1_flat], cw1_acc[acc1_flat])

    # -------------------------------------------------------------
    # Normalization & Output Masking
    # -------------------------------------------------------------
    cov_mask_flat = accum_weights > config.weight_threshold
    weights_safe = torch.where(cov_mask_flat, accum_weights, torch.ones_like(accum_weights))

    out_depth_flat = torch.where(
        cov_mask_flat,
        accum_depths / weights_safe,
        torch.full_like(accum_depths, float("inf")),
    )
    out_colors_flat = torch.where(
        cov_mask_flat.unsqueeze(0),
        accum_colors / weights_safe.unsqueeze(0),
        torch.zeros_like(accum_colors),
    )

    return SplatViewResult(
        color=out_colors_flat.view(3, h, w).clamp(0.0, 1.0),
        depth=out_depth_flat.view(h, w),
        coverage_mask=cov_mask_flat.view(h, w),
        accum_weights=accum_weights.view(h, w),
    )
