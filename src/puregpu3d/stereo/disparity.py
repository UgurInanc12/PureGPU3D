"""Signed depth-to-disparity mapping for PureGPU3D stereo renderer.

Conventions and contracts:
  - Canonical depth z: positive finite distance (z > 0), where smaller z is nearer.
  - Normalized inverse depth q in [0, 1]: q = robust_norm(1 / z), where q=1 is nearest.
  - Parallax zero plane: q_screen in [0, 1]. Points with q > q_screen are nearer than
    the screen plane (d > 0); points with q < q_screen are behind the screen (d < 0).
  - Signed disparity convention:
      d = clamp(strength * image_width * (q - q_screen), lower_limit, upper_limit)
      x_left  = x + d / 2
      x_right = x - d / 2
      d = x_left - x_right  (near positive)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class DisparityConfig:
    """Configuration for depth-to-disparity computation.

    Attributes:
        strength: Disparity scale as a fraction of image width (e.g. 0.03 = 3% of width).
                  Setting strength=0.0 produces zero disparity everywhere.
        q_screen: Screen convergence plane in normalized inverse depth space [0, 1].
                  Default 0.6 places the majority of scene content behind the screen.
        percentile_min: Lower percentile of inverse depth for robust normalization.
        percentile_max: Upper percentile of inverse depth for robust normalization.
        max_disparity_fraction: Hard limit on positive disparity as fraction of image width.
        min_disparity_fraction: Hard limit on negative disparity as fraction of image width.
                                If None, defaults to -max_disparity_fraction.
        normalization_bounds: Optional external normalization bounds (q_low, q_high) in
                              inverse depth space. When provided, replaces per-frame
                              percentile computation to avoid temporal depth pumping.
    """

    strength: float = 0.03
    q_screen: float = 0.6
    percentile_min: float = 1.0
    percentile_max: float = 99.0
    max_disparity_fraction: float = 0.05
    min_disparity_fraction: Optional[float] = None
    normalization_bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"Disparity strength must be non-negative, got {self.strength}")
        if not (0.0 <= self.q_screen <= 1.0):
            raise ValueError(f"q_screen must be in [0, 1], got {self.q_screen}")
        if not (0.0 <= self.percentile_min < self.percentile_max <= 100.0):
            raise ValueError(
                f"Invalid percentiles: {self.percentile_min} must be < {self.percentile_max} in [0, 100]"
            )
        if self.max_disparity_fraction <= 0.0:
            raise ValueError(
                f"max_disparity_fraction must be positive, got {self.max_disparity_fraction}"
            )
        if self.min_disparity_fraction is not None and self.min_disparity_fraction > 0.0:
            raise ValueError(
                f"min_disparity_fraction must be <= 0.0, got {self.min_disparity_fraction}"
            )
        if self.normalization_bounds is not None:
            if len(self.normalization_bounds) != 2:
                raise ValueError(
                    f"normalization_bounds must be a tuple of 2 floats (q_low, q_high), got {self.normalization_bounds}"
                )
            b_low, b_high = float(self.normalization_bounds[0]), float(self.normalization_bounds[1])
            import math
            if not (math.isfinite(b_low) and math.isfinite(b_high)):
                raise ValueError(
                    f"normalization_bounds must be finite floats, got ({b_low}, {b_high})"
                )
            if b_low >= b_high:
                raise ValueError(
                    f"normalization_bounds low ({b_low}) must be strictly less than high ({b_high})"
                )


@dataclass
class DisparityResult:
    """Outputs from depth-to-disparity mapping.

    Attributes:
        disparity: Signed disparity map d = x_left - x_right, shape (H, W).
        q: Normalized inverse depth map in [0, 1], shape (H, W).
        shift_left: Horizontal pixel shift for left eye (+d / 2), shape (H, W).
        shift_right: Horizontal pixel shift for right eye (-d / 2), shape (H, W).
        q_screen: Convergence plane value used.
        d_min: Minimum disparity value in pixels (None if deferred/fastpath).
        d_max: Maximum disparity value in pixels (None if deferred/fastpath).
        normalization_bounds: Normalization bounds (q_low, q_high) in inverse depth space.
    """

    disparity: torch.Tensor
    q: torch.Tensor
    shift_left: torch.Tensor
    shift_right: torch.Tensor
    q_screen: float
    d_min: Optional[float] = None
    d_max: Optional[float] = None
    normalization_bounds: Optional[Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]]] = None


def validate_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    """Validate depth tensor properties defensively.

    Raises:
        TypeError: If depth is not a torch.Tensor or not floating point.
        ValueError: If depth has invalid dimensions, non-finite values, or non-positive values.
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Depth must be a torch.Tensor, got {type(depth).__name__}")
    if not depth.is_floating_point():
        raise TypeError(f"Depth tensor must be floating-point, got {depth.dtype}")
    if depth.ndim not in (2, 3):
        raise ValueError(f"Depth tensor must have 2 or 3 dimensions (H, W) or (1, H, W), got shape {depth.shape}")
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(f"3D depth tensor must have leading channel 1, got shape {depth.shape}")
        depth_2d = depth.squeeze(0)
    else:
        depth_2d = depth

    if depth_2d.shape[0] < 1 or depth_2d.shape[1] < 1:
        raise ValueError(f"Depth tensor dimensions must be >= 1, got shape {depth_2d.shape}")

    # Combined finite and positive validation: single GPU reduction
    is_valid = torch.all(torch.isfinite(depth_2d) & (depth_2d > 0.0))
    if not is_valid:
        if not torch.all(torch.isfinite(depth_2d)):
            raise ValueError("Depth tensor contains non-finite values (NaN or Inf)")
        min_val = float(depth_2d.min().item())
        raise ValueError(f"Depth tensor contains non-positive values (min depth = {min_val} <= 0)")

    return depth_2d


def compute_disparity(
    depth: torch.Tensor,
    config: Optional[DisparityConfig] = None,
    normalization_bounds: Optional[Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]]] = None,
    compute_diagnostics: bool = True,
) -> DisparityResult:
    """Compute signed stereo disparity from canonical positive depth map.

    Args:
        depth: Canonical depth tensor (H, W) or (1, H, W) where depth > 0.
        config: DisparityConfig parameters. If None, default config is used.
        normalization_bounds: Optional (q_low, q_high) bounds for inverse depth normalization.
                              Can be float tuple or CUDA scalar tensors.
                              If provided or set in config, bypasses per-frame quantile calculation.
        compute_diagnostics: If True, computes d_min, d_max, and strict bounds validation synchronously.
                             If False, defers/omits per-frame scalar extraction to avoid host stalls.

    Returns:
        DisparityResult with disparity, normalized inverse depth q, eye shifts, and bounds.
    """
    if config is None:
        config = DisparityConfig()

    depth_2d = validate_depth_tensor(depth)
    h, w = depth_2d.shape
    device = depth_2d.device

    # Resolve effective normalization bounds
    effective_bounds = normalization_bounds if normalization_bounds is not None else config.normalization_bounds
    if effective_bounds is not None:
        if len(effective_bounds) != 2:
            raise ValueError(f"normalization_bounds must have length 2, got {effective_bounds}")
        if compute_diagnostics:
            b_low, b_high = float(effective_bounds[0]), float(effective_bounds[1])
            import math
            if not (math.isfinite(b_low) and math.isfinite(b_high)):
                raise ValueError(f"normalization_bounds must be finite floats, got ({b_low}, {b_high})")
            if b_low >= b_high:
                raise ValueError(f"normalization_bounds low ({b_low}) must be < high ({b_high})")
            effective_bounds = (b_low, b_high)

    # Zero strength short-circuit: exact zero disparity
    if config.strength == 0.0:
        zeros = torch.zeros((h, w), device=device, dtype=depth_2d.dtype)
        inv_z = 1.0 / depth_2d
        if effective_bounds is not None:
            q_low, q_high = effective_bounds
        else:
            if compute_diagnostics:
                z_min = float(inv_z.min().item())
                z_max = float(inv_z.max().item())
                q_low, q_high = z_min, z_max
            else:
                q_low, q_high = inv_z.min(), inv_z.max()

        denom = q_high - q_low
        if isinstance(denom, torch.Tensor):
            is_flat = denom.abs() <= 1e-7
            denom_safe = torch.where(is_flat, torch.ones_like(denom), denom)
            q_calc = torch.clamp((inv_z - q_low) / denom_safe, 0.0, 1.0)
            q = torch.where(is_flat, torch.full((h, w), config.q_screen, device=device, dtype=depth_2d.dtype), q_calc)
        else:
            if abs(denom) < 1e-7:
                q = torch.full((h, w), config.q_screen, device=device, dtype=depth_2d.dtype)
            else:
                q = torch.clamp((inv_z - q_low) / denom, 0.0, 1.0)

        if compute_diagnostics:
            d_min: Optional[float] = 0.0
            d_max: Optional[float] = 0.0
            if isinstance(q_low, torch.Tensor) and isinstance(q_high, torch.Tensor):
                norm_bounds_out: Any = (float(q_low.item()), float(q_high.item()))
            else:
                norm_bounds_out = (float(q_low), float(q_high))
        else:
            d_min = None
            d_max = None
            norm_bounds_out = (q_low, q_high)

        return DisparityResult(
            disparity=zeros,
            q=q,
            shift_left=zeros.clone(),
            shift_right=zeros.clone(),
            q_screen=config.q_screen,
            d_min=d_min,
            d_max=d_max,
            normalization_bounds=norm_bounds_out,
        )

    # Compute inverse depth: small z (near) -> large inv_z
    inv_z = 1.0 / depth_2d

    if effective_bounds is not None:
        q_low, q_high = effective_bounds
    else:
        p_min = config.percentile_min / 100.0
        p_max = config.percentile_max / 100.0
        inv_flat = inv_z.view(-1)
        if inv_flat.numel() == 1:
            q_low_t = inv_flat[0]
            q_high_t = inv_flat[0]
        else:
            q_low_t = torch.quantile(inv_flat, p_min)
            q_high_t = torch.quantile(inv_flat, p_max)

        if compute_diagnostics:
            q_low = float(q_low_t.item())
            q_high = float(q_high_t.item())
        else:
            q_low = q_low_t
            q_high = q_high_t

    denom = q_high - q_low
    if isinstance(denom, torch.Tensor):
        is_flat = denom.abs() <= 1e-7
        denom_safe = torch.where(is_flat, torch.ones_like(denom), denom)
        q_calc = torch.clamp((inv_z - q_low) / denom_safe, 0.0, 1.0)
        q = torch.where(is_flat, torch.full((h, w), config.q_screen, device=device, dtype=depth_2d.dtype), q_calc)
    else:
        if denom <= 1e-7:
            q = torch.full((h, w), config.q_screen, device=device, dtype=depth_2d.dtype)
        else:
            q = torch.clamp((inv_z - q_low) / denom, 0.0, 1.0)

    # Disparity: d = strength * width * (q - q_screen)
    # Near points (q > q_screen) have d > 0
    # Far points (q < q_screen) have d < 0
    raw_disparity = config.strength * float(w) * (q - config.q_screen)

    max_disp = config.max_disparity_fraction * float(w)
    min_disp = (
        config.min_disparity_fraction * float(w)
        if config.min_disparity_fraction is not None
        else -max_disp
    )

    disparity = torch.clamp(raw_disparity, min=min_disp, max=max_disp)

    # Symmetric eye shifts:
    # Left eye: x_left = x + d / 2
    # Right eye: x_right = x - d / 2
    # So x_left - x_right = d
    shift_left = disparity / 2.0
    shift_right = -disparity / 2.0

    if compute_diagnostics:
        d_min = float(disparity.min().item())
        d_max = float(disparity.max().item())
        if isinstance(q_low, torch.Tensor) and isinstance(q_high, torch.Tensor):
            norm_bounds_res: Any = (float(q_low.item()), float(q_high.item()))
        else:
            norm_bounds_res = (float(q_low), float(q_high))
    else:
        d_min = None
        d_max = None
        norm_bounds_res = (q_low, q_high)

    return DisparityResult(
        disparity=disparity,
        q=q,
        shift_left=shift_left,
        shift_right=shift_right,
        q_screen=config.q_screen,
        d_min=d_min,
        d_max=d_max,
        normalization_bounds=norm_bounds_res,
    )
