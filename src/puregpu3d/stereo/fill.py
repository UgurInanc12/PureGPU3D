"""Conservative background-aware hole filling for PureGPU3D stereo renderer.

Implements pure PyTorch vectorized horizontal hole filling:
  - Preserves exact raw coverage mask before fill (coverage_before_fill).
  - Determines hole boundaries along each row using vectorized prefix/suffix scans (cummax / cummin).
  - Background-side selection: Compares left vs right boundary depths. Disocclusion holes
    are filled strictly from the demonstrable farther surface (larger z), preventing
    foreground edge bleed into background.
  - Equal-depth interpolation: When boundaries are at approximately equal depth (within tolerance),
    interpolates smoothly between boundaries.
  - Edge extension: Holes reaching image borders are filled conservatively from the available boundary.
  - Large-hole diagnostics: Measures exact hole width and area. Flags holes wider than
    max_hole_width with an explicit diagnostic warning rather than silently hallucinating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HoleDiagnostics:
    """Diagnostic statistics for disocclusion holes in a rendered eye view.

    Attributes:
        total_hole_pixels: Total count of pixels uncovered before fill.
        hole_fraction: Fraction of total image pixels that were holes (total_hole_pixels / (H * W)).
        max_hole_width: Maximum contiguous horizontal hole width in pixels.
        large_hole_count: Number of hole spans exceeding max_hole_width.
        warning: Optional human-readable warning message if large holes are detected.
        deferred: Whether diagnostics were deferred to avoid per-frame synchronization.
    """

    total_hole_pixels: Optional[int] = None
    hole_fraction: Optional[float] = None
    max_hole_width: Optional[int] = None
    large_hole_count: Optional[int] = None
    warning: Optional[str] = None
    deferred: bool = False


@dataclass(frozen=True)
class FillConfig:
    """Configuration for conservative background-aware hole filling.

    Attributes:
        max_hole_width: Threshold in pixels above which a hole is considered a large disocclusion hole.
        depth_relative_tolerance: Margin by which one boundary's depth must exceed the other to be
                                   demonstrably farther (background). E.g. 0.05 = 5%.
        fill_large_holes: If True, fills large holes with conservative background extension while
                          issuing a diagnostic warning. If False, large holes remain unfilled.
    """

    max_hole_width: int = 16
    depth_relative_tolerance: float = 0.05
    fill_large_holes: bool = True

    def __post_init__(self) -> None:
        if self.max_hole_width < 1:
            raise ValueError(f"max_hole_width must be >= 1, got {self.max_hole_width}")
        if self.depth_relative_tolerance < 0.0:
            raise ValueError(
                f"depth_relative_tolerance must be non-negative, got {self.depth_relative_tolerance}"
            )


@dataclass
class FillResult:
    """Outputs from hole filling operation.

    Attributes:
        color: Filled color image (3, H, W) in float32 [0, 1].
        depth: Filled depth buffer (H, W) in float32.
        coverage_before_fill: Raw boolean mask before filling (True where originally covered).
        filled_mask: Boolean mask (H, W) of pixels that were holes and got filled.
        unfilled_mask: Boolean mask (H, W) of pixels that remained unfilled.
        diagnostics: HoleDiagnostics containing hole metrics and any warnings.
    """

    color: torch.Tensor
    depth: torch.Tensor
    coverage_before_fill: torch.Tensor
    filled_mask: torch.Tensor
    unfilled_mask: torch.Tensor
    diagnostics: HoleDiagnostics


def fill_holes(
    color: torch.Tensor,
    depth: torch.Tensor,
    coverage_mask: torch.Tensor,
    config: Optional[FillConfig] = None,
    compute_diagnostics: bool = True,
) -> FillResult:
    """Fill disocclusion holes conservatively from demonstrable background surfaces.

    Args:
        color: Splatted color tensor (3, H, W), float32 [0, 1].
        depth: Splatted depth tensor (H, W), float32.
        coverage_mask: Boolean coverage mask (H, W), True where covered.
        config: FillConfig parameters. If None, default config is used.
        compute_diagnostics: If True, computes detailed hole metrics synchronously.
                             If False, defers diagnostic scalar synchronization to avoid host stalls.

    Returns:
        FillResult with filled color, filled depth, coverage masks, and diagnostics.
    """
    if config is None:
        config = FillConfig()

    if not isinstance(color, torch.Tensor) or not isinstance(depth, torch.Tensor) or not isinstance(coverage_mask, torch.Tensor):
        raise TypeError("color, depth, and coverage_mask must all be torch.Tensors")

    if color.ndim != 3 or color.shape[0] != 3:
        raise ValueError(f"color tensor must have shape (3, H, W), got {color.shape}")
    if depth.ndim != 2:
        raise ValueError(f"depth tensor must have shape (H, W), got {depth.shape}")
    if coverage_mask.ndim != 2:
        raise ValueError(f"coverage_mask tensor must have shape (H, W), got {coverage_mask.shape}")
    if coverage_mask.dtype != torch.bool:
        raise TypeError(f"coverage_mask must have dtype bool, got {coverage_mask.dtype}")

    c, h, w = color.shape
    if depth.shape != (h, w):
        raise ValueError(f"depth shape {depth.shape} does not match color ({h}, {w})")
    if coverage_mask.shape != (h, w):
        raise ValueError(f"coverage_mask shape {coverage_mask.shape} does not match color ({h}, {w})")

    device = color.device
    raw_coverage = coverage_mask.clone()

    # Fully covered shortcut: no holes to fill (synchronous check only if computing diagnostics)
    if compute_diagnostics and torch.all(raw_coverage):
        diag = HoleDiagnostics(
            total_hole_pixels=0,
            hole_fraction=0.0,
            max_hole_width=0,
            large_hole_count=0,
            warning=None,
            deferred=False,
        )
        empty_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
        return FillResult(
            color=color.clone(),
            depth=depth.clone(),
            coverage_before_fill=raw_coverage,
            filled_mask=empty_mask,
            unfilled_mask=empty_mask.clone(),
            diagnostics=diag,
        )

    is_hole = ~raw_coverage
    x_coords = torch.arange(w, device=device, dtype=torch.long).unsqueeze(0).expand(h, w)

    # -----------------------------------------------------------------
    # Vectorized boundary search along rows via cummax and cummin
    # -----------------------------------------------------------------
    # Left boundary: maximum covered index <= x
    idx_left = torch.where(raw_coverage, x_coords, -1)
    left_b, _ = torch.cummax(idx_left, dim=1)

    # Right boundary: minimum covered index >= x
    idx_right = torch.where(raw_coverage, x_coords, w)
    cummin_rev, _ = torch.cummin(idx_right.flip(1), dim=1)
    right_b = cummin_rev.flip(1)

    has_left = (left_b >= 0)
    has_right = (right_b < w)

    # Compute hole widths and diagnostics if requested or needed for gating large holes
    is_large_hole: Optional[torch.Tensor] = None
    if compute_diagnostics or (not config.fill_large_holes):
        w_both = right_b - left_b - 1
        w_left_edge = right_b
        w_right_edge = (w - 1) - left_b
        w_empty = torch.full((h, w), fill_value=w, device=device, dtype=torch.long)

        hole_width = torch.where(
            has_left & has_right,
            w_both,
            torch.where(
                has_right,
                w_left_edge,
                torch.where(has_left, w_right_edge, w_empty),
            ),
        )
        hole_width = torch.where(is_hole, hole_width, torch.zeros_like(hole_width))
        is_large_hole = is_hole & (hole_width > config.max_hole_width)

        if compute_diagnostics:
            total_holes = int(is_hole.sum().item())
            hole_fraction = total_holes / float(h * w)
            max_hole_w = int(hole_width.max().item()) if total_holes > 0 else 0
            prev_is_large = torch.cat(
                [torch.zeros((h, 1), dtype=torch.bool, device=device), is_large_hole[:, :-1]],
                dim=1,
            )
            large_hole_starts = is_large_hole & (~prev_is_large)
            large_hole_count = int(large_hole_starts.sum().item())

            warning: Optional[str] = None
            if large_hole_count > 0:
                warning = (
                    f"Large disocclusion holes detected: max width {max_hole_w}px exceeds threshold "
                    f"{config.max_hole_width}px ({large_hole_count} distinct spans, {total_holes} total pixels, "
                    f"{hole_fraction:.1%})."
                )

            diag = HoleDiagnostics(
                total_hole_pixels=total_holes,
                hole_fraction=hole_fraction,
                max_hole_width=max_hole_w,
                large_hole_count=large_hole_count,
                warning=warning,
                deferred=False,
            )
        else:
            diag = HoleDiagnostics(
                total_hole_pixels=None,
                hole_fraction=None,
                max_hole_width=None,
                large_hole_count=None,
                warning=None,
                deferred=True,
            )
    else:
        diag = HoleDiagnostics(
            total_hole_pixels=None,
            hole_fraction=None,
            max_hole_width=None,
            large_hole_count=None,
            warning=None,
            deferred=True,
        )

    # -----------------------------------------------------------------
    # Boundary depths and colors
    # -----------------------------------------------------------------
    left_b_clamped = left_b.clamp(min=0)
    right_b_clamped = right_b.clamp(max=w - 1)

    z_l = torch.gather(depth, 1, left_b_clamped)
    z_r = torch.gather(depth, 1, right_b_clamped)

    c_l = torch.gather(color, 2, left_b_clamped.unsqueeze(0).expand(3, -1, -1))
    c_r = torch.gather(color, 2, right_b_clamped.unsqueeze(0).expand(3, -1, -1))

    # Demonstrable background selection:
    # In depth z, larger z is farther (background).
    tol = config.depth_relative_tolerance
    both_boundaries = has_left & has_right
    left_farther = both_boundaries & (z_l > z_r * (1.0 + tol))
    right_farther = both_boundaries & (z_r > z_l * (1.0 + tol))
    both_similar = both_boundaries & (~left_farther) & (~right_farther)

    # Linear interpolation when depths are similar
    dist_l = (x_coords - left_b).float()
    dist_r = (right_b - x_coords).float()
    span = (dist_l + dist_r).clamp(min=1.0)
    t = (dist_l / span).clamp(0.0, 1.0)

    c_interp = (1.0 - t).unsqueeze(0) * c_l + t.unsqueeze(0) * c_r
    z_interp = (1.0 - t) * z_l + t * z_r

    # Candidate filled values
    candidate_color = torch.where(
        left_farther.unsqueeze(0),
        c_l,
        torch.where(
            right_farther.unsqueeze(0),
            c_r,
            torch.where(
                both_similar.unsqueeze(0),
                c_interp,
                torch.where(
                    has_left.unsqueeze(0),
                    c_l,
                    torch.where(has_right.unsqueeze(0), c_r, color),
                ),
            ),
        ),
    )

    candidate_depth = torch.where(
        left_farther,
        z_l,
        torch.where(
            right_farther,
            z_r,
            torch.where(
                both_similar,
                z_interp,
                torch.where(has_left, z_l, torch.where(has_right, z_r, depth)),
            ),
        ),
    )

    # -----------------------------------------------------------------
    # Gate filling by config.fill_large_holes
    # -----------------------------------------------------------------
    can_fill = has_left | has_right
    if not config.fill_large_holes and is_large_hole is not None:
        can_fill = can_fill & (~is_large_hole)

    fill_active = is_hole & can_fill
    filled_mask = fill_active
    unfilled_mask = is_hole & (~fill_active)

    out_color = torch.where(fill_active.unsqueeze(0), candidate_color, color)
    out_depth = torch.where(fill_active, candidate_depth, depth)

    return FillResult(
        color=out_color.clamp(0.0, 1.0),
        depth=out_depth,
        coverage_before_fill=raw_coverage,
        filled_mask=filled_mask,
        unfilled_mask=unfilled_mask,
        diagnostics=diag,
    )
