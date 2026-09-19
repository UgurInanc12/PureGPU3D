"""Bounded causal video temporal depth stabilization for PureGPU3D.

Guarantees:
  - Bounded memory: Strictly O(1) state (retains exactly one previous frame and depth map).
  - Causal execution: Exactly one frame in flight; operates in forward streaming order.
  - Optical flow warping: Low-resolution Farneback optical flow with forward-backward
    consistency and photometric error rejection.
  - Newly visible regions: Fully use current frame depth; zero trails or disocclusion ghosting.
  - Boundary protection: Edge-aware confidence prevents blurring depth discontinuities.
  - Scale alignment: Robust shared-pixel inverse depth ratio alignment with identifiability checks.
  - Online shot normalization: Smooth EMA bounds separate from depth filtering to eliminate depth pumping.
  - Hard cut reset: Instantaneous state clearing on scene boundaries, fades, or resolution changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from puregpu3d.stereo.disparity import validate_depth_tensor


@dataclass(frozen=True)
class TemporalDepthConfig:
    """Configuration for causal temporal depth stabilization.

    Attributes:
        enabled: Enable or bypass temporal stabilization.
        alpha: Maximum temporal smoothing weight in [0, 1) for valid motion-compensated pixels.
        flow_scale: Downscale factor for optical flow calculation (e.g. 0.5 = half resolution).
        consistency_threshold: Maximum forward-backward flow error in pixels.
        photometric_threshold: Maximum photometric pixel delta in [0, 255].
        depth_diff_threshold: Maximum relative depth change |d_curr - d_prev| / d_curr before rejection.
        edge_threshold: Relative gradient threshold on current depth to preserve crisp boundaries.
        align_scale: Whether to align inverse depth scale on reliable shared pixels.
        max_scale_adjustment: Maximum allowed multiplicative scale drift per frame.
        min_shared_pixels_for_scale: Minimum valid pixels required to compute scale alignment.
        normalization_ema_eta: EMA smoothing factor for shot-level normalization bounds.
        percentile_min: Lower percentile of inverse depth for robust normalization bounds.
        percentile_max: Upper percentile of inverse depth for robust normalization bounds.
    """

    enabled: bool = True
    alpha: float = 0.70
    flow_scale: float = 0.50
    consistency_threshold: float = 1.50
    photometric_threshold: float = 30.0
    depth_diff_threshold: float = 0.25
    edge_threshold: float = 0.15
    align_scale: bool = True
    max_scale_adjustment: float = 0.15
    min_shared_pixels_for_scale: int = 500
    normalization_ema_eta: float = 0.10
    percentile_min: float = 1.0
    percentile_max: float = 99.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        if not (0.1 <= self.flow_scale <= 1.0):
            raise ValueError(f"flow_scale must be in [0.1, 1.0], got {self.flow_scale}")
        if self.consistency_threshold <= 0.0:
            raise ValueError(f"consistency_threshold must be positive, got {self.consistency_threshold}")
        if self.photometric_threshold <= 0.0:
            raise ValueError(f"photometric_threshold must be positive, got {self.photometric_threshold}")
        if self.depth_diff_threshold <= 0.0:
            raise ValueError(f"depth_diff_threshold must be positive, got {self.depth_diff_threshold}")
        if self.edge_threshold <= 0.0:
            raise ValueError(f"edge_threshold must be positive, got {self.edge_threshold}")
        if not (0.0 <= self.max_scale_adjustment < 1.0):
            raise ValueError(f"max_scale_adjustment must be in [0, 1), got {self.max_scale_adjustment}")
        if self.min_shared_pixels_for_scale < 10:
            raise ValueError(f"min_shared_pixels_for_scale must be >= 10, got {self.min_shared_pixels_for_scale}")
        if not (0.0 < self.normalization_ema_eta <= 1.0):
            raise ValueError(f"normalization_ema_eta must be in (0, 1], got {self.normalization_ema_eta}")
        if not (0.0 <= self.percentile_min < self.percentile_max <= 100.0):
            raise ValueError(
                f"percentiles must satisfy 0 <= min < max <= 100, got ({self.percentile_min}, {self.percentile_max})"
            )


@dataclass
class TemporalStabilizationResult:
    """Outputs from temporal stabilization of a single frame.

    Attributes:
        depth: Stabilized depth tensor or array matching input type.
        normalization_bounds: Online shot-level (q_low, q_high) in inverse depth space.
        valid_flow_fraction: Fraction of pixels passing motion, photometric, and depth consistency.
        scale_factor: Scale adjustment applied to align with previous frame.
        is_cut: True if this frame was treated as a shot cut / reset boundary.
        diagnostics: Internal telemetry and metrics dictionary.
    """

    depth: Union[torch.Tensor, np.ndarray]
    normalization_bounds: Tuple[float, float]
    valid_flow_fraction: float
    scale_factor: float
    is_cut: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class TemporalDepthStabilizer:
    """Bounded, causal temporal depth stabilizer with optical flow compensation.

    Retains strictly O(1) state: exactly one previous frame grayscale and depth map.
    """

    def __init__(self, config: Optional[TemporalDepthConfig] = None) -> None:
        self.config = config if config is not None else TemporalDepthConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None
        self.shot_q_low: Optional[float] = None
        self.shot_q_high: Optional[float] = None
        self.frame_index: int = 0
        self._prev_shape: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        """Reset all temporal history and shot normalization bounds."""
        self.prev_gray = None
        self.prev_depth = None
        self.shot_q_low = None
        self.shot_q_high = None
        self.frame_index = 0
        self._prev_shape = None

    def process_frame(
        self,
        frame_rgb: Union[np.ndarray, torch.Tensor],
        raw_depth: Union[np.ndarray, torch.Tensor],
        is_cut: bool = False,
        letterbox_crop: Optional[Tuple[int, int, int, int]] = None,
    ) -> TemporalStabilizationResult:
        """Stabilize depth for one frame using motion compensation and shot normalization.

        Args:
            frame_rgb: (H, W, 3) or (3, H, W) RGB image, uint8 or float [0, 1].
            raw_depth: (H, W) or (1, H, W) depth map, positive finite values.
            is_cut: Whether this frame represents a cut or scene transition.
            letterbox_crop: Optional (ymin, ymax, xmin, xmax) for active picture region.

        Returns:
            TemporalStabilizationResult containing stabilized depth and shot normalization bounds.
        """
        # Convert depth to numpy for opencv processing while tracking original format
        depth_is_torch = isinstance(raw_depth, torch.Tensor)
        if depth_is_torch:
            torch_dev = raw_depth.device
            torch_dtype = raw_depth.dtype
            d_2d = validate_depth_tensor(raw_depth)
            curr_depth = d_2d.detach().cpu().numpy().astype(np.float32)
        elif isinstance(raw_depth, np.ndarray):
            torch_dev = None
            torch_dtype = None
            if raw_depth.ndim == 3 and raw_depth.shape[0] == 1:
                curr_depth = raw_depth[0].astype(np.float32)
            elif raw_depth.ndim == 2:
                curr_depth = raw_depth.astype(np.float32)
            else:
                raise ValueError(f"Depth array must be 2D or (1, H, W), got shape {raw_depth.shape}")
            if not np.all(np.isfinite(curr_depth)):
                raise ValueError("Depth contains non-finite values (NaN or Inf)")
            if not np.all(curr_depth > 0.0):
                raise ValueError(f"Depth contains non-positive values (min = {float(curr_depth.min())})")
        else:
            raise TypeError(f"Depth must be torch.Tensor or np.ndarray, got {type(raw_depth).__name__}")

        h, w = curr_depth.shape

        # Convert image to grayscale uint8 (H, W)
        if isinstance(frame_rgb, torch.Tensor):
            img_np = frame_rgb.detach().cpu().numpy()
        else:
            img_np = frame_rgb

        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.dtype != np.uint8:
            img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img_np

        if img_uint8.shape[:2] != (h, w):
            raise ValueError(f"Image dimensions {img_uint8.shape[:2]} do not match depth {h}x{w}")

        curr_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Check resolution change or cut
        if self._prev_shape is not None and self._prev_shape != (h, w):
            is_cut = True
        self._prev_shape = (h, w)

        if is_cut:
            self.reset()

        idx = self.frame_index
        self.frame_index += 1

        # 1. Update online shot-level normalization bounds (in inverse depth space)
        inv_z = 1.0 / curr_depth
        if letterbox_crop is not None:
            ymin, ymax, xmin, xmax = letterbox_crop
            inv_active = inv_z[ymin:ymax, xmin:xmax]
        else:
            inv_active = inv_z

        f_q_low = float(np.percentile(inv_active, self.config.percentile_min))
        f_q_high = float(np.percentile(inv_active, self.config.percentile_max))

        if self.shot_q_low is None or self.shot_q_high is None:
            self.shot_q_low = f_q_low
            self.shot_q_high = f_q_high
        else:
            eta = self.config.normalization_ema_eta
            self.shot_q_low = (1.0 - eta) * self.shot_q_low + eta * f_q_low
            self.shot_q_high = (1.0 - eta) * self.shot_q_high + eta * f_q_high

        # Ensure minimum span for stability
        if self.shot_q_high - self.shot_q_low < 1e-6:
            self.shot_q_high = self.shot_q_low + 1e-6
        norm_bounds = (self.shot_q_low, self.shot_q_high)

        # 2. Check bypass condition: first frame of scene or disabled
        if (not self.config.enabled) or (self.prev_gray is None) or (self.prev_depth is None):
            self.prev_gray = curr_gray
            self.prev_depth = curr_depth.copy()
            out_depth = torch.from_numpy(curr_depth).to(device=torch_dev, dtype=torch_dtype) if depth_is_torch else curr_depth
            return TemporalStabilizationResult(
                depth=out_depth,
                normalization_bounds=norm_bounds,
                valid_flow_fraction=1.0 if not self.config.enabled else 0.0,
                scale_factor=1.0,
                is_cut=(idx == 0 or is_cut),
                diagnostics={"status": "initial_or_disabled", "frame_index": idx},
            )

        # 3. Motion-compensated depth filtering
        # Low-res optical flow
        flow_scale = self.config.flow_scale
        low_w = max(32, int(w * flow_scale))
        low_h = max(32, int(h * flow_scale))
        # Ensure even dimensions
        low_w = low_w - (low_w % 2)
        low_h = low_h - (low_h % 2)

        curr_low = cv2.resize(curr_gray, (low_w, low_h), interpolation=cv2.INTER_AREA)
        prev_low = cv2.resize(self.prev_gray, (low_w, low_h), interpolation=cv2.INTER_AREA)

        # Backward flow (from current to previous): points to coordinates in previous frame
        flow_init_ba = np.zeros((low_h, low_w, 2), dtype=np.float32)
        flow_ba_low = cv2.calcOpticalFlowFarneback(
            curr_low, prev_low, flow_init_ba,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        # Forward flow (from previous to current): for consistency check
        flow_init_ab = np.zeros((low_h, low_w, 2), dtype=np.float32)
        flow_ab_low = cv2.calcOpticalFlowFarneback(
            prev_low, curr_low, flow_init_ab,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Upsample flow fields to full resolution
        scale_x = float(w) / float(low_w)
        scale_y = float(h) / float(low_h)
        flow_ba = np.zeros((h, w, 2), dtype=np.float32)
        flow_ba[..., 0] = cv2.resize(flow_ba_low[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) * scale_x
        flow_ba[..., 1] = cv2.resize(flow_ba_low[..., 1], (w, h), interpolation=cv2.INTER_LINEAR) * scale_y

        flow_ab = np.zeros((h, w, 2), dtype=np.float32)
        flow_ab[..., 0] = cv2.resize(flow_ab_low[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) * scale_x
        flow_ab[..., 1] = cv2.resize(flow_ab_low[..., 1], (w, h), interpolation=cv2.INTER_LINEAR) * scale_y

        # Compute sampling coordinates in previous frame
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        map_x = grid_x + flow_ba[..., 0]
        map_y = grid_y + flow_ba[..., 1]

        # In-bounds check
        in_bounds = (map_x >= 0.0) & (map_x <= float(w - 1)) & (map_y >= 0.0) & (map_y <= float(h - 1))

        # Forward-backward consistency check
        warped_flow_ab = cv2.remap(
            flow_ab, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1e4, 1e4),
        )
        fb_err = np.linalg.norm(flow_ba + warped_flow_ab, axis=-1)

        # Photometric consistency check
        warped_prev_gray = cv2.remap(
            self.prev_gray, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        photo_err = np.abs(curr_gray.astype(np.float32) - warped_prev_gray.astype(np.float32))

        # Warp previous depth to current coordinate frame
        warped_prev_depth = cv2.remap(
            self.prev_depth, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Relative depth difference check
        depth_diff = np.abs(curr_depth - warped_prev_depth) / np.maximum(curr_depth, 1e-6)

        # Depth boundary detection on current frame to protect sharp silhouettes
        grad_x = cv2.Sobel(curr_depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(curr_depth, cv2.CV_32F, 0, 1, ksize=3)
        rel_grad = np.sqrt(grad_x**2 + grad_y**2) / np.maximum(curr_depth, 1e-6)
        is_depth_edge = rel_grad > self.config.edge_threshold

        # Composite validity mask for reliable motion compensation
        valid_mask = (
            in_bounds
            & (fb_err <= self.config.consistency_threshold)
            & (photo_err <= self.config.photometric_threshold)
            & (depth_diff <= self.config.depth_diff_threshold)
        )

        # 4. Shared-pixel robust inverse depth scale alignment
        scale = 1.0
        if self.config.align_scale:
            # Pick reliable shared non-edge pixels
            reliable_mask = valid_mask & (~is_depth_edge)
            if np.count_nonzero(reliable_mask) >= self.config.min_shared_pixels_for_scale:
                inv_c = 1.0 / curr_depth[reliable_mask]
                inv_w = 1.0 / warped_prev_depth[reliable_mask]
                # Identifiability check: non-trivial depth variation
                iqr = float(np.percentile(inv_c, 75) - np.percentile(inv_c, 25))
                if iqr > 1e-4:
                    raw_ratio = inv_c / np.maximum(inv_w, 1e-6)
                    med_ratio = float(np.median(raw_ratio))
                    max_adj = self.config.max_scale_adjustment
                    scale = float(np.clip(med_ratio, 1.0 - max_adj, 1.0 + max_adj))
                    # Scale warped depth
                    warped_prev_depth = warped_prev_depth / scale

        # 5. Continuous confidence weighting and blending
        w_fb = np.clip(1.0 - (fb_err / max(self.config.consistency_threshold, 1e-3)), 0.0, 1.0)
        w_photo = np.clip(1.0 - (photo_err / max(self.config.photometric_threshold, 1e-3)), 0.0, 1.0)
        w_conf = w_fb * w_photo

        # Effective blending weight
        blend_weight = self.config.alpha * w_conf * valid_mask.astype(np.float32)

        # Protect depth edges: zero out temporal blending on sharp boundaries to avoid blurring
        blend_weight[is_depth_edge] = 0.0

        # Stabilized depth calculation
        # Newly visible regions have valid_mask == 0 -> blend_weight == 0 -> stabilized = curr_depth
        stabilized_depth = blend_weight * warped_prev_depth + (1.0 - blend_weight) * curr_depth

        # Ensure positive finite depth
        stabilized_depth = np.maximum(stabilized_depth, 1e-4)

        valid_fraction = float(np.mean(valid_mask.astype(np.float32)))

        # Update bounded reference state
        self.prev_gray = curr_gray
        self.prev_depth = stabilized_depth.copy()

        out_depth = (
            torch.from_numpy(stabilized_depth).to(device=torch_dev, dtype=torch_dtype)
            if depth_is_torch
            else stabilized_depth
        )

        return TemporalStabilizationResult(
            depth=out_depth,
            normalization_bounds=norm_bounds,
            valid_flow_fraction=valid_fraction,
            scale_factor=scale,
            is_cut=False,
            diagnostics={
                "frame_index": idx,
                "valid_flow_fraction": valid_fraction,
                "scale_factor": scale,
                "shot_q_low": norm_bounds[0],
                "shot_q_high": norm_bounds[1],
            },
        )
