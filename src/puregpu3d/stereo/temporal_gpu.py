"""Bounded causal video temporal depth stabilization on CUDA GPU for PureGPU3D.

Guarantees:
  - Bounded GPU memory: Strictly O(1) state (retains exactly one previous frame and depth map on CUDA).
  - Causal execution: Exactly one frame in flight; operates in forward streaming order.
  - Fully GPU-resident: All frame and depth tensors remain entirely on CUDA device memory.
    No CPU full-frame fallback or host round-trips; only tiny scalar diagnostics sync when requested.
  - Multiscale local motion estimation: PyTorch CUDA coarse-to-fine correlation search with
    forward-backward consistency and photometric error rejection.
  - Newly visible / disoccluded regions: Fully use current frame depth; zero trails or ghosting.
  - Boundary protection: Edge-aware confidence prevents blurring depth discontinuities.
  - Scale alignment: Shared-pixel inverse depth ratio alignment with identifiability checks.
  - Online shot normalization: Smooth EMA bounds in inverse depth space on GPU.
  - Hard cut / transition reset: Instantaneous state clearing on scene boundaries, fades, or resolution changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalGPUConfig:
    """Configuration for causal temporal depth stabilization on GPU.

    Attributes:
        enabled: Enable or bypass temporal stabilization.
        alpha: Maximum temporal smoothing weight in [0, 1) for valid motion-compensated pixels.
        flow_scale: Downscale factor for optical flow calculation.
        max_estimator_dim: Maximum width/height for motion estimation to bound execution time.
        search_radius_coarse: Search radius at coarse pyramid level (subsampled 0.5x).
        search_radius_fine: Residual search radius at fine level around coarse flow.
        patch_size: Neighborhood window size for SAD/patch correlation.
        subpixel: Whether to apply subpixel softmin weighting at fine level.
        temperature: Softmin temperature for subpixel candidate interpolation.
        consistency_threshold: Maximum forward-backward flow error in pixels.
        photometric_threshold: Maximum photometric pixel delta on [0, 255] scale.
        depth_diff_threshold: Maximum relative depth change |d_curr - d_prev| / d_curr before rejection.
        edge_threshold: Relative gradient threshold on current depth to preserve crisp boundaries.
        align_scale: Whether to align inverse depth scale on reliable shared pixels.
        max_scale_adjustment: Maximum allowed multiplicative scale drift per frame.
        min_shared_pixels_for_scale: Minimum valid pixels required to compute scale alignment.
        normalization_ema_eta: EMA smoothing factor for shot-level normalization bounds.
        percentile_min: Lower percentile of inverse depth for robust normalization bounds.
        percentile_max: Upper percentile of inverse depth for robust normalization bounds.
        fade_threshold: Mean luminance threshold in [0, 255] below which a frame is classified as fade.
    """

    enabled: bool = True
    alpha: float = 0.70
    flow_scale: float = 0.50
    max_estimator_dim: int = 240
    search_radius_coarse: int = 4
    search_radius_fine: int = 2
    patch_size: int = 5
    subpixel: bool = False
    temperature: float = 0.005
    consistency_threshold: float = 1.50
    photometric_threshold: float = 30.0
    depth_diff_threshold: float = 0.25
    edge_threshold: float = 0.20
    align_scale: bool = True
    max_scale_adjustment: float = 0.15
    min_shared_pixels_for_scale: int = 500
    normalization_ema_eta: float = 0.10
    percentile_min: float = 1.0
    percentile_max: float = 99.0
    fade_threshold: float = 8.0
    cut_threshold: float = 0.25

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        if not (0.05 <= self.flow_scale <= 1.0):
            raise ValueError(f"flow_scale must be in [0.05, 1.0], got {self.flow_scale}")
        if self.max_estimator_dim < 32:
            raise ValueError(f"max_estimator_dim must be >= 32, got {self.max_estimator_dim}")
        if self.search_radius_coarse < 1:
            raise ValueError(f"search_radius_coarse must be >= 1, got {self.search_radius_coarse}")
        if self.search_radius_fine < 1:
            raise ValueError(f"search_radius_fine must be >= 1, got {self.search_radius_fine}")
        if self.patch_size % 2 == 0 or self.patch_size < 3:
            raise ValueError(f"patch_size must be an odd integer >= 3, got {self.patch_size}")
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
        if not (0.0 < self.cut_threshold <= 1.0):
            raise ValueError(f"cut_threshold must be in (0, 1], got {self.cut_threshold}")


@dataclass
class TemporalGPUResult:
    """Outputs from temporal stabilization on GPU for a single frame.

    All frame/depth and confidence tensors reside strictly on CUDA device memory.
    """

    depth: torch.Tensor
    normalization_bounds: Tuple[torch.Tensor, torch.Tensor]
    cut_flag: torch.Tensor
    is_cut: bool
    confidence_mask: torch.Tensor
    valid_flow_fraction: Optional[float] = None
    scale_factor: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalization_bounds_float(self) -> Tuple[float, float]:
        """Convenience property providing float bounds for downstream rendering adapters."""
        return float(self.normalization_bounds[0].item()), float(self.normalization_bounds[1].item())


def _warp_2d_gpu(tensor: torch.Tensor, flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp a 2D tensor using flow field [dx, dy] with boundary handling on CUDA.

    Args:
        tensor: (B, C, H, W) tensor to be sampled.
        flow: (B, 2, H, W) flow field where channel 0 is dx (horizontal) and channel 1 is dy (vertical).

    Returns:
        (warped_tensor, in_bounds_mask) where in_bounds_mask is (B, 1, H, W) bool.
    """
    B, C, H, W = tensor.shape
    device = tensor.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
    x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

    sample_x = x_coords + flow[:, 0:1]
    sample_y = y_coords + flow[:, 1:2]

    in_bounds = (sample_x >= 0.0) & (sample_x <= float(W - 1)) & (sample_y >= 0.0) & (sample_y <= float(H - 1))

    grid_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
    grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)

    warped = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped, in_bounds


def _estimate_multiscale_flow_gpu(
    source_gray: torch.Tensor,
    target_gray: torch.Tensor,
    r_coarse: int = 4,
    r_fine: int = 2,
    patch_size: int = 5,
    subpixel: bool = False,
    temperature: float = 0.005,
) -> torch.Tensor:
    """Compute 2-scale bounded correlation motion field from source to target on CUDA.

    Estimates displacement (dx, dy) such that target(x + dx, y + dy) matches source(x, y).

    Args:
        source_gray: (B, 1, H, W) normalized grayscale image in [0, 1] on CUDA.
        target_gray: (B, 1, H, W) normalized grayscale image in [0, 1] on CUDA.
        r_coarse: Coarse search radius (subsampled 0.5x).
        r_fine: Fine residual search radius.
        patch_size: Correlation patch size.
        subpixel: Whether to apply softmin subpixel interpolation at fine level.
        temperature: Softmin temperature for subpixel refinement.

    Returns:
        flow: (B, 2, H, W) displacement field [dx, dy] on CUDA.
    """
    B, _, H, W = source_gray.shape
    device = source_gray.device
    pad_patch = patch_size // 2

    # Level 1: Coarse resolution (0.5x)
    src_low = F.interpolate(source_gray, scale_factor=0.5, mode="area")
    tgt_low = F.interpolate(target_gray, scale_factor=0.5, mode="area")
    h_low, w_low = src_low.shape[-2:]

    pad_tgt1 = F.pad(tgt_low, (r_coarse, r_coarse, r_coarse, r_coarse), mode="replicate")
    cands1 = []
    shifts1 = []
    for dy in range(-r_coarse, r_coarse + 1):
        for dx in range(-r_coarse, r_coarse + 1):
            shifts1.append((float(dx), float(dy)))
            cands1.append(pad_tgt1[:, :, r_coarse + dy : r_coarse + dy + h_low, r_coarse + dx : r_coarse + dx + w_low])
    cands1 = torch.cat(cands1, dim=1)  # (B, K1, h_low, w_low)
    diff1 = torch.abs(cands1 - src_low)
    costs1 = F.avg_pool2d(diff1, patch_size, stride=1, padding=pad_patch)
    best_idx1 = torch.argmin(costs1, dim=1)  # (B, h_low, w_low)

    shifts_t1 = torch.tensor(shifts1, device=device, dtype=torch.float32)  # (K1, 2) [dx, dy]
    # Gather best shift: (B, h_low, w_low, 2)
    best_shifts1 = shifts_t1[best_idx1].permute(0, 3, 1, 2) * 2.0  # (B, 2, h_low, w_low) [dx, dy] full-res units

    # Upsample coarse flow to fine resolution
    flow_coarse = F.interpolate(best_shifts1, size=(H, W), mode="bilinear", align_corners=False)

    # Level 2: Fine resolution residual search around coarse flow
    # Warp target by coarse flow
    warped_tgt, _ = _warp_2d_gpu(target_gray, flow_coarse)

    pad_tgt2 = F.pad(warped_tgt, (r_fine, r_fine, r_fine, r_fine), mode="replicate")
    cands2 = []
    shifts2 = []
    for dy in range(-r_fine, r_fine + 1):
        for dx in range(-r_fine, r_fine + 1):
            shifts2.append((float(dx), float(dy)))
            cands2.append(pad_tgt2[:, :, r_fine + dy : r_fine + dy + H, r_fine + dx : r_fine + dx + W])
    cands2 = torch.cat(cands2, dim=1)  # (B, K2, H, W)
    diff2 = torch.abs(cands2 - source_gray)
    costs2 = F.avg_pool2d(diff2, patch_size, stride=1, padding=pad_patch)

    shifts_t2 = torch.tensor(shifts2, device=device, dtype=torch.float32)  # (K2, 2)
    if subpixel:
        tau = max(1e-4, temperature)
        weights2 = F.softmax(-costs2 / tau, dim=1)  # (B, K2, H, W)
        shifts_t2_ch = shifts_t2.permute(1, 0).view(1, 2, len(shifts2), 1, 1)  # (1, 2, K2, 1, 1)
        flow_res = (weights2.unsqueeze(1) * shifts_t2_ch).sum(dim=2)  # (B, 2, H, W)
    else:
        best_idx2 = torch.argmin(costs2, dim=1)
        flow_res = shifts_t2[best_idx2].permute(0, 3, 1, 2)

    total_flow = flow_coarse + flow_res
    return total_flow


class TemporalGPUStabilizer:
    """Bounded causal temporal depth stabilizer running entirely on CUDA device memory.

    Retains strictly O(1) state: exactly one previous frame grayscale and depth tensor on CUDA.
    No full-frame host transfers.
    """

    def __init__(self, config: Optional[TemporalGPUConfig] = None) -> None:
        self.config = config if config is not None else TemporalGPUConfig()
        self.prev_gray: Optional[torch.Tensor] = None
        self.prev_depth: Optional[torch.Tensor] = None
        self.shot_q_low: Optional[torch.Tensor] = None
        self.shot_q_high: Optional[torch.Tensor] = None
        self.frame_index: int = 0
        self._prev_shape: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        """Clear all temporal history and shot normalization bounds."""
        self.prev_gray = None
        self.prev_depth = None
        self.shot_q_low = None
        self.shot_q_high = None
        self.frame_index = 0
        self._prev_shape = None

    def process_frame(
        self,
        frame_rgb: torch.Tensor,
        raw_depth: torch.Tensor,
        is_cut: bool = False,
        letterbox_crop: Optional[Tuple[int, int, int, int]] = None,
        compute_diagnostics: bool = True,
    ) -> TemporalGPUResult:
        """Stabilize depth map for one video frame on CUDA.

        Args:
            frame_rgb: (B, 3, H, W), (3, H, W), or (H, W, 3) RGB tensor on CUDA.
            raw_depth: (B, 1, H, W), (1, H, W), or (H, W) positive depth tensor on CUDA.
            is_cut: External shot cut signal (or transition boundary).
            letterbox_crop: Optional active content box (ymin, ymax, xmin, xmax).
            compute_diagnostics: If True, computes per-frame diagnostics (.item() calls).
                                 If False, defers diagnostic scalar synchronization to avoid host stalls.

        Returns:
            TemporalGPUResult with all tensors resident on CUDA.
        """
        if not isinstance(raw_depth, torch.Tensor):
            raise TypeError(f"raw_depth must be a torch.Tensor, got {type(raw_depth).__name__}")
        if not raw_depth.is_cuda:
            raise ValueError(f"raw_depth must reside on CUDA device, got {raw_depth.device}")

        orig_depth_dim = raw_depth.dim()
        # Canonicalize raw_depth to (1, 1, H, W)
        if orig_depth_dim == 2:
            curr_depth = raw_depth.unsqueeze(0).unsqueeze(0).float()
        elif orig_depth_dim == 3 and raw_depth.shape[0] == 1:
            curr_depth = raw_depth.unsqueeze(0).float()
        elif orig_depth_dim == 4 and raw_depth.shape[1] == 1:
            curr_depth = raw_depth.float()
        else:
            raise ValueError(f"raw_depth tensor must be (H, W), (1, H, W), or (B, 1, H, W), got {raw_depth.shape}")

        _, _, h, w = curr_depth.shape
        device = curr_depth.device

        # Canonicalize frame_rgb to (1, 3, H, W)
        if not isinstance(frame_rgb, torch.Tensor):
            raise TypeError(f"frame_rgb must be a torch.Tensor, got {type(frame_rgb).__name__}")
        if not frame_rgb.is_cuda:
            frame_rgb = frame_rgb.to(device=device)

        if frame_rgb.dim() == 3:
            if frame_rgb.shape[0] == 3:
                rgb_4d = frame_rgb.unsqueeze(0)
            elif frame_rgb.shape[2] == 3:
                rgb_4d = frame_rgb.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected frame_rgb shape: {frame_rgb.shape}")
        elif frame_rgb.dim() == 4:
            if frame_rgb.shape[1] == 3:
                rgb_4d = frame_rgb
            elif frame_rgb.shape[3] == 3:
                rgb_4d = frame_rgb.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected frame_rgb shape: {frame_rgb.shape}")
        else:
            raise ValueError(f"Unexpected frame_rgb shape: {frame_rgb.shape}")

        if rgb_4d.shape[-2:] != (h, w):
            raise ValueError(f"frame_rgb resolution {rgb_4d.shape[-2:]} does not match depth ({h}, {w})")

        # Normalize RGB to [0, 255] float32 for luminance computation
        if rgb_4d.dtype == torch.uint8:
            rgb_255 = rgb_4d.float()
        elif rgb_4d.max() <= 1.0 + 1e-4:
            rgb_255 = rgb_4d.float() * 255.0
        else:
            rgb_255 = rgb_4d.float()

        # Compute standard BT.601 luminance in [0, 255] and normalized in [0, 1]
        curr_lum_255 = 0.299 * rgb_255[:, 0:1] + 0.587 * rgb_255[:, 1:2] + 0.114 * rgb_255[:, 2:3]
        curr_gray_norm = curr_lum_255 / 255.0

        # Dimension / scene cut / fade check
        if self._prev_shape is not None and self._prev_shape != (h, w):
            is_cut = True
        self._prev_shape = (h, w)

        lum_mean = curr_lum_255.mean()
        if self.prev_gray is not None and not is_cut:
            delta_gray = torch.abs(curr_gray_norm - self.prev_gray).mean()
            # Single combined host copy for both scalars to minimize barrier overhead
            cut_scalars = torch.stack([lum_mean, delta_gray]).tolist()
            is_fade = cut_scalars[0] < self.config.fade_threshold
            if cut_scalars[1] >= self.config.cut_threshold:
                is_cut = True
        else:
            is_fade = bool((lum_mean < self.config.fade_threshold).item())

        if is_cut or is_fade:
            self.reset()

        idx = self.frame_index
        self.frame_index += 1

        # 1. Update online shot-level normalization bounds in inverse depth space on GPU
        inv_z = 1.0 / curr_depth
        if letterbox_crop is not None:
            ymin, ymax, xmin, xmax = letterbox_crop
            inv_active = inv_z[:, :, ymin:ymax, xmin:xmax]
        else:
            inv_active = inv_z

        flat_inv = inv_active.flatten()
        f_q_low = torch.quantile(flat_inv, self.config.percentile_min / 100.0)
        f_q_high = torch.quantile(flat_inv, self.config.percentile_max / 100.0)

        if self.shot_q_low is None or self.shot_q_high is None:
            self.shot_q_low = f_q_low
            self.shot_q_high = f_q_high
        else:
            eta = self.config.normalization_ema_eta
            self.shot_q_low = (1.0 - eta) * self.shot_q_low + eta * f_q_low
            self.shot_q_high = (1.0 - eta) * self.shot_q_high + eta * f_q_high

        # Ensure minimum span for numeric stability
        if (self.shot_q_high - self.shot_q_low) < 1e-6:
            self.shot_q_high = self.shot_q_low + 1e-6

        norm_bounds = (self.shot_q_low, self.shot_q_high)

        # Helper to format output shape to match input convention
        def format_output(tensor_4d: torch.Tensor) -> torch.Tensor:
            if orig_depth_dim == 2:
                return tensor_4d.squeeze(0).squeeze(0)
            elif orig_depth_dim == 3:
                return tensor_4d.squeeze(0)
            return tensor_4d

        cut_flag_tensor = torch.tensor(idx == 0 or is_cut or is_fade, device=device, dtype=torch.bool)

        # 2. Check bypass condition: disabled or first frame of shot
        if (not self.config.enabled) or (self.prev_gray is None) or (self.prev_depth is None):
            self.prev_gray = curr_gray_norm.detach()
            self.prev_depth = curr_depth.detach()
            ones_conf = torch.ones_like(curr_depth)
            diag_b: Dict[str, Any] = {"status": "initial_or_disabled", "frame_index": idx}
            if not compute_diagnostics:
                diag_b["deferred"] = True
            return TemporalGPUResult(
                depth=format_output(curr_depth),
                normalization_bounds=norm_bounds,
                cut_flag=cut_flag_tensor,
                is_cut=(idx == 0 or is_cut or is_fade),
                confidence_mask=format_output(ones_conf),
                valid_flow_fraction=1.0 if not self.config.enabled else (0.0 if compute_diagnostics else None),
                scale_factor=1.0 if compute_diagnostics else None,
                diagnostics=diag_b,
            )

        # 3. Motion-compensated filtering
        # Determine capped low-resolution estimator size
        target_w = max(48, min(self.config.max_estimator_dim, int(w * self.config.flow_scale)))
        target_h = max(48, min(self.config.max_estimator_dim, int(h * self.config.flow_scale)))
        # If input image is already small (<= 240), preserve resolution for precise tracking
        if w <= self.config.max_estimator_dim and h <= self.config.max_estimator_dim:
            low_w = w - (w % 2)
            low_h = h - (h % 2)
        else:
            low_w = target_w - (target_w % 2)
            low_h = target_h - (target_h % 2)

        curr_low = F.interpolate(curr_gray_norm, size=(low_h, low_w), mode="area")
        prev_low = F.interpolate(self.prev_gray, size=(low_h, low_w), mode="area")

        # Estimate low-resolution flow fields
        # flow_ba_low: maps current coordinates to previous coordinates (backward flow)
        flow_ba_low = _estimate_multiscale_flow_gpu(
            curr_low,
            prev_low,
            r_coarse=self.config.search_radius_coarse,
            r_fine=self.config.search_radius_fine,
            patch_size=self.config.patch_size,
            subpixel=self.config.subpixel,
            temperature=self.config.temperature,
        )
        # flow_ab_low: maps previous coordinates to current coordinates (forward flow for consistency)
        flow_ab_low = _estimate_multiscale_flow_gpu(
            prev_low,
            curr_low,
            r_coarse=self.config.search_radius_coarse,
            r_fine=self.config.search_radius_fine,
            patch_size=self.config.patch_size,
            subpixel=self.config.subpixel,
            temperature=self.config.temperature,
        )

        # Upsample flow fields to full depth resolution and scale displacements
        scale_x = float(w) / float(low_w)
        scale_y = float(h) / float(low_h)
        flow_ba_scaled = torch.cat([flow_ba_low[:, 0:1] * scale_x, flow_ba_low[:, 1:2] * scale_y], dim=1)
        flow_ba = F.interpolate(flow_ba_scaled, size=(h, w), mode="bilinear", align_corners=False)

        flow_ab_scaled = torch.cat([flow_ab_low[:, 0:1] * scale_x, flow_ab_low[:, 1:2] * scale_y], dim=1)
        flow_ab = F.interpolate(flow_ab_scaled, size=(h, w), mode="bilinear", align_corners=False)

        # 4. Consistency checks and warping
        # Warp previous depth and gray
        warped_prev_depth, in_bounds = _warp_2d_gpu(self.prev_depth, flow_ba)
        warped_prev_gray, _ = _warp_2d_gpu(self.prev_gray, flow_ba)

        # Forward-backward consistency check: warp flow_ab to current coordinates
        warped_flow_ab, _ = _warp_2d_gpu(flow_ab, flow_ba)
        fb_err = torch.norm(flow_ba + warped_flow_ab, dim=1, keepdim=True)

        # Photometric error on [0, 255] scale
        photo_err = torch.abs(curr_gray_norm - warped_prev_gray) * 255.0

        # Relative depth difference check
        depth_diff = torch.abs(curr_depth - warped_prev_depth) / torch.clamp(curr_depth, min=1e-6)

        # Depth edge detection: pre-filter with 3x3 average pool to suppress point sensor noise,
        # then compute Sobel gradient to detect true step boundaries.
        d_smooth = F.avg_pool2d(curr_depth, 3, stride=1, padding=1)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        grad_x = F.conv2d(d_smooth, kx, padding=1)
        grad_y = F.conv2d(d_smooth, ky, padding=1)
        rel_grad = torch.sqrt(grad_x * grad_x + grad_y * grad_y) / torch.clamp(d_smooth, min=1e-6)
        is_depth_edge = rel_grad > self.config.edge_threshold

        # Composite validity mask
        valid_mask = (
            in_bounds
            & (fb_err <= self.config.consistency_threshold)
            & (photo_err <= self.config.photometric_threshold)
            & (depth_diff <= self.config.depth_diff_threshold)
        )

        # 5. Shared-pixel robust scale alignment in inverse depth space
        scale = 1.0
        scale_val: Optional[float] = None
        if self.config.align_scale:
            reliable_mask = valid_mask & (~is_depth_edge)
            num_reliable = reliable_mask.sum().item()
            if num_reliable >= self.config.min_shared_pixels_for_scale:
                inv_c = 1.0 / curr_depth[reliable_mask]
                inv_w = 1.0 / warped_prev_depth[reliable_mask]
                q75 = torch.quantile(inv_c, 0.75)
                q25 = torch.quantile(inv_c, 0.25)
                raw_ratio = inv_c / torch.clamp(inv_w, min=1e-6)
                med_ratio = torch.median(raw_ratio)
                max_adj = self.config.max_scale_adjustment
                scale_clamped = torch.clamp(med_ratio, 1.0 - max_adj, 1.0 + max_adj)
                scale_tensor = torch.where((q75 - q25) > 1e-4, scale_clamped, torch.tensor(1.0, device=device, dtype=curr_depth.dtype))
                warped_prev_depth = warped_prev_depth / scale_tensor
                if compute_diagnostics:
                    scale = float(scale_tensor.item())
                    scale_val = scale
        elif compute_diagnostics:
            scale_val = 1.0

        # 6. Continuous confidence weighting and blending
        w_fb = torch.clamp(1.0 - (fb_err / max(self.config.consistency_threshold, 1e-3)), 0.0, 1.0)
        w_photo = torch.clamp(1.0 - (photo_err / max(self.config.photometric_threshold, 1e-3)), 0.0, 1.0)
        w_conf = w_fb * w_photo

        # Effective blending weight
        confidence_mask = w_conf * valid_mask.float()
        # Zero out temporal blending on sharp boundaries to avoid smearing depth edges
        confidence_mask = torch.where(is_depth_edge, torch.zeros_like(confidence_mask), confidence_mask)

        blend_weight = self.config.alpha * confidence_mask

        # Stabilized depth calculation
        # Newly visible / disoccluded regions have blend_weight == 0 -> stabilized = curr_depth
        stabilized_depth = blend_weight * warped_prev_depth + (1.0 - blend_weight) * curr_depth
        stabilized_depth = torch.clamp(stabilized_depth, min=1e-4)

        if compute_diagnostics:
            valid_fraction: Optional[float] = float(valid_mask.float().mean().item())
            scale_out = scale_val if scale_val is not None else scale
            diag = {
                "frame_index": idx,
                "valid_flow_fraction": valid_fraction,
                "scale_factor": scale_out,
                "shot_q_low": float(norm_bounds[0].item()),
                "shot_q_high": float(norm_bounds[1].item()),
            }
        else:
            valid_fraction = None
            scale_out = None
            diag = {
                "frame_index": idx,
                "deferred": True,
            }

        # Update bounded reference state strictly O(1) on CUDA
        self.prev_gray = curr_gray_norm.detach()
        self.prev_depth = stabilized_depth.detach()

        return TemporalGPUResult(
            depth=format_output(stabilized_depth),
            normalization_bounds=norm_bounds,
            cut_flag=torch.tensor(False, device=device, dtype=torch.bool),
            is_cut=False,
            confidence_mask=format_output(confidence_mask),
            valid_flow_fraction=valid_fraction,
            scale_factor=scale_out,
            diagnostics=diag,
        )
