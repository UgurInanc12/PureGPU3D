from __future__ import annotations

import numpy as np

from ...config.types import DepthConfig


def _split_nv12(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (height * 3 // 2, width)
    if frame.shape != expected_shape:
        raise ValueError(f"NV12 frame shape mismatch. Expected {expected_shape}, got {frame.shape}")
    y_plane = frame[:height, :]
    uv_plane = frame[height:, :]
    return y_plane, uv_plane


def compute_disparity_map(y_plane: np.ndarray, depth_cfg: DepthConfig) -> np.ndarray:
    h, _ = y_plane.shape
    edge_w, luma_w, vertical_w = depth_cfg.normalized_weights()

    y_i16 = y_plane.astype(np.int16)
    grad_x = np.abs(np.diff(y_i16, axis=1, prepend=y_i16[:, :1]))
    grad_y = np.abs(np.diff(y_i16, axis=0, prepend=y_i16[:1, :]))

    edge = np.clip((grad_x + grad_y).astype(np.float32) / 510.0, 0.0, 1.0)
    luma = y_plane.astype(np.float32) / 255.0
    vertical = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

    score = (edge * edge_w) + (luma * luma_w) + (vertical * vertical_w)
    score = np.clip(score * depth_cfg.depth_strength, 0.0, 1.0)

    disparity = np.rint(score * float(depth_cfg.max_disparity_px)).astype(np.int16)
    return np.clip(disparity, 0, depth_cfg.max_disparity_px)


def convert_nv12_to_sbs_cpu(
    frame: np.ndarray,
    width: int,
    height: int,
    depth_cfg: DepthConfig,
) -> np.ndarray:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("NV12 conversion requires even width/height")

    y_plane, uv_plane = _split_nv12(frame, width, height)
    disparity = compute_disparity_map(y_plane, depth_cfg)

    x = np.arange(width, dtype=np.int16)[None, :]
    left_src = np.clip(x - disparity, 0, width - 1)
    right_src = np.clip(x + disparity, 0, width - 1)

    left_y = np.take_along_axis(y_plane, left_src, axis=1)
    right_y = np.take_along_axis(y_plane, right_src, axis=1)
    out_y = np.concatenate((left_y, right_y), axis=1)

    uv_pairs = uv_plane.reshape(height // 2, width // 2, 2)
    disparity_uv = (disparity[::2, ::2] // 2).astype(np.int16)
    x_uv = np.arange(width // 2, dtype=np.int16)[None, :]

    left_uv_src = np.clip(x_uv - disparity_uv, 0, (width // 2) - 1)
    right_uv_src = np.clip(x_uv + disparity_uv, 0, (width // 2) - 1)

    left_uv = np.take_along_axis(uv_pairs, left_uv_src[..., None], axis=1)
    right_uv = np.take_along_axis(uv_pairs, right_uv_src[..., None], axis=1)
    out_uv = np.concatenate((left_uv, right_uv), axis=1).reshape(height // 2, width * 2)

    out = np.empty((height * 3 // 2, width * 2), dtype=np.uint8)
    out[:height, :] = out_y
    out[height:, :] = out_uv
    return out


def make_black_nv12(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    frame[height:, :] = 128
    return frame
