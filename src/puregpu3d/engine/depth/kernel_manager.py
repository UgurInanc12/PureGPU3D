from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...config.types import DepthConfig
from .cpu_kernel import convert_nv12_to_sbs_cpu


@dataclass(slots=True)
class KernelTimingBreakdown:
    h2d_ms: float = 0.0
    kernel_compute_ms: float = 0.0
    d2h_ms: float = 0.0


@dataclass(slots=True)
class KernelManager:
    prefer_cuda: bool = True
    backend_name: str = "cpu_numpy"
    _cuda_available: bool = False
    _cp: Any | None = None
    _cuda_init_error: str | None = None
    _raw_module: Any | None = None
    _disparity_kernel: Any | None = None
    _y_kernel: Any | None = None
    _uv_kernel: Any | None = None
    _cached_resolution: tuple[int, int] | None = None
    _in_gpu: Any | None = None
    _out_gpu: Any | None = None
    _disparity_gpu: Any | None = None
    _pinned_output_mem: Any | None = None
    _pinned_output_view: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.prefer_cuda:
            return

        try:
            cp = __import__("cupy")
            if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                return

            _ = cp.arange(1, dtype=cp.float32)
            _.sum().item()

            source = self.kernel_source_path.read_text(encoding="utf-8")
            raw_module = cp.RawModule(
                code=source,
                options=("--std=c++11",),
                name_expressions=(
                    "nv12_disparity_kernel",
                    "nv12_sbs_y_kernel",
                    "nv12_sbs_uv_kernel",
                ),
            )

            self._cp = cp
            self._raw_module = raw_module
            self._disparity_kernel = raw_module.get_function("nv12_disparity_kernel")
            self._y_kernel = raw_module.get_function("nv12_sbs_y_kernel")
            self._uv_kernel = raw_module.get_function("nv12_sbs_uv_kernel")
            self._cuda_available = True
            self.backend_name = "cupy_cuda_raw"
            self._cuda_init_error = None
        except Exception as exc:
            self._disable_cuda(exc)

    @property
    def kernel_source_path(self) -> Path:
        return Path(__file__).with_name("kernels.cu")

    @property
    def cuda_init_error(self) -> str | None:
        return self._cuda_init_error

    def _disable_cuda(self, exc: Exception) -> None:
        self._cuda_available = False
        self.backend_name = "cpu_numpy"
        self._cuda_init_error = str(exc)
        self._cp = None
        self._raw_module = None
        self._disparity_kernel = None
        self._y_kernel = None
        self._uv_kernel = None
        self._cached_resolution = None
        self._in_gpu = None
        self._out_gpu = None
        self._disparity_gpu = None
        self._pinned_output_mem = None
        self._pinned_output_view = None

    def _ensure_gpu_buffers(self, width: int, height: int) -> None:
        assert self._cp is not None
        cp = self._cp
        if self._cached_resolution == (width, height):
            return
        self._cached_resolution = (width, height)
        self._in_gpu = cp.empty((height * 3 // 2, width), dtype=cp.uint8)
        self._out_gpu = cp.empty((height * 3 // 2, width * 2), dtype=cp.uint8)
        self._disparity_gpu = cp.empty((height, width), dtype=cp.int16)

        self._pinned_output_mem = None
        self._pinned_output_view = None
        element_count = (height * 3 // 2) * (width * 2)
        try:
            self._pinned_output_mem = cp.cuda.alloc_pinned_memory(element_count)
            self._pinned_output_view = np.frombuffer(
                self._pinned_output_mem,
                dtype=np.uint8,
                count=element_count,
            ).reshape((height * 3 // 2, width * 2))
        except Exception:
            self._pinned_output_mem = None
            self._pinned_output_view = None

    def _process_nv12_cupy_with_timing(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        depth_cfg: DepthConfig,
    ) -> tuple[np.ndarray, KernelTimingBreakdown]:
        assert self._cp is not None
        assert self._disparity_kernel is not None
        assert self._y_kernel is not None
        assert self._uv_kernel is not None
        cp = self._cp

        if width % 2 != 0 or height % 2 != 0:
            raise ValueError("NV12 conversion requires even width/height")
        if frame.shape != (height * 3 // 2, width):
            raise ValueError(
                f"NV12 frame shape mismatch. Expected {(height * 3 // 2, width)}, got {frame.shape}"
            )

        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        self._ensure_gpu_buffers(width, height)
        assert self._in_gpu is not None
        assert self._out_gpu is not None
        assert self._disparity_gpu is not None

        h2d_start = cp.cuda.Event()
        h2d_end = cp.cuda.Event()
        kernel_start = cp.cuda.Event()
        kernel_end = cp.cuda.Event()
        d2h_start = cp.cuda.Event()
        d2h_end = cp.cuda.Event()

        h2d_start.record()
        self._in_gpu.set(frame)
        h2d_end.record()
        h2d_end.synchronize()
        h2d_ms = float(cp.cuda.get_elapsed_time(h2d_start, h2d_end))

        src_y = self._in_gpu[:height, :]
        src_uv = self._in_gpu[height:, :]
        dst_y = self._out_gpu[:height, :]
        dst_uv = self._out_gpu[height:, :]

        edge_w, luma_w, vertical_w = depth_cfg.normalized_weights()
        block = (32, 8, 1)
        grid_y = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1], 1)
        grid_uv = (
            ((width // 2) + block[0] - 1) // block[0],
            ((height // 2) + block[1] - 1) // block[1],
            1,
        )

        disparity_args = (
            src_y,
            self._disparity_gpu,
            np.int32(width),
            np.int32(height),
            np.int32(depth_cfg.max_disparity_px),
            np.float32(depth_cfg.depth_strength),
            np.float32(edge_w),
            np.float32(luma_w),
            np.float32(vertical_w),
        )
        y_args = (
            src_y,
            self._disparity_gpu,
            dst_y,
            np.int32(width),
            np.int32(height),
        )
        uv_args = (
            src_uv,
            self._disparity_gpu,
            dst_uv,
            np.int32(width),
            np.int32(height),
        )

        kernel_start.record()
        self._disparity_kernel(grid_y, block, disparity_args)
        self._y_kernel(grid_y, block, y_args)
        self._uv_kernel(grid_uv, block, uv_args)
        kernel_end.record()
        kernel_end.synchronize()
        kernel_compute_ms = float(cp.cuda.get_elapsed_time(kernel_start, kernel_end))

        d2h_start.record()
        if self._pinned_output_view is not None:
            self._out_gpu.get(out=self._pinned_output_view)
            out_frame = self._pinned_output_view.copy()
        else:
            out_frame = self._out_gpu.get()
        d2h_end.record()
        d2h_end.synchronize()
        d2h_ms = float(cp.cuda.get_elapsed_time(d2h_start, d2h_end))

        return out_frame, KernelTimingBreakdown(
            h2d_ms=h2d_ms,
            kernel_compute_ms=kernel_compute_ms,
            d2h_ms=d2h_ms,
        )

    def process_nv12_with_timing(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        depth_cfg: DepthConfig,
    ) -> tuple[np.ndarray, KernelTimingBreakdown]:
        if self._cuda_available:
            try:
                return self._process_nv12_cupy_with_timing(frame, width, height, depth_cfg)
            except Exception as exc:
                self._disable_cuda(exc)

        cpu_started = perf_counter()
        out = convert_nv12_to_sbs_cpu(frame, width, height, depth_cfg)
        cpu_ms = (perf_counter() - cpu_started) * 1000.0
        return out, KernelTimingBreakdown(
            h2d_ms=0.0,
            kernel_compute_ms=cpu_ms,
            d2h_ms=0.0,
        )

    def process_nv12(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        depth_cfg: DepthConfig,
    ) -> np.ndarray:
        out_frame, _ = self.process_nv12_with_timing(frame, width, height, depth_cfg)
        return out_frame
