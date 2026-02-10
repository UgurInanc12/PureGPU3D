from __future__ import annotations

import numpy as np

from ...config.types import DepthConfig
from ..depth.kernel_manager import KernelManager, KernelTimingBreakdown


class StereoProcessor:
    def __init__(self, kernel_manager: KernelManager | None = None) -> None:
        self._kernel_manager = kernel_manager or KernelManager(prefer_cuda=True)

    @property
    def backend_name(self) -> str:
        return self._kernel_manager.backend_name

    @property
    def backend_init_error(self) -> str | None:
        return self._kernel_manager.cuda_init_error

    def process_frame(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        depth_cfg: DepthConfig,
    ) -> np.ndarray:
        return self._kernel_manager.process_nv12(frame, width, height, depth_cfg)

    def process_frame_with_timing(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        depth_cfg: DepthConfig,
    ) -> tuple[np.ndarray, KernelTimingBreakdown]:
        return self._kernel_manager.process_nv12_with_timing(frame, width, height, depth_cfg)
