"""Bounded GPU device-memory ring buffer and decode prefetch scheduling for PureGPU3D.

Constraints & Safety Contracts:
  - Bounded ring slots: fixed preallocated device tensors (default 3 slots), strictly bounded.
  - Zero full-frame host transfers: all frames remain strictly on GPU device memory.
  - Safe surface reuse contract:
      PyNvVideoCodec SimpleDecoder reuses internal borrowed surfaces across dec[i] calls.
      Device-to-device copy into owned slot.tensor on the decode stream is strictly required
      and preserved.
  - Asynchronous cross-stream synchronization via CUDA events:
      * ready_event: recorded on decode_stream after DtoD copy; compute_stream waits on it.
      * consumed_event: recorded on compute_stream after compute kernels reading slot.tensor
        are submitted; decode_stream waits on it before overwriting the slot.
  - Producer-consumer ownership:
      * Producer worker thread owns SimpleDecoder calls and decode_stream.
      * Consumer thread owns DA3 inference, temporal stabilization, and stereo splatting.
      * Consumer maintains temporal frame ordering and rational CFR presentation.
  - Clean error propagation and worker joining:
      Cancellation and exceptions propagate promptly to consumer; worker threads are joined
      with timeout, leaving no dangling threads or orphaned memory.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GpuRingSlot:
    """Individual slot in the bounded GPU device memory ring buffer."""

    slot_id: int
    tensor: torch.Tensor
    ready_event: Any
    consumed_event: Any
    has_been_used: bool = False


class GpuDecodePrefetchRing:
    """Bounded GPU decode prefetch ring buffer with cross-stream event scheduling."""

    def __init__(
        self,
        decoder: Any,
        target_device: torch.device,
        in_height: int,
        in_width: int,
        total_frames: int,
        num_slots: int = 3,
        decode_stream: Optional[Any] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        if num_slots < 2:
            raise ValueError(f"num_slots must be at least 2 for pipelining, got {num_slots}")

        self.decoder = decoder
        self.target_device = target_device
        self.in_height = in_height
        self.in_width = in_width
        self.total_frames = total_frames
        self.num_slots = num_slots
        self.cancel_callback = cancel_callback

        self.decode_stream: Any = decode_stream or torch.cuda.Stream(device=target_device)

        # Preallocate fixed device ring slots
        self.slots: List[GpuRingSlot] = []
        with torch.cuda.stream(self.decode_stream):
            for i in range(num_slots):
                t = torch.empty((in_height, in_width, 3), dtype=torch.uint8, device=target_device)
                self.slots.append(
                    GpuRingSlot(
                        slot_id=i,
                        tensor=t,
                        ready_event=torch.cuda.Event(),
                        consumed_event=torch.cuda.Event(),
                        has_been_used=False,
                    )
                )

        # Bounded queues for flow control
        self.free_slots: queue.Queue[int] = queue.Queue(maxsize=num_slots)
        for i in range(num_slots):
            self.free_slots.put(i)

        # Ready queue passes (frame_idx, slot_idx, plane_ptr, raw_ptr, ptrs_match, decode_wall_s)
        self.ready_queue: queue.Queue[
            Optional[Tuple[int, int, int, int, bool, float]]
        ] = queue.Queue(maxsize=num_slots)

        self.stop_event = threading.Event()
        self.error_holder: List[Exception] = []
        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background decode producer thread."""
        self.worker_thread = threading.Thread(
            target=self._producer_loop,
            name="gpu-decode-prefetch-producer",
            daemon=True,
        )
        self.worker_thread.start()

    def _producer_loop(self) -> None:
        """Background worker: decodes frames, performs DtoD copy, and records ready events."""
        try:
            with torch.cuda.stream(self.decode_stream):
                for frame_idx in range(self.total_frames):
                    if self.stop_event.is_set():
                        break
                    if self.cancel_callback is not None and self.cancel_callback():
                        self.stop_event.set()
                        break

                    # Acquire a free slot index with timeout to observe stop_event
                    slot_idx: Optional[int] = None
                    while not self.stop_event.is_set():
                        try:
                            slot_idx = self.free_slots.get(timeout=0.1)
                            break
                        except queue.Empty:
                            if self.cancel_callback is not None and self.cancel_callback():
                                self.stop_event.set()
                                break
                            continue

                    if slot_idx is None or self.stop_event.is_set():
                        break

                    slot = self.slots[slot_idx]

                    # If slot was used previously, wait on GPU until compute finished reading it
                    if slot.has_been_used:
                        self.decode_stream.wait_event(slot.consumed_event)

                    # NVDEC decode
                    t_d0 = time.perf_counter()
                    dec_frame = self.decoder[frame_idx]
                    t_d1 = time.perf_counter()
                    decode_wall_s = t_d1 - t_d0

                    # DLPack zero-copy pointer extraction
                    plane_ptr = int(dec_frame.GetPtrToPlane(0))
                    t_raw = torch.from_dlpack(dec_frame)
                    raw_ptr = int(t_raw.data_ptr())
                    ptrs_match = (plane_ptr == raw_ptr)

                    # Device-to-device copy into owned preallocated slot tensor
                    slot.tensor.copy_(t_raw)
                    slot.ready_event.record(self.decode_stream)
                    slot.has_been_used = True

                    item = (frame_idx, slot_idx, plane_ptr, raw_ptr, ptrs_match, decode_wall_s)

                    # Enqueue ready item with timeout to observe stop_event
                    while not self.stop_event.is_set():
                        try:
                            self.ready_queue.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            if self.cancel_callback is not None and self.cancel_callback():
                                self.stop_event.set()
                                break
                            continue

        except Exception as exc:
            if not self.stop_event.is_set():
                logger.error("Decode prefetch worker encountered error: %s", exc)
                self.error_holder.append(exc)
        finally:
            # Enqueue sentinel to signal completion
            try:
                self.ready_queue.put(None, timeout=1.0)
            except Exception:
                pass

    def acquire_next_frame(
        self, timeout: float = 30.0
    ) -> Optional[Tuple[int, int, int, int, bool, float]]:
        """Acquire the next decoded frame descriptor from the producer queue."""
        if self.error_holder:
            raise self.error_holder[0]

        start_t = time.perf_counter()
        while not self.stop_event.is_set():
            if self.error_holder:
                raise self.error_holder[0]
            try:
                item = self.ready_queue.get(timeout=0.1)
                return item
            except queue.Empty:
                if (time.perf_counter() - start_t) > timeout:
                    raise TimeoutError(f"GpuDecodePrefetchRing timed out waiting for frame ({timeout}s)")
                continue

        if self.error_holder:
            raise self.error_holder[0]
        return None

    def release_slot(self, slot_idx: int, compute_stream: Any) -> None:
        """Consumer signals that compute kernels reading slot have been submitted."""
        slot = self.slots[slot_idx]
        slot.consumed_event.record(compute_stream)
        self.free_slots.put(slot_idx)

    def close(self) -> None:
        """Stop background worker and join thread safely."""
        self.stop_event.set()
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

    def __enter__(self) -> GpuDecodePrefetchRing:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
