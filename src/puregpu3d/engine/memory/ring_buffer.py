from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from time import perf_counter
from typing import Any

from ...config.types import OverflowPolicy


@dataclass(slots=True)
class SurfaceSlot:
    slot_id: int
    in_frame: Any | None = None
    out_frame: Any | None = None


class RingBuffer:
    def __init__(self, size: int, overflow_policy: OverflowPolicy) -> None:
        if size < 2:
            raise ValueError("RingBuffer size must be >= 2")
        if overflow_policy != OverflowPolicy.BLOCK:
            raise ValueError("This engine version supports only BLOCK overflow policy")
        self._slots = [SurfaceSlot(slot_id=i) for i in range(size)]
        self._available: Queue[int] = Queue(maxsize=size)
        for i in range(size):
            self._available.put(i)

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def depth(self) -> int:
        return self.capacity - self._available.qsize()

    def acquire(self, timeout_s: float | None = None) -> tuple[SurfaceSlot, float]:
        start = perf_counter()
        slot_id = self._available.get(block=True, timeout=timeout_s)
        wait_ms = (perf_counter() - start) * 1000.0
        slot = self._slots[slot_id]
        slot.in_frame = None
        slot.out_frame = None
        return slot, wait_ms

    def release(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        slot.in_frame = None
        slot.out_frame = None
        self._available.put(slot_id)
