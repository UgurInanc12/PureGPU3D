from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable, Iterable

import numpy as np

from ..config.types import CorruptFrameFallback, EngineConfig, FpsMode
from ..metrics.collector import MetricsCollector
from .codec.base import CodecBackend
from .codec.factory import create_backend
from .depth.cpu_kernel import make_black_nv12
from .memory.ring_buffer import RingBuffer
from .pipeline.processor import StereoProcessor
from .types import TranscodeResult


class TranscodeCancelledError(RuntimeError):
    """Raised when an active transcode is cancelled by user request."""


@dataclass(slots=True)
class _DecodedPacket:
    frame: np.ndarray | None = None
    error: Exception | None = None
    done: bool = False


class StereoEngine:
    def __init__(self, backend: CodecBackend | None = None) -> None:
        self._fixed_backend = backend
        self.last_metrics: MetricsCollector | None = None
        self._active_lock = Lock()
        self._active_backend: CodecBackend | None = None
        self._active_abort_event: Event | None = None

    def _set_active_run(self, backend: CodecBackend, abort_event: Event) -> None:
        with self._active_lock:
            self._active_backend = backend
            self._active_abort_event = abort_event

    def _clear_active_run(self, backend: CodecBackend, abort_event: Event) -> None:
        with self._active_lock:
            if self._active_backend is backend and self._active_abort_event is abort_event:
                self._active_backend = None
                self._active_abort_event = None

    def abort_active_run(self) -> None:
        with self._active_lock:
            backend = self._active_backend
            abort_event = self._active_abort_event
        if abort_event is not None:
            abort_event.set()
        if backend is not None:
            try:
                backend.request_abort()
            except Exception:
                pass

    def _is_cancel_requested(
        self,
        cancel_requested: Callable[[], bool] | None,
        run_abort_event: Event,
    ) -> bool:
        if run_abort_event.is_set():
            return True
        if cancel_requested is None:
            return False
        try:
            requested = bool(cancel_requested())
        except Exception:
            requested = True
        if requested:
            run_abort_event.set()
            return True
        return False

    def _build_fallback_frame(
        self,
        *,
        fallback: CorruptFrameFallback,
        last_good: np.ndarray | None,
        width: int,
        height: int,
    ) -> np.ndarray:
        if fallback == CorruptFrameFallback.LAST_GOOD and last_good is not None:
            return last_good.copy()
        return make_black_nv12(width=width, height=height)

    def _resolve_backend(self, preference: str) -> tuple[CodecBackend, list[str]]:
        if self._fixed_backend is not None:
            return self._fixed_backend, []
        return create_backend(preference=preference)

    def _await_encode_future(
        self,
        future: Future[None],
        *,
        timeout_s: float,
        cancel_requested: Callable[[], bool],
    ) -> None:
        deadline = time.perf_counter() + timeout_s
        while True:
            if cancel_requested():
                raise TranscodeCancelledError("Transcode cancelled by user.")
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("Video encode write timed out")
            try:
                future.result(timeout=min(0.05, remaining))
                return
            except FutureTimeoutError:
                continue
            except TranscodeCancelledError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Video encode write failed: {exc}") from exc

    def _queue_put_with_cancel(
        self,
        q: Queue[_DecodedPacket],
        *,
        packet: _DecodedPacket,
        timeout_s: float,
        cancel_requested: Callable[[], bool],
    ) -> bool:
        deadline = time.perf_counter() + timeout_s
        while True:
            if cancel_requested():
                return False
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            try:
                q.put(packet, timeout=min(0.05, remaining))
                return True
            except Full:
                continue

    def _queue_get_with_cancel(
        self,
        q: Queue[_DecodedPacket],
        *,
        timeout_s: float,
        cancel_requested: Callable[[], bool],
    ) -> _DecodedPacket:
        deadline = time.perf_counter() + timeout_s
        while True:
            if cancel_requested():
                raise TranscodeCancelledError("Transcode cancelled by user.")
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("Decode queue timed out while waiting for next frame.")
            try:
                return q.get(timeout=min(0.05, remaining))
            except Empty:
                continue

    def run_file(
        self,
        config: EngineConfig,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> TranscodeResult:
        config.validate()
        config.ensure_output_parent()

        backend, backend_warnings = self._resolve_backend(config.backend_preference)
        probe = backend.probe(config.input_path)
        total_frames_hint = probe.frame_count
        if total_frames_hint <= 0 and probe.duration_s and probe.fps > 0:
            total_frames_hint = int(round(probe.duration_s * probe.fps))
        out_width = probe.width * 2
        out_height = probe.height

        if out_width % 2 != 0 or out_height % 2 != 0:
            raise ValueError("SBS output requires even dimensions")

        metrics = MetricsCollector()
        self.last_metrics = metrics
        warnings: list[str] = [*backend_warnings]
        warnings.append(f"Encode queue depth: {config.buffer_slots}")
        if probe.bit_depth > 8:
            warnings.append(
                f"Source is {probe.bit_depth}-bit ({probe.pix_fmt}). Current processing path is NV12 8-bit; "
                "exact HDR precision cannot be fully preserved."
            )
        if probe.color_transfer in {"smpte2084", "arib-std-b67"}:
            warnings.append(
                "HDR transfer metadata detected. Color metadata is preserved on output, "
                "but frame processing remains in 8-bit NV12."
            )

        processor = StereoProcessor()
        if processor.backend_name != "cpu_numpy":
            warnings.append(f"Depth backend: {processor.backend_name}")
        elif processor.backend_init_error:
            warnings.append(f"Depth backend fell back to CPU: {processor.backend_init_error}")
        else:
            warnings.append("Depth backend: cpu_numpy")
        ring = RingBuffer(size=config.buffer_slots, overflow_policy=config.overflow_policy)

        temp_video_path = config.output_path.with_suffix(".video_only.mp4")
        if temp_video_path == config.output_path:
            temp_video_path = config.output_path.with_name(config.output_path.stem + ".video_only.mp4")

        writer = backend.open_writer(
            output_path=temp_video_path,
            width=out_width,
            height=out_height,
            fps=probe.fps,
            codec=config.codec,
            video_encoder=config.video_encoder,
            video_bitrate_mbps=config.video_bitrate_mbps,
            source_probe=probe if config.preserve_color_metadata else None,
        )
        selected_encoder = getattr(backend, "last_selected_encoder", None)
        if selected_encoder:
            warnings.append(f"Video encoder: {selected_encoder}")
        if config.video_bitrate_mbps is not None:
            warnings.append(f"Target bitrate: {config.video_bitrate_mbps:.3f} Mbps")

        frames_in = 0
        frames_out = 0
        last_good: np.ndarray | None = None
        pending_writes: deque[Future[None]] = deque()
        max_inflight_writes = max(2, config.buffer_slots)
        encode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="puregpu3d-encode")

        realtime_next_deadline = time.perf_counter()
        frame_interval_s = (1.0 / probe.fps) if (config.fps_mode == FpsMode.REALTIME and probe.fps > 0) else 0.0

        run_abort_event = Event()
        self._set_active_run(backend, run_abort_event)

        cancel_predicate = lambda: self._is_cancel_requested(cancel_requested, run_abort_event)
        decode_queue: Queue[_DecodedPacket] = Queue(maxsize=max(2, config.buffer_slots))
        decode_thread_error: list[Exception] = []

        def _decode_worker() -> None:
            try:
                for frame in backend.decode_iter(config.input_path, probe):
                    if cancel_predicate():
                        break
                    ok = self._queue_put_with_cancel(
                        decode_queue,
                        packet=_DecodedPacket(frame=frame),
                        timeout_s=config.watchdog_timeout_s,
                        cancel_requested=cancel_predicate,
                    )
                    if not ok:
                        return
                self._queue_put_with_cancel(
                    decode_queue,
                    packet=_DecodedPacket(done=True),
                    timeout_s=config.watchdog_timeout_s,
                    cancel_requested=lambda: False,
                )
            except Exception as exc:
                decode_thread_error.append(exc)
                self._queue_put_with_cancel(
                    decode_queue,
                    packet=_DecodedPacket(error=exc),
                    timeout_s=config.watchdog_timeout_s,
                    cancel_requested=lambda: False,
                )

        decode_thread = Thread(target=_decode_worker, name="puregpu3d-decode", daemon=True)
        decode_thread.start()

        last_progress = time.perf_counter()
        loop_error: Exception | None = None
        try:
            while True:
                if cancel_predicate():
                    self.abort_active_run()
                    raise TranscodeCancelledError("Transcode cancelled by user.")

                decode_wait_started = time.perf_counter()
                packet = self._queue_get_with_cancel(
                    decode_queue,
                    timeout_s=config.watchdog_timeout_s,
                    cancel_requested=cancel_predicate,
                )
                decode_wait_ms = (time.perf_counter() - decode_wait_started) * 1000.0

                if packet.error is not None:
                    raise packet.error
                if packet.done:
                    break
                if packet.frame is None:
                    raise RuntimeError("Decode worker returned an empty frame packet")

                frame = packet.frame
                frames_in += 1
                frame_started = time.perf_counter()
                slot, block_wait_ms = ring.acquire(timeout_s=config.watchdog_timeout_s)
                encode_wait_ms = 0.0

                try:
                    slot.in_frame = frame
                    out_frame, kernel_timing = processor.process_frame_with_timing(
                        frame, probe.width, probe.height, config.depth
                    )
                    kernel_ms = kernel_timing.kernel_compute_ms
                    slot.out_frame = out_frame

                    write_future = encode_executor.submit(writer.write, out_frame)
                    pending_writes.append(write_future)
                    if len(pending_writes) >= max_inflight_writes:
                        wait_started = time.perf_counter()
                        self._await_encode_future(
                            pending_writes.popleft(),
                            timeout_s=config.watchdog_timeout_s,
                            cancel_requested=cancel_predicate,
                        )
                        encode_wait_ms += (time.perf_counter() - wait_started) * 1000.0

                    last_good = out_frame
                    frames_out += 1

                    frame_ms = (time.perf_counter() - frame_started) * 1000.0
                    metrics.record_frame(
                        frame_index=frames_in,
                        frame_ms=frame_ms,
                        kernel_ms=kernel_ms,
                        queue_depth=ring.depth,
                        block_wait_ms=block_wait_ms,
                        decode_wait_ms=decode_wait_ms,
                        encode_wait_ms=encode_wait_ms,
                        h2d_ms=kernel_timing.h2d_ms,
                        kernel_compute_ms=kernel_timing.kernel_compute_ms,
                        d2h_ms=kernel_timing.d2h_ms,
                    )
                    if progress_callback is not None:
                        progress_callback(frames_in, total_frames_hint, frame_ms, kernel_ms)
                except Exception as exc:
                    if isinstance(exc, TranscodeCancelledError):
                        raise
                    warn = f"Frame {frames_in} processing failed: {exc}"
                    metrics.add_warning(warn)

                    fallback_frame = self._build_fallback_frame(
                        fallback=config.corrupt_frame_fallback,
                        last_good=last_good,
                        width=out_width,
                        height=out_height,
                    )
                    write_future = encode_executor.submit(writer.write, fallback_frame)
                    pending_writes.append(write_future)
                    if len(pending_writes) >= max_inflight_writes:
                        wait_started = time.perf_counter()
                        self._await_encode_future(
                            pending_writes.popleft(),
                            timeout_s=config.watchdog_timeout_s,
                            cancel_requested=cancel_predicate,
                        )
                        encode_wait_ms += (time.perf_counter() - wait_started) * 1000.0
                    frames_out += 1

                    frame_ms = (time.perf_counter() - frame_started) * 1000.0
                    metrics.record_frame(
                        frame_index=frames_in,
                        frame_ms=frame_ms,
                        kernel_ms=0.0,
                        queue_depth=ring.depth,
                        block_wait_ms=block_wait_ms,
                        decode_wait_ms=decode_wait_ms,
                        encode_wait_ms=encode_wait_ms,
                        h2d_ms=0.0,
                        kernel_compute_ms=0.0,
                        d2h_ms=0.0,
                    )
                    if progress_callback is not None:
                        progress_callback(frames_in, total_frames_hint, frame_ms, 0.0)
                finally:
                    ring.release(slot.slot_id)

                now = time.perf_counter()
                if now - last_progress > config.watchdog_timeout_s:
                    raise TimeoutError("Pipeline watchdog timeout: no frame progress")
                last_progress = now

                if frame_interval_s > 0:
                    realtime_next_deadline += frame_interval_s
                    sleep_s = realtime_next_deadline - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        except Exception as exc:
            loop_error = exc
            if isinstance(exc, TranscodeCancelledError):
                self.abort_active_run()
            else:
                run_abort_event.set()
                try:
                    backend.request_abort()
                except Exception:
                    pass
        finally:
            decode_thread.join(timeout=1.0)

        drain_error: Exception | None = None
        try:
            if loop_error is None:
                while pending_writes:
                    self._await_encode_future(
                        pending_writes.popleft(),
                        timeout_s=config.watchdog_timeout_s,
                        cancel_requested=cancel_predicate,
                    )
            elif isinstance(loop_error, TranscodeCancelledError):
                for future in pending_writes:
                    future.cancel()
            else:
                while pending_writes:
                    self._await_encode_future(
                        pending_writes.popleft(),
                        timeout_s=config.watchdog_timeout_s,
                        cancel_requested=lambda: False,
                    )
        except Exception as exc:
            drain_error = exc
        finally:
            encode_executor.shutdown(wait=True, cancel_futures=False)
            try:
                writer.close()
            except Exception as exc:
                if drain_error is None:
                    drain_error = exc
            self._clear_active_run(backend, run_abort_event)

        if loop_error is None and decode_thread_error:
            loop_error = decode_thread_error[-1]

        if loop_error is not None:
            temp_video_path.unlink(missing_ok=True)
            raise loop_error
        if drain_error is not None:
            temp_video_path.unlink(missing_ok=True)
            raise drain_error

        if metrics.frame_metrics:
            avg_frame_ms = float(np.mean([m.frame_ms for m in metrics.frame_metrics]))
            avg_kernel_ms = float(np.mean([m.kernel_ms for m in metrics.frame_metrics]))
            avg_decode_wait_ms = float(np.mean([m.decode_wait_ms for m in metrics.frame_metrics]))
            avg_encode_wait_ms = float(np.mean([m.encode_wait_ms for m in metrics.frame_metrics]))
            avg_h2d_ms = float(np.mean([m.h2d_ms for m in metrics.frame_metrics]))
            avg_kernel_compute_ms = float(np.mean([m.kernel_compute_ms for m in metrics.frame_metrics]))
            avg_d2h_ms = float(np.mean([m.d2h_ms for m in metrics.frame_metrics]))
            avg_non_kernel_ms = max(0.0, avg_frame_ms - avg_kernel_compute_ms)
            warnings.append(
                "Stage timing avg (ms): "
                f"frame={avg_frame_ms:.2f}, kernel={avg_kernel_ms:.2f}, "
                f"decode_wait={avg_decode_wait_ms:.2f}, encode_wait={avg_encode_wait_ms:.2f}, "
                f"h2d={avg_h2d_ms:.2f}, kernel_compute={avg_kernel_compute_ms:.2f}, d2h={avg_d2h_ms:.2f}, "
                f"non_kernel={avg_non_kernel_ms:.2f}"
            )
            if (avg_decode_wait_ms + avg_encode_wait_ms) > (avg_kernel_compute_ms * 1.2):
                warnings.append("Likely bottleneck: decode/encode stages are slower than kernel compute.")

        if cancel_predicate():
            temp_video_path.unlink(missing_ok=True)
            raise TranscodeCancelledError("Transcode cancelled by user.")

        audio_enabled = config.audio_passthrough and probe.has_audio
        if config.audio_passthrough and not probe.has_audio:
            warnings.append("Source has no audio stream. Output is video-only.")
        remux_warnings = backend.remux_with_audio(
            encoded_video_path=temp_video_path,
            source_input_path=config.input_path,
            output_path=config.output_path,
            codec=config.codec,
            audio_passthrough=audio_enabled,
            audio_codec=config.audio_codec,
        )
        warnings.extend(remux_warnings)
        temp_video_path.unlink(missing_ok=True)
        warnings.extend(getattr(backend, "runtime_warnings", []))

        return metrics.build_result(
            ok=True,
            output_path=config.output_path,
            backend_name=backend.name,
            frames_in=frames_in,
            frames_out=frames_out,
            extra_warnings=warnings,
        )

    def run_batch(
        self,
        *,
        inputs: Iterable[Path],
        output_dir: Path,
        base_config: EngineConfig,
    ) -> dict[Path, TranscodeResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[Path, TranscodeResult] = {}

        for input_path in inputs:
            output_path = output_dir / f"{input_path.stem}_sbs_full.mp4"
            config = replace(base_config, input_path=input_path, output_path=output_path)
            config.allow_missing_input = False
            results[input_path] = self.run_file(config)

        return results
