"""Safe atomic output file transactions with preflight validation, locking, and rollback.

Protects source files and existing outputs against corruption:
  - Strict collision rejection (identical paths, canonical symlink/alias resolution,
    case-insensitive paths on Windows, hardlinks).
  - Explicit overwrite gate with destination identity snapshots; existing output files
    remain intact until the new file passes validation, and concurrent modification
    by another process aborts promotion.
  - Per-destination advisory process locking (WindowsFileLock) preventing racing
    jobs from targeting the same output.
  - Unique same-volume staging files ensuring atomic promotion.
  - Non-clobber atomic rename for overwrite=False.
  - Comprehensive pre-promotion validation: format decodability (ffmpeg -xerror with
    duration-aware timeout and bounded stderr drainer), exact dimensions, rational
    frame rate preservation, non-zero duration sanity, exact decoded frame count
    (-count_frames, 0-tolerance contract), and audio stream count & properties policy.
  - Automatic cleanup of temporary staging files on exception or cancellation.
"""

from __future__ import annotations

import collections
import fractions
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Deque, Dict, List, Optional, Type, Union

from puregpu3d.models.store import WindowsFileLock

logger = logging.getLogger(__name__)


class TransactionError(RuntimeError):
    """Base error raised for transaction failures."""
    pass


class PathCollisionError(TransactionError, ValueError):
    """Raised when source and destination point to the same physical file or path alias."""
    pass


class ValidationError(TransactionError):
    """Raised when the rendered staging file fails post-conversion validation."""
    pass


class TransactionCancelledError(TransactionError):
    """Raised when transaction validation is cancelled."""
    pass


def _terminate_process_safely(proc: Optional[subprocess.Popen], name: str = "child") -> None:
    """Terminate and reap child process safely without leaking resources or blocking pipes."""
    if proc is None:
        return
    # Terminate / kill FIRST before touching pipes
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except OSError:
                pass
        proc.poll()
        logger.debug(f"Process {name} (PID {proc.pid}) terminated.")

    # Then close buffered pipes
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe and not pipe.closed:
            try:
                pipe.close()
            except Exception:
                pass

    try:
        proc.wait(timeout=0.2)
    except Exception:
        pass


def paths_refer_to_same_file(path_a: Union[str, Path], path_b: Union[str, Path]) -> bool:
    """Check whether two paths refer to the same file or canonical location.

    Evaluates:
      1. String/path equivalence.
      2. Resolved canonical absolute paths.
      3. Case-insensitive normalization (on Windows).
      4. OS-level samefile inode/file index check if both files exist.
    """
    p_a = Path(path_a)
    p_b = Path(path_b)

    if p_a == p_b:
        return True

    res_a = p_a.resolve()
    res_b = p_b.resolve()
    if res_a == res_b:
        return True

    # Check case-normalized path representations
    norm_a = os.path.normcase(str(res_a))
    norm_b = os.path.normcase(str(res_b))
    if norm_a == norm_b:
        return True

    # If both files physically exist, check OS-level file identity (hardlinks/symlinks)
    try:
        if p_a.exists() and p_b.exists():
            if os.path.samefile(p_a, p_b):
                return True
    except (OSError, ValueError):
        pass

    return False


class OutputTransaction:
    """Context manager controlling safe atomic staging and validation for exported media.

    Attributes:
        source_path: Path to input media.
        destination_path: Final destination path.
        overwrite: If True, allows replacing an existing destination upon successful validation.
        expected_width: Expected output video width (e.g. 2 * input_width).
        expected_height: Expected output video height.
        expected_frames: Expected exact frame count (contract enforces exact match).
        expect_audio: Whether output must contain at least one audio stream.
        expected_audio_streams: Optional exact count of audio streams expected.
        expected_duration: Optional expected video duration in seconds.
        expected_frame_rate: Optional expected rational frame rate.
        staging_path: Temporary staging path on the destination volume.
        committed: Boolean indicating whether validate_and_promote() succeeded.
    """

    def __init__(
        self,
        source_path: Union[str, Path],
        destination_path: Union[str, Path],
        *,
        overwrite: bool = False,
        expected_width: Optional[int] = None,
        expected_height: Optional[int] = None,
        expected_frames: Optional[int] = None,
        expect_audio: bool = False,
        expected_audio_streams: Optional[int] = None,
        expected_duration: Optional[float] = None,
        expected_frame_rate: Optional[Union[fractions.Fraction, str]] = None,
        ffprobe_path: Optional[Union[str, Path]] = None,
        ffmpeg_path: Optional[Union[str, Path]] = None,
        decode_timeout_s: Optional[float] = None,
        lock_timeout_s: float = 30.0,
    ) -> None:
        self.source_path = Path(source_path).resolve()
        self.destination_path = Path(destination_path).resolve()
        self.overwrite = overwrite
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.expected_frames = expected_frames
        self.expect_audio = expect_audio
        self.expected_audio_streams = expected_audio_streams
        self.expected_duration = expected_duration
        if expected_frame_rate is not None:
            from puregpu3d.video.probe import parse_fraction
            self.expected_frame_rate = parse_fraction(str(expected_frame_rate))
        else:
            self.expected_frame_rate = None
        self.ffprobe_path = ffprobe_path
        self.ffmpeg_path = ffmpeg_path
        self.decode_timeout_s = decode_timeout_s
        self.lock_timeout_s = lock_timeout_s

        self.committed = False
        self._staging_path: Optional[Path] = None
        self._dest_snapshot: Dict[str, Any] = {}
        self._lock: Optional[WindowsFileLock] = None

        dest_dir = self.destination_path.parent
        self._lock_file = dest_dir / f".{self.destination_path.name}.lock"

        # Preflight validation
        self._preflight()

    def _take_destination_snapshot(self) -> Dict[str, Any]:
        """Record destination file identity (existence, mtime, size, inode)."""
        if not self.destination_path.exists():
            return {"exists": False}
        try:
            st = self.destination_path.stat()
            return {
                "exists": True,
                "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
                "size": st.st_size,
                "ino": getattr(st, "st_ino", 0),
            }
        except OSError:
            return {"exists": False}

    def _preflight(self) -> None:
        """Validate path collision and overwrite protection upfront."""
        if paths_refer_to_same_file(self.source_path, self.destination_path):
            raise PathCollisionError(
                f"Source and destination refer to the same physical file: '{self.destination_path}'. "
                f"In-place overwrite of input files is strictly rejected to prevent irreversible data loss."
            )

        if self.destination_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Destination file already exists and overwrite=False: '{self.destination_path}'. "
                f"Specify overwrite=True to allow replacement upon successful completion."
            )

        # Destination parent directory must exist or be creatable
        dest_dir = self.destination_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Record initial snapshot
        self._dest_snapshot = self._take_destination_snapshot()

        # Staging path placed on the same parent volume to guarantee atomic promotion
        token = uuid.uuid4().hex[:10]
        staging_name = f".staging_{token}_{self.destination_path.name}"
        self._staging_path = dest_dir / staging_name

    @property
    def staging_path(self) -> Path:
        """Return the active staging file path."""
        if self._staging_path is None:
            raise RuntimeError("Transaction not initialized.")
        return self._staging_path

    def __enter__(self) -> OutputTransaction:
        # Acquire per-destination advisory process lock
        self._lock = WindowsFileLock(self._lock_file, timeout=self.lock_timeout_s)
        try:
            self._lock.acquire()
        except Exception as err:
            raise TransactionError(
                f"Could not acquire process lock for destination '{self.destination_path}': {err}"
            ) from err

        # Verify destination existence under lock
        if self.destination_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Destination file already exists and overwrite=False: '{self.destination_path}'."
            )

        # Update destination identity snapshot under lock
        self._dest_snapshot = self._take_destination_snapshot()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        # If an error occurred or promotion was never committed, clean up staging file
        try:
            if not self.committed:
                self._cleanup_staging()
        finally:
            if self._lock is not None:
                try:
                    self._lock.release()
                except Exception:
                    pass
                self._lock = None
        return False  # Do not suppress exceptions

    def _cleanup_staging(self) -> None:
        """Unlink staging file if it exists."""
        if self._staging_path and self._staging_path.exists():
            for attempt in range(5):
                try:
                    self._staging_path.unlink()
                    logger.info(f"Cleaned up temporary staging file: {self._staging_path}")
                    break
                except OSError as err:
                    if attempt == 4:
                        logger.warning(f"Failed to remove temporary staging file {self._staging_path}: {err}")
                    else:
                        time.sleep(0.05)

    def validate_and_promote(
        self,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> Path:
        """Validate rendered staging file properties and promote atomically to destination.

        Checks:
          1. Staging file exists and is non-empty.
          2. Probe staging file using ffprobe with -count_frames for exact decoded frames.
          3. Probed dimensions match expected_width and expected_height.
          4. Probed duration > 0 and matches expected_duration within frame tolerance.
          5. Frame count matches expected_frames exactly (0-tolerance contract).
          6. Frame rate matches expected_frame_rate rational Fraction.
          7. Audio stream count and audio stream properties match policy.
          8. Decodability via ffmpeg -xerror null sink with bounded stderr drainer
             and duration-aware timeout.
          9. Destination identity verified against preflight snapshot before atomic commit.

        Promotes:
          Atomic no-clobber rename (overwrite=False) or atomic os.replace (overwrite=True).

        Returns:
            Promoted destination Path.

        Raises:
            ValidationError: If any validation check fails.
            TransactionCancelledError: If cancelled during validation.
            FileExistsError: If destination appeared with overwrite=False.
            TransactionError: If destination identity changed unexpectedly or promotion fails.
        """
        if cancel_callback is not None and cancel_callback():
            raise TransactionCancelledError("Validation aborted: cancellation requested.")

        if not self.staging_path.exists():
            raise ValidationError(f"Rendered staging file does not exist: '{self.staging_path}'")

        file_size = self.staging_path.stat().st_size
        if file_size <= 0:
            raise ValidationError(f"Rendered staging file is empty (0 bytes): '{self.staging_path}'")

        # 1. Probe staging file with -count_frames for exact stream decoding inspection
        from puregpu3d.video.probe import find_ffmpeg, find_ffprobe, parse_fraction
        ffprobe = find_ffprobe(self.ffprobe_path)
        probe_cmd = [
            str(ffprobe),
            "-v", "error",
            "-count_frames",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_error",
            str(self.staging_path),
        ]
        try:
            res = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                encoding="utf-8",
                timeout=max(30.0, float(self.decode_timeout_s or 60.0)),
            )
        except subprocess.TimeoutExpired as err:
            raise ValidationError(f"ffprobe staging inspection timed out: {err}") from err

        if res.returncode != 0:
            raise ValidationError(
                f"ffprobe failed on staging output '{self.staging_path}' (code {res.returncode}): "
                f"{res.stderr.strip()}"
            )

        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError as err:
            raise ValidationError(f"Failed to parse ffprobe JSON for staging output: {err}") from err

        if "error" in data:
            raise ValidationError(f"ffprobe reported error for staging output: {data['error']}")

        streams = data.get("streams", [])
        format_info = data.get("format", {})

        video_stream: Optional[Dict[str, Any]] = None
        audio_streams: List[Dict[str, Any]] = []

        for s in streams:
            if s.get("codec_type") == "video" and video_stream is None:
                video_stream = s
            elif s.get("codec_type") == "audio":
                audio_streams.append(s)

        if video_stream is None:
            raise ValidationError("No video stream found in rendered staging output.")

        # 2. Dimensions check
        probed_w = int(video_stream.get("width", 0))
        probed_h = int(video_stream.get("height", 0))
        if self.expected_width is not None and probed_w != self.expected_width:
            raise ValidationError(
                f"Dimension mismatch in staging output: width is {probed_w}, expected {self.expected_width}"
            )
        if self.expected_height is not None and probed_h != self.expected_height:
            raise ValidationError(
                f"Dimension mismatch in staging output: height is {probed_h}, expected {self.expected_height}"
            )

        # 3. Precise decoded frame count check (exact contract, no tolerance)
        nb_read_frames = video_stream.get("nb_read_frames")
        nb_frames = video_stream.get("nb_frames")
        if nb_read_frames and str(nb_read_frames).isdigit():
            actual_frames = int(nb_read_frames)
        elif nb_frames and str(nb_frames).isdigit():
            actual_frames = int(nb_frames)
        else:
            actual_frames = 0

        if self.expected_frames is not None:
            if actual_frames != self.expected_frames:
                raise ValidationError(
                    f"Frame count mismatch in staging output: produced {actual_frames} frames, "
                    f"expected {self.expected_frames} frames."
                )

        # 4. Rational frame rate check
        r_fps = parse_fraction(video_stream.get("r_frame_rate"))
        avg_fps = parse_fraction(video_stream.get("avg_frame_rate"))
        probed_fps = r_fps if r_fps is not None else avg_fps

        if self.expected_frame_rate is not None:
            if probed_fps != self.expected_frame_rate:
                raise ValidationError(
                    f"Frame rate mismatch in staging output: produced {probed_fps}, "
                    f"expected {self.expected_frame_rate}."
                )

        # 5. Duration check
        dur_str = video_stream.get("duration") or format_info.get("duration")
        probed_duration = float(dur_str) if dur_str and dur_str != "N/A" else 0.0

        if probed_duration <= 0.0:
            raise ValidationError(
                f"Invalid duration in staging output: {probed_duration}s (must be > 0)"
            )

        if self.expected_duration is not None and self.expected_duration > 0:
            fps_val = float(probed_fps) if probed_fps else 24.0
            frame_tolerance = max(0.1, 1.5 / fps_val)
            if abs(probed_duration - self.expected_duration) > frame_tolerance:
                raise ValidationError(
                    f"Duration mismatch in staging output: produced {probed_duration:.3f}s, "
                    f"expected {self.expected_duration:.3f}s (tolerance {frame_tolerance:.3f}s)"
                )

        # 6. Audio stream count and policy check
        if self.expected_audio_streams is not None:
            if len(audio_streams) != self.expected_audio_streams:
                raise ValidationError(
                    f"Audio stream count mismatch in staging output: produced {len(audio_streams)} audio streams, "
                    f"expected {self.expected_audio_streams} audio streams."
                )
            for a in audio_streams:
                codec = a.get("codec_name")
                channels = int(a.get("channels", 0))
                sample_rate = int(a.get("sample_rate", 0))
                if not codec or channels <= 0 or sample_rate <= 0:
                    raise ValidationError(
                        f"Invalid audio stream #{a.get('index')}: codec={codec}, "
                        f"channels={channels}, sample_rate={sample_rate}"
                    )
        elif self.expect_audio:
            if len(audio_streams) == 0:
                raise ValidationError(
                    "Source media contained audio, but rendered staging output contains no audio streams."
                )
            for a in audio_streams:
                codec = a.get("codec_name")
                channels = int(a.get("channels", 0))
                sample_rate = int(a.get("sample_rate", 0))
                if not codec or channels <= 0 or sample_rate <= 0:
                    raise ValidationError(
                        f"Invalid audio stream #{a.get('index')}: codec={codec}, "
                        f"channels={channels}, sample_rate={sample_rate}"
                    )
        else:
            # Silent output expected
            if len(audio_streams) > 0:
                raise ValidationError(
                    f"Silent output expected, but rendered staging output contains {len(audio_streams)} audio streams."
                )

        # 7. Stream decodability test using ffmpeg null sink with -xerror and bounded stderr
        ffmpeg = find_ffmpeg(self.ffmpeg_path)
        decode_cmd = [
            str(ffmpeg),
            "-v", "error",
            "-xerror",
            "-i", str(self.staging_path),
            "-f", "null",
            "-",
        ]

        if self.decode_timeout_s is not None:
            timeout_s = float(self.decode_timeout_s)
        else:
            timeout_s = max(60.0, probed_duration * 2.5)

        decode_proc = subprocess.Popen(
            decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        decode_stderr_lines: Deque[str] = collections.deque(maxlen=100)

        def _drain_stderr() -> None:
            try:
                assert decode_proc.stderr is not None
                for line in iter(decode_proc.stderr.readline, b""):
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        decode_stderr_lines.append(decoded)
            except Exception:
                pass
            finally:
                try:
                    if decode_proc.stderr and not decode_proc.stderr.closed:
                        decode_proc.stderr.close()
                except Exception:
                    pass

        drainer = threading.Thread(target=_drain_stderr, daemon=True)
        drainer.start()

        start_time = time.monotonic()
        try:
            while True:
                ret = decode_proc.poll()
                if ret is not None:
                    break
                if cancel_callback is not None and cancel_callback():
                    _terminate_process_safely(decode_proc, "decode_validator")
                    raise TransactionCancelledError("Decode validation cancelled by user request.")
                if time.monotonic() - start_time > timeout_s:
                    _terminate_process_safely(decode_proc, "decode_validator")
                    raise ValidationError(
                        f"Staging file decode validation timed out after {timeout_s:.1f}s."
                    )
                time.sleep(0.05)
        finally:
            _terminate_process_safely(decode_proc, "decode_validator")
            if drainer.is_alive():
                drainer.join(timeout=0.5)

        if decode_proc.returncode != 0:
            err_text = "\n".join(list(decode_stderr_lines)[-30:])
            raise ValidationError(
                f"Staging file failed decodability validation (exit code {decode_proc.returncode}):\n{err_text}"
            )

        # 8. Destination identity check against preflight snapshot and atomic promotion
        if not self.overwrite:
            # Overwrite=False: Destination must NOT exist
            if self.destination_path.exists():
                raise FileExistsError(
                    f"Destination file appeared after preflight check with overwrite=False: '{self.destination_path}'"
                )
            try:
                if sys.platform == "win32":
                    # On Windows, os.rename is an atomic no-clobber operation failing if destination exists
                    os.rename(self.staging_path, self.destination_path)
                else:
                    # On POSIX, atomic link fails with FileExistsError if destination exists
                    try:
                        os.link(self.staging_path, self.destination_path)
                        self.staging_path.unlink()
                    except FileExistsError:
                        raise
                    except OSError:
                        if self.destination_path.exists():
                            raise FileExistsError(f"Destination already exists: '{self.destination_path}'")
                        os.replace(self.staging_path, self.destination_path)
                self.committed = True
                logger.info(
                    f"Successfully validated and promoted '{self.staging_path.name}' -> '{self.destination_path}'"
                )
            except FileExistsError as err:
                raise FileExistsError(
                    f"Atomic promotion failed because destination already exists and overwrite=False: '{self.destination_path}'"
                ) from err
            except OSError as err:
                raise TransactionError(
                    f"Atomic promotion failed during rename('{self.staging_path}', '{self.destination_path}'): {err}"
                ) from err
        else:
            # Overwrite=True: Verify destination has not been modified or replaced by another process
            if self._dest_snapshot.get("exists"):
                if not self.destination_path.exists():
                    raise TransactionError(
                        f"Destination file '{self.destination_path}' existed at transaction start but was removed before promotion."
                    )
                curr_stat = self.destination_path.stat()
                curr_mtime_ns = getattr(curr_stat, "st_mtime_ns", int(curr_stat.st_mtime * 1e9))
                curr_size = curr_stat.st_size
                curr_ino = getattr(curr_stat, "st_ino", 0)

                snap_mtime_ns = self._dest_snapshot["mtime_ns"]
                snap_size = self._dest_snapshot["size"]
                snap_ino = self._dest_snapshot["ino"]

                if (
                    curr_mtime_ns != snap_mtime_ns
                    or curr_size != snap_size
                    or (snap_ino != 0 and curr_ino != 0 and curr_ino != snap_ino)
                ):
                    raise TransactionError(
                        f"Destination file '{self.destination_path}' was modified or replaced by another process after transaction start. "
                        f"Promotion aborted to prevent overwriting concurrent changes."
                    )
            else:
                if self.destination_path.exists():
                    raise TransactionError(
                        f"Destination file '{self.destination_path}' did not exist at transaction start but appeared before promotion. "
                        f"Promotion aborted to prevent overwriting concurrent creation."
                    )

            try:
                os.replace(self.staging_path, self.destination_path)
                self.committed = True
                logger.info(
                    f"Successfully validated and promoted '{self.staging_path.name}' -> '{self.destination_path}'"
                )
            except OSError as err:
                raise TransactionError(
                    f"Atomic promotion failed during os.replace('{self.staging_path}', '{self.destination_path}'): {err}"
                ) from err

        return self.destination_path
