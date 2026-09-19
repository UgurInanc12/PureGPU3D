"""Application-local verified model store for PureGPU3D."""

from __future__ import annotations

import datetime
import enum
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import fcntl
except ImportError:
    fcntl = None

from puregpu3d.models.catalog import (
    ModelCatalogEntry,
    get_model_entry,
    list_catalog_entries,
    load_catalog,
)
from puregpu3d.models.download import (
    ChecksumMismatchError,
    DownloadCancelledError,
    DownloadError,
    DownloadProgress,
    compute_file_sha256,
    download_file_resumable,
)
from puregpu3d.runtime.paths import AppPaths, get_app_paths, validate_subpath

logger = logging.getLogger(__name__)


class ModelStoreStatus(enum.Enum):
    """Lifecycle status of a model in the application store."""

    MISSING = "missing"
    AWAITING_ACKNOWLEDGMENT = "awaiting_acknowledgment"
    DOWNLOADING = "downloading"
    VERIFIED_DOWNLOAD = "verified_download"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class LicenseAcknowledgmentRequiredError(Exception):
    """Raised when a model requires non-commercial/conflict license acknowledgment before download."""


class ModelLockedError(Exception):
    """Raised when an exclusive model operation cannot acquire lock."""


class WindowsFileLock:
    """Cross-platform, OS-level advisory file lock with crash recovery.

    Uses Windows byte locking (msvcrt.locking) on Windows and advisory flock
    (fcntl.flock) on POSIX. The lock file persists on disk and is not unlinked
    to prevent race conditions and unintended unlock of concurrent owners.
    """

    def __init__(self, lock_file_path: Union[str, Path], timeout: float = 30.0) -> None:
        self.lock_file_path = Path(lock_file_path)
        self.timeout = float(timeout)
        self._fd: Optional[int] = None
        self._is_locked: bool = False

    def acquire(self) -> None:
        """Acquire advisory byte lock, waiting up to timeout seconds."""
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()

        if self._fd is None:
            self._fd = os.open(str(self.lock_file_path), os.O_CREAT | os.O_RDWR)

        while True:
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                if msvcrt is not None:
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                elif fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                self._is_locked = True
                return
            except (BlockingIOError, PermissionError, OSError):
                if time.monotonic() - start >= self.timeout:
                    if self._fd is not None:
                        try:
                            os.close(self._fd)
                        except OSError:
                            pass
                        self._fd = None
                    raise ModelLockedError(
                        f"Timed out after {self.timeout}s waiting for lock at {self.lock_file_path}"
                    )
                time.sleep(0.05)

    def release(self) -> None:
        """Release lock held by this instance.

        If this instance does not hold the lock, calling release is a safe no-op
        and does not touch any other process's lock.
        """
        if not self._is_locked or self._fd is None:
            return

        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            if msvcrt is not None:
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]
        except OSError:
            pass
        finally:
            self._is_locked = False
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> WindowsFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class ModelStore:
    """Manages application-local model directories, downloads, integrity, and readiness."""

    def __init__(
        self,
        paths: Optional[AppPaths] = None,
        *,
        catalog: Optional[Dict[str, ModelCatalogEntry]] = None,
        allow_http_test_transport: bool = False,
        base_download_url: Optional[str] = None,
    ) -> None:
        """Initialize model store.

        Args:
            paths: AppPaths instance. If None, initialized from environment/exe.
            catalog: Model catalog mapping. If None, loaded from resources/models.json.
            allow_http_test_transport: Allow unencrypted HTTP (loopback test fixture servers only).
            base_download_url: Optional override for base download URL (for tests).
        """
        self.paths = paths or get_app_paths()
        self.catalog = catalog or load_catalog()
        self.allow_http_test_transport = allow_http_test_transport
        self.base_download_url = base_download_url

        self._thread_locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self._active_status: Dict[str, ModelStoreStatus] = {}

    def _get_thread_lock(self, model_id: str) -> threading.Lock:
        with self._meta_lock:
            if model_id not in self._thread_locks:
                self._thread_locks[model_id] = threading.Lock()
            return self._thread_locks[model_id]

    def _acknowledgments_file(self) -> Path:
        return self.paths.data / "license_acknowledgments.json"

    def is_license_acknowledged(self, entry: ModelCatalogEntry) -> bool:
        """Check if required license acknowledgment has been recorded for this model."""
        if not entry.license_info.noncommercial_ack_required and not entry.license_info.license_conflict:
            return True

        ack_file = self._acknowledgments_file()
        if not ack_file.is_file():
            return False

        try:
            with open(ack_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get(entry.id, {}).get("acknowledged", False))
        except (json.JSONDecodeError, OSError):
            return False

    def record_license_acknowledgment(
        self,
        identifier: str,
        *,
        acknowledged: bool = True,
        acknowledged_by: str = "user",
    ) -> None:
        """Persist user acknowledgment for non-commercial or conflicting license terms."""
        entry = get_model_entry(identifier, self.catalog)
        ack_file = self._acknowledgments_file()
        ack_file.parent.mkdir(parents=True, exist_ok=True)

        with self._meta_lock:
            data: Dict[str, Any] = {}
            if ack_file.is_file():
                try:
                    with open(ack_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            data[entry.id] = {
                "model_id": entry.id,
                "repo_id": entry.repo_id,
                "license": entry.license_info.license,
                "license_type": entry.license_info.license_type,
                "acknowledged": acknowledged,
                "acknowledged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "acknowledged_by": acknowledged_by,
                "conflict_warning": entry.license_info.conflict_details,
            }

            temp_ack = ack_file.with_name(f".ack_{os.getpid()}_{time.time_ns()}.tmp")
            with open(temp_ack, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_ack.replace(ack_file)

    def get_model_dir(self, identifier: str) -> Path:
        """Return the base directory for the given model ID, validating against traversal."""
        entry = get_model_entry(identifier, self.catalog)
        return validate_subpath(self.paths.models, entry.id)

    def get_revision_dir(self, identifier: str) -> Path:
        """Return the revision directory for the given model, validating against traversal."""
        entry = get_model_entry(identifier, self.catalog)
        model_dir = self.get_model_dir(identifier)
        return validate_subpath(model_dir, entry.revision)

    def get_status(
        self,
        identifier: str,
        *,
        verify_hashes: bool = False,
    ) -> ModelStoreStatus:
        """Inspect and return current model lifecycle status.

        Args:
            identifier: Model ID or name.
            verify_hashes: If True, physically verify SHA-256 for all model files,
                           bypassing any in-memory cached readiness.

        Returns:
            ModelStoreStatus enum value.
        """
        entry = get_model_entry(identifier, self.catalog)

        # In-memory status is only consulted when verify_hashes is False,
        # and only for transient in-progress states (e.g. DOWNLOADING, LOADING).
        if not verify_hashes:
            with self._meta_lock:
                if entry.id in self._active_status:
                    active = self._active_status[entry.id]
                    if active in (ModelStoreStatus.DOWNLOADING, ModelStoreStatus.LOADING):
                        return active

        if not self.is_license_acknowledged(entry):
            return ModelStoreStatus.AWAITING_ACKNOWLEDGMENT

        rev_dir = self.get_revision_dir(identifier)
        if not rev_dir.is_dir():
            return ModelStoreStatus.MISSING

        # Check required files exist and sizes match
        for filename, spec in entry.files.items():
            file_path = rev_dir / filename
            if not file_path.is_file():
                return ModelStoreStatus.MISSING
            if file_path.stat().st_size != spec.bytes:
                return ModelStoreStatus.MISSING
            if verify_hashes:
                if compute_file_sha256(file_path).lower() != spec.sha256.lower():
                    return ModelStoreStatus.FAILED

        # Check READY marker and manifest with strong verification
        ready_file = rev_dir / "READY"
        manifest_file = rev_dir / "manifest.json"

        if ready_file.is_file() and manifest_file.is_file():
            try:
                ready_text = ready_file.read_text(encoding="utf-8")
                if f"READY {entry.revision}" in ready_text:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    # Require strong evidence for READY:
                    # 1. Matching revision
                    # 2. Matching model identity (repo_id or id)
                    # 3. Explicit runtime probe verification
                    # 4. Matching file checksum manifest
                    is_rev_match = manifest.get("revision") == entry.revision
                    is_id_match = manifest.get("model_id") in (entry.repo_id, entry.id)
                    is_probe_verified = manifest.get("runtime_probe_verified") is True

                    manifest_files = manifest.get("files")
                    is_manifest_files_valid = isinstance(manifest_files, dict)
                    if is_manifest_files_valid:
                        for fname, fspec in entry.files.items():
                            m_entry = manifest_files.get(fname)
                            if not isinstance(m_entry, dict):
                                is_manifest_files_valid = False
                                break
                            m_sha = str(m_entry.get("sha256", "")).lower()
                            m_bytes = m_entry.get("bytes")
                            if m_sha != fspec.sha256.lower() or m_bytes != fspec.bytes:
                                is_manifest_files_valid = False
                                break

                    if is_rev_match and is_id_match and is_probe_verified and is_manifest_files_valid:
                        return ModelStoreStatus.READY
            except (json.JSONDecodeError, OSError):
                pass

        # Files exist and match required sizes (and hashes if checked),
        # but lack full verified runtime probe READY status.
        return ModelStoreStatus.VERIFIED_DOWNLOAD

    def verify_offline_readiness(
        self,
        identifier: str,
        *,
        verify_hashes: bool = True,
    ) -> bool:
        """Verify model is fully ready for offline use without any network access.

        Performs strictly local file system checks. Does not perform HTTP HEAD or requests.

        Args:
            identifier: Model ID.
            verify_hashes: Whether to compute and verify SHA-256 checksums of all weights.

        Returns:
            True if model is present, verified, and has READY marker.
        """
        status = self.get_status(identifier, verify_hashes=verify_hashes)
        return status == ModelStoreStatus.READY

    def _construct_download_url(self, entry: ModelCatalogEntry, filename: str) -> str:
        """Build download URL for an official Hugging Face file or test server override."""
        if self.base_download_url:
            base = self.base_download_url.rstrip("/")
            return f"{base}/{entry.repo_id}/{entry.revision}/{filename}"
        return f"https://huggingface.co/{entry.repo_id}/resolve/{entry.revision}/{filename}"

    def prepare_model(
        self,
        *,
        identifier: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        load_verifier: Optional[Callable[[Path], None]] = None,
        acknowledge_license: bool = False,
    ) -> Path:
        """Ensure model is downloaded, verified, and optionally loaded.

        Args:
            identifier: Model ID or name.
            progress_callback: Optional byte progress listener.
            is_cancelled: Optional cancellation predicate.
            load_verifier: Optional callback `fn(revision_dir)` to probe runtime loading.
                           READY marker is ONLY written if this callback succeeds.
            acknowledge_license: Set True to automatically record license acknowledgment.

        Returns:
            Path to the local revision directory.

        Raises:
            LicenseAcknowledgmentRequiredError: If noncommercial/conflict license not acknowledged.
            DownloadCancelledError: If cancelled.
            ChecksumMismatchError: If file hash verification fails.
            DownloadError: On network or download failure.
        """
        entry = get_model_entry(identifier, self.catalog)

        if acknowledge_license:
            self.record_license_acknowledgment(entry.id)

        if not self.is_license_acknowledged(entry):
            raise LicenseAcknowledgmentRequiredError(
                f"Model '{entry.id}' requires explicit acknowledgment of license terms: "
                f"'{entry.license_info.license}'. "
                f"{entry.license_info.conflict_details or ''}".strip()
            )

        thread_lock = self._get_thread_lock(entry.id)
        model_dir = self.get_model_dir(entry.id)
        rev_dir = self.get_revision_dir(entry.id)
        lock_file = model_dir / ".download.lock"

        with thread_lock:
            with WindowsFileLock(lock_file):
                current_status = self.get_status(entry.id, verify_hashes=True)

                if current_status == ModelStoreStatus.READY:
                    return rev_dir

                if current_status == ModelStoreStatus.VERIFIED_DOWNLOAD:
                    # Files already exist and are verified; if load_verifier supplied, run it now
                    if load_verifier is not None:
                        self._run_load_verifier_and_mark_ready(entry, rev_dir, load_verifier)
                    return rev_dir

                # Download required: stage into temporary staging directory
                staging_dir = model_dir / f".staging_{entry.revision}_{time.time_ns()}"
                staging_dir.mkdir(parents=True, exist_ok=True)

                with self._meta_lock:
                    self._active_status[entry.id] = ModelStoreStatus.DOWNLOADING

                temp_old: Optional[Path] = None
                try:
                    for filename, file_spec in entry.files.items():
                        if is_cancelled and is_cancelled():
                            raise DownloadCancelledError("Download cancelled before file start.")

                        target_file = staging_dir / filename
                        url = self._construct_download_url(entry, filename)

                        download_file_resumable(
                            url=url,
                            destination_path=target_file,
                            expected_sha256=file_spec.sha256,
                            expected_bytes=file_spec.bytes,
                            allow_http_test_transport=self.allow_http_test_transport,
                            progress_callback=progress_callback,
                            is_cancelled=is_cancelled,
                        )

                    # All files downloaded and verified in staging_dir
                    # Prepare rollback-safe promotion
                    if rev_dir.exists():
                        temp_old = model_dir / f".old_{entry.revision}_{time.time_ns()}"
                        try:
                            rev_dir.replace(temp_old)
                        except OSError:
                            pass

                    # Promote staging_dir to rev_dir atomically
                    try:
                        staging_dir.replace(rev_dir)
                    except Exception:
                        # Restore previous valid rev_dir if promotion failed
                        if temp_old and temp_old.exists() and not rev_dir.exists():
                            try:
                                temp_old.replace(rev_dir)
                            except OSError:
                                pass
                        raise

                    # If load verifier is provided, execute it before cleaning up previous version
                    if load_verifier is not None:
                        try:
                            self._run_load_verifier_and_mark_ready(entry, rev_dir, load_verifier)
                        except Exception:
                            # If load verifier fails on an update, restore the prior working directory
                            if temp_old and temp_old.exists():
                                shutil.rmtree(rev_dir, ignore_errors=True)
                                try:
                                    temp_old.replace(rev_dir)
                                except OSError:
                                    pass
                            raise

                    # Clean up superseded directory only after replacement succeeds
                    if temp_old and temp_old.exists():
                        shutil.rmtree(temp_old, ignore_errors=True)

                    return rev_dir

                except Exception:
                    with self._meta_lock:
                        self._active_status[entry.id] = ModelStoreStatus.FAILED
                    raise

                finally:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    with self._meta_lock:
                        self._active_status.pop(entry.id, None)

    def _run_load_verifier_and_mark_ready(
        self,
        entry: ModelCatalogEntry,
        rev_dir: Path,
        load_verifier: Callable[[Path], None],
    ) -> None:
        """Run load verification callback and write READY marker on success."""
        with self._meta_lock:
            self._active_status[entry.id] = ModelStoreStatus.LOADING

        try:
            logger.info("Executing load probe verification for %s at %s", entry.id, rev_dir)
            load_verifier(rev_dir)

            # Write manifest.json
            manifest = {
                "model_id": entry.repo_id,
                "revision": entry.revision,
                "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "files": {
                    name: {"sha256": spec.sha256, "bytes": spec.bytes}
                    for name, spec in entry.files.items()
                },
                "license": entry.license_info.license,
                "runtime_probe_verified": True,
            }
            manifest_path = rev_dir / "manifest.json"
            manifest_tmp = rev_dir / f".manifest_{os.getpid()}_{time.time_ns()}.tmp"
            with open(manifest_tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            manifest_tmp.replace(manifest_path)

            # Write READY file
            ready_path = rev_dir / "READY"
            ready_tmp = rev_dir / f".ready_{os.getpid()}_{time.time_ns()}.tmp"
            with open(ready_tmp, "w", encoding="utf-8") as f:
                f.write(f"READY {entry.revision}\n")
                f.write("manifest_version: 1.0.0\n")
                f.write(f"verified_at: {manifest['verified_at']}\n")
            ready_tmp.replace(ready_path)

        finally:
            with self._meta_lock:
                self._active_status.pop(entry.id, None)

    def mark_ready_after_load_validation(
        self,
        identifier: str,
        load_verifier: Callable[[Path], None],
    ) -> Path:
        """Validate runtime load for a verified_download model and promote it to READY.

        Args:
            identifier: Model ID.
            load_verifier: Callback `fn(rev_dir)` that successfully loads model weights.

        Returns:
            Path to revision directory.
        """
        entry = get_model_entry(identifier, self.catalog)
        model_dir = self.get_model_dir(entry.id)
        rev_dir = self.get_revision_dir(identifier)
        if not rev_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found for {entry.id} at {rev_dir}")

        lock_file = model_dir / ".download.lock"
        thread_lock = self._get_thread_lock(entry.id)

        with thread_lock:
            with WindowsFileLock(lock_file):
                # Check integrity under lock before running load probe
                status = self.get_status(entry.id, verify_hashes=True)
                if status not in (ModelStoreStatus.VERIFIED_DOWNLOAD, ModelStoreStatus.READY):
                    raise ValueError(
                        f"Cannot mark {entry.id} as READY: current status is {status.value}"
                    )

                self._run_load_verifier_and_mark_ready(entry, rev_dir, load_verifier)
                return rev_dir
