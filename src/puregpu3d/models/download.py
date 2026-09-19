"""Secure, resumable HTTP/HTTPS downloader for PureGPU3D models."""

from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import requests
import socket
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import NewConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from puregpu3d.runtime.paths import validate_subpath


class IPv4HTTPSConnection(HTTPSConnection):
    """Avoid broken IPv6 routes without changing global DNS behavior."""

    def _new_conn(self):
        last_error = None
        for family, kind, proto, _, address in socket.getaddrinfo(
            self._dns_host, self.port, socket.AF_INET, socket.SOCK_STREAM
        ):
            sock = socket.socket(family, kind, proto)
            try:
                sock.settimeout(self.timeout)
                for option in self.socket_options or []:
                    sock.setsockopt(*option)
                if self.source_address:
                    sock.bind(self.source_address)
                sock.connect(address)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        raise NewConnectionError(self, f"IPv4 connection failed: {last_error}")


class IPv4HTTPSPool(HTTPSConnectionPool):
    ConnectionCls = IPv4HTTPSConnection


class ModelDownloadAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = dict(self.poolmanager.pool_classes_by_scheme)
        self.poolmanager.pool_classes_by_scheme["https"] = IPv4HTTPSPool


class DownloadError(Exception):
    """Base exception for model download errors."""


class DownloadCancelledError(DownloadError):
    """Raised when download operation is cancelled by the caller."""


class ChecksumMismatchError(DownloadError):
    """Raised when downloaded file fails SHA-256 integrity verification."""


class InsecureTransportError(DownloadError):
    """Raised when non-HTTPS transport is attempted without explicit test authorization."""


class UntrustedRedirectError(DownloadError):
    """Raised when an HTTP redirect targets an unapproved domain."""


@dataclass(frozen=True, kw_only=True)
class DownloadProgress:
    """Byte progress update event."""

    file_name: str
    downloaded_bytes: int
    total_bytes: int
    speed_bps: float
    eta_seconds: Optional[float]

    @property
    def fraction(self) -> float:
        """Fraction of completion between 0.0 and 1.0."""
        if self.total_bytes <= 0:
            return 0.0
        return min(1.0, max(0.0, self.downloaded_bytes / self.total_bytes))


# Official Hugging Face and approved CDN/storage endpoints
DEFAULT_ALLOWED_DOMAINS: Set[str] = {
    "huggingface.co",
    "hf.co",
    "xethub.com",
    "xethub.hf.co",
}

DEFAULT_ALLOWED_SUFFIXES: Tuple[str, ...] = (
    ".huggingface.co",
    ".hf.co",
    ".xethub.com",
    ".xethub.hf.co",
)


def is_domain_trusted(hostname: str, *, allow_localhost: bool = False) -> bool:
    """Check if destination hostname is an approved source or CDN."""
    host = hostname.lower()
    if allow_localhost and host in ("127.0.0.1", "localhost", "::1"):
        return True
    if host in DEFAULT_ALLOWED_DOMAINS:
        return True
    for suffix in DEFAULT_ALLOWED_SUFFIXES:
        if host.endswith(suffix):
            return True
    return False


def validate_download_url(
    url: str,
    *,
    allow_http_test_transport: bool = False,
) -> None:
    """Validate URL protocol and host safety.

    Args:
        url: URL string.
        allow_http_test_transport: Whether unencrypted HTTP is allowed (loopback test fixtures only).

    Raises:
        InsecureTransportError: If scheme is HTTP and allow_http_test_transport is False or non-loopback.
        UntrustedRedirectError: If domain is not an approved HF/CDN host.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""

    if scheme != "https":
        if scheme == "http" and allow_http_test_transport and host.lower() in ("127.0.0.1", "localhost", "::1"):
            pass
        else:
            raise InsecureTransportError(
                f"Insecure transport '{scheme}' rejected for URL: {url}. "
                "Production downloads require HTTPS with certificate verification "
                "(HTTP test transport is strictly restricted to loopback)."
            )

    if not is_domain_trusted(host, allow_localhost=allow_http_test_transport):
        raise UntrustedRedirectError(
            f"Host '{host}' is not an approved model repository or CDN distribution domain."
        )


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _parse_content_range(header_val: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Parse Content-Range header (e.g. 'bytes 100-199/200')."""
    if not header_val:
        return None
    m = re.match(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", header_val.strip(), re.IGNORECASE)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        total = int(m.group(3)) if m.group(3) != "*" else -1
        return start, end, total
    return None


def download_file_resumable(
    *,
    url: str,
    destination_path: Union[str, Path],
    expected_sha256: str,
    expected_bytes: int,
    allow_http_test_transport: bool = False,
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    max_retries: int = 3,
    chunk_size: int = 128 * 1024,
    timeout: Tuple[float, float] = (10.0, 30.0),
    session: Optional[requests.Session] = None,
) -> Path:
    """Download a file with Range resume, integrity check, and cancellation support."""
    dest = Path(destination_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial_path = dest.with_name(dest.name + ".partial")

    # If destination already exists and matches hash, return immediately
    if dest.is_file() and dest.stat().st_size == expected_bytes:
        existing_hash = compute_file_sha256(dest)
        if existing_hash.lower() == expected_sha256.lower():
            if progress_callback:
                progress_callback(
                    DownloadProgress(
                        file_name=dest.name,
                        downloaded_bytes=expected_bytes,
                        total_bytes=expected_bytes,
                        speed_bps=0.0,
                        eta_seconds=0.0,
                    )
                )
            return dest

    owned_session: Optional[requests.Session] = None
    if session is None:
        owned_session = requests.Session()
        adapter = ModelDownloadAdapter(max_retries=Retry(total=0, connect=0, read=0))
        owned_session.mount("https://", adapter)
        owned_session.mount("http://", adapter)
        req_session = owned_session
    else:
        req_session = session

    expected_sha_lower = expected_sha256.lower().strip()
    attempt = 0

    try:
        while attempt < max_retries:
            attempt += 1
            if is_cancelled and is_cancelled():
                raise DownloadCancelledError("Download cancelled before request.")

            # Determine existing partial byte offset
            existing_bytes = 0
            if partial_path.is_file():
                existing_bytes = partial_path.stat().st_size
                if existing_bytes > expected_bytes:
                    # Oversized partial file: discard and restart
                    partial_path.unlink(missing_ok=True)
                    existing_bytes = 0
                elif existing_bytes == expected_bytes:
                    # Already complete? Verify hash
                    if compute_file_sha256(partial_path).lower() == expected_sha_lower:
                        partial_path.replace(dest)
                        return dest
                    # Corrupted completed file: restart
                    partial_path.unlink(missing_ok=True)
                    existing_bytes = 0

            headers: Dict[str, str] = {"Accept-Encoding": "identity"}
            if progress_callback:
                progress_callback(DownloadProgress(
                    file_name=dest.name, downloaded_bytes=existing_bytes,
                    total_bytes=expected_bytes, speed_bps=0.0, eta_seconds=None,
                ))
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            resp: Optional[requests.Response] = None
            try:
                # Handle redirects manually to rigorously validate every destination hop
                current_url = url
                redirect_count = 0
                max_redirects = 5

                while True:
                    if redirect_count > max_redirects:
                        raise DownloadError(f"Exceeded maximum redirect limit of {max_redirects} for {url}")

                    validate_download_url(
                        current_url,
                        allow_http_test_transport=allow_http_test_transport,
                    )

                    resp = req_session.get(
                        current_url,
                        headers=headers,
                        stream=True,
                        timeout=timeout,
                        allow_redirects=False,
                    )

                    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("Location")
                        resp.close()
                        resp = None
                        if not loc:
                            raise DownloadError("Redirect response missing Location header.")
                        next_url = urllib.parse.urljoin(current_url, loc)
                        validate_download_url(
                            next_url,
                            allow_http_test_transport=allow_http_test_transport,
                        )
                        current_url = next_url
                        redirect_count += 1
                        continue
                    break

                if resp is None:
                    raise DownloadError("No response received from server.")

                status = resp.status_code

                if status == 416:  # Range Not Satisfiable
                    resp.close()
                    resp = None
                    if partial_path.is_file():
                        if (
                            partial_path.stat().st_size == expected_bytes
                            and compute_file_sha256(partial_path).lower() == expected_sha_lower
                        ):
                            partial_path.replace(dest)
                            return dest
                        partial_path.unlink(missing_ok=True)
                    # Restart from scratch
                    existing_bytes = 0
                    continue

                if status not in (200, 206):
                    resp.raise_for_status()

                # Strictly require identity encoding for byte-exact raw range downloads
                content_encoding = resp.headers.get("Content-Encoding", "").strip().lower()
                if content_encoding and content_encoding != "identity":
                    resp.close()
                    resp = None
                    raise DownloadError(
                        f"Unexpected Content-Encoding '{content_encoding}' received from server; "
                        "identity encoding is required for byte-exact range downloads."
                    )

                write_mode = "ab"
                bytes_downloaded = existing_bytes

                if status == 200:
                    # Server sent full response (Range ignored or fresh request)
                    if "Content-Length" in resp.headers:
                        try:
                            server_total = int(resp.headers["Content-Length"])
                            if server_total != expected_bytes:
                                resp.close()
                                resp = None
                                partial_path.unlink(missing_ok=True)
                                raise DownloadError(
                                    f"Remote entity Content-Length ({server_total}) does not match "
                                    f"pinned expected bytes ({expected_bytes})"
                                )
                        except (ValueError, TypeError):
                            pass
                    write_mode = "wb"
                    bytes_downloaded = 0

                elif status == 206:
                    cr_header = resp.headers.get("Content-Range")
                    content_range = _parse_content_range(cr_header)
                    if not content_range:
                        # Missing or malformed Content-Range on 206 response
                        resp.close()
                        resp = None
                        partial_path.unlink(missing_ok=True)
                        existing_bytes = 0
                        continue

                    start_offset, end_offset, total_offset = content_range

                    # Check if remote entity total mismatches pinned catalog expected_bytes
                    if total_offset != -1 and total_offset != expected_bytes:
                        resp.close()
                        resp = None
                        partial_path.unlink(missing_ok=True)
                        raise DownloadError(
                            f"Remote entity size ({total_offset}) does not match "
                            f"pinned expected bytes ({expected_bytes})"
                        )

                    # Reject mismatched range boundaries (do not zero-pad or extend)
                    if (
                        start_offset != existing_bytes
                        or end_offset >= expected_bytes
                        or end_offset < start_offset
                    ):
                        resp.close()
                        resp = None
                        partial_path.unlink(missing_ok=True)
                        existing_bytes = 0
                        continue

                    # Check Content-Length if present
                    if "Content-Length" in resp.headers:
                        try:
                            cl = int(resp.headers["Content-Length"])
                            if cl != (end_offset - start_offset + 1):
                                resp.close()
                                resp = None
                                partial_path.unlink(missing_ok=True)
                                existing_bytes = 0
                                continue
                        except (ValueError, TypeError):
                            pass

                start_time = time.monotonic()
                last_progress_time = 0.0
                bytes_this_attempt = 0
                read_chunk_size = min(chunk_size, 32 * 1024)

                if resp is None:
                    raise DownloadError("No response received from server.")

                try:
                    with open(partial_path, write_mode) as f:
                        while True:
                            if is_cancelled and is_cancelled():
                                f.flush()
                                raise DownloadCancelledError("Download cancelled by user.")

                            chunk = resp.raw.read(read_chunk_size)
                            if not chunk:
                                break

                            f.write(chunk)
                            chunk_len = len(chunk)
                            bytes_downloaded += chunk_len
                            bytes_this_attempt += chunk_len

                            if bytes_downloaded > expected_bytes:
                                f.flush()
                                raise DownloadError(
                                    f"Received more bytes ({bytes_downloaded}) than expected ({expected_bytes})"
                                )

                            now = time.monotonic()
                            if (
                                now - last_progress_time >= 0.05
                                or bytes_downloaded == expected_bytes
                                or last_progress_time == 0.0
                            ):
                                elapsed = max(0.001, now - start_time)
                                speed = bytes_this_attempt / elapsed
                                remaining_bytes = max(0, expected_bytes - bytes_downloaded)
                                eta = remaining_bytes / speed if speed > 0 else None
                                if progress_callback:
                                    progress_callback(
                                        DownloadProgress(
                                            file_name=dest.name,
                                            downloaded_bytes=bytes_downloaded,
                                            total_bytes=expected_bytes,
                                            speed_bps=speed,
                                            eta_seconds=eta,
                                        )
                                    )
                                last_progress_time = now

                            if is_cancelled and is_cancelled():
                                f.flush()
                                raise DownloadCancelledError("Download cancelled by user.")

                        f.flush()
                        os.fsync(f.fileno())
                finally:
                    if resp is not None:
                        resp.close()
                        resp = None

                if bytes_downloaded < expected_bytes:
                    raise DownloadError(
                        f"Download stream closed prematurely: received {bytes_downloaded} of {expected_bytes} bytes."
                    )

                actual_sha = compute_file_sha256(partial_path).lower()
                if actual_sha != expected_sha_lower:
                    partial_path.unlink(missing_ok=True)
                    raise ChecksumMismatchError(
                        f"SHA-256 verification failed for {dest.name}: "
                        f"expected {expected_sha_lower}, computed {actual_sha}"
                    )

                partial_path.replace(dest)
                return dest

            except (DownloadCancelledError, ChecksumMismatchError, InsecureTransportError, UntrustedRedirectError):
                raise
            except Exception as exc:
                if attempt >= max_retries:
                    raise DownloadError(
                        f"Download failed after {attempt} attempts for {url}: {exc}"
                    ) from exc
                time.sleep(0.2 * (2 ** (attempt - 1)))
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                    resp = None
    finally:
        if owned_session is not None:
            try:
                owned_session.close()
            except Exception:
                pass

    raise DownloadError(f"Download failed after {max_retries} attempts: {url}")
