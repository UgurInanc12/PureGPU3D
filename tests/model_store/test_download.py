"""Comprehensive unit tests for the resumable downloader using a local HTTP fixture server."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for p in (SRC_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.download import (
    ChecksumMismatchError,
    DownloadCancelledError,
    DownloadError,
    DownloadProgress,
    InsecureTransportError,
    UntrustedRedirectError,
    download_file_resumable,
)
from tests.model_store.download_fixtures import LocalHttpServerFixture, MockDownloadHandler


class TestResumableDownloader(unittest.TestCase):
    """Test suite exercising network edge cases against the local HTTP server fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server_port = LocalHttpServerFixture.start()

    @classmethod
    def tearDownClass(cls) -> None:
        LocalHttpServerFixture.stop()

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.target_dir = Path(self.temp_dir.name)
        MockDownloadHandler.reset_state()

        self.base_url = f"http://127.0.0.1:{self.server_port}/model.safetensors"
        self.raw_data = MockDownloadHandler.payload_data
        self.raw_sha256 = hashlib.sha256(self.raw_data).hexdigest()
        self.raw_len = len(self.raw_data)

    def tearDown(self) -> None:
        MockDownloadHandler.reset_state()
        self.temp_dir.cleanup()

    def test_clean_download_success(self) -> None:
        """Full uninterrupted download completes and validates checksum."""
        dest = self.target_dir / "weights.bin"
        promoted = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
        )
        self.assertEqual(promoted, dest)
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.stat().st_size, self.raw_len)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), self.raw_data)

    def test_resumable_download_after_drop(self) -> None:
        """Partial download interrupted mid-stream resumes seamlessly using Range 206."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")

        # 1. First attempt: simulate drop at 500 bytes
        MockDownloadHandler.simulate_drop_at_byte = 500
        with self.assertRaises(DownloadError):
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )

        self.assertTrue(partial.is_file())
        self.assertEqual(partial.stat().st_size, 500)
        self.assertFalse(dest.is_file())

        # 2. Second attempt: server clean, client resumes from byte 500
        MockDownloadHandler.simulate_drop_at_byte = None
        promoted = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
            max_retries=1,
        )

        self.assertTrue(dest.is_file())
        self.assertFalse(partial.is_file())
        self.assertEqual(dest.stat().st_size, self.raw_len)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), self.raw_data)

    def test_range_ignored_200_restart(self) -> None:
        """When server responds 200 OK to a Range request, partial file is safely restarted."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")

        partial.write_bytes(self.raw_data[:300])
        self.assertEqual(partial.stat().st_size, 300)

        MockDownloadHandler.force_200_on_range = True

        promoted = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
        )

        self.assertTrue(dest.is_file())
        self.assertEqual(dest.stat().st_size, self.raw_len)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), self.raw_data)

    def test_checksum_mismatch_rejected(self) -> None:
        """Corrupted payload fails hash verification and raises ChecksumMismatchError."""
        dest = self.target_dir / "weights.bin"
        MockDownloadHandler.corrupt_body = True

        with self.assertRaises(ChecksumMismatchError) as ctx:
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )

        self.assertIn("verification failed", str(ctx.exception).lower())
        self.assertFalse(dest.is_file())

    def test_oversized_content_rejected(self) -> None:
        """Payload exceeding expected_bytes raises DownloadError and aborts."""
        dest = self.target_dir / "weights.bin"
        MockDownloadHandler.serve_oversized = True

        with self.assertRaises(DownloadError):
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )
        self.assertFalse(dest.is_file())

    def test_short_content_rejected(self) -> None:
        """Payload shorter than expected_bytes raises DownloadError."""
        dest = self.target_dir / "weights.bin"
        MockDownloadHandler.serve_short = True

        with self.assertRaises(DownloadError) as ctx:
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )
        err_msg = str(ctx.exception).lower()
        self.assertTrue(
            "closed prematurely" in err_msg or "content-length" in err_msg,
            f"Unexpected error message: {err_msg}",
        )
        self.assertFalse(dest.is_file())

    def test_cancellation_mid_stream(self) -> None:
        """Cancellation flag immediately aborts download and retains partial data."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")

        calls = 0

        def should_cancel() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 3

        with self.assertRaises(DownloadCancelledError):
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                is_cancelled=should_cancel,
                chunk_size=64,
                max_retries=1,
            )

        self.assertTrue(partial.is_file())
        self.assertFalse(dest.is_file())

    def test_insecure_transport_rejected_by_default(self) -> None:
        """Plain HTTP rejected unless allow_http_test_transport is explicitly enabled."""
        dest = self.target_dir / "weights.bin"
        with self.assertRaises(InsecureTransportError) as ctx:
            download_file_resumable(
                url="http://huggingface.co/model.bin",
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=False,
            )
        self.assertIn("Insecure transport", str(ctx.exception))

    def test_untrusted_redirect_domain_rejected(self) -> None:
        """Redirect to untrusted external host is intercepted and rejected without connecting."""
        dest = self.target_dir / "weights.bin"
        MockDownloadHandler.redirect_target = "https://malicious-site.example.com/payload.bin"

        with self.assertRaises(UntrustedRedirectError):
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )
        self.assertFalse(dest.is_file())

    def test_changed_entity_metadata_rejected(self) -> None:
        """Mismatched entity total in Content-Range is rejected in favor of pinned catalog."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")
        partial.write_bytes(self.raw_data[:200])

        # Server claims total is 99999 instead of raw_len
        MockDownloadHandler.mismatched_content_range_total = 99999

        with self.assertRaises(DownloadError) as ctx:
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )
        self.assertIn("does not match pinned", str(ctx.exception))

    def test_strict_range_missing_content_range_restarts(self) -> None:
        """HTTP 206 missing Content-Range header safely discards partial and restarts cleanly."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")
        partial.write_bytes(self.raw_data[:200])

        # First request will be 206 without Content-Range; second request will be fresh 200
        MockDownloadHandler.missing_content_range_on_206 = True

        promoted = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
            max_retries=2,
        )
        self.assertEqual(promoted, dest)
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.stat().st_size, self.raw_len)

    def test_strict_range_mismatched_start_restarts_no_zero_padding(self) -> None:
        """HTTP 206 with mismatched start offset discards partial and restarts without zero-padding."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")
        partial.write_bytes(self.raw_data[:200])

        # Server returns start offset 800 instead of 200
        MockDownloadHandler.mismatched_content_range_start = 800

        # After discarding partial, next attempt sends no Range and receives 200 OK
        promoted = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
            max_retries=2,
        )
        self.assertEqual(promoted, dest)
        self.assertEqual(dest.stat().st_size, self.raw_len)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), self.raw_data)

    def test_http_416_range_not_satisfiable_handling(self) -> None:
        """HTTP 416 Range Not Satisfiable discards invalid partial or promotes if already complete."""
        dest = self.target_dir / "weights.bin"
        partial = dest.with_name(dest.name + ".partial")

        # 1. Partial is already complete and matches hash -> 416 promotes directly
        partial.write_bytes(self.raw_data)
        MockDownloadHandler.force_416_on_range = True

        promoted = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
            max_retries=1,
        )
        self.assertEqual(promoted, dest)
        self.assertTrue(dest.is_file())
        self.assertFalse(partial.is_file())

        # 2. Corrupted partial receives 416 on Range query -> partial deleted and retried without Range
        dest.unlink(missing_ok=True)
        partial.write_bytes(b"corrupted_partial_data_exceeding_offset")
        # force_416_on_range will send 416 on the first Range attempt, then client unlinks partial and retries clean 200
        promoted2 = download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
            max_retries=2,
        )
        self.assertEqual(promoted2, dest)
        self.assertEqual(dest.stat().st_size, self.raw_len)

    def test_max_redirect_limit_exceeded(self) -> None:
        """Redirect chains exceeding 5 hops explicitly fail with DownloadError."""
        dest = self.target_dir / "weights.bin"
        # Chain of 7 loopback redirects
        chain = [
            f"http://127.0.0.1:{self.server_port}/hop{i}"
            for i in range(7)
        ]
        MockDownloadHandler.redirect_chain = chain
        MockDownloadHandler.redirect_counter = 0

        with self.assertRaises(DownloadError) as ctx:
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )
        self.assertIn("maximum redirect limit", str(ctx.exception).lower())

    def test_http_test_transport_rejects_non_loopback(self) -> None:
        """allow_http_test_transport=True strictly rejects plain HTTP to external domains."""
        dest = self.target_dir / "weights.bin"
        with self.assertRaises(InsecureTransportError) as ctx:
            download_file_resumable(
                url="http://huggingface.co/model.bin",
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
            )
        self.assertIn("strictly restricted to loopback", str(ctx.exception))

    def test_untrusted_s3_and_cloudfront_rejected(self) -> None:
        """Arbitrary AWS S3 and CloudFront URLs are rejected without blanket whitelist."""
        dest = self.target_dir / "weights.bin"
        with self.assertRaises(UntrustedRedirectError):
            download_file_resumable(
                url="https://arbitrary-bucket.s3.amazonaws.com/model.bin",
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
            )

        with self.assertRaises(UntrustedRedirectError):
            download_file_resumable(
                url="https://d12345abcdef.cloudfront.net/model.bin",
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
            )

    def test_request_sends_identity_encoding(self) -> None:
        """Verify client explicitly requests Accept-Encoding: identity."""
        dest = self.target_dir / "weights.bin"
        download_file_resumable(
            url=self.base_url,
            destination_path=dest,
            expected_sha256=self.raw_sha256,
            expected_bytes=self.raw_len,
            allow_http_test_transport=True,
        )
        headers = MockDownloadHandler.last_request_headers
        self.assertIsNotNone(headers)
        assert headers is not None
        self.assertEqual(headers.get("Accept-Encoding"), "identity")

    def test_unexpected_gzip_content_encoding_rejected(self) -> None:
        """Verify that server returning Content-Encoding: gzip is rejected with DownloadError."""
        dest = self.target_dir / "weights.bin"
        MockDownloadHandler.serve_gzip = True
        with self.assertRaises(DownloadError) as ctx:
            download_file_resumable(
                url=self.base_url,
                destination_path=dest,
                expected_sha256=self.raw_sha256,
                expected_bytes=self.raw_len,
                allow_http_test_transport=True,
                max_retries=1,
            )
        self.assertIn("content-encoding", str(ctx.exception).lower())
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
