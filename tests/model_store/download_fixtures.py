"""Test fixtures and mock HTTP server for model store and download tests."""

from __future__ import annotations

import http.server
import re
import socketserver
import threading
from typing import Optional


class MockDownloadHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP handler simulating range queries, drops, and edge conditions."""

    payload_data: bytes = b"PureGPU3D_Verified_Model_Weight_Data_Chunk_0123456789" * 40  # 2160 bytes
    simulate_drop_at_byte: Optional[int] = None
    force_200_on_range: bool = False
    corrupt_body: bool = False
    serve_oversized: bool = False
    serve_short: bool = False
    redirect_target: Optional[str] = None
    redirect_chain: Optional[list[str]] = None
    redirect_counter: int = 0
    missing_content_range_on_206: bool = False
    mismatched_content_range_start: Optional[int] = None
    mismatched_content_range_total: Optional[int] = None
    force_416: bool = False
    force_416_on_range: bool = False
    serve_gzip: bool = False
    last_request_headers: Optional[dict] = None

    def log_message(self, format: str, *args: object) -> None:
        pass

    @classmethod
    def reset_state(cls) -> None:
        cls.simulate_drop_at_byte = None
        cls.force_200_on_range = False
        cls.corrupt_body = False
        cls.serve_oversized = False
        cls.serve_short = False
        cls.redirect_target = None
        cls.redirect_chain = None
        cls.redirect_counter = 0
        cls.missing_content_range_on_206 = False
        cls.mismatched_content_range_start = None
        cls.mismatched_content_range_total = None
        cls.force_416 = False
        cls.force_416_on_range = False
        cls.serve_gzip = False
        cls.last_request_headers = None

    def do_GET(self) -> None:
        MockDownloadHandler.last_request_headers = dict(self.headers)
        if self.redirect_chain is not None:
            if self.redirect_counter < len(self.redirect_chain):
                target = self.redirect_chain[self.redirect_counter]
                MockDownloadHandler.redirect_counter += 1
                self.send_response(302)
                self.send_header("Location", target)
                self.end_headers()
                return

        if self.redirect_target:
            self.send_response(302)
            self.send_header("Location", self.redirect_target)
            self.end_headers()
            return

        body = self.payload_data
        if self.corrupt_body:
            body = b"corrupted_" + body[10:]
        elif self.serve_oversized:
            body = body + b"_EXTRA_OVERSIZED_BYTES"
        elif self.serve_short:
            body = body[: len(body) // 2]

        total_length = len(body)
        range_header = self.headers.get("Range")

        if self.serve_gzip:
            import gzip
            compressed = gzip.compress(body)
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(compressed)
            return

        if self.force_416 or (range_header and self.force_416_on_range):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total_length}")
            self.end_headers()
            return

        if range_header and not self.force_200_on_range:
            m = re.match(r"^bytes=(\d+)-(\d*)", range_header.strip())
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else total_length - 1
                if start >= total_length:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{total_length}")
                    self.end_headers()
                    return

                chunk = body[start : end + 1]
                self.send_response(206)

                if not self.missing_content_range_on_206:
                    reported_start = (
                        self.mismatched_content_range_start
                        if self.mismatched_content_range_start is not None
                        else start
                    )
                    reported_total = (
                        self.mismatched_content_range_total
                        if self.mismatched_content_range_total is not None
                        else total_length
                    )
                    self.send_header(
                        "Content-Range",
                        f"bytes {reported_start}-{end}/{reported_total}",
                    )

                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()

                if self.simulate_drop_at_byte is not None and self.simulate_drop_at_byte > 0:
                    slice_len = min(len(chunk), self.simulate_drop_at_byte)
                    self.wfile.write(chunk[:slice_len])
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(2)
                        self.connection.close()
                    except OSError:
                        pass
                    return

                self.wfile.write(chunk)
                return

        # Normal 200 response
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        if self.simulate_drop_at_byte is not None and self.simulate_drop_at_byte > 0:
            slice_len = min(len(body), self.simulate_drop_at_byte)
            self.wfile.write(body[:slice_len])
            self.wfile.flush()
            self.close_connection = True
            try:
                self.connection.shutdown(2)
                self.connection.close()
            except OSError:
                pass
            return

        self.wfile.write(body)


class LocalHttpServerFixture:
    """Manages lifecycle of local mock HTTP server."""

    server: Optional[socketserver.TCPServer] = None
    server_port: int = 0
    server_thread: Optional[threading.Thread] = None

    @classmethod
    def start(cls) -> int:
        if cls.server is None:
            cls.server = socketserver.TCPServer(("127.0.0.1", 0), MockDownloadHandler)
            cls.server_port = cls.server.server_address[1]
            cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
            cls.server_thread.start()
        return cls.server_port

    @classmethod
    def stop(cls) -> None:
        if cls.server is not None:
            cls.server.shutdown()
            cls.server.server_close()
            cls.server = None
            cls.server_port = 0
            cls.server_thread = None
