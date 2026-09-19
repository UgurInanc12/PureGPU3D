#!/usr/bin/env python3
"""
Reproducible setup and asset download script for DA3 Small feasibility probe.
Downloads config.json and model.safetensors for revision e08cab65ca0ec38e7826075418411ab90cab4da3.
Forces IPv4 to avoid Windows IPv6 stalls, verifies byte size and SHA-256 integrity,
and writes manifest.json and READY marker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

# Workaround for Windows IPv6 stalls in urllib3/requests
try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
except ImportError:
    pass

import requests

MODEL_ID = "depth-anything/DA3-SMALL"
REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"

EXPECTED_FILES = {
    "config.json": {
        "url": f"https://huggingface.co/{MODEL_ID}/raw/{REVISION}/config.json",
        "sha256": "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2",
        "bytes": 1202,
    },
    "model.safetensors": {
        "url": f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/model.safetensors",
        "sha256": "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf",
        "bytes": 137248940,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, expected_bytes: int, expected_sha256: str, timeout: int = 60) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = dest_path.with_suffix(dest_path.suffix + ".partial")

    if dest_path.exists():
        actual_bytes = dest_path.stat().st_size
        if actual_bytes == expected_bytes:
            actual_sha = sha256_file(dest_path)
            if actual_sha == expected_sha256:
                print(f"[OK] Already present and verified: {dest_path.name}")
                return
            else:
                print(f"[WARN] Checksum mismatch for existing {dest_path.name}, re-downloading...")
        else:
            print(f"[WARN] Size mismatch for existing {dest_path.name} ({actual_bytes} != {expected_bytes}), re-downloading...")

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    headers = {"User-Agent": "PureGPU3D-Setup/1.0"}
    session = requests.Session()
    session.trust_env = True

    start_time = time.time()
    with session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        downloaded = 0
        h = hashlib.sha256()
        with open(partial_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    h.update(chunk)
                    downloaded += len(chunk)
                    if expected_bytes > 0:
                        pct = (downloaded / expected_bytes) * 100
                        print(f"\r  Progress: {downloaded / 1024 / 1024:.1f} MB / {expected_bytes / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="", flush=True)

    print()
    elapsed = time.time() - start_time
    print(f"Downloaded {downloaded} bytes in {elapsed:.1f}s ({(downloaded / 1024 / 1024) / max(elapsed, 0.001):.2f} MB/s)")

    actual_sha = h.hexdigest()
    if expected_bytes > 0 and downloaded != expected_bytes:
        partial_path.unlink(missing_ok=True)
        raise ValueError(f"Byte count mismatch for {dest_path.name}: expected {expected_bytes}, got {downloaded}")

    if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
        partial_path.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {dest_path.name}: expected {expected_sha256}, got {actual_sha}")

    if dest_path.exists():
        dest_path.unlink()
    partial_path.rename(dest_path)
    print(f"[VERIFIED] {dest_path.name} sha256={actual_sha}")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models" / "DA3-SMALL"
    rev_dir = models_dir / REVISION

    print(f"Setting up DA3 Small checkpoint at {rev_dir}...")
    rev_dir.mkdir(parents=True, exist_ok=True)

    for fname, meta in EXPECTED_FILES.items():
        dest = rev_dir / fname
        download_file(meta["url"], dest, meta["bytes"], meta["sha256"])
        # Also mirror/copy to top-level models/DA3-SMALL/ for convenience
        top_dest = models_dir / fname
        if not top_dest.exists() or top_dest.stat().st_size != dest.stat().st_size:
            shutil.copy2(dest, top_dest)

    # Write manifest.json
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            fname: {
                "sha256": meta["sha256"],
                "bytes": meta["bytes"],
            }
            for fname, meta in EXPECTED_FILES.items()
        },
        "license": "Apache-2.0",
    }
    manifest_path = rev_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest written to {manifest_path}")

    # Write READY marker
    ready_marker = rev_dir / "READY"
    with open(ready_marker, "w", encoding="utf-8") as f:
        f.write(f"READY {REVISION}\n")
    print(f"[OK] READY marker written to {ready_marker}")

    # Also top level READY
    top_ready = models_dir / "READY"
    with open(top_ready, "w", encoding="utf-8") as f:
        f.write(f"READY {REVISION}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
