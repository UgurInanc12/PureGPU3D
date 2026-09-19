"""Tests for PureGPU3D model store, catalog, and secure downloader."""

import sys
from pathlib import Path

# Ensure src is reachable without external PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
