#!/usr/bin/env python3
"""Convenience launcher script for PureGPU3D desktop application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure package src and DA3 vendor src are available in development mode
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for path in (SRC_DIR, VENDOR_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puregpu3d.desktop.app import run_desktop

if __name__ == "__main__":
    sys.exit(run_desktop())
