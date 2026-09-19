#!/usr/bin/env python3
"""Desktop application entrypoint for PureGPU3D PyInstaller packaging."""

from __future__ import annotations

import sys
from puregpu3d.desktop.app import run_desktop

if __name__ == "__main__":
    sys.exit(run_desktop())
