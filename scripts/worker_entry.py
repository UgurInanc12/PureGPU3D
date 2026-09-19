#!/usr/bin/env python3
"""Dedicated conversion worker entrypoint for PureGPU3D PyInstaller packaging."""

from __future__ import annotations

import sys
from puregpu3d.runtime.worker import main

if __name__ == "__main__":
    sys.exit(main())
