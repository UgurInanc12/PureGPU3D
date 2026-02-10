from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Profile must be a mapping: {path}")
    return raw
