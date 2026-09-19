"""Audited Depth Anything 3 model catalog definitions and metadata."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True, kw_only=True)
class ModelFileSpec:
    """Specification for an audited, pinned model file."""

    bytes: int
    sha256: str


@dataclass(frozen=True, kw_only=True)
class ModelLicenseInfo:
    """License disclosure and commercial clearance status."""

    license: str
    license_type: str  # 'permissive', 'noncommercial', 'conflict'
    noncommercial_ack_required: bool
    license_conflict: bool
    conflict_details: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class ModelCatalogEntry:
    """Catalog entry for a verified Depth Anything 3 model checkpoint."""

    id: str
    repo_id: str
    ui_name: str
    revision: str
    parameters: str
    role: str
    category: str  # 'general' or 'specialist'
    license_info: ModelLicenseInfo
    files: Dict[str, ModelFileSpec]

    @property
    def total_weight_bytes(self) -> int:
        """Total size of model weights in bytes."""
        wt = self.files.get("model.safetensors")
        return wt.bytes if wt else 0

    @property
    def total_bytes(self) -> int:
        """Total size of all required files in bytes."""
        return sum(f.bytes for f in self.files.values())

    @property
    def is_specialist(self) -> bool:
        """True if the model is a specialist (mono/metric), not general-purpose any-view."""
        return self.category == "specialist"

    @property
    def commercial_clearance_blocked(self) -> bool:
        """True if commercial clearance cannot be claimed (non-commercial or conflicting license)."""
        return self.license_info.license_conflict or self.license_info.license_type == "noncommercial"


def _find_catalog_resource(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Locate resources/models.json file across development and packaged layouts."""
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Custom catalog file not found: {p}")
        return p

    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        bundle_dir = Path(getattr(sys, "_MEIPASS", exe_dir))
        candidates.extend([
            exe_dir / "resources" / "models.json",
            bundle_dir / "resources" / "models.json",
            bundle_dir / "puregpu3d" / "resources" / "models.json",
        ])

    candidates.extend([
        # Relative to this source file: src/puregpu3d/models/catalog.py -> repo_root/resources/models.json
        Path(__file__).resolve().parent.parent.parent.parent / "resources" / "models.json",
        # Package internal resources if vendored
        Path(__file__).resolve().parent / "resources" / "models.json",
    ])

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not locate resources/models.json in repository or package resources."
    )


def _entry_from_dict(data: Dict[str, Any]) -> ModelCatalogEntry:
    """Construct a ModelCatalogEntry from a dictionary."""
    files = {
        name: ModelFileSpec(
            bytes=int(meta["bytes"]),
            sha256=str(meta["sha256"]).lower(),
        )
        for name, meta in data["files"].items()
    }

    license_info = ModelLicenseInfo(
        license=str(data["license"]),
        license_type=str(data["license_type"]),
        noncommercial_ack_required=bool(data["noncommercial_ack_required"]),
        license_conflict=bool(data["license_conflict"]),
        conflict_details=data.get("conflict_details"),
    )

    return ModelCatalogEntry(
        id=str(data["id"]),
        repo_id=str(data["repo_id"]),
        ui_name=str(data["ui_name"]),
        revision=str(data["revision"]),
        parameters=str(data["parameters"]),
        role=str(data["role"]),
        category=str(data["category"]),
        license_info=license_info,
        files=files,
    )


def load_catalog(
    catalog_path: Optional[Union[str, Path]] = None,
) -> Dict[str, ModelCatalogEntry]:
    """Load the official model catalog from resources/models.json.

    Returns:
        Mapping from model ID (e.g. 'DA3-SMALL') to ModelCatalogEntry.
    """
    path = _find_catalog_resource(catalog_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_models = payload.get("models", [])
    catalog: Dict[str, ModelCatalogEntry] = {}
    for item in raw_models:
        entry = _entry_from_dict(item)
        catalog[entry.id] = entry

    return catalog


def list_catalog_entries(
    category: Optional[str] = None,
    catalog: Optional[Dict[str, ModelCatalogEntry]] = None,
) -> List[ModelCatalogEntry]:
    """List catalog entries, optionally filtered by category ('general' or 'specialist')."""
    cat = catalog if catalog is not None else load_catalog()
    entries = list(cat.values())
    if category is not None:
        entries = [e for e in entries if e.category == category]
    return entries


def get_model_entry(
    identifier: str,
    catalog: Optional[Dict[str, ModelCatalogEntry]] = None,
) -> ModelCatalogEntry:
    """Retrieve a catalog entry by ID, repo_id, or ui_name.

    Args:
        identifier: e.g. 'DA3-SMALL', 'depth-anything/DA3-SMALL', or 'Small'.
        catalog: Optional preloaded catalog mapping.

    Returns:
        The matched ModelCatalogEntry.

    Raises:
        KeyError: If identifier does not match any catalog entry.
    """
    cat = catalog if catalog is not None else load_catalog()
    # 1. Direct ID match
    if identifier in cat:
        return cat[identifier]

    # 2. Case-insensitive or repo_id / ui_name lookup
    ident_lower = identifier.strip().lower()
    for entry in cat.values():
        if (
            entry.id.lower() == ident_lower
            or entry.repo_id.lower() == ident_lower
            or entry.ui_name.lower() == ident_lower
            or entry.repo_id.split("/")[-1].lower() == ident_lower
        ):
            return entry

    available = ", ".join(cat.keys())
    raise KeyError(f"Unknown model identifier '{identifier}'. Available models: {available}")
