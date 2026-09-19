"""App-local path resolution and directory safety for PureGPU3D."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


class ReadOnlyAppRootError(PermissionError):
    """Raised when the PureGPU3D application root directory is read-only or not writable."""

    def __init__(self, root: Path, detail: str = "") -> None:
        self.root = root
        self.detail = detail
        msg = (
            f"PureGPU3D portable application root is not writable: '{root}'.\n"
            "PureGPU3D is a self-contained portable application that stores models, "
            "temporary job files, and application logs adjacent to the application executable.\n"
            "Action required: Move or extract the entire PureGPU3D application folder to a writable "
            "location (for example: C:\\PureGPU3D or a folder under your user profile with full write permissions).\n"
            "PureGPU3D refuses to silently redirect downloads into global user caches or elevate system permissions."
        )
        if detail:
            msg += f"\nDetail: {detail}"
        super().__init__(msg)


def validate_subpath(base_dir: Union[str, Path], subpath: Union[str, Path]) -> Path:
    """Validate that subpath does not escape base_dir via directory traversal.

    Args:
        base_dir: Base directory that must contain the resolved target.
        subpath: Relative subpath or user/manifest-provided component.

    Returns:
        Resolved Path guaranteed to reside inside base_dir.

    Raises:
        ValueError: If path traversal or escaping base_dir is detected.
    """
    base_resolved = Path(base_dir).resolve()
    # Normalize subpath string: reject suspicious traversal patterns early
    subpath_str = str(subpath).strip()
    if not subpath_str:
        return base_resolved

    # Reject null bytes and raw traversal components
    if "\0" in subpath_str:
        raise ValueError(f"Null byte detected in subpath: {subpath!r}")

    # Check for drive letters or UNC prefixes on Windows when given as relative
    parts = Path(subpath_str).parts
    if any(p in ("..", "..\\", "../") for p in parts):
        raise ValueError(f"Directory traversal component '..' detected in subpath: {subpath_str!r}")

    resolved = (base_resolved / subpath_str).resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as err:
        raise ValueError(
            f"Path traversal detected: '{subpath_str}' escapes base directory '{base_resolved}'"
        ) from err

    return resolved


def check_directory_writable(directory: Path) -> None:
    """Check that directory can be written to by creating and removing a probe file.

    Args:
        directory: Directory to test.

    Raises:
        ReadOnlyAppRootError: If probe creation or removal fails due to permissions.
    """
    directory.mkdir(parents=True, exist_ok=True)
    probe_name = f".write_probe_{os.getpid()}_{os.urandom(4).hex()}"
    probe_path = directory / probe_name
    try:
        with open(probe_path, "wb") as f:
            f.write(b"probe")
        if probe_path.exists():
            probe_path.unlink()
    except (PermissionError, OSError) as err:
        raise ReadOnlyAppRootError(directory, detail=str(err)) from err


def resolve_app_root(
    explicit_root: Optional[Union[str, Path]] = None,
    *,
    enforce_writable: bool = True,
) -> Path:
    """Resolve the application root directory.

    Priority:
    1. Explicit root (injected via argument or environment variable).
    2. Frozen executable directory (sys.executable parent, NOT PyInstaller _MEIPASS).
    3. Development repository root (inferred from package layout).

    Args:
        explicit_root: Explicit override path.
        enforce_writable: Whether to check and enforce write access to the root.

    Returns:
        Resolved Path to application root.

    Raises:
        ReadOnlyAppRootError: If root is read-only and enforce_writable is True.
    """
    if explicit_root is not None:
        root = Path(explicit_root).resolve()
    elif "PUREGPU3D_APP_ROOT" in os.environ:
        root = Path(os.environ["PUREGPU3D_APP_ROOT"]).resolve()
    elif getattr(sys, "frozen", False):
        # Frozen executable: use executable directory
        root = Path(sys.executable).resolve().parent
    else:
        # Development mode: find repository root from this file
        # this file is at src/puregpu3d/runtime/paths.py
        pkg_root = Path(__file__).resolve().parent.parent.parent.parent
        root = pkg_root.resolve()

    if enforce_writable:
        check_directory_writable(root)

    return root


@dataclass(frozen=True, kw_only=True)
class AppPaths:
    """Container for canonical PureGPU3D application paths."""

    root: Path
    models: Path
    data: Path
    downloads: Path
    licenses: Path
    logs: Path
    cache: Path

    @classmethod
    def from_root(
        cls,
        root: Union[str, Path],
        *,
        enforce_writable: bool = True,
    ) -> AppPaths:
        """Create AppPaths from a given root path."""
        resolved_root = Path(root).resolve()
        if enforce_writable:
            check_directory_writable(resolved_root)

        data = resolved_root / "data"
        return cls(
            root=resolved_root,
            models=resolved_root / "models",
            data=data,
            downloads=data / "downloads",
            licenses=resolved_root / "licenses",
            logs=data / "logs",
            cache=data / "cache",
        )

    def ensure_dirs(self) -> None:
        """Create standard application subdirectories."""
        for path in (
            self.root,
            self.models,
            self.data,
            self.downloads,
            self.licenses,
            self.logs,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_app_paths(
    explicit_root: Optional[Union[str, Path]] = None,
    *,
    enforce_writable: bool = True,
) -> AppPaths:
    """Get initialized AppPaths instance for the current application environment."""
    root = resolve_app_root(explicit_root=explicit_root, enforce_writable=enforce_writable)
    paths = AppPaths.from_root(root, enforce_writable=enforce_writable)
    paths.ensure_dirs()
    return paths
