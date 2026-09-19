"""Unit tests for PureGPU3D application path resolution and safety."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.runtime.paths import (
    AppPaths,
    ReadOnlyAppRootError,
    check_directory_writable,
    get_app_paths,
    resolve_app_root,
    validate_subpath,
)


class TestAppPaths(unittest.TestCase):
    """Test path resolution, writable enforcement, and directory traversal guards."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_explicit_root_resolution(self) -> None:
        """Explicit root parameter overrides default detection and builds standard paths."""
        paths = get_app_paths(explicit_root=self.root)
        self.assertEqual(paths.root, self.root)
        self.assertEqual(paths.models, self.root / "models")
        self.assertEqual(paths.data, self.root / "data")
        self.assertEqual(paths.downloads, self.root / "data" / "downloads")
        self.assertEqual(paths.licenses, self.root / "licenses")
        self.assertEqual(paths.logs, self.root / "data" / "logs")
        self.assertEqual(paths.cache, self.root / "data" / "cache")
        # Ensure directories were created
        self.assertTrue(paths.models.is_dir())
        self.assertTrue(paths.downloads.is_dir())

    def test_read_only_root_rejection(self) -> None:
        """Read-only root must raise actionable ReadOnlyAppRootError."""
        # Simulate read-only directory by mocking check_directory_writable to raise PermissionError
        with patch("puregpu3d.runtime.paths.check_directory_writable") as mock_check:
            mock_check.side_effect = ReadOnlyAppRootError(self.root, detail="Access denied")
            with self.assertRaises(ReadOnlyAppRootError) as ctx:
                resolve_app_root(explicit_root=self.root, enforce_writable=True)

            self.assertIn("PureGPU3D portable application root is not writable", str(ctx.exception))
            self.assertIn("Action required", str(ctx.exception))

    def test_validate_subpath_valid(self) -> None:
        """Valid subpaths inside base directory resolve safely."""
        base = self.root / "models"
        base.mkdir()
        target = validate_subpath(base, "DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3")
        self.assertEqual(
            target,
            (base / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3").resolve(),
        )

    def test_validate_subpath_traversal_parent_rejected(self) -> None:
        """Paths attempting to traverse upward outside base directory are rejected."""
        base = self.root / "models"
        base.mkdir()
        with self.assertRaises(ValueError) as ctx:
            validate_subpath(base, "../secret.txt")
        self.assertIn("traversal", str(ctx.exception).lower())

    def test_validate_subpath_nested_traversal_rejected(self) -> None:
        """Deep traversal attempts using .. are caught and rejected."""
        base = self.root / "models"
        base.mkdir()
        with self.assertRaises(ValueError):
            validate_subpath(base, "DA3-SMALL/../../passwords.txt")

    def test_validate_subpath_null_byte_rejected(self) -> None:
        """Null byte injection is rejected."""
        base = self.root / "models"
        base.mkdir()
        with self.assertRaises(ValueError) as ctx:
            validate_subpath(base, "DA3-SMALL\0payload")
        self.assertIn("null byte", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
