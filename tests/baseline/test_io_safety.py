"""Characterization tests for input/output safety, path collisions, and file transaction boundaries.

Documents missing guards in the baseline implementation without destructive same-file runs.
"""

import tempfile
import unittest
from pathlib import Path

from puregpu3d.config.types import EngineConfig


class TestIoSafety(unittest.TestCase):
    """Characterize input/output safety properties and identify gaps for Phase 3 safe transaction."""

    def test_input_equals_output_unprotected_in_baseline(self) -> None:
        """Baseline does NOT reject input_path == output_path.

        If executed, FFmpeg remux with -y would overwrite source media.
        Phase 3 must add canonical path collision rejection.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            media_path = tmp / "sample.mp4"
            media_path.write_bytes(b"dummy_media_bytes")

            # Identical Path objects
            cfg_identical = EngineConfig(input_path=media_path, output_path=media_path)
            # Baseline behavior: passes validate() without error or warning
            try:
                cfg_identical.validate()
                unprotected = True
            except (ValueError, RuntimeError):
                unprotected = False

            self.assertTrue(
                unprotected,
                "Baseline was expected to allow identical input/output path (documenting missing guard).",
            )

            # Equivalent aliased Path objects
            aliased_path = tmp / "." / "sample.mp4"
            self.assertEqual(media_path.resolve(), aliased_path.resolve())

            cfg_alias = EngineConfig(input_path=media_path, output_path=aliased_path)
            try:
                cfg_alias.validate()
                alias_unprotected = True
            except (ValueError, RuntimeError):
                alias_unprotected = False

            self.assertTrue(
                alias_unprotected,
                "Baseline was expected to allow aliased input/output path without collision detection.",
            )

    def test_existing_output_overwrite_unprotected_in_baseline(self) -> None:
        """Baseline does NOT verify whether output_path already exists.

        Baseline uses FFmpeg -y unconditionally, which overwrites without confirmation.
        Phase 3 must introduce destination locks and confirmation guards.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            media_in = tmp / "input.mp4"
            media_in.write_bytes(b"input_data")

            media_out = tmp / "output_already_exists.mp4"
            media_out.write_bytes(b"pre_existing_completed_job")

            cfg = EngineConfig(input_path=media_in, output_path=media_out)
            # Baseline passes validation even when output_path pre-exists
            cfg.validate()
            self.assertTrue(media_out.exists())

    def test_staging_path_construction(self) -> None:
        """Characterize baseline staging naming convention for intermediate video."""
        out_standard = Path("E:/Media/export.mp4")
        staged = out_standard.with_suffix(".video_only.mp4")
        self.assertEqual(staged, Path("E:/Media/export.video_only.mp4"))

        # Collision avoidance when output already has .video_only.mp4 suffix
        out_collision = Path("E:/Media/export.video_only.mp4")
        if out_collision.with_suffix(".video_only.mp4") == out_collision:
            staged_fallback = out_collision.with_name(out_collision.stem + ".video_only.mp4")
        else:
            staged_fallback = out_collision.with_suffix(".video_only.mp4")
        self.assertEqual(staged_fallback, Path("E:/Media/export.video_only.video_only.mp4"))


if __name__ == "__main__":
    unittest.main()
