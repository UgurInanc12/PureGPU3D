"""Unit and safety tests for transactional staging, preflight collision, and validation."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.jobs.output_transaction import (
    OutputTransaction,
    PathCollisionError,
    TransactionError,
    ValidationError,
    paths_refer_to_same_file,
)
from puregpu3d.video.probe import find_ffmpeg


class TestOutputTransaction(unittest.TestCase):
    """Test suite for OutputTransaction safety invariants and atomic promotion."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = find_ffmpeg()

    def _create_dummy_video(self, path: Path, width: int = 320, height: int = 180, duration: float = 0.5) -> Path:
        cmd = [
            str(self.ffmpeg),
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate=12",
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            str(path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(res.returncode, 0)
        return path

    def test_identical_source_and_destination_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "source.mp4"
            media.write_bytes(b"dummy")

            with self.assertRaises(PathCollisionError) as ctx:
                OutputTransaction(media, media)
            self.assertIn("same physical file", str(ctx.exception))

    def test_canonical_and_relative_alias_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "video.mp4"
            media.write_bytes(b"dummy")

            alias = Path(tmpdir) / "." / "video.mp4"
            self.assertTrue(paths_refer_to_same_file(media, alias))

            with self.assertRaises(PathCollisionError) as ctx:
                OutputTransaction(media, alias)
            self.assertIn("same physical file", str(ctx.exception))

    def test_case_insensitive_path_collision_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "video_clip.mp4"
            media.write_bytes(b"dummy")

            # Mixed-case alias
            cased = Path(tmpdir) / "VIDEO_CLIP.MP4"
            self.assertTrue(paths_refer_to_same_file(media, cased))

            with self.assertRaises(PathCollisionError):
                OutputTransaction(media, cased)

    def test_existing_destination_rejected_when_overwrite_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"
            dst.write_bytes(b"existing_dest_bytes")

            with self.assertRaises(FileExistsError) as ctx:
                OutputTransaction(src, dst, overwrite=False)
            self.assertIn("already exists and overwrite=False", str(ctx.exception))

            # Destination must remain completely unchanged
            self.assertEqual(dst.read_bytes(), b"existing_dest_bytes")

    def test_staging_cleanup_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            staging_path: Path | None = None
            try:
                with OutputTransaction(src, dst) as txn:
                    staging_path = txn.staging_path
                    # Simulate writing partial data to staging
                    staging_path.write_bytes(b"incomplete data")
                    self.assertTrue(staging_path.exists())
                    # Simulate an unhandled exception or abort
                    raise RuntimeError("Simulated processing error")
            except RuntimeError:
                pass

            # Staging file must be automatically removed
            assert staging_path is not None
            self.assertFalse(staging_path.exists())
            # Destination must not exist
            self.assertFalse(dst.exists())

    def test_existing_destination_preserved_on_failure_with_overwrite_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"
            dst.write_bytes(b"original_precious_data")

            try:
                with OutputTransaction(src, dst, overwrite=True) as txn:
                    txn.staging_path.write_bytes(b"partial failure")
                    raise ValueError("Simulated crash")
            except ValueError:
                pass

            # Existing destination MUST be untouched
            self.assertTrue(dst.exists())
            self.assertEqual(dst.read_bytes(), b"original_precious_data")

    def test_validation_dimension_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            with OutputTransaction(
                src,
                dst,
                expected_width=640,
                expected_height=360,
            ) as txn:
                # Write video with wrong dimensions (320x180 instead of 640x360)
                self._create_dummy_video(txn.staging_path, width=320, height=180)

                with self.assertRaises(ValidationError) as err_ctx:
                    txn.validate_and_promote()
                self.assertIn("Dimension mismatch", str(err_ctx.exception))

            # Destination must not exist after failed validation
            self.assertFalse(dst.exists())

    def test_successful_validation_and_atomic_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            promoted: Path | None = None
            with OutputTransaction(
                src,
                dst,
                expected_width=320,
                expected_height=180,
            ) as txn:
                self._create_dummy_video(txn.staging_path, width=320, height=180)
                promoted = txn.validate_and_promote()

            self.assertEqual(promoted, dst)
            self.assertTrue(dst.exists())
            self.assertGreater(dst.stat().st_size, 0)
            self.assertFalse(txn.staging_path.exists())

    def test_exact_frame_count_contract_rejection(self) -> None:
        """Exact frame count contract: discrepancy of 1 must fail (no tolerance)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            # Dummy video has 6 frames (0.5s @ 12fps)
            # Tolerance=1 in old implementation allowed 5 or 7 frames to pass
            with OutputTransaction(src, dst, expected_frames=5) as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                with self.assertRaises(ValidationError) as ctx:
                    txn.validate_and_promote()
                self.assertIn("Frame count mismatch", str(ctx.exception))

            with OutputTransaction(src, dst, expected_frames=7) as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                with self.assertRaises(ValidationError) as ctx:
                    txn.validate_and_promote()
                self.assertIn("Frame count mismatch", str(ctx.exception))

            # Exact match (6 frames) must succeed
            with OutputTransaction(src, dst, expected_frames=6) as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                promoted = txn.validate_and_promote()
                self.assertEqual(promoted, dst)

    def test_destination_appears_after_preflight_overwrite_false(self) -> None:
        """When overwrite=False, destination created during processing must be refused."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            with OutputTransaction(src, dst, overwrite=False) as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                # Another process creates the destination file before promotion
                dst.write_bytes(b"concurrent_file_created")

                with self.assertRaises(FileExistsError) as ctx:
                    txn.validate_and_promote()
                self.assertIn("appeared after preflight", str(ctx.exception))

            # Existing destination must remain uncorrupted
            self.assertEqual(dst.read_bytes(), b"concurrent_file_created")

    def test_destination_modified_after_preflight_overwrite_true(self) -> None:
        """When overwrite=True, destination modified during processing must be refused."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"
            dst.write_bytes(b"original_precious_data")

            with OutputTransaction(src, dst, overwrite=True) as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                # Another process modifies the destination file before promotion
                import time
                time.sleep(0.02)
                dst.write_bytes(b"modified_data_by_other_process")

                with self.assertRaises(TransactionError) as ctx:
                    txn.validate_and_promote()
                self.assertIn("modified or replaced by another process", str(ctx.exception))

            # Modified destination must NOT be overwritten
            self.assertEqual(dst.read_bytes(), b"modified_data_by_other_process")

    def test_per_destination_process_lock_mutual_exclusion(self) -> None:
        """Per-destination advisory process lock prevents concurrent conflicting transactions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            with OutputTransaction(src, dst, lock_timeout_s=0.2) as txn1:
                # Concurrent transaction attempting to target the same destination must be locked out
                with self.assertRaises(TransactionError) as ctx:
                    with OutputTransaction(src, dst, lock_timeout_s=0.2) as txn2:
                        pass
                self.assertIn("Could not acquire process lock", str(ctx.exception))

    def test_frame_rate_mismatch_rejected(self) -> None:
        """Rational frame rate mismatch is strictly rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            # Video has 12 fps, expect 24/1
            with OutputTransaction(src, dst, expected_frame_rate="24/1") as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                with self.assertRaises(ValidationError) as ctx:
                    txn.validate_and_promote()
                self.assertIn("Frame rate mismatch", str(ctx.exception))

    def test_audio_stream_count_policy_mismatch_rejected(self) -> None:
        """Expected audio stream count mismatch is strictly rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.mp4"
            src.write_bytes(b"src")
            dst = Path(tmpdir) / "dest.mp4"

            # Video created without audio, expect 1 audio stream
            with OutputTransaction(src, dst, expected_audio_streams=1) as txn:
                self._create_dummy_video(txn.staging_path, duration=0.5)
                with self.assertRaises(ValidationError) as ctx:
                    txn.validate_and_promote()
                self.assertIn("Audio stream count mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
