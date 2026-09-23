"""Unit tests for DesktopController logic, state machine, and model constraints."""

import tempfile
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

from puregpu3d.desktop.controller import DesktopController, DesktopState

# Ensure QApplication exists for Qt signals
app = QApplication.instance() or QApplication(["-platform", "offscreen"])

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"


class TestDesktopController(unittest.TestCase):
    """Test desktop controller state machine, model filtering, and media validation."""

    def setUp(self) -> None:
        self.controller = DesktopController()

    def test_initial_state(self) -> None:
        self.assertEqual(self.controller.state, DesktopState.IDLE)
        self.assertEqual(self.controller.selected_model_id, "DA3-SMALL")
        self.assertAlmostEqual(self.controller.disparity_strength, 0.001)
        self.assertAlmostEqual(self.controller.q_screen, 0.6)
        self.assertIsNone(self.controller.input_path)
        self.assertIsNone(self.controller.output_path)

    def test_seven_catalog_models_present(self) -> None:
        entries = self.controller.get_catalog_entries()
        self.assertEqual(len(entries), 7)

        ids = [e.id for e in entries]
        expected_ids = [
            "DA3-SMALL",
            "DA3-BASE",
            "DA3-LARGE-1.1",
            "DA3-GIANT-1.1",
            "DA3NESTED-GIANT-LARGE-1.1",
            "DA3MONO-LARGE",
            "DA3METRIC-LARGE",
        ]
        for expected in expected_ids:
            self.assertIn(expected, ids)

    def test_four_models_supported_and_three_unsupported_for_conversion(self) -> None:
        # Four models are supported: Small, Base, Mono Large, Metric Large
        supported_ids = [
            "DA3-SMALL",
            "DA3-BASE",
            "DA3MONO-LARGE",
            "DA3METRIC-LARGE",
        ]
        for sid in supported_ids:
            info = self.controller.get_model_info(sid)
            self.assertTrue(info["is_supported"], f"Model {sid} must be marked supported")
            self.assertIn("ready", info["support_note"].lower())

        # Three models MUST NOT be marked supported: Large 1.1, Giant 1.1, Nested Giant+Large 1.1
        unsupported_ids = [
            "DA3-LARGE-1.1",
            "DA3-GIANT-1.1",
            "DA3NESTED-GIANT-LARGE-1.1",
        ]
        for uid in unsupported_ids:
            info = self.controller.get_model_info(uid)
            self.assertFalse(info["is_supported"], f"Model {uid} must not be marked supported yet")
            self.assertIn("not yet integrated", info["support_note"].lower())

    def test_unsupported_model_blocks_conversion(self) -> None:
        if FIXTURE_VIDEO.is_file():
            self.controller.set_input_path(FIXTURE_VIDEO)
            self.controller.set_output_path(FIXTURE_VIDEO.with_name("out.mp4"))

            # When Large 1.1 is selected, validate_for_conversion must refuse
            self.controller.set_selected_model("DA3-LARGE-1.1")
            valid, reason = self.controller.validate_for_conversion()
            self.assertFalse(valid)
            self.assertIn("not yet integrated", reason.lower())

            # When Base is selected, validate_for_conversion succeeds
            self.controller.set_selected_model("DA3-BASE")
            valid, reason = self.controller.validate_for_conversion()
            self.assertTrue(valid, f"Validation failed for Base: {reason}")

            # When Small is selected, validate_for_conversion succeeds
            self.controller.set_selected_model("DA3-SMALL")
            valid, reason = self.controller.validate_for_conversion()
            self.assertTrue(valid, f"Validation failed for Small: {reason}")

    def test_input_probing_and_output_suggestion(self) -> None:
        if not FIXTURE_VIDEO.is_file():
            raise unittest.SkipTest(f"Fixture video not found at {FIXTURE_VIDEO}")

        success, msg = self.controller.set_input_path(FIXTURE_VIDEO)
        self.assertTrue(success)
        self.assertIsNotNone(self.controller.input_probe)
        probe = self.controller.input_probe
        assert probe is not None

        self.assertEqual(probe.width, 1920)
        self.assertEqual(probe.height, 1080)
        self.assertFalse(probe.is_hdr)
        self.assertFalse(probe.is_vfr)

        # Output path automatically suggested as FullSBS_LR
        self.assertIsNotNone(self.controller.output_path)
        assert self.controller.output_path is not None
        self.assertTrue(self.controller.output_path.name.endswith("_FullSBS_LR.mp4"))

    def test_input_equals_output_rejected(self) -> None:
        if not FIXTURE_VIDEO.is_file():
            raise unittest.SkipTest(f"Fixture video not found at {FIXTURE_VIDEO}")

        self.controller.set_input_path(FIXTURE_VIDEO)
        success, msg = self.controller.set_output_path(FIXTURE_VIDEO)
        self.assertFalse(success)
        self.assertIn("identical", msg.lower())

    def test_overwrite_confirmation_required(self) -> None:
        if not FIXTURE_VIDEO.is_file():
            raise unittest.SkipTest(f"Fixture video not found at {FIXTURE_VIDEO}")

        with tempfile.TemporaryDirectory() as tmpdir:
            existing_out = Path(tmpdir) / "existing.mp4"
            existing_out.write_bytes(b"dummy")

            self.controller.set_input_path(FIXTURE_VIDEO)
            self.controller.set_output_path(existing_out)

            # Attempt conversion without overwrite confirmation
            started, reason = self.controller.start_conversion(overwrite_confirmed=False)
            self.assertFalse(started)
            self.assertIn("already exists", reason.lower())


if __name__ == "__main__":
    unittest.main()
