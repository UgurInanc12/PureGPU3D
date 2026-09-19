"""Unit tests for Depth Anything 3 catalog metadata and licensing policies."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.models.catalog import (
    get_model_entry,
    list_catalog_entries,
    load_catalog,
)


class TestModelCatalog(unittest.TestCase):
    """Test suite for the seven-model production catalog."""

    def setUp(self) -> None:
        self.catalog = load_catalog()

    def test_catalog_contains_seven_models(self) -> None:
        """Catalog must define exactly seven production models."""
        expected_ids = {
            "DA3-SMALL",
            "DA3-BASE",
            "DA3-LARGE-1.1",
            "DA3-GIANT-1.1",
            "DA3NESTED-GIANT-LARGE-1.1",
            "DA3MONO-LARGE",
            "DA3METRIC-LARGE",
        }
        self.assertEqual(set(self.catalog.keys()), expected_ids)

    def test_model_categories(self) -> None:
        """Five general-purpose models and two specialist models must be segregated."""
        general = list_catalog_entries(category="general", catalog=self.catalog)
        specialist = list_catalog_entries(category="specialist", catalog=self.catalog)

        self.assertEqual(len(general), 5)
        self.assertEqual(len(specialist), 2)

        general_ids = {m.id for m in general}
        specialist_ids = {m.id for m in specialist}

        self.assertIn("DA3-SMALL", general_ids)
        self.assertIn("DA3-BASE", general_ids)
        self.assertIn("DA3-LARGE-1.1", general_ids)
        self.assertIn("DA3-GIANT-1.1", general_ids)
        self.assertIn("DA3NESTED-GIANT-LARGE-1.1", general_ids)

        self.assertIn("DA3MONO-LARGE", specialist_ids)
        self.assertIn("DA3METRIC-LARGE", specialist_ids)

    def test_pinned_hashes_and_files_present(self) -> None:
        """Every model must specify config.json and model.safetensors with verified hashes."""
        for entry in self.catalog.values():
            self.assertIn("config.json", entry.files, f"{entry.id} missing config.json")
            self.assertIn("model.safetensors", entry.files, f"{entry.id} missing model.safetensors")

            cfg = entry.files["config.json"]
            self.assertGreater(cfg.bytes, 0)
            self.assertEqual(len(cfg.sha256), 64)

            wt = entry.files["model.safetensors"]
            self.assertGreater(wt.bytes, 100_000_000)
            self.assertEqual(len(wt.sha256), 64)

            # Revisions must be 40-character git commit hashes
            self.assertEqual(len(entry.revision), 40)

    def test_large_1_1_license_conflict_flagged(self) -> None:
        """Large 1.1 has contradictory upstream licenses and must block commercial claims."""
        large = get_model_entry("DA3-LARGE-1.1", self.catalog)
        self.assertTrue(large.license_info.license_conflict)
        self.assertTrue(large.commercial_clearance_blocked)
        self.assertTrue(large.license_info.noncommercial_ack_required)
        self.assertIsNotNone(large.license_info.conflict_details)
        self.assertIn("CC BY-NC 4.0", large.license_info.conflict_details or "")

    def test_giant_and_nested_noncommercial(self) -> None:
        """Giant and Nested models require non-commercial acknowledgment."""
        giant = get_model_entry("DA3-GIANT-1.1", self.catalog)
        nested = get_model_entry("DA3NESTED-GIANT-LARGE-1.1", self.catalog)

        self.assertTrue(giant.commercial_clearance_blocked)
        self.assertTrue(giant.license_info.noncommercial_ack_required)
        self.assertEqual(giant.license_info.license_type, "noncommercial")

        self.assertTrue(nested.commercial_clearance_blocked)
        self.assertTrue(nested.license_info.noncommercial_ack_required)
        self.assertEqual(nested.license_info.license_type, "noncommercial")

    def test_permissive_models(self) -> None:
        """Small, Base, Mono, and Metric must be Apache-2.0 without conflict."""
        for model_id in ("DA3-SMALL", "DA3-BASE", "DA3MONO-LARGE", "DA3METRIC-LARGE"):
            entry = get_model_entry(model_id, self.catalog)
            self.assertFalse(entry.commercial_clearance_blocked)
            self.assertFalse(entry.license_info.noncommercial_ack_required)
            self.assertFalse(entry.license_info.license_conflict)
            self.assertEqual(entry.license_info.license_type, "permissive")

    def test_model_entry_lookup_flexibility(self) -> None:
        """Lookup works by model ID, full repo ID, and friendly UI name."""
        by_id = get_model_entry("DA3-BASE", self.catalog)
        by_repo = get_model_entry("depth-anything/DA3-BASE", self.catalog)
        by_name = get_model_entry("Base", self.catalog)

        self.assertEqual(by_id.id, "DA3-BASE")
        self.assertEqual(by_repo.id, "DA3-BASE")
        self.assertEqual(by_name.id, "DA3-BASE")

    def test_unknown_identifier_raises(self) -> None:
        """Invalid or unapproved model identifier must raise KeyError."""
        with self.assertRaises(KeyError) as ctx:
            get_model_entry("DA3-FABRICATED-99", self.catalog)
        self.assertIn("DA3-FABRICATED-99", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
