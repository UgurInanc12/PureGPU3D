"""Tests for Depth Anything 3 catalog alignment, license status, and metadata integrity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from puregpu3d.models.catalog import get_model_entry, list_catalog_entries, load_catalog
from puregpu3d.models.da3_adapter import DA3DepthAdapter


class TestCatalogAlignment(unittest.TestCase):
    """Verify catalog specifications across all 7 supported DA3 models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_catalog_contains_all_seven_models(self) -> None:
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
        self.assertEqual(set(DA3DepthAdapter.get_supported_model_ids()), expected_ids)

    def test_verified_and_unverified_model_statuses(self) -> None:
        verified = DA3DepthAdapter.get_verified_model_ids()
        unverified = DA3DepthAdapter.get_unverified_model_ids()

        self.assertEqual(set(verified), {"DA3-SMALL", "DA3-BASE", "DA3MONO-LARGE", "DA3METRIC-LARGE"})
        self.assertEqual(set(unverified), {"DA3-LARGE-1.1", "DA3-GIANT-1.1", "DA3NESTED-GIANT-LARGE-1.1"})

        # Verified models must all be permissive Apache-2.0
        for m_id in verified:
            entry = self.catalog[m_id]
            self.assertEqual(entry.license_info.license_type, "permissive")
            self.assertFalse(entry.license_info.noncommercial_ack_required)
            self.assertFalse(entry.license_info.license_conflict)
            self.assertFalse(entry.commercial_clearance_blocked)

        # Unverified models must require noncommercial acknowledgment or flag conflict
        for m_id in unverified:
            entry = self.catalog[m_id]
            self.assertTrue(entry.license_info.noncommercial_ack_required)
            self.assertTrue(entry.commercial_clearance_blocked)

    def test_required_file_specs_present(self) -> None:
        for m_id, entry in self.catalog.items():
            self.assertIn("config.json", entry.files)
            self.assertIn("model.safetensors", entry.files)
            self.assertGreater(entry.files["config.json"].bytes, 0)
            self.assertGreater(entry.files["model.safetensors"].bytes, 0)
            self.assertEqual(len(entry.files["config.json"].sha256), 64)
            self.assertEqual(len(entry.files["model.safetensors"].sha256), 64)

    def test_specialist_classification(self) -> None:
        mono = get_model_entry("DA3MONO-LARGE", self.catalog)
        metric = get_model_entry("DA3METRIC-LARGE", self.catalog)
        base = get_model_entry("DA3-BASE", self.catalog)
        small = get_model_entry("DA3-SMALL", self.catalog)

        self.assertTrue(mono.is_specialist)
        self.assertTrue(metric.is_specialist)
        self.assertFalse(base.is_specialist)
        self.assertFalse(small.is_specialist)


if __name__ == "__main__":
    unittest.main()
