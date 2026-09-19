"""Tests for generalized parameter alias detection, strict loading, and security verification."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
from safetensors.torch import load_file

from puregpu3d.models.catalog import load_catalog
from puregpu3d.models.da3_adapter import DA3DepthAdapter, ensure_da3_vendor_import

ensure_da3_vendor_import()
from depth_anything_3.cfg import create_object


class TestGeneralizedAliasResolution(unittest.TestCase):
    """Deep verification of generalized alias resolution across model architectures."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.base_dir = REPO_ROOT / "models" / "DA3-BASE" / cls.catalog["DA3-BASE"].revision
        cls.mono_dir = REPO_ROOT / "models" / "DA3MONO-LARGE" / cls.catalog["DA3MONO-LARGE"].revision

        with open(cls.base_dir / "config.json", "r", encoding="utf-8") as f:
            cls.base_cfg = json.load(f)["config"]
        cls.base_model = create_object(cls.base_cfg)
        cls.base_tensors = load_file(str(cls.base_dir / "model.safetensors"))

    def test_alias_group_discovery_in_base_model(self) -> None:
        """Verify that DualDPT head parameter sharing is correctly detected by identity."""
        param_groups = {}
        for name, param in self.base_model.named_parameters(remove_duplicate=False):
            param_groups.setdefault(id(param), []).append(name)

        aliased_groups = [names for names in param_groups.values() if len(names) > 1]
        self.assertEqual(len(aliased_groups), 2)  # weight and bias for output_conv2_aux

        for group in aliased_groups:
            self.assertEqual(len(group), 4)  # 4 aux levels
            # Verify data pointers match exactly across all 4 levels
            base_p = self.base_model.get_parameter(group[0])
            for alias_name in group[1:]:
                alias_p = self.base_model.get_parameter(alias_name)
                self.assertIs(alias_p, base_p)
                self.assertEqual(alias_p.data_ptr(), base_p.data_ptr())

    def test_alias_restoration_produces_strict_loadable_state_dict(self) -> None:
        """Verify that resolve_aliases populates deduplicated keys for strict load."""
        resolved = DA3DepthAdapter.resolve_aliases(self.base_model, self.base_tensors)

        # Check all 4 levels exist in resolved state_dict
        for level in range(4):
            self.assertIn(f"head.scratch.output_conv2_aux.{level}.2.weight", resolved)
            self.assertIn(f"head.scratch.output_conv2_aux.{level}.2.bias", resolved)

        # Strict load must pass with 0 missing and 0 unexpected keys
        load_res = self.base_model.load_state_dict(resolved, strict=True)
        self.assertEqual(len(load_res.missing_keys), 0)
        self.assertEqual(len(load_res.unexpected_keys), 0)

    def test_rejection_of_contradictory_alias_weights(self) -> None:
        """Verify that if weights file contains conflicting values for aliased parameters, it is rejected."""
        tensors_copy = dict(self.base_tensors)

        # Simulate a corrupted checkpoint where level 0 and level 1 weights were both saved but differ
        k0 = "model.head.scratch.output_conv2_aux.0.2.weight"
        k1 = "model.head.scratch.output_conv2_aux.1.2.weight"
        tensors_copy[k1] = tensors_copy[k0].clone() + 1.0  # Deliberately contradict

        with self.assertRaises(RuntimeError) as ctx:
            DA3DepthAdapter.resolve_aliases(self.base_model, tensors_copy)

        self.assertIn("Contradictory alias tensors detected", str(ctx.exception))

    def test_genuine_missing_parameter_rejection(self) -> None:
        """Verify that deleting a non-aliased parameter causes strict loading to fail."""
        resolved = DA3DepthAdapter.resolve_aliases(self.base_model, self.base_tensors)
        key_to_delete = "backbone.pretrained.cls_token"
        self.assertIn(key_to_delete, resolved)
        del resolved[key_to_delete]

        with self.assertRaises(RuntimeError) as ctx:
            self.base_model.load_state_dict(resolved, strict=True)
        self.assertIn(key_to_delete, str(ctx.exception))

    def test_security_rejection_of_untrusted_module_imports(self) -> None:
        """Verify that configs requesting arbitrary module imports outside whitelist are rejected."""
        malicious_config = {
            "__object__": {
                "path": "subprocess.run",
                "name": "run",
            }
        }
        with self.assertRaises(ValueError) as ctx:
            DA3DepthAdapter._audit_config_security(malicious_config)
        self.assertIn("Untrusted module path in model config", str(ctx.exception))

    def test_adapter_rejects_unregistered_model_directory(self) -> None:
        """Verify that arbitrary unverified model directories are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg_file = tmp_path / "config.json"
            wt_file = tmp_path / "model.safetensors"
            cfg_file.write_text('{"config": {}}', encoding="utf-8")
            wt_file.write_bytes(b"dummy")

            with self.assertRaises(ValueError) as ctx:
                DA3DepthAdapter(model_dir=tmp_path, identifier=None, verify_hashes=False)
            self.assertIn("does not match any audited catalog entry", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
