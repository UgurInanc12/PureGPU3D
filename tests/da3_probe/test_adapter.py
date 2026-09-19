"""Unit and integration tests for DA3 Small depth adapter and inference probe.

Verifies:
  - Checkpoint manifest and SHA-256 hash validation
  - In-memory parameter identity for shared LayerNorm across DualDPT aux levels
  - Exact Safetensors alias resolution under strict state dict loading
  - Rejection of genuinely missing or unexpected parameters
  - Canonical preprocessing (aspect ratio preservation, 14-multiple rounding, ImageNet normalization)
  - Actual CPU float32 and CUDA float16 inference execution
  - Numerical consistency across CPU and CUDA
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure repository root and vendor are reachable without external PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
VENDOR_SRC = REPO_ROOT / "third_party" / "depth_anything_3" / "src"

for p in (SRC_DIR, VENDOR_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from safetensors.torch import load_file

from puregpu3d.models.da3_adapter import (
    DEFAULT_PROCESS_RES,
    DEFAULT_REVISION,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_WEIGHTS_SHA256,
    PATCH_SIZE,
    DA3SmallDepthAdapter,
    DepthPredictionResult,
    compute_sha256,
)

CHECKPOINT_DIR = REPO_ROOT / "models" / "DA3-SMALL" / DEFAULT_REVISION
SAMPLE_IMAGE_PATH = REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"


class TestCheckpointIntegrity(unittest.TestCase):
    """Test checkpoint verification against pinned hashes and manifests."""

    def test_checkpoint_files_exist(self) -> None:
        self.assertTrue(CHECKPOINT_DIR.is_dir(), f"Directory missing: {CHECKPOINT_DIR}")
        config_path = CHECKPOINT_DIR / "config.json"
        weights_path = CHECKPOINT_DIR / "model.safetensors"
        manifest_path = CHECKPOINT_DIR / "manifest.json"
        self.assertTrue(config_path.is_file(), f"Missing {config_path}")
        self.assertTrue(weights_path.is_file(), f"Missing {weights_path}")
        self.assertTrue(manifest_path.is_file(), f"Missing {manifest_path}")

    def test_pinned_hashes_match(self) -> None:
        config_path = CHECKPOINT_DIR / "config.json"
        weights_path = CHECKPOINT_DIR / "model.safetensors"
        self.assertEqual(compute_sha256(config_path), EXPECTED_CONFIG_SHA256)
        self.assertEqual(compute_sha256(weights_path), EXPECTED_WEIGHTS_SHA256)

    def test_manifest_schema_and_values(self) -> None:
        manifest_path = CHECKPOINT_DIR / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["revision"], DEFAULT_REVISION)
        self.assertEqual(manifest["files"]["config.json"]["sha256"], EXPECTED_CONFIG_SHA256)
        self.assertEqual(manifest["files"]["model.safetensors"]["sha256"], EXPECTED_WEIGHTS_SHA256)
        self.assertEqual(manifest["files"]["config.json"]["bytes"], (CHECKPOINT_DIR / "config.json").stat().st_size)
        self.assertEqual(manifest["files"]["model.safetensors"]["bytes"], (CHECKPOINT_DIR / "model.safetensors").stat().st_size)

    def test_invalid_hash_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corrupt_cfg = tmp_path / "config.json"
            corrupt_wt = tmp_path / "model.safetensors"
            corrupt_cfg.write_text('{"config": {}}', encoding="utf-8")
            corrupt_wt.write_bytes(b"dummy corrupted weights")

            with self.assertRaises(ValueError) as ctx:
                DA3SmallDepthAdapter(model_dir=tmp_path, verify_hashes=True)
            self.assertIn("mismatch", str(ctx.exception).lower())


class TestSharedLayerNormAliasResolution(unittest.TestCase):
    """Deep verification of DualDPT parameter identity and strict alias resolution."""

    @classmethod
    def setUpClass(cls) -> None:
        from depth_anything_3.cfg import create_object
        with open(CHECKPOINT_DIR / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cls.raw_model = create_object(cfg["config"])
        cls.raw_tensors = load_file(str(CHECKPOINT_DIR / "model.safetensors"))

    def test_parameter_identity_in_architecture(self) -> None:
        """Verify that the instantiated DualDPT head reuses identical nn.LayerNorm instances."""
        aux_list = self.raw_model.head.scratch.output_conv2_aux
        self.assertEqual(len(aux_list), 4)

        base_ln = aux_list[0][2]
        self.assertIsInstance(base_ln, torch.nn.LayerNorm)

        for level in range(1, 4):
            lvl_ln = aux_list[level][2]
            # Object identity
            self.assertIs(lvl_ln, base_ln, f"Level {level} LayerNorm is not the identical object to level 0")
            # Parameter object identity
            self.assertIs(lvl_ln.weight, base_ln.weight)
            self.assertIs(lvl_ln.bias, base_ln.bias)
            # Memory address identity
            self.assertEqual(lvl_ln.weight.data_ptr(), base_ln.weight.data_ptr())
            self.assertEqual(lvl_ln.bias.data_ptr(), base_ln.bias.data_ptr())

    def test_safetensors_lacks_duplicate_aux_keys(self) -> None:
        """Safetensors deduplicates shared memory; verify level 1, 2, 3 LN keys are absent in file."""
        self.assertIn("model.head.scratch.output_conv2_aux.0.2.weight", self.raw_tensors)
        self.assertIn("model.head.scratch.output_conv2_aux.0.2.bias", self.raw_tensors)

        for level in range(1, 4):
            self.assertNotIn(f"model.head.scratch.output_conv2_aux.{level}.2.weight", self.raw_tensors)
            self.assertNotIn(f"model.head.scratch.output_conv2_aux.{level}.2.bias", self.raw_tensors)

    def test_unresolved_strict_load_fails(self) -> None:
        """Loading without alias restoration must fail strict=True with exact missing keys."""
        state_dict = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in self.raw_tensors.items()
        }

        with self.assertRaises(RuntimeError) as ctx:
            self.raw_model.load_state_dict(state_dict, strict=True)

        err_msg = str(ctx.exception)
        self.assertIn("Missing key(s) in state_dict", err_msg)
        for level in range(1, 4):
            self.assertIn(f"head.scratch.output_conv2_aux.{level}.2.weight", err_msg)
            self.assertIn(f"head.scratch.output_conv2_aux.{level}.2.bias", err_msg)

    def test_resolved_strict_load_succeeds(self) -> None:
        """Restoring only the shared aliases enables 100% strict loading with zero missing/unexpected."""
        adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cpu", verify_hashes=False)
        self.assertIsNotNone(adapter.model)

    def test_genuine_missing_key_rejection(self) -> None:
        """Ensure genuine missing learned parameters are strictly rejected and not ignored."""
        state_dict = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in self.raw_tensors.items()
        }
        for level in range(1, 4):
            state_dict[f"head.scratch.output_conv2_aux.{level}.2.weight"] = state_dict["head.scratch.output_conv2_aux.0.2.weight"]
            state_dict[f"head.scratch.output_conv2_aux.{level}.2.bias"] = state_dict["head.scratch.output_conv2_aux.0.2.bias"]

        # Delete a genuine learned parameter from backbone
        target_key = "backbone.pretrained.cls_token"
        self.assertIn(target_key, state_dict)
        del state_dict[target_key]

        with self.assertRaises(RuntimeError) as ctx:
            self.raw_model.load_state_dict(state_dict, strict=True)
        self.assertIn("backbone.pretrained.cls_token", str(ctx.exception))


class TestPreprocessing(unittest.TestCase):
    """Test canonical DA3 input preprocessing."""

    def test_divisible_by_14_and_aspect_preservation(self) -> None:
        # Test with arbitrary dimensions: 680x1208
        img = np.zeros((680, 1208, 3), dtype=np.uint8)
        tensor, orig_shape, proc_shape = DA3SmallDepthAdapter.preprocess_image(img, target_size=504)

        self.assertEqual(orig_shape, (680, 1208))
        proc_h, proc_w = proc_shape
        self.assertEqual(proc_w, 504)
        self.assertEqual(proc_w % PATCH_SIZE, 0)
        self.assertEqual(proc_h % PATCH_SIZE, 0)
        self.assertEqual(tensor.shape, (1, 1, 3, proc_h, proc_w))

    def test_odd_dimension_handling(self) -> None:
        # Non-standard dimensions: 405x719
        img = np.random.randint(0, 255, (405, 719, 3), dtype=np.uint8)
        tensor, orig_shape, proc_shape = DA3SmallDepthAdapter.preprocess_image(img, target_size=504)

        self.assertEqual(orig_shape, (405, 719))
        proc_h, proc_w = proc_shape
        self.assertEqual(proc_w % PATCH_SIZE, 0)
        self.assertEqual(proc_h % PATCH_SIZE, 0)
        self.assertEqual(tensor.shape, (1, 1, 3, proc_h, proc_w))


class TestInferenceExecution(unittest.TestCase):
    """Test actual CPU and CUDA inference on real sample image."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_path = SAMPLE_IMAGE_PATH

    def test_cpu_inference(self) -> None:
        adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cpu", verify_hashes=True)
        result = adapter.infer(self.image_path, target_size=504, return_original_size=True)

        self.assertIsInstance(result, DepthPredictionResult)
        self.assertEqual(result.input_shape, (680, 1208))
        self.assertEqual(result.depth.shape, (680, 1208))
        self.assertEqual(result.depth_raw.shape, (280, 504))
        self.assertGreater(result.min_depth, 0.0)
        self.assertFalse(np.isnan(result.depth).any())
        self.assertFalse(np.isinf(result.depth).any())
        self.assertGreater(result.latency_ms, 0.0)

    def test_cuda_inference_and_cross_check(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this machine")

        cpu_adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cpu", verify_hashes=False)
        cpu_res = cpu_adapter.infer(self.image_path, target_size=504, return_original_size=True)

        cuda_adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cuda:0", verify_hashes=False)
        cuda_res = cuda_adapter.infer(self.image_path, target_size=504, return_original_size=True, autocast=True)

        self.assertEqual(cuda_res.depth.shape, (680, 1208))
        self.assertGreater(cuda_res.min_depth, 0.0)
        self.assertFalse(np.isnan(cuda_res.depth).any())

        # Numerical agreement check: MAE should be very small (< 0.005)
        mae = float(np.mean(np.abs(cpu_res.depth - cuda_res.depth)))
        self.assertLess(mae, 0.005, f"Discrepancy between CPU and CUDA too large: MAE={mae}")

    def test_save_outputs(self) -> None:
        adapter = DA3SmallDepthAdapter(model_dir=CHECKPOINT_DIR, device="cpu", verify_hashes=False)
        result = adapter.infer(self.image_path, target_size=504, return_original_size=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = DA3SmallDepthAdapter.save_depth_outputs(result, out_dir=tmp_dir, base_name="test_out")
            self.assertTrue(saved["npy"].is_file())
            self.assertTrue(saved["png_u16"].is_file())
            self.assertTrue(saved["png_color"].is_file())
            self.assertTrue(saved["metrics"].is_file())

            # Verify saved npy matches
            loaded_npy = np.load(saved["npy"])
            np.testing.assert_allclose(loaded_npy, result.depth)


if __name__ == "__main__":
    unittest.main()
