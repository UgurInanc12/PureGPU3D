"""Tests for real model inference, preprocessing geometry, and metric classification across DA3 models."""

from __future__ import annotations

import gc
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

import cv2
import numpy as np
import torch

from puregpu3d.models.catalog import load_catalog
from puregpu3d.models.da3_adapter import (
    DEFAULT_PROCESS_RES,
    PATCH_SIZE,
    DA3DepthAdapter,
    DepthPredictionResult,
)

SAMPLE_IMAGE = REPO_ROOT / "third_party" / "depth_anything_3" / "assets" / "examples" / "SOH" / "000.png"


class TestPreprocessingGeometry(unittest.TestCase):
    """Test canonical DA3 input preprocessing and geometry constraints."""

    def test_aspect_preservation_and_divisibility(self) -> None:
        # Non-square image: 720x1280
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        tensor, orig_shape, proc_shape = DA3DepthAdapter.preprocess_image(img, target_size=504)

        self.assertEqual(orig_shape, (720, 1280))
        h, w = proc_shape
        self.assertEqual(w, 504)
        self.assertEqual(w % PATCH_SIZE, 0)
        self.assertEqual(h % PATCH_SIZE, 0)
        self.assertEqual(tensor.shape, (1, 1, 3, h, w))

    def test_odd_shapes(self) -> None:
        # 333x555
        img = np.random.randint(0, 255, (333, 555, 3), dtype=np.uint8)
        tensor, orig_shape, proc_shape = DA3DepthAdapter.preprocess_image(img, target_size=504)

        self.assertEqual(orig_shape, (333, 555))
        h, w = proc_shape
        self.assertEqual(w % PATCH_SIZE, 0)
        self.assertEqual(h % PATCH_SIZE, 0)
        self.assertEqual(tensor.shape, (1, 1, 3, h, w))


class TestModelsInference(unittest.TestCase):
    """Test actual CPU/CUDA depth inference across Base, Mono Large, and Metric Large."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.has_cuda = torch.cuda.is_available()

    def _get_rev_dir(self, model_id: str) -> Path:
        entry = self.catalog[model_id]
        return REPO_ROOT / "models" / model_id / entry.revision

    def test_base_inference(self) -> None:
        rev_dir = self._get_rev_dir("DA3-BASE")
        device = "cuda:0" if self.has_cuda else "cpu"

        adapter = DA3DepthAdapter(model_dir=rev_dir, identifier="DA3-BASE", device=device, verify_hashes=False)
        result = adapter.infer(SAMPLE_IMAGE, target_size=504, return_original_size=True)

        self.assertIsInstance(result, DepthPredictionResult)
        self.assertEqual(result.model_id, "DA3-BASE")
        self.assertFalse(result.is_metric)
        self.assertEqual(result.input_shape, (680, 1208))
        self.assertEqual(result.depth.shape, (680, 1208))
        self.assertTrue(np.isfinite(result.depth).all())
        self.assertGreater(result.min_depth, 0.0)

        del adapter
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()

    def test_mono_large_inference(self) -> None:
        rev_dir = self._get_rev_dir("DA3MONO-LARGE")
        device = "cuda:0" if self.has_cuda else "cpu"

        adapter = DA3DepthAdapter(model_dir=rev_dir, identifier="DA3MONO-LARGE", device=device, verify_hashes=False)
        result = adapter.infer(SAMPLE_IMAGE, target_size=504, return_original_size=True)

        self.assertIsInstance(result, DepthPredictionResult)
        self.assertEqual(result.model_id, "DA3MONO-LARGE")
        self.assertFalse(result.is_metric)
        self.assertEqual(result.depth.shape, (680, 1208))
        self.assertTrue(np.isfinite(result.depth).all())
        self.assertGreater(result.min_depth, 0.0)

        del adapter
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()

    def test_metric_large_inference_and_schema(self) -> None:
        rev_dir = self._get_rev_dir("DA3METRIC-LARGE")
        device = "cuda:0" if self.has_cuda else "cpu"

        adapter = DA3DepthAdapter(model_dir=rev_dir, identifier="DA3METRIC-LARGE", device=device, verify_hashes=False)
        result = adapter.infer(SAMPLE_IMAGE, target_size=504, return_original_size=True)

        self.assertIsInstance(result, DepthPredictionResult)
        self.assertEqual(result.model_id, "DA3METRIC-LARGE")
        # DA3METRIC-LARGE requires camera focal length (metric_depth = focal * net_output / 300).
        # Without focal provided, raw output is unscaled/focal-dependent and MUST NOT claim is_metric=True or meters.
        self.assertFalse(result.is_metric, "DA3METRIC-LARGE without focal scaling must NOT claim is_metric=True")
        self.assertEqual(result.depth_units, "focal_dependent_unscaled")
        self.assertTrue(adapter.is_metric_family)
        self.assertFalse(adapter.is_metric)
        self.assertEqual(result.depth.shape, (680, 1208))
        self.assertTrue(np.isfinite(result.depth).all())
        self.assertGreater(result.min_depth, 0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = DA3DepthAdapter.save_depth_outputs(result, out_dir=tmp_dir, base_name="metric_test")
            self.assertTrue(saved["npy"].is_file())
            self.assertTrue(saved["png_u16"].is_file())
            self.assertTrue(saved["png_color"].is_file())
            self.assertTrue(saved["metrics"].is_file())

        del adapter
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()

    def test_metric_large_without_focal_never_reports_meters(self) -> None:
        """Regression test: verifying that absence of camera focal length leaves is_metric False and units unscaled."""
        rev_dir = self._get_rev_dir("DA3METRIC-LARGE")
        device = "cuda:0" if self.has_cuda else "cpu"
        adapter = DA3DepthAdapter(model_dir=rev_dir, identifier="DA3METRIC-LARGE", device=device, verify_hashes=False)
        result = adapter.infer(SAMPLE_IMAGE, target_size=504)
        self.assertFalse(result.is_metric, "Must not report is_metric=True when focal length is not provided")
        self.assertNotEqual(result.depth_units, "meters", "Must never claim 'meters' without focal calibration")
        self.assertEqual(result.depth_units, "focal_dependent_unscaled")
        del adapter
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
