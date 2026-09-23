"""Depth Anything 3 model adapter for PureGPU3D.

Provides verified loading, general parameter alias restoration, canonical
preprocessing, and CPU/CUDA inference across Depth Anything 3 catalog checkpoints
(Small, Base, Mono Large, Metric Large, and architecture support for Large 1.1,
Giant 1.1, and Nested Giant+Large).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from puregpu3d.models.catalog import (
    ModelCatalogEntry,
    get_model_entry,
    load_catalog,
)
from puregpu3d.models.geometry import (
    DepthGeometry,
    compute_depth_geometry,
    pad_image_for_depth,
    unpad_depth_map,
)

logger = logging.getLogger(__name__)

# Pinned metadata for verified DA3 Small checkpoint (backward-compatibility constants)
DEFAULT_MODEL_ID = "depth-anything/DA3-SMALL"
DEFAULT_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
EXPECTED_CONFIG_SHA256 = "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
EXPECTED_WEIGHTS_SHA256 = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
PATCH_SIZE = 14
DEFAULT_PROCESS_RES = 504

# Canonical ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Strict whitelist of approved module import paths in config.json
ALLOWED_MODULE_PATHS: Set[str] = {
    "depth_anything_3.model.da3",
    "depth_anything_3.model.dinov2.dinov2",
    "depth_anything_3.model.dualdpt",
    "depth_anything_3.model.dpt",
    "depth_anything_3.model.cam_enc",
    "depth_anything_3.model.cam_dec",
    "depth_anything_3.model.gsdpt",
    "depth_anything_3.model.gs_adapter",
}


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def ensure_da3_vendor_import() -> Path:
    """Ensure depth_anything_3 src directory is available on sys.path."""
    try:
        import depth_anything_3
        if hasattr(depth_anything_3, "__file__") and depth_anything_3.__file__:
            return Path(depth_anything_3.__file__).resolve().parent
        return Path(sys.executable).resolve().parent
    except ImportError:
        pass

    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        candidates = [
            base_dir / "depth_anything_3",
            base_dir / "third_party" / "depth_anything_3" / "src",
        ]
        for candidate in candidates:
            if candidate.exists():
                candidate_str = str(candidate if candidate.name == "src" else base_dir)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return candidate

    repo_root = Path(__file__).resolve().parents[3]
    vendor_src = repo_root / "third_party" / "depth_anything_3" / "src"
    if not vendor_src.exists():
        raise FileNotFoundError(
            f"Upstream DA3 source directory not found at {vendor_src}."
        )
    vendor_str = str(vendor_src)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    return vendor_src


@dataclass
class DepthPredictionResult:
    """Container for model inference outputs and profiling metrics."""

    depth: np.ndarray
    depth_raw: np.ndarray
    input_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    latency_ms: float
    device: str
    dtype: str
    min_depth: float
    max_depth: float
    mean_depth: float
    is_metric: bool = False
    metric_scale: Optional[float] = None
    model_id: str = ""
    depth_units: str = "relative"
    geometry: Optional[DepthGeometry] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata dictionary."""
        d = {
            "input_shape": list(self.input_shape),
            "processed_shape": list(self.processed_shape),
            "latency_ms": round(self.latency_ms, 3),
            "device": self.device,
            "dtype": self.dtype,
            "min_depth": round(float(self.min_depth), 6),
            "max_depth": round(float(self.max_depth), 6),
            "mean_depth": round(float(self.mean_depth), 6),
            "is_metric": bool(self.is_metric),
            "depth_units": self.depth_units,
        }
        if self.geometry is not None:
            d["geometry"] = {
                "scale": self.geometry.scale,
                "scale_factor": self.geometry.scale_factor,
                "req_shape": list(self.geometry.req_shape),
                "padded_shape": list(self.geometry.padded_shape),
                "pad_right": self.geometry.pad_right,
                "pad_bottom": self.geometry.pad_bottom,
            }
        if self.metric_scale is not None:
            d["metric_scale"] = round(float(self.metric_scale), 6)
        if self.model_id:
            d["model_id"] = self.model_id
        return d


@dataclass
class DepthTensorResult:
    """Container for GPU-resident depth model inference outputs and metadata.

    Retains all depth tensors directly on CUDA device memory without copying
    to host CPU, eliminating PCIe round-trips for downstream video pipelines.
    """

    depth: torch.Tensor
    depth_raw: torch.Tensor
    input_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    latency_ms: float
    device: str
    dtype: str
    is_metric: bool = False
    metric_scale: Optional[float] = None
    model_id: str = ""
    depth_units: str = "relative"
    geometry: Optional[DepthGeometry] = None
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None
    mean_depth: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata dictionary."""
        d: Dict[str, Any] = {
            "input_shape": list(self.input_shape),
            "processed_shape": list(self.processed_shape),
            "latency_ms": round(self.latency_ms, 3),
            "device": self.device,
            "dtype": self.dtype,
            "is_metric": bool(self.is_metric),
            "depth_units": self.depth_units,
            "tensor_shape": list(self.depth.shape),
            "raw_shape": list(self.depth_raw.shape),
        }
        if self.min_depth is not None:
            d["min_depth"] = round(float(self.min_depth), 6)
        if self.max_depth is not None:
            d["max_depth"] = round(float(self.max_depth), 6)
        if self.mean_depth is not None:
            d["mean_depth"] = round(float(self.mean_depth), 6)
        if self.geometry is not None:
            d["geometry"] = {
                "scale": self.geometry.scale,
                "scale_factor": self.geometry.scale_factor,
                "req_shape": list(self.geometry.req_shape),
                "padded_shape": list(self.geometry.padded_shape),
                "pad_right": self.geometry.pad_right,
                "pad_bottom": self.geometry.pad_bottom,
            }
        if self.metric_scale is not None:
            d["metric_scale"] = round(float(self.metric_scale), 6)
        if self.model_id:
            d["model_id"] = self.model_id
        return d

    def compute_summary_stats(self) -> Tuple[float, float, float]:
        """Compute (min, max, mean) scalar statistics with explicit host synchronization.

        Caution: triggers a GPU->CPU synchronization point. Only use for logging/diagnostics.
        """
        d_min = float(self.depth_raw.min().item())
        d_max = float(self.depth_raw.max().item())
        d_mean = float(self.depth_raw.mean().item())
        self.min_depth = d_min
        self.max_depth = d_max
        self.mean_depth = d_mean
        return d_min, d_max, d_mean

    def to_cpu_prediction(self) -> DepthPredictionResult:
        """Convert GPU result to CPU numpy DepthPredictionResult (triggers host copy)."""
        d_np = self.depth.detach().cpu().numpy()
        raw_np = self.depth_raw.detach().cpu().numpy()
        min_d = float(d_np.min()) if self.min_depth is None else self.min_depth
        max_d = float(d_np.max()) if self.max_depth is None else self.max_depth
        mean_d = float(d_np.mean()) if self.mean_depth is None else self.mean_depth
        return DepthPredictionResult(
            depth=d_np,
            depth_raw=raw_np,
            input_shape=self.input_shape,
            processed_shape=self.processed_shape,
            latency_ms=self.latency_ms,
            device=self.device,
            dtype=self.dtype,
            min_depth=min_d,
            max_depth=max_d,
            mean_depth=mean_d,
            is_metric=self.is_metric,
            metric_scale=self.metric_scale,
            model_id=self.model_id,
            depth_units=self.depth_units,
            geometry=self.geometry,
        )


class DA3DepthAdapter:
    """General adapter for Depth Anything 3 depth estimation models."""

    ALLOWED_MODULE_PATHS: Set[str] = ALLOWED_MODULE_PATHS

    def __init__(
        self,
        model_dir: Union[str, Path],
        identifier: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        verify_hashes: bool = True,
    ) -> None:
        """Initialize adapter from local model directory and audited catalog entry.

        Args:
            model_dir: Path to directory containing config.json and model.safetensors.
            identifier: Model ID or repo_id from catalog (e.g. 'DA3-BASE', 'DA3MONO-LARGE').
                        If None, inferred from manifest.json or directory path.
            device: Target torch device ('cpu', 'cuda', 'cuda:0', etc.).
            verify_hashes: Whether to strictly verify SHA-256 hashes against pinned catalog.
        """
        self.model_dir = Path(model_dir).resolve()
        self.config_path = self.model_dir / "config.json"
        self.weights_path = self.model_dir / "model.safetensors"
        self.manifest_path = self.model_dir / "manifest.json"

        if not self.config_path.is_file():
            raise FileNotFoundError(f"Missing config.json at {self.config_path}")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Missing model.safetensors at {self.weights_path}")

        self.catalog = load_catalog()
        self.entry = self._resolve_catalog_entry(identifier)

        if verify_hashes:
            self._verify_checkpoint_integrity()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        ensure_da3_vendor_import()
        from depth_anything_3.cfg import create_object

        self._create_object = create_object
        self.model = self._load_and_resolve_weights()
        self.model.to(self.device)
        self.model.eval()

    @property
    def identifier(self) -> str:
        """Return catalog entry ID."""
        return self.entry.id

    @property
    def is_specialist(self) -> bool:
        """Return whether model is a specialist (mono or metric)."""
        return self.entry.is_specialist

    @property
    def is_metric(self) -> bool:
        """Return whether model natively outputs true metric depth in meters without focal scaling.

        Note: While DA3NESTED models produce native metric depth in meters,
        DA3METRIC-LARGE produces unscaled output requiring camera focal length
        (metric_depth = focal * net_output / 300). Without validated focal length,
        raw output is not in meters and cannot be claimed as metric depth.
        """
        return "nested" in self.entry.id.lower()

    @property
    def is_metric_family(self) -> bool:
        """Return whether model belongs to the metric depth family (specialist or nested)."""
        return "metric" in self.entry.id.lower() or "nested" in self.entry.id.lower()

    @property
    def depth_units(self) -> str:
        """Return native depth units of raw model output: 'meters', 'focal_dependent_unscaled', or 'relative'."""
        if "nested" in self.entry.id.lower():
            return "meters"
        if "metric" in self.entry.id.lower():
            return "focal_dependent_unscaled"
        return "relative"

    @classmethod
    def get_supported_model_ids(cls) -> List[str]:
        """Return list of all 7 catalog model IDs structurally supported by the adapter."""
        return list(load_catalog().keys())

    @classmethod
    def get_verified_model_ids(cls) -> List[str]:
        """Return list of model IDs verified with actual weights and GPU execution in this slice."""
        return ["DA3-SMALL", "DA3-BASE", "DA3MONO-LARGE", "DA3METRIC-LARGE"]

    @classmethod
    def get_unverified_model_ids(cls) -> List[str]:
        """Return list of model IDs requiring noncommercial acknowledgment or pending verification."""
        return ["DA3-LARGE-1.1", "DA3-GIANT-1.1", "DA3NESTED-GIANT-LARGE-1.1"]

    def _resolve_catalog_entry(self, identifier: Optional[str]) -> ModelCatalogEntry:
        """Resolve catalog entry strictly from provided identifier or local metadata."""
        if identifier is not None:
            return get_model_entry(identifier, self.catalog)

        # Infer from manifest if available
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                m_id = manifest.get("model_id")
                if m_id:
                    try:
                        return get_model_entry(m_id, self.catalog)
                    except KeyError:
                        pass
                m_rev = manifest.get("revision")
                if m_rev:
                    for entry in self.catalog.values():
                        if entry.revision == m_rev:
                            return entry
            except (json.JSONDecodeError, OSError):
                pass

        # Infer from directory name or parent folder
        candidates = [self.model_dir.name, self.model_dir.parent.name]
        for cand in candidates:
            try:
                return get_model_entry(cand, self.catalog)
            except KeyError:
                pass

        available = ", ".join(self.catalog.keys())
        raise ValueError(
            f"Model directory '{self.model_dir}' does not match any audited catalog entry, "
            f"and no valid identifier was specified. Available models: {available}"
        )

    def _verify_checkpoint_integrity(self) -> None:
        """Verify checkpoint files match pinned SHA-256 hashes from the catalog."""
        expected_cfg_spec = self.entry.files.get("config.json")
        expected_wt_spec = self.entry.files.get("model.safetensors")

        if not expected_cfg_spec or not expected_wt_spec:
            raise ValueError(f"Catalog entry for {self.entry.id} missing required file specifications")

        # Verify manifest if present
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                files_meta = manifest.get("files", {})
                if "config.json" in files_meta:
                    m_cfg_sha = files_meta["config.json"].get("sha256")
                    if m_cfg_sha and m_cfg_sha.lower() != expected_cfg_spec.sha256.lower():
                        raise ValueError(
                            f"Manifest config.json SHA-256 {m_cfg_sha} mismatch with catalog {expected_cfg_spec.sha256}"
                        )
                if "model.safetensors" in files_meta:
                    m_wt_sha = files_meta["model.safetensors"].get("sha256")
                    if m_wt_sha and m_wt_sha.lower() != expected_wt_spec.sha256.lower():
                        raise ValueError(
                            f"Manifest model.safetensors SHA-256 {m_wt_sha} mismatch with catalog {expected_wt_spec.sha256}"
                        )
            except json.JSONDecodeError as err:
                raise ValueError(f"Corrupt manifest.json at {self.manifest_path}") from err

        # Verify physical file size and checksum
        cfg_size = self.config_path.stat().st_size
        if cfg_size != expected_cfg_spec.bytes:
            raise ValueError(
                f"Config byte count mismatch for {self.config_path}: expected {expected_cfg_spec.bytes}, got {cfg_size}"
            )
        cfg_sha = compute_sha256(self.config_path)
        if cfg_sha.lower() != expected_cfg_spec.sha256.lower():
            raise ValueError(
                f"Config SHA-256 mismatch for {self.config_path}: expected {expected_cfg_spec.sha256}, got {cfg_sha}"
            )

        wt_size = self.weights_path.stat().st_size
        if wt_size != expected_wt_spec.bytes:
            raise ValueError(
                f"Weights byte count mismatch for {self.weights_path}: expected {expected_wt_spec.bytes}, got {wt_size}"
            )
        wt_sha = compute_sha256(self.weights_path)
        if wt_sha.lower() != expected_wt_spec.sha256.lower():
            raise ValueError(
                f"Weights SHA-256 mismatch for {self.weights_path}: expected {expected_wt_spec.sha256}, got {wt_sha}"
            )

    @classmethod
    def _audit_config_security(cls, node: Any) -> None:
        """Enforce strict whitelist on module paths in config.json to prevent untrusted dynamic imports."""
        if isinstance(node, dict):
            if "__object__" in node:
                obj_info = node["__object__"]
                if not isinstance(obj_info, dict):
                    raise ValueError(f"Malformed __object__ definition: {obj_info}")
                path = obj_info.get("path")
                if not path or path not in cls.ALLOWED_MODULE_PATHS:
                    raise ValueError(
                        f"Untrusted module path in model config: '{path}'. "
                        f"Must be one of: {sorted(cls.ALLOWED_MODULE_PATHS)}"
                    )
            for v in node.values():
                cls._audit_config_security(v)
        elif isinstance(node, list):
            for item in node:
                cls._audit_config_security(item)

    @staticmethod
    def resolve_aliases(
        model: nn.Module,
        raw_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Detect identical parameter objects across model modules and resolve deduplicated weights.

        Safetensors deduplicates shared memory tensors so only one parameter name per
        aliased group is written to disk.

        This method:
          1. Groups parameter names by underlying Parameter object identity and data pointer.
          2. Strips upstream training prefixes ('model.', 'module.').
          3. For each aliased group (len > 1):
             - Rejects if contradictory tensors exist for parameters sharing memory.
             - Reconstructs missing alias entries from the single persisted tensor.
             - Leaves genuinely missing parameter groups untouched so strict loading fails.
        """
        # 1. Group parameter names by parameter object identity
        param_groups: Dict[int, List[str]] = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            param_groups.setdefault(id(param), []).append(name)

        # Verify object identity and memory sharing for grouped names
        for pid, names in param_groups.items():
            if len(names) > 1:
                base_param = model.get_parameter(names[0])
                base_ptr = base_param.data_ptr()
                for alias_name in names[1:]:
                    alias_param = model.get_parameter(alias_name)
                    if alias_param is not base_param or alias_param.data_ptr() != base_ptr:
                        raise RuntimeError(
                            f"Parameter alias inconsistency: '{alias_name}' does not share memory with '{names[0]}'"
                        )

        # 2. Normalize raw_tensors by stripping 'model.' or 'module.' prefixes
        state_dict: Dict[str, torch.Tensor] = {}
        for k, v in raw_tensors.items():
            if k.startswith("model."):
                state_dict[k[len("model.") :]] = v
            elif k.startswith("module."):
                state_dict[k[len("module.") :]] = v
            else:
                state_dict[k] = v

        # 3. Restore deduplicated aliases and reject contradictory tensors
        for pid, names in param_groups.items():
            if len(names) <= 1:
                continue

            present = [n for n in names if n in state_dict]
            if len(present) == 0:
                # Genuine missing: none of the aliased parameters are in the weights file
                continue
            elif len(present) == 1:
                # Exactly one tensor stored; restore all other aliases
                src_key = present[0]
                src_tensor = state_dict[src_key]
                for n in names:
                    if n != src_key:
                        state_dict[n] = src_tensor
            else:
                # Multiple alias keys present: verify they are not contradictory
                base_key = present[0]
                base_tensor = state_dict[base_key]
                for other_key in present[1:]:
                    other_tensor = state_dict[other_key]
                    if (
                        base_tensor.shape != other_tensor.shape
                        or base_tensor.dtype != other_tensor.dtype
                        or not torch.equal(base_tensor, other_tensor)
                    ):
                        raise RuntimeError(
                            f"Contradictory alias tensors detected in weights for parameters sharing identity: "
                            f"'{base_key}' and '{other_key}' differ in shape, dtype, or values."
                        )
                # Ensure all names in group are populated
                for n in names:
                    if n not in state_dict:
                        state_dict[n] = base_tensor

        return state_dict

    def _load_and_resolve_weights(self) -> nn.Module:
        """Instantiate network from audited config and strictly load resolved weights."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "config" not in cfg:
            raise ValueError(f"Invalid config.json at {self.config_path}: missing top-level 'config' key")

        # Audit configuration against untrusted imports
        self._audit_config_security(cfg["config"])

        model = self._create_object(cfg["config"])
        if not isinstance(model, nn.Module):
            raise TypeError(f"Created object is not an nn.Module: {type(model)}")

        raw_tensors = load_file(str(self.weights_path))

        # Resolve parameter aliases
        resolved_state_dict = self.resolve_aliases(model, raw_tensors)

        # Strict load: reject any genuine missing or unexpected keys
        load_result = model.load_state_dict(resolved_state_dict, strict=True)
        if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
            raise RuntimeError(
                f"Strict load failed for {self.identifier}. "
                f"Missing keys: {load_result.missing_keys}, "
                f"Unexpected keys: {load_result.unexpected_keys}"
            )

        return model

    @staticmethod
    def preprocess_image(
        image: Union[str, Path, np.ndarray, Image.Image],
        target_size: Optional[int] = DEFAULT_PROCESS_RES,
        depth_scale: Optional[Union[str, float]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """Canonical upstream preprocessing for Depth Anything 3.

        Modes:
          1. Explicit depth_scale: scales dimensions aspect-preservingly, pads upward
             to patch_size (14) multiples, normalizes with ImageNet constants.
          2. Legacy target_size: scales longest dimension to target_size, rounds to
             nearest 14 multiple, normalizes with ImageNet constants.

        Returns:
            (tensor, original_shape, processed_shape) where shapes are (H, W).
        """
        if isinstance(image, (str, Path)):
            path_str = str(image)
            bgr = cv2.imread(path_str)
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image at {path_str}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:
                    rgb = image
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported array dimensions: {image.ndim}")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = rgb.shape[:2]

        # Explicit depth_scale route: aspect-preserving padding
        if depth_scale is not None:
            geom = compute_depth_geometry(orig_w, orig_h, scale=depth_scale)
            padded = pad_image_for_depth(rgb, geom)
            norm_img = ((padded.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(norm_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            return tensor, (orig_h, orig_w), (geom.padded_height, geom.padded_width)

        # Legacy target_size route: longest-side resize
        effective_target = target_size if target_size is not None else DEFAULT_PROCESS_RES
        longest = max(orig_h, orig_w)
        scale = float(effective_target) / float(longest)
        scaled_w = max(1, int(round(orig_w * scale)))
        scaled_h = max(1, int(round(orig_h * scale)))

        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=interp)

        # Enforce divisibility by PATCH_SIZE (14) using nearest multiple
        def round_to_multiple(dim: int, patch: int = PATCH_SIZE) -> int:
            down = (dim // patch) * patch
            up = down + patch
            return up if abs(up - dim) <= abs(dim - down) else down

        final_w = max(PATCH_SIZE, round_to_multiple(scaled_w))
        final_h = max(PATCH_SIZE, round_to_multiple(scaled_h))

        if final_w != scaled_w or final_h != scaled_h:
            patch_interp = cv2.INTER_CUBIC if (final_w > scaled_w or final_h > scaled_h) else cv2.INTER_AREA
            processed = cv2.resize(scaled, (final_w, final_h), interpolation=patch_interp)
        else:
            processed = scaled

        # Normalize with ImageNet mean and std
        norm_img = processed.astype(np.float32) / 255.0
        norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD

        # Shape: (1, 1, 3, H, W)
        tensor = torch.from_numpy(norm_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return tensor, (orig_h, orig_w), (final_h, final_w)

    @staticmethod
    def preprocess_tensor(
        image_tensor: torch.Tensor,
        target_size: Optional[int] = DEFAULT_PROCESS_RES,
        depth_scale: Optional[Union[str, float]] = None,
        interpolate_mode: str = "area",
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], Optional[DepthGeometry]]:
        """GPU-native preprocessing for Depth Anything 3 on existing device tensor.

        Transforms a CUDA RGB tensor into the normalized (1, 1, 3, H_pad, W_pad) 5D tensor
        expected by DA3 vision transformer backbones. All operations (color conversion,
        resizing, patch padding, and ImageNet normalization) occur entirely on the device
        with zero host copies.

        Supported input layouts:
          - HWC: (H, W, 3) uint8 [0..255] or float32 [0.0..1.0]
          - CHW: (3, H, W) uint8 [0..255] or float32 [0.0..1.0]
          - BCHW: (1, 3, H, W) uint8 [0..255] or float32 [0.0..1.0]
          - BHWC: (1, H, W, 3) uint8 [0..255] or float32 [0.0..1.0]

        Args:
            image_tensor: Input torch.Tensor residing on target device (CUDA).
            target_size: Legacy longest-dimension size (default 504).
            depth_scale: Explicit spatial depth scale ('1/4', '1/2', '1/1').
            interpolate_mode: Interpolation mode ('area', 'bilinear', or 'bicubic').
                             Defaults to 'area' for downsampling, matching cv2.INTER_AREA.

        Returns:
            (model_tensor, original_shape, processed_shape, geometry)
            where model_tensor has shape (1, 1, 3, padded_h, padded_w).
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"image_tensor must be a torch.Tensor, got {type(image_tensor)}")

        # Parse layout to standard (1, 3, H, W) float32
        if image_tensor.ndim == 3:
            if image_tensor.shape[2] == 3:
                # HWC
                orig_h, orig_w = int(image_tensor.shape[0]), int(image_tensor.shape[1])
                x = image_tensor.permute(2, 0, 1).unsqueeze(0)
            elif image_tensor.shape[0] == 3:
                # CHW
                orig_h, orig_w = int(image_tensor.shape[1]), int(image_tensor.shape[2])
                x = image_tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected 3 color channels in dim 0 or dim 2, got shape {tuple(image_tensor.shape)}"
                )
        elif image_tensor.ndim == 4:
            if image_tensor.shape[0] != 1:
                raise ValueError(
                    f"Batch size must be 1 for single frame inference, got shape {tuple(image_tensor.shape)}"
                )
            if image_tensor.shape[1] == 3:
                # BCHW
                orig_h, orig_w = int(image_tensor.shape[2]), int(image_tensor.shape[3])
                x = image_tensor
            elif image_tensor.shape[3] == 3:
                # BHWC
                orig_h, orig_w = int(image_tensor.shape[1]), int(image_tensor.shape[2])
                x = image_tensor.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Expected 3 color channels in dim 1 or dim 3, got shape {tuple(image_tensor.shape)}"
                )
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(image_tensor.shape)}")

        # Dtype normalization to [0.0, 1.0] float32
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32) / 255.0
        elif x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = x.to(dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported tensor dtype {x.dtype}. Expected torch.uint8 or float32/16")

        geom: Optional[DepthGeometry] = None
        if depth_scale is not None:
            geom = compute_depth_geometry(orig_w, orig_h, scale=depth_scale)
            target_w, target_h = geom.req_width, geom.req_height
        else:
            effective_target = target_size if target_size is not None else DEFAULT_PROCESS_RES
            longest = max(orig_h, orig_w)
            scale = float(effective_target) / float(longest)
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))

            def round_to_multiple(dim: int, patch: int = PATCH_SIZE) -> int:
                down = (dim // patch) * patch
                up = down + patch
                return up if abs(up - dim) <= abs(dim - down) else down

            final_w = max(PATCH_SIZE, round_to_multiple(scaled_w))
            final_h = max(PATCH_SIZE, round_to_multiple(scaled_h))
            target_w, target_h = final_w, final_h

        # GPU resize
        if (orig_h, orig_w) != (target_h, target_w):
            if target_w < orig_w and target_h < orig_h and interpolate_mode == "area":
                x = F.interpolate(x, size=(target_h, target_w), mode="area")
            else:
                mode = interpolate_mode if interpolate_mode in ("bilinear", "bicubic") else "bilinear"
                x = F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False)

        # Padding to patch size (14)
        if geom is not None:
            if geom.pad_right > 0 or geom.pad_bottom > 0:
                if geom.req_width > geom.pad_right and geom.req_height > geom.pad_bottom:
                    x = F.pad(x, (0, geom.pad_right, 0, geom.pad_bottom), mode="reflect")
                else:
                    x = F.pad(x, (0, geom.pad_right, 0, geom.pad_bottom), mode="replicate")
            padded_h, padded_w = geom.padded_height, geom.padded_width
        else:
            padded_h, padded_w = target_h, target_w

        # ImageNet normalization on GPU
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 5D tensor: (1, 1, 3, H, W)
        tensor = x.unsqueeze(0)
        return tensor, (orig_h, orig_w), (padded_h, padded_w), geom

    @classmethod
    def preprocess_tensor_batch(
        cls,
        image_tensors: Union[torch.Tensor, Sequence[torch.Tensor], List[torch.Tensor]],
        target_size: Optional[int] = DEFAULT_PROCESS_RES,
        depth_scale: Optional[Union[str, float]] = None,
        interpolate_mode: str = "area",
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], Optional[DepthGeometry]]:
        """GPU-native batched preprocessing for Depth Anything 3 on device tensors.

        Transforms a batch of CUDA RGB tensors into the normalized (B, 1, 3, H_pad, W_pad)
        5D tensor expected by DA3 vision transformer backbones. All operations (resizing,
        patch padding, and ImageNet normalization) occur entirely on the device with zero
        host copies.

        Each sample in the batch is represented with sequence length S=1, guaranteeing
        independent-frame processing without multi-view cross-attention or view reordering.

        Supported input formats:
          - Sequence/List of 3D tensors: [(H, W, 3), ...] or [(3, H, W), ...]
          - Sequence/List of single-frame 4D tensors: [(1, 3, H, W), ...] or [(1, H, W, 3), ...]
          - Single 4D batched tensor: (B, 3, H, W) or (B, H, W, 3)
          - Single 5D batched tensor with S=1: (B, 1, 3, H, W) or (B, 1, H, W, 3)
          - Single 3D tensor: (H, W, 3) or (3, H, W) (treated as B=1)

        Returns:
            (model_tensor, original_shape, processed_shape, geometry)
            where model_tensor has shape (B, 1, 3, padded_h, padded_w).
        """
        orig_h: Optional[int] = None
        orig_w: Optional[int] = None
        x: torch.Tensor

        if isinstance(image_tensors, (list, tuple)):
            if len(image_tensors) == 0:
                raise ValueError("image_tensors sequence must not be empty.")
            bchw_list: List[torch.Tensor] = []
            device = None
            for idx, t in enumerate(image_tensors):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"Item {idx} in image_tensors must be torch.Tensor, got {type(t)}")
                if device is None:
                    device = t.device
                elif t.device != device:
                    raise ValueError(
                        f"Item {idx} device '{t.device}' does not match first tensor device '{device}'. "
                        f"All batch elements must reside on the same device."
                    )

                if t.ndim == 3:
                    if t.shape[2] == 3:
                        h, w = int(t.shape[0]), int(t.shape[1])
                        t_norm = t.permute(2, 0, 1).unsqueeze(0).contiguous()
                    elif t.shape[0] == 3:
                        h, w = int(t.shape[1]), int(t.shape[2])
                        t_norm = t.unsqueeze(0).contiguous()
                    else:
                        raise ValueError(f"Expected 3 color channels in dim 0 or 2, got shape {tuple(t.shape)}")
                elif t.ndim == 4:
                    if t.shape[0] != 1:
                        raise ValueError(
                            f"Item {idx} in image_tensors sequence must have batch dimension 1, got shape {tuple(t.shape)}"
                        )
                    if t.shape[1] == 3:
                        h, w = int(t.shape[2]), int(t.shape[3])
                        t_norm = t.contiguous()
                    elif t.shape[3] == 3:
                        h, w = int(t.shape[1]), int(t.shape[2])
                        t_norm = t.permute(0, 3, 1, 2).contiguous()
                    else:
                        raise ValueError(f"Expected 3 color channels in dim 1 or 3, got shape {tuple(t.shape)}")
                else:
                    raise ValueError(f"Expected 3D or 4D tensor for item {idx}, got shape {tuple(t.shape)}")

                if orig_h is None or orig_w is None:
                    orig_h, orig_w = h, w
                elif (h, w) != (orig_h, orig_w):
                    raise ValueError(
                        f"All frames in batch must have identical spatial dimensions. "
                        f"Frame 0 has ({orig_h}, {orig_w}), frame {idx} has ({h}, {w})"
                    )
                bchw_list.append(t_norm)
            x = torch.cat(bchw_list, dim=0)

        elif isinstance(image_tensors, torch.Tensor):
            if image_tensors.ndim == 3:
                if image_tensors.shape[2] == 3:
                    orig_h, orig_w = int(image_tensors.shape[0]), int(image_tensors.shape[1])
                    x = image_tensors.permute(2, 0, 1).unsqueeze(0).contiguous()
                elif image_tensors.shape[0] == 3:
                    orig_h, orig_w = int(image_tensors.shape[1]), int(image_tensors.shape[2])
                    x = image_tensors.unsqueeze(0).contiguous()
                else:
                    raise ValueError(f"Expected 3 color channels in dim 0 or 2, got shape {tuple(image_tensors.shape)}")
            elif image_tensors.ndim == 4:
                if image_tensors.shape[1] == 3:
                    orig_h, orig_w = int(image_tensors.shape[2]), int(image_tensors.shape[3])
                    x = image_tensors.contiguous()
                elif image_tensors.shape[3] == 3:
                    orig_h, orig_w = int(image_tensors.shape[1]), int(image_tensors.shape[2])
                    x = image_tensors.permute(0, 3, 1, 2).contiguous()
                else:
                    raise ValueError(f"Expected 3 color channels in dim 1 or 3, got shape {tuple(image_tensors.shape)}")
            elif image_tensors.ndim == 5:
                if image_tensors.shape[1] != 1:
                    raise ValueError(
                        f"Dimension 1 (sequence length S) must be 1 for independent frame inference, "
                        f"got shape {tuple(image_tensors.shape)}. Multi-view S>1 causes cross-frame attention leaks."
                    )
                if image_tensors.shape[2] == 3:
                    orig_h, orig_w = int(image_tensors.shape[3]), int(image_tensors.shape[4])
                    x = image_tensors.squeeze(1).contiguous()
                elif image_tensors.shape[4] == 3:
                    orig_h, orig_w = int(image_tensors.shape[2]), int(image_tensors.shape[3])
                    x = image_tensors.squeeze(1).permute(0, 3, 1, 2).contiguous()
                else:
                    raise ValueError(f"Expected 3 color channels in dim 2 or 4, got shape {tuple(image_tensors.shape)}")
            else:
                raise ValueError(f"Expected 3D, 4D, or 5D tensor, got shape {tuple(image_tensors.shape)}")
        else:
            raise TypeError(f"Expected torch.Tensor or Sequence[torch.Tensor], got {type(image_tensors)}")

        assert orig_h is not None and orig_w is not None
        batch_size = x.shape[0]
        if batch_size < 1:
            raise ValueError("Batch size must be >= 1.")

        # Dtype normalization to [0.0, 1.0] float32
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32) / 255.0
        elif x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = x.to(dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported tensor dtype {x.dtype}. Expected torch.uint8 or float32/16")

        geom: Optional[DepthGeometry] = None
        if depth_scale is not None:
            geom = compute_depth_geometry(orig_w, orig_h, scale=depth_scale)
            target_w, target_h = geom.req_width, geom.req_height
        else:
            effective_target = target_size if target_size is not None else DEFAULT_PROCESS_RES
            longest = max(orig_h, orig_w)
            scale = float(effective_target) / float(longest)
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))

            def round_to_multiple(dim: int, patch: int = PATCH_SIZE) -> int:
                down = (dim // patch) * patch
                up = down + patch
                return up if abs(up - dim) <= abs(dim - down) else down

            final_w = max(PATCH_SIZE, round_to_multiple(scaled_w))
            final_h = max(PATCH_SIZE, round_to_multiple(scaled_h))
            target_w, target_h = final_w, final_h

        # GPU resize across all batch items in parallel
        if (orig_h, orig_w) != (target_h, target_w):
            if target_w < orig_w and target_h < orig_h and interpolate_mode == "area":
                x = F.interpolate(x, size=(target_h, target_w), mode="area")
            else:
                mode = interpolate_mode if interpolate_mode in ("bilinear", "bicubic") else "bilinear"
                x = F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False)

        # Padding to patch size (14)
        if geom is not None:
            if geom.pad_right > 0 or geom.pad_bottom > 0:
                if geom.req_width > geom.pad_right and geom.req_height > geom.pad_bottom:
                    x = F.pad(x, (0, geom.pad_right, 0, geom.pad_bottom), mode="reflect")
                else:
                    x = F.pad(x, (0, geom.pad_right, 0, geom.pad_bottom), mode="replicate")
            padded_h, padded_w = geom.padded_height, geom.padded_width
        else:
            padded_h, padded_w = target_h, target_w

        # ImageNet normalization on GPU
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 5D tensor: (B, 1, 3, H, W) - independent samples with S=1
        tensor = x.unsqueeze(1)
        return tensor, (orig_h, orig_w), (padded_h, padded_w), geom

    def infer(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        target_size: Optional[int] = DEFAULT_PROCESS_RES,
        depth_scale: Optional[Union[str, float]] = None,
        return_original_size: bool = True,
        autocast: bool = True,
    ) -> DepthPredictionResult:
        """Run depth estimation on input image.

        Args:
            image: Image input (filepath, PIL image, or numpy array).
            target_size: Longest dimension size for processing (legacy default 504).
            depth_scale: Explicit spatial depth scale ('1/4', '1/2', '1/1').
            return_original_size: If True, interpolates final depth map to match input (H, W).
            autocast: If True and on CUDA, enables torch.autocast(fp16).

        Returns:
            DepthPredictionResult containing depth arrays, latency measurements, and metric metadata.
        """
        geom: Optional[DepthGeometry] = None
        if depth_scale is not None:
            tensor, orig_shape, proc_shape = self.preprocess_image(image, depth_scale=depth_scale)
            geom = compute_depth_geometry(orig_shape[1], orig_shape[0], scale=depth_scale)
        else:
            tensor, orig_shape, proc_shape = self.preprocess_image(image, target_size=target_size)
        tensor = tensor.to(self.device)

        is_cuda = self.device.type == "cuda"
        dtype_str = "float16" if (is_cuda and autocast) else "float32"

        if is_cuda:
            torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()

        with torch.no_grad():
            if is_cuda and autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(tensor, infer_gs=False)
            else:
                out = self.model(tensor, infer_gs=False)

        if is_cuda:
            torch.cuda.synchronize(self.device)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Extract and validate depth tensor from output schema
        if hasattr(out, "depth"):
            depth_tensor = out.depth
        elif isinstance(out, dict) and "depth" in out:
            depth_tensor = out["depth"]
        else:
            raise ValueError(
                f"Model output did not contain 'depth' field. "
                f"Available fields: {list(out.keys()) if hasattr(out, 'keys') else dir(out)}"
            )

        if depth_tensor.ndim == 4 and depth_tensor.shape[0] == 1 and depth_tensor.shape[1] == 1:
            depth_tensor = depth_tensor.squeeze(0).squeeze(0)
        elif depth_tensor.ndim == 3 and depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor.squeeze(0)
        elif depth_tensor.ndim != 2:
            raise ValueError(f"Unexpected depth tensor shape: {depth_tensor.shape}")

        depth_tensor = depth_tensor.detach().cpu().float()

        # Strict finite validation
        if not torch.isfinite(depth_tensor).all():
            raise ValueError("Depth prediction produced non-finite values (NaN or Inf).")

        raw_depth_np = depth_tensor.numpy()

        if geom is not None:
            depth_raw_np = unpad_depth_map(raw_depth_np, geom)
        else:
            depth_raw_np = raw_depth_np

        if return_original_size and (depth_raw_np.shape[0] != orig_shape[0] or depth_raw_np.shape[1] != orig_shape[1]):
            orig_h, orig_w = orig_shape
            depth_np = cv2.resize(depth_raw_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_np = depth_raw_np.copy()

        # Extract metric metadata without guessing (handle addict.Dict safely)
        # As documented in official DA3 specifications:
        # - DA3NESTED-GIANT-LARGE outputs metric depth directly in meters.
        # - DA3METRIC-LARGE outputs unscaled depth where metric_depth = focal * net_output / 300.
        #   Without camera intrinsics (focal length in pixels), raw metric output cannot be
        #   converted to meters. We never fabricate focal lengths.
        # - Therefore, DA3METRIC-LARGE raw output is strictly marked depth_units='focal_dependent_unscaled'
        #   and is_metric=False until an actual validated focal conversion is performed.
        is_metric_out = False
        if isinstance(out, dict):
            raw_metric = out.get("is_metric")
            if isinstance(raw_metric, (bool, int, float)):
                is_metric_out = bool(raw_metric)
        is_metric_pred = bool(is_metric_out or self.is_metric)

        scale_val: Optional[float] = None
        if isinstance(out, dict):
            raw_scale = out.get("scale_factor")
            if isinstance(raw_scale, (int, float)):
                scale_val = float(raw_scale)
            elif isinstance(raw_scale, torch.Tensor) and raw_scale.numel() == 1:
                scale_val = float(raw_scale.item())

        units = "meters" if is_metric_pred else self.depth_units

        return DepthPredictionResult(
            depth=depth_np,
            depth_raw=depth_raw_np,
            input_shape=orig_shape,
            processed_shape=proc_shape,
            latency_ms=latency_ms,
            device=str(self.device),
            dtype=dtype_str,
            min_depth=float(depth_np.min()),
            max_depth=float(depth_np.max()),
            mean_depth=float(depth_np.mean()),
            is_metric=is_metric_pred,
            metric_scale=scale_val,
            model_id=self.entry.id,
            depth_units=units,
            geometry=geom,
        )

    def infer_tensor(
        self,
        image_tensor: torch.Tensor,
        target_size: Optional[int] = DEFAULT_PROCESS_RES,
        depth_scale: Optional[Union[str, float]] = None,
        return_original_size: bool = True,
        autocast: bool = True,
        interpolate_mode: str = "area",
        validate_finite: bool = False,
        ensure_positive: bool = False,
        compute_stats: bool = False,
        timing: bool = True,
    ) -> DepthTensorResult:
        """Run depth estimation directly on a GPU-resident image tensor.

        Guarantees zero CPU-memory copies for image input and depth output.
        The returned DepthTensorResult holds PyTorch tensors residing on the
        model's target device.

        Fast-path characteristics:
          - No full-frame .cpu() or .numpy() transfers.
          - No device-wide torch.cuda.synchronize().
          - Optional CUDA events timing avoids host stalls.
          - By default, validate_finite=False and compute_stats=False avoid
            scalar GPU->CPU synchronization points.

        Args:
            image_tensor: Input tensor on device (H, W, 3) or (3, H, W) or (1, 3, H, W).
                          Must reside on self.device.
            target_size: Longest dimension size for processing (legacy default 504).
            depth_scale: Explicit spatial depth scale ('1/4', '1/2', '1/1').
            return_original_size: If True, interpolates fullres depth to match input (H, W) on GPU.
            autocast: If True and on CUDA, enables torch.autocast(fp16).
            interpolate_mode: Downscale interpolation mode ('area', 'bilinear', 'bicubic').
            validate_finite: If True, checks torch.isfinite() with a scalar sync.
            ensure_positive: If True, clamps depth values to min=1e-6 entirely on GPU.
            compute_stats: If True, calculates (min, max, mean) with explicit scalar sync.
            timing: If True, records latency in milliseconds using CUDA events (or perf_counter on CPU).

        Returns:
            DepthTensorResult containing GPU depth tensors, latency, and model metadata.
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(image_tensor)}")

        if image_tensor.device != self.device:
            raise ValueError(
                f"Input tensor device '{image_tensor.device}' does not match adapter device '{self.device}'. "
                f"Tensors must already reside on device to guarantee zero host copies."
            )

        is_cuda = self.device.type == "cuda"
        dtype_str = "float16" if (is_cuda and autocast) else "float32"

        tensor, orig_shape, proc_shape, geom = self.preprocess_tensor(
            image_tensor,
            target_size=target_size,
            depth_scale=depth_scale,
            interpolate_mode=interpolate_mode,
        )

        start_event: Any = None
        end_event: Any = None
        t_start: float = 0.0

        if timing:
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t_start = time.perf_counter()

        with torch.no_grad():
            if is_cuda and autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(tensor, infer_gs=False)
            else:
                out = self.model(tensor, infer_gs=False)

        latency_ms = 0.0
        if timing:
            if is_cuda and start_event is not None and end_event is not None:
                end_event.record()
                end_event.synchronize()
                latency_ms = float(start_event.elapsed_time(end_event))
            else:
                latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Extract depth tensor
        if hasattr(out, "depth"):
            depth_tensor = out.depth
        elif isinstance(out, dict) and "depth" in out:
            depth_tensor = out["depth"]
        else:
            raise ValueError(
                f"Model output did not contain 'depth' field. "
                f"Available fields: {list(out.keys()) if hasattr(out, 'keys') else dir(out)}"
            )

        if depth_tensor.ndim == 4 and depth_tensor.shape[0] == 1 and depth_tensor.shape[1] == 1:
            depth_tensor = depth_tensor.squeeze(0).squeeze(0)
        elif depth_tensor.ndim == 3 and depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor.squeeze(0)
        elif depth_tensor.ndim != 2:
            raise ValueError(f"Unexpected depth tensor shape: {depth_tensor.shape}")

        depth_tensor = depth_tensor.detach().float()

        # Unpad back to requested content dimensions
        if geom is not None:
            depth_raw = depth_tensor[: geom.req_height, : geom.req_width]
        else:
            depth_raw = depth_tensor

        # Clamping / positive handling on GPU if requested
        if ensure_positive:
            depth_raw = torch.clamp(depth_raw, min=1e-6)

        # Full-resolution GPU depth
        orig_h, orig_w = orig_shape
        if return_original_size and (depth_raw.shape[0] != orig_h or depth_raw.shape[1] != orig_w):
            depth_full = F.interpolate(
                depth_raw.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            if ensure_positive:
                depth_full = torch.clamp(depth_full, min=1e-6)
        else:
            depth_full = depth_raw

        # Optional scalar synchronization checks
        if validate_finite:
            if not torch.isfinite(depth_raw).all().item():
                raise ValueError("Depth prediction produced non-finite values (NaN or Inf).")

        d_min: Optional[float] = None
        d_max: Optional[float] = None
        d_mean: Optional[float] = None
        if compute_stats:
            d_min = float(depth_full.min().item())
            d_max = float(depth_full.max().item())
            d_mean = float(depth_full.mean().item())

        # Metadata extraction
        is_metric_out = False
        if isinstance(out, dict):
            raw_metric = out.get("is_metric")
            if isinstance(raw_metric, (bool, int, float)):
                is_metric_out = bool(raw_metric)
        is_metric_pred = bool(is_metric_out or self.is_metric)

        scale_val: Optional[float] = None
        if isinstance(out, dict):
            raw_scale = out.get("scale_factor")
            if isinstance(raw_scale, (int, float)):
                scale_val = float(raw_scale)
            elif isinstance(raw_scale, torch.Tensor) and raw_scale.numel() == 1:
                scale_val = float(raw_scale.item())

        units = "meters" if is_metric_pred else self.depth_units

        return DepthTensorResult(
            depth=depth_full,
            depth_raw=depth_raw,
            input_shape=orig_shape,
            processed_shape=proc_shape,
            latency_ms=latency_ms,
            device=str(self.device),
            dtype=dtype_str,
            is_metric=is_metric_pred,
            metric_scale=scale_val,
            model_id=self.entry.id,
            depth_units=units,
            geometry=geom,
            min_depth=d_min,
            max_depth=d_max,
            mean_depth=d_mean,
        )

    def infer_tensor_batch(
        self,
        image_tensors: Union[torch.Tensor, Sequence[torch.Tensor], List[torch.Tensor]],
        target_size: Optional[int] = DEFAULT_PROCESS_RES,
        depth_scale: Optional[Union[str, float]] = None,
        return_original_size: bool = True,
        autocast: bool = True,
        interpolate_mode: str = "area",
        validate_finite: bool = False,
        ensure_positive: bool = False,
        compute_stats: bool = False,
        timing: bool = True,
    ) -> List[DepthTensorResult]:
        """Run depth estimation directly on GPU-resident image tensors in batches.

        Guarantees zero CPU-memory copies for image inputs and depth outputs.
        Processes B independent frames simultaneously with sequence length S=1,
        eliminating multi-view cross-attention leakage while leveraging GPU parallelism.

        Args:
            image_tensors: Batched tensor (B, 3, H, W) / (B, H, W, 3) or Sequence of tensors.
                           All tensors must reside on self.device.
            target_size: Longest dimension size for processing (legacy default 504).
            depth_scale: Explicit spatial depth scale ('1/4', '1/2', '1/1').
            return_original_size: If True, interpolates fullres depth to match input (H, W) on GPU.
            autocast: If True and on CUDA, enables torch.autocast(fp16).
            interpolate_mode: Downscale interpolation mode ('area', 'bilinear', 'bicubic').
            validate_finite: If True, checks torch.isfinite() with a scalar sync.
            ensure_positive: If True, clamps depth values to min=1e-6 entirely on GPU.
            compute_stats: If True, calculates per-frame (min, max, mean) with explicit sync.
            timing: If True, records latency in milliseconds using CUDA events.
                    Each DepthTensorResult.latency_ms reports amortized per-frame latency
                    (total_batch_latency_ms / batch_size).

        Returns:
            List of DepthTensorResult objects of length B, one for each frame in the batch.
        """
        if isinstance(image_tensors, torch.Tensor):
            if image_tensors.device != self.device:
                raise ValueError(
                    f"Input tensor device '{image_tensors.device}' does not match adapter device '{self.device}'. "
                    f"Tensors must already reside on device to guarantee zero host copies."
                )
        elif isinstance(image_tensors, (list, tuple)):
            if len(image_tensors) == 0:
                raise ValueError("image_tensors must not be empty.")
            for idx, t in enumerate(image_tensors):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"Item {idx} in image_tensors must be torch.Tensor, got {type(t)}")
                if t.device != self.device:
                    raise ValueError(
                        f"Item {idx} device '{t.device}' does not match adapter device '{self.device}'. "
                        f"Tensors must already reside on device to guarantee zero host copies."
                    )
        else:
            raise TypeError(f"Expected torch.Tensor or Sequence[torch.Tensor], got {type(image_tensors)}")

        tensor, orig_shape, proc_shape, geom = self.preprocess_tensor_batch(
            image_tensors,
            target_size=target_size,
            depth_scale=depth_scale,
            interpolate_mode=interpolate_mode,
        )
        batch_size = tensor.shape[0]

        is_cuda = self.device.type == "cuda"
        dtype_str = "float16" if (is_cuda and autocast) else "float32"

        start_event: Any = None
        end_event: Any = None
        t_start: float = 0.0

        if timing:
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t_start = time.perf_counter()

        with torch.no_grad():
            if is_cuda and autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(tensor, infer_gs=False)
            else:
                out = self.model(tensor, infer_gs=False)

        total_latency_ms = 0.0
        if timing:
            if is_cuda and start_event is not None and end_event is not None:
                end_event.record()
                end_event.synchronize()
                total_latency_ms = float(start_event.elapsed_time(end_event))
            else:
                total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        per_frame_latency = total_latency_ms / batch_size if batch_size > 0 else total_latency_ms

        # Extract depth tensor
        if hasattr(out, "depth"):
            depth_tensor = out.depth
        elif isinstance(out, dict) and "depth" in out:
            depth_tensor = out["depth"]
        else:
            raise ValueError(
                f"Model output did not contain 'depth' field. "
                f"Available fields: {list(out.keys()) if hasattr(out, 'keys') else dir(out)}"
            )

        # Shape from model is (B, 1, H_pad, W_pad)
        if depth_tensor.ndim == 4 and depth_tensor.shape[1] == 1:
            depth_tensor = depth_tensor.squeeze(1)
        elif depth_tensor.ndim == 3 and depth_tensor.shape[0] == batch_size:
            pass
        elif depth_tensor.ndim == 2 and batch_size == 1:
            depth_tensor = depth_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected depth tensor shape: {depth_tensor.shape} for batch size {batch_size}")

        depth_tensor = depth_tensor.detach().float()

        # Unpad back to requested content dimensions
        if geom is not None:
            depth_raw = depth_tensor[:, : geom.req_height, : geom.req_width]
        else:
            depth_raw = depth_tensor

        # Clamping / positive handling on GPU if requested
        if ensure_positive:
            depth_raw = torch.clamp(depth_raw, min=1e-6)

        # Full-resolution GPU depth
        orig_h, orig_w = orig_shape
        if return_original_size and (depth_raw.shape[1] != orig_h or depth_raw.shape[2] != orig_w):
            depth_full = F.interpolate(
                depth_raw.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if ensure_positive:
                depth_full = torch.clamp(depth_full, min=1e-6)
        else:
            depth_full = depth_raw

        # Optional scalar synchronization checks
        if validate_finite:
            if not torch.isfinite(depth_raw).all().item():
                raise ValueError("Depth prediction produced non-finite values (NaN or Inf).")

        d_mins: List[Optional[float]] = [None] * batch_size
        d_maxs: List[Optional[float]] = [None] * batch_size
        d_means: List[Optional[float]] = [None] * batch_size
        if compute_stats:
            d_mins_t = depth_full.amin(dim=(-2, -1)).tolist()
            d_maxs_t = depth_full.amax(dim=(-2, -1)).tolist()
            d_means_t = depth_full.mean(dim=(-2, -1)).tolist()
            d_mins = [float(x) for x in d_mins_t]
            d_maxs = [float(x) for x in d_maxs_t]
            d_means = [float(x) for x in d_means_t]

        # Metadata extraction
        is_metric_out = False
        if isinstance(out, dict):
            raw_metric = out.get("is_metric")
            if isinstance(raw_metric, (bool, int, float)):
                is_metric_out = bool(raw_metric)
        is_metric_pred = bool(is_metric_out or self.is_metric)

        scale_val: Optional[float] = None
        if isinstance(out, dict):
            raw_scale = out.get("scale_factor")
            if isinstance(raw_scale, (int, float)):
                scale_val = float(raw_scale)
            elif isinstance(raw_scale, torch.Tensor) and raw_scale.numel() == 1:
                scale_val = float(raw_scale.item())

        units = "meters" if is_metric_pred else self.depth_units

        results: List[DepthTensorResult] = []
        for i in range(batch_size):
            results.append(
                DepthTensorResult(
                    depth=depth_full[i],
                    depth_raw=depth_raw[i],
                    input_shape=orig_shape,
                    processed_shape=proc_shape,
                    latency_ms=per_frame_latency,
                    device=str(self.device),
                    dtype=dtype_str,
                    is_metric=is_metric_pred,
                    metric_scale=scale_val,
                    model_id=self.entry.id,
                    depth_units=units,
                    geometry=geom,
                    min_depth=d_mins[i],
                    max_depth=d_maxs[i],
                    mean_depth=d_means[i],
                )
            )
        return results

    @staticmethod
    def colorize_depth(depth: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """Render a 2D float depth array as a normalized colormapped RGB image."""
        d_min = float(depth.min())
        d_max = float(depth.max())
        if abs(d_max - d_min) < 1e-8:
            norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            norm = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(norm, colormap)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_depth_outputs(
        result: DepthPredictionResult,
        out_dir: Union[str, Path],
        base_name: str = "depth",
    ) -> Dict[str, Path]:
        """Save depth outputs to disk: raw .npy, 16-bit grayscale .png, colorized .png, and metrics .json."""
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        npy_path = out_path / f"{base_name}.npy"
        np.save(npy_path, result.depth)

        # 16-bit normalized grayscale PNG
        d_min = float(result.depth.min())
        d_max = float(result.depth.max())
        if abs(d_max - d_min) < 1e-8:
            norm_u16 = np.zeros_like(result.depth, dtype=np.uint16)
        else:
            norm_u16 = ((result.depth - d_min) / (d_max - d_min) * 65535.0).astype(np.uint16)
        png_u16_path = out_path / f"{base_name}_u16.png"
        cv2.imwrite(str(png_u16_path), norm_u16)

        # Colorized RGB PNG
        color_rgb = DA3DepthAdapter.colorize_depth(result.depth)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        color_path = out_path / f"{base_name}_color.png"
        cv2.imwrite(str(color_path), color_bgr)

        # Metrics JSON
        metrics_path = out_path / f"{base_name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return {
            "npy": npy_path,
            "png_u16": png_u16_path,
            "png_color": color_path,
            "metrics": metrics_path,
        }


class DA3SmallDepthAdapter(DA3DepthAdapter):
    """Adapter for Depth Anything 3 Small model inference (backward compatible wrapper)."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        verify_hashes: bool = True,
    ) -> None:
        """Initialize adapter strictly for DA3-SMALL checkpoint."""
        super().__init__(
            model_dir=model_dir,
            identifier="DA3-SMALL",
            device=device,
            verify_hashes=verify_hashes,
        )
