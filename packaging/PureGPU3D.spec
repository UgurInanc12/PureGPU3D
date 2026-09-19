# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for PureGPU3D native desktop onedir distribution.

Builds:
  1. PureGPU3D.exe (GUI application, windowed/console=False)
  2. PureGPU3D-worker.exe (Dedicated worker, console=True)
Both share the common _internal runtime directory in dist/PureGPU3D.
"""

from pathlib import Path
from importlib.util import find_spec
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

repo_root = Path(SPECPATH).resolve().parent
src_dir = repo_root / "src"
da3_src = repo_root / "third_party" / "depth_anything_3" / "src"

# Proven import closure for desktop UI, worker process, video pipeline, and DA3 models
hidden_imports = [
    # Upstream DA3 model and architecture modules
    "depth_anything_3",
    "depth_anything_3.cfg",
    "depth_anything_3.model",
    "depth_anything_3.model.cam_dec",
    "depth_anything_3.model.cam_enc",
    "depth_anything_3.model.da3",
    "depth_anything_3.model.dinov2",
    "depth_anything_3.model.dinov2.dinov2",
    "depth_anything_3.model.dinov2.layers",
    "depth_anything_3.model.dinov2.layers.attention",
    "depth_anything_3.model.dinov2.layers.block",
    "depth_anything_3.model.dinov2.layers.drop_path",
    "depth_anything_3.model.dinov2.layers.layer_scale",
    "depth_anything_3.model.dinov2.layers.mlp",
    "depth_anything_3.model.dinov2.layers.patch_embed",
    "depth_anything_3.model.dinov2.layers.rope",
    "depth_anything_3.model.dinov2.layers.swiglu_ffn",
    "depth_anything_3.model.dinov2.vision_transformer",
    "depth_anything_3.model.dpt",
    "depth_anything_3.model.dualdpt",
    "depth_anything_3.model.reference_view_selector",
    "depth_anything_3.model.utils",
    "depth_anything_3.model.utils.attention",
    "depth_anything_3.model.utils.block",
    "depth_anything_3.model.utils.head_utils",
    "depth_anything_3.model.utils.transform",
    "depth_anything_3.utils",
    "depth_anything_3.utils.alignment",
    "depth_anything_3.utils.constants",
    "depth_anything_3.utils.geometry",
    "depth_anything_3.utils.logger",
    "depth_anything_3.utils.ray_utils",
    # PureGPU3D packages
    "puregpu3d",
    "puregpu3d.config",
    "puregpu3d.config.types",
    "puregpu3d.desktop",
    "puregpu3d.desktop.app",
    "puregpu3d.desktop.controller",
    "puregpu3d.desktop.window",
    "puregpu3d.jobs",
    "puregpu3d.jobs.output_transaction",
    "puregpu3d.metrics",
    "puregpu3d.metrics.collector",
    "puregpu3d.metrics.reporters",
    "puregpu3d.models",
    "puregpu3d.models.catalog",
    "puregpu3d.models.da3_adapter",
    "puregpu3d.models.download",
    "puregpu3d.models.store",
    "puregpu3d.runtime",
    "puregpu3d.runtime.paths",
    "puregpu3d.runtime.protocol",
    "puregpu3d.runtime.worker",
    "puregpu3d.stereo",
    "puregpu3d.stereo.disparity",
    "puregpu3d.stereo.fill",
    "puregpu3d.stereo.splat",
    "puregpu3d.video",
    "puregpu3d.video.convert",
    "puregpu3d.video.probe",
    # External libraries
    "safetensors",
    "safetensors.torch",
    "einops",
    "cv2",
    "PIL",
    "PIL.Image",
    "torch",
    "torchvision",
    "yaml",
    "omegaconf",
    "addict",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Dynamic libraries and data files
torch_binaries = collect_dynamic_libs("torch")
torch_datas = collect_data_files("torch")

datas = [
    (str(da3_src / "depth_anything_3"), "depth_anything_3"),
    (str(repo_root / "resources" / "models.json"), "resources"),
    (str(repo_root / "resources" / "models.json"), "puregpu3d/resources"),
] + torch_datas

binaries = list(torch_binaries)

# NVIDIA selects an ABI-specific extension by filename at runtime; static
# import discovery only finds VersionCheck, not the codec implementations.
codec_spec = find_spec("PyNvVideoCodec")
if codec_spec is None or codec_spec.origin is None:
    raise RuntimeError("Build requires the pinned PyNvVideoCodec dependency")
codec_dir = Path(codec_spec.origin).parent
codec_extensions = list(codec_dir.glob("PyNvVideoCodec_*.pyd"))
if not codec_extensions:
    raise RuntimeError("PyNvVideoCodec native implementations are missing")
binaries += [(str(p), "PyNvVideoCodec") for p in codec_extensions]
binaries += collect_dynamic_libs("PyNvVideoCodec")

excludes = [
    "puregpu3d.cli",
    "puregpu3d.ui.gradio_app",
    "tkinter",
    "matplotlib",
    "scipy",
    "open3d",
    "trimesh",
    "viser",
    "gradio",
    "uvicorn",
    "fastapi",
    "pandas",
    "IPython",
    "pytest",
    "typer",
]

# 1. Main Desktop UI Analysis
a_gui = Analysis(
    [str(repo_root / "scripts" / "desktop_entry.py")],
    pathex=[str(src_dir), str(da3_src)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz_gui = PYZ(a_gui.pure)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="PureGPU3D",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 2. Worker Console Analysis
a_worker = Analysis(
    [str(repo_root / "scripts" / "worker_entry.py")],
    pathex=[str(src_dir), str(da3_src)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz_worker = PYZ(a_worker.pure)

exe_worker = EXE(
    pyz_worker,
    a_worker.scripts,
    [],
    exclude_binaries=True,
    name="PureGPU3D-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 3. Combined Onedir Collection
coll = COLLECT(
    exe_gui,
    a_gui.binaries,
    a_gui.datas,
    exe_worker,
    a_worker.binaries,
    a_worker.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PureGPU3D",
)
