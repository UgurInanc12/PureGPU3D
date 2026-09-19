# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for PureGPU3D DA3 Small frozen feasibility probe."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

repo_root = Path(SPECPATH).resolve().parent
src_dir = repo_root / "src"
da3_src = repo_root / "third_party" / "depth_anything_3" / "src"

# Proven minimal import closure for DA3 Small inference
da3_hidden_imports = [
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
    "puregpu3d",
    "puregpu3d.models",
    "puregpu3d.models.da3_adapter",
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
]

# Collect torch binaries and data
torch_binaries = collect_dynamic_libs("torch")
torch_datas = collect_data_files("torch")

# Datas: include depth_anything_3 source tree into bundle
datas = [
    (str(da3_src / "depth_anything_3"), "depth_anything_3"),
] + torch_datas

binaries = list(torch_binaries)

a = Analysis(
    [str(repo_root / "scripts" / "frozen_da3_probe.py")],
    pathex=[str(src_dir), str(da3_src)],
    binaries=binaries,
    datas=datas,
    hiddenimports=da3_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="da3_probe",
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

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="da3_probe",
)
