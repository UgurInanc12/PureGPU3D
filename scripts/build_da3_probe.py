#!/usr/bin/env python3
"""Build script for PureGPU3D DA3 Small frozen feasibility probe.

Invokes PyInstaller to create a minimal onedir Windows distribution,
bundles ffprobe with license provenance, and stages the verified adjacent
DA3 Small model checkpoint.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = REPO_ROOT / "packaging" / "da3_probe.spec"
DIST_DIR = REPO_ROOT / "dist" / "da3_probe"
BUILD_DIR = REPO_ROOT / "build" / "da3_probe"
MODEL_SRC = REPO_ROOT / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"


def find_system_ffprobe() -> Optional[Path]:
    """Discover installed ffprobe executable."""
    # Known winget location on this system
    winget_cand = Path(
        r"C:\Users\uguri\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffprobe.exe"
    )
    if winget_cand.is_file():
        return winget_cand

    # Fallback to PATH
    shutil_which = shutil.which("ffprobe")
    if shutil_which:
        return Path(shutil_which).resolve()

    return None


def get_ffprobe_provenance(ffprobe_path: Path) -> str:
    """Query ffprobe for version information and generate provenance text."""
    try:
        proc = subprocess.run(
            [str(ffprobe_path), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        first_line = proc.stdout.strip().splitlines()[0] if proc.stdout else "ffprobe version unknown"
    except Exception as exc:
        first_line = f"ffprobe (version query failed: {exc})"

    return (
        f"Component: ffprobe\n"
        f"Binary Path: bin/ffprobe.exe\n"
        f"Source: Gyan.dev FFmpeg Windows Full Build\n"
        f"Discovered Host Path: {ffprobe_path}\n"
        f"Version: {first_line}\n"
        f"License: GPLv3 / LGPL (depends on build configuration flags; built with --enable-gpl --enable-version3)\n"
        f"Packaging Date: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )


def compute_directory_size_mb(path: Path) -> float:
    """Compute total size of directory in megabytes."""
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
    return total_bytes / (1024 * 1024)


def build_da3_probe() -> int:
    """Execute complete onedir build pipeline for DA3 Small probe."""
    print("=" * 70)
    print("Building PureGPU3D DA3 Small Frozen Feasibility Probe")
    print("=" * 70)
    t_start = time.time()

    # 1. Verify environment and prerequisites
    if not SPEC_FILE.is_file():
        print(f"ERROR: PyInstaller spec file not found at {SPEC_FILE}")
        return 1

    if not MODEL_SRC.is_dir() or not (MODEL_SRC / "model.safetensors").is_file():
        print(f"ERROR: DA3 Small model checkpoint not found at {MODEL_SRC}")
        return 1

    ffprobe_src = find_system_ffprobe()
    if not ffprobe_src:
        print("ERROR: ffprobe.exe could not be discovered on host system.")
        return 1
    print(f"Discovered host ffprobe: {ffprobe_src}")

    # 2. Run PyInstaller
    print(f"\nRunning PyInstaller on {SPEC_FILE}...")
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(REPO_ROOT / "dist"),
        "--workpath",
        str(REPO_ROOT / "build"),
        str(SPEC_FILE),
    ]
    print(f"Command: {' '.join(pyinstaller_cmd)}")
    build_proc = subprocess.run(pyinstaller_cmd, cwd=str(REPO_ROOT))
    if build_proc.returncode != 0:
        print(f"ERROR: PyInstaller failed with return code {build_proc.returncode}")
        return build_proc.returncode

    exe_path = DIST_DIR / "da3_probe.exe"
    if not exe_path.is_file():
        print(f"ERROR: Built executable not found at expected location: {exe_path}")
        return 1
    print(f"PyInstaller build succeeded: {exe_path}")

    # 3. Stage bundled bin/ffprobe.exe
    bin_dir = DIST_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target_ffprobe = bin_dir / "ffprobe.exe"
    print(f"Bundling ffprobe to {target_ffprobe}...")
    shutil.copy2(ffprobe_src, target_ffprobe)

    provenance_text = get_ffprobe_provenance(ffprobe_src)
    provenance_file = bin_dir / "PROVENANCE.txt"
    provenance_file.write_text(provenance_text, encoding="utf-8")
    print(f"Wrote ffprobe provenance to {provenance_file}")

    # 4. Stage adjacent model directory
    target_model_dir = DIST_DIR / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"
    print(f"Staging adjacent DA3 Small model to {target_model_dir}...")
    target_model_dir.mkdir(parents=True, exist_ok=True)
    for model_file in ["config.json", "manifest.json", "model.safetensors"]:
        src_f = MODEL_SRC / model_file
        if src_f.is_file():
            dst_f = target_model_dir / model_file
            if not dst_f.is_file() or dst_f.stat().st_size != src_f.stat().st_size:
                print(f"  Copying {model_file} ({src_f.stat().st_size / (1024*1024):.1f} MB)...")
                shutil.copy2(src_f, dst_f)
            else:
                print(f"  {model_file} already present and matched size.")

    # 5. Measure and summarize
    elapsed = time.time() - t_start
    total_size_mb = compute_directory_size_mb(DIST_DIR)
    print("\n" + "=" * 70)
    print("Build Complete")
    print(f"Artifact Directory: {DIST_DIR}")
    print(f"Main Executable:    {exe_path}")
    print(f"Bundled ffprobe:    {target_ffprobe}")
    print(f"Adjacent Model:     {target_model_dir}")
    print(f"Total Bundle Size:  {total_size_mb:.1f} MB")
    print(f"Build Elapsed Time: {elapsed:.1f} s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(build_da3_probe())
