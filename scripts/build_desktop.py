#!/usr/bin/env python3
"""Build script for PureGPU3D native desktop onedir Windows distribution.

Invokes PyInstaller to create a complete onedir Windows application containing:
  - PureGPU3D.exe (GUI application entrypoint)
  - PureGPU3D-worker.exe (Dedicated conversion worker entrypoint)
  - _internal (Shared runtime components: PyTorch, PySide6, OpenCV, DA3 vendor)
  - bin/ffmpeg.exe and bin/ffprobe.exe (Discovered and bundled media binaries)
  - bin/PROVENANCE.txt (Media binary provenance and license gap documentation)
  - resources/models.json (Audited model catalog)
  - models/ (App-root model directory, packaged empty by default)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = REPO_ROOT / "packaging" / "PureGPU3D.spec"
DIST_DIR = REPO_ROOT / "dist" / "PureGPU3D"
BUILD_DIR = REPO_ROOT / "build" / "desktop"


def find_system_media_binaries() -> Tuple[Optional[Path], Optional[Path]]:
    """Discover host ffmpeg and ffprobe executables."""
    winget_bin_dir = Path(
        r"C:\Users\uguri\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin"
    )
    ffmpeg_cand = winget_bin_dir / "ffmpeg.exe"
    ffprobe_cand = winget_bin_dir / "ffprobe.exe"

    ffmpeg_path: Optional[Path] = ffmpeg_cand if ffmpeg_cand.is_file() else None
    ffprobe_path: Optional[Path] = ffprobe_cand if ffprobe_cand.is_file() else None

    if not ffmpeg_path:
        w = shutil.which("ffmpeg")
        if w:
            ffmpeg_path = Path(w).resolve()

    if not ffprobe_path:
        w = shutil.which("ffprobe")
        if w:
            ffprobe_path = Path(w).resolve()

    return ffmpeg_path, ffprobe_path


def get_binary_version(bin_path: Path) -> str:
    """Query executable for version string."""
    try:
        proc = subprocess.run(
            [str(bin_path), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        return proc.stdout.strip().splitlines()[0] if proc.stdout else "version unknown"
    except Exception as exc:
        return f"version query failed: {exc}"


def generate_provenance(ffmpeg_path: Path, ffprobe_path: Path) -> str:
    """Generate detailed PROVENANCE.txt content for bundled media binaries."""
    ffmpeg_ver = get_binary_version(ffmpeg_path)
    ffprobe_ver = get_binary_version(ffprobe_path)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return (
        "PureGPU3D Bundled Media Binaries Provenance & Licensing Disclosure\n"
        "===================================================================\n\n"
        f"Packaging Date: {timestamp}\n\n"
        "Component: ffmpeg\n"
        "  Target Path: bin/ffmpeg.exe\n"
        f"  Source Host Path: {ffmpeg_path}\n"
        f"  Version: {ffmpeg_ver}\n"
        "  Build Origin: Gyan.dev FFmpeg Windows Full Build\n"
        "  License: GPLv3 (built with --enable-gpl --enable-version3)\n\n"
        "Component: ffprobe\n"
        "  Target Path: bin/ffprobe.exe\n"
        f"  Source Host Path: {ffprobe_path}\n"
        f"  Version: {ffprobe_ver}\n"
        "  Build Origin: Gyan.dev FFmpeg Windows Full Build\n"
        "  License: GPLv3 (built with --enable-gpl --enable-version3)\n\n"
        "Licensing Notice & Legal Clearance Gap:\n"
        "  The bundled ffmpeg and ffprobe binaries are pre-compiled full builds from Gyan.dev\n"
        "  licensed under the GNU General Public License version 3 (GPLv3). They are bundled\n"
        "  here for portable local runtime execution and verification.\n"
        "  Redistribution clearance for commercial distribution is NOT claimed as final legal\n"
        "  release clearance. A commercial distribution would require replacing GPL-dependent\n"
        "  libraries with an LGPLv2.1+ build or obtaining requisite patent/codec licenses.\n"
    )


def compute_directory_size_mb(path: Path) -> float:
    """Compute total size of directory in megabytes."""
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
    return total_bytes / (1024 * 1024)


def build_desktop() -> int:
    """Execute complete onedir build pipeline for PureGPU3D desktop distribution."""
    print("=" * 75)
    print("Building PureGPU3D Native Desktop Onedir Windows Distribution")
    print("=" * 75)
    t_start = time.time()

    # 1. Verify prerequisites
    if not SPEC_FILE.is_file():
        print(f"ERROR: PyInstaller spec file not found at {SPEC_FILE}")
        return 1

    ffmpeg_src, ffprobe_src = find_system_media_binaries()
    if not ffmpeg_src or not ffprobe_src:
        print(f"ERROR: Host ffmpeg ({ffmpeg_src}) or ffprobe ({ffprobe_src}) not found.")
        return 1
    print(f"Discovered ffmpeg:  {ffmpeg_src}")
    print(f"Discovered ffprobe: {ffprobe_src}")

    catalog_src = REPO_ROOT / "resources" / "models.json"
    if not catalog_src.is_file():
        print(f"ERROR: Model catalog not found at {catalog_src}")
        return 1
    print(f"Discovered catalog: {catalog_src}")

    # 2. Run PyInstaller
    print(f"\nRunning PyInstaller on {SPEC_FILE}...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(REPO_ROOT / "dist"),
        "--workpath",
        str(BUILD_DIR),
        str(SPEC_FILE),
    ]
    print(f"Command: {' '.join(pyinstaller_cmd)}")

    build_env = os.environ.copy()
    build_env.pop("PYTHONPATH", None)
    build_env["PYTHONDONTWRITEBYTECODE"] = "1"

    build_proc = subprocess.run(pyinstaller_cmd, cwd=str(REPO_ROOT), env=build_env)
    if build_proc.returncode != 0:
        print(f"ERROR: PyInstaller failed with return code {build_proc.returncode}")
        return build_proc.returncode

    exe_gui = DIST_DIR / "PureGPU3D.exe"
    exe_worker = DIST_DIR / "PureGPU3D-worker.exe"

    if not exe_gui.is_file():
        print(f"ERROR: Built GUI executable not found at {exe_gui}")
        return 1
    if not exe_worker.is_file():
        print(f"ERROR: Built worker executable not found at {exe_worker}")
        return 1
    print(f"PyInstaller build succeeded:\n  GUI:    {exe_gui}\n  Worker: {exe_worker}")

    # 3. Stage bundled media binaries (bin/ffmpeg.exe, bin/ffprobe.exe, bin/PROVENANCE.txt)
    bin_dir = DIST_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    target_ffmpeg = bin_dir / "ffmpeg.exe"
    target_ffprobe = bin_dir / "ffprobe.exe"
    print(f"\nBundling media binaries into {bin_dir}...")
    shutil.copy2(ffmpeg_src, target_ffmpeg)
    shutil.copy2(ffprobe_src, target_ffprobe)

    provenance_content = generate_provenance(ffmpeg_src, ffprobe_src)
    provenance_path = bin_dir / "PROVENANCE.txt"
    provenance_path.write_text(provenance_content, encoding="utf-8")
    print(f"Wrote media provenance to {provenance_path}")

    # 4. Stage adjacent resources/models.json
    res_dir = DIST_DIR / "resources"
    res_dir.mkdir(parents=True, exist_ok=True)
    target_catalog = res_dir / "models.json"
    print(f"Staging adjacent model catalog to {target_catalog}...")
    shutil.copy2(catalog_src, target_catalog)

    # 5. Ensure adjacent models/ directory exists (packaged empty by default)
    models_dir = DIST_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prepared adjacent models directory (empty by default): {models_dir}")

    # 6. Final verification and summary
    elapsed = time.time() - t_start
    total_size_mb = compute_directory_size_mb(DIST_DIR)

    print("\n" + "=" * 75)
    print("PureGPU3D Desktop Distribution Build Complete")
    print("=" * 75)
    print(f"Artifact Directory:  {DIST_DIR}")
    print(f"GUI Executable:      {exe_gui}")
    print(f"Worker Executable:   {exe_worker}")
    print(f"Bundled FFmpeg:      {target_ffmpeg}")
    print(f"Bundled FFprobe:     {target_ffprobe}")
    print(f"Media Provenance:    {provenance_path}")
    print(f"Bundled Catalog:     {target_catalog}")
    print(f"Adjacent Models Dir: {models_dir}")
    print(f"Total Bundle Size:   {total_size_mb:.1f} MB")
    print(f"Build Elapsed Time:  {elapsed:.1f} s")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(build_desktop())
