"""Tests for PureGPU3D DA3 Small frozen packaging artifact."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "dist" / "da3_probe"
EXE_PATH = DIST_DIR / "da3_probe.exe"


class TestDA3PackagingArtifact(unittest.TestCase):
    """Verify onedir build integrity, layout, bundled ffprobe, and isolated execution."""

    temp_stage_dir: Path
    staged_exe: Path
    staged_bin_ffprobe: Path
    staged_models_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        if not EXE_PATH.is_file():
            raise unittest.SkipTest(f"Frozen executable not found at {EXE_PATH}. Run build_da3_probe.py first.")

        # Stage bundle to temporary directory outside repository to verify path isolation
        cls.temp_stage_dir = Path(tempfile.mkdtemp(prefix="puregpu3d_stage_"))
        dst_bundle = cls.temp_stage_dir / "da3_probe"
        shutil.copytree(DIST_DIR, dst_bundle)
        cls.staged_exe = dst_bundle / "da3_probe.exe"
        cls.staged_bin_ffprobe = dst_bundle / "bin" / "ffprobe.exe"
        cls.staged_models_dir = dst_bundle / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temp_stage_dir") and cls.temp_stage_dir.exists():
            shutil.rmtree(cls.temp_stage_dir, ignore_errors=True)

    def test_staged_bundle_structure(self) -> None:
        """Verify layout of staged bundle outside repo."""
        self.assertTrue(self.staged_exe.is_file(), "Main executable must exist in staged bundle")
        self.assertTrue(self.staged_bin_ffprobe.is_file(), "Bundled ffprobe must exist in bin/")
        provenance_txt = self.staged_exe.parent / "bin" / "PROVENANCE.txt"
        self.assertTrue(provenance_txt.is_file(), "PROVENANCE.txt must exist in bin/")
        self.assertTrue((self.staged_models_dir / "config.json").is_file(), "config.json missing in staged models")
        self.assertTrue((self.staged_models_dir / "model.safetensors").is_file(), "model.safetensors missing in staged models")

    def test_bundled_ffprobe_execution(self) -> None:
        """Verify bundled ffprobe executes and reports valid version info."""
        proc = subprocess.run(
            [str(self.staged_bin_ffprobe), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Bundled ffprobe failed: {proc.stderr}")
        self.assertIn("ffprobe version", proc.stdout)
        self.assertIn("gyan.dev", proc.stdout.lower())

    def test_isolated_hostile_path_execution_both_devices(self) -> None:
        """Run self-test from staged directory with hostile PATH (no python, no git, no repo)."""
        bundle_dir = self.staged_exe.parent
        output_dir = bundle_dir / "test_out_both"
        json_out = output_dir / "self_test_result.json"

        # Hostile PATH: only Windows system and the staged bundle directories
        hostile_path = ";".join([
            str(bundle_dir),
            str(bundle_dir / "bin"),
            r"C:\Windows\System32",
            r"C:\Windows",
        ])

        env = os.environ.copy()
        env["PATH"] = hostile_path
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.pop("VIRTUAL_ENV", None)

        cmd = [
            str(self.staged_exe),
            "--self-test",
            "--device", "both",
            "--output-dir", str(output_dir),
            "--json-out", str(json_out),
            "--quiet",
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(bundle_dir),
            env=env,
            timeout=120,
        )

        self.assertEqual(proc.returncode, 0, f"Process exited with code {proc.returncode}: {proc.stderr}")
        self.assertTrue(json_out.is_file(), f"Expected JSON report at {json_out}")

        data = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["frozen"])
        self.assertTrue(data["bundled_ffprobe"]["found"])
        self.assertTrue(data["da3_vendor_import"]["success"])
        self.assertTrue(data["model"]["found"])

        # Check CPU inference metrics
        self.assertIsNotNone(data["cpu_inference"])
        self.assertEqual(data["cpu_inference"]["device"], "cpu")
        self.assertGreater(data["cpu_inference"]["mean_depth"], 0.0)

        # Check CUDA inference metrics (if machine has CUDA)
        if data["cuda_available"]:
            self.assertIsNotNone(data["cuda_inference"])
            self.assertIn("cuda", data["cuda_inference"]["device"])
            self.assertGreater(data["cuda_inference"]["mean_depth"], 0.0)
            self.assertIsNotNone(data["comparison"])
            self.assertLess(data["comparison"]["mae"], 0.01)

        # Check output depth files were written
        self.assertTrue((output_dir / "depth_cpu.npy").is_file())
        self.assertTrue((output_dir / "depth_cpu_color.png").is_file())
        self.assertTrue((output_dir / "depth_cpu_u16.png").is_file())
        if data["cuda_available"]:
            self.assertTrue((output_dir / "depth_cuda.npy").is_file())
            self.assertTrue((output_dir / "depth_cuda_color.png").is_file())
            self.assertTrue((output_dir / "depth_cuda_u16.png").is_file())

    def test_isolated_hostile_path_cpu_only(self) -> None:
        """Run self-test forcing CPU only under hostile PATH."""
        bundle_dir = self.staged_exe.parent
        output_dir = bundle_dir / "test_out_cpu"
        json_out = output_dir / "cpu_result.json"

        hostile_path = ";".join([
            str(bundle_dir),
            str(bundle_dir / "bin"),
            r"C:\Windows\System32",
            r"C:\Windows",
        ])

        env = os.environ.copy()
        env["PATH"] = hostile_path
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.pop("VIRTUAL_ENV", None)

        cmd = [
            str(self.staged_exe),
            "--self-test",
            "--device", "cpu",
            "--output-dir", str(output_dir),
            "--json-out", str(json_out),
            "--quiet",
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(bundle_dir),
            env=env,
            timeout=120,
        )

        self.assertEqual(proc.returncode, 0, f"Process exited with code {proc.returncode}: {proc.stderr}")
        data = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "success")
        self.assertIsNotNone(data["cpu_inference"])
        self.assertIsNone(data["cuda_inference"])
        self.assertTrue((output_dir / "depth_cpu.npy").is_file())
        self.assertTrue((output_dir / "depth_cpu_color.png").is_file())
        self.assertTrue((output_dir / "depth_cpu_u16.png").is_file())


if __name__ == "__main__":
    unittest.main()
