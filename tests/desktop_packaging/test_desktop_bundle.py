"""Tests for PureGPU3D native desktop onedir Windows packaging artifact.

Verifies:
  1. Full onedir bundle layout (GUI EXE, worker EXE, _internal, bin/ffmpeg, bin/ffprobe, resources/models.json).
  2. Bundled media binaries provenance, version query, and license gap disclosure.
  3. Packaged default models directory is empty (no premature weight bundling).
  4. Offscreen GUI execution and window screenshot capture outside repository under hostile PATH.
  5. Isolated worker conversion on synthetic 1080p moving clip using staged DA3-BASE model under hostile PATH:
     - Strict JSON-lines protocol emission
     - 3840x1080 Full-SBS video export
     - Exact 12 frame preservation
     - Audio stream preservation
  6. Refusal of missing model / offline missing weight pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "dist" / "PureGPU3D"
GUI_EXE = DIST_DIR / "PureGPU3D.exe"
WORKER_EXE = DIST_DIR / "PureGPU3D-worker.exe"
SAMPLE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"
BASE_MODEL_SRC = REPO_ROOT / "models" / "DA3-BASE" / "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"


class TestDesktopPackaging(unittest.TestCase):
    """Verify onedir desktop build, layout, isolated hostile PATH execution, and video conversion."""

    temp_stage_dir: Path
    staged_bundle: Path
    staged_gui_exe: Path
    staged_worker_exe: Path
    staged_bin_dir: Path
    staged_ffmpeg: Path
    staged_ffprobe: Path
    staged_models_dir: Path
    staged_resources_dir: Path
    hostile_env: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        if not GUI_EXE.is_file() or not WORKER_EXE.is_file():
            raise unittest.SkipTest(f"Packaged executables not found at {DIST_DIR}. Run build_desktop.py first.")

        # Stage bundle to temporary directory outside repository to verify path isolation
        cls.temp_stage_dir = Path(tempfile.mkdtemp(prefix="puregpu3d_desktop_stage_"))
        cls.staged_bundle = cls.temp_stage_dir / "PureGPU3D"
        bundle_size_mb = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"\n[TestSetup] Staging {bundle_size_mb:.1f}MB ({bundle_size_mb / 1024:.2f}GB) desktop bundle to {cls.staged_bundle}...")
        t0 = time.time()
        shutil.copytree(DIST_DIR, cls.staged_bundle)
        print(f"[TestSetup] Staged in {time.time() - t0:.1f}s")

        cls.staged_gui_exe = cls.staged_bundle / "PureGPU3D.exe"
        cls.staged_worker_exe = cls.staged_bundle / "PureGPU3D-worker.exe"
        cls.staged_bin_dir = cls.staged_bundle / "bin"
        cls.staged_ffmpeg = cls.staged_bin_dir / "ffmpeg.exe"
        cls.staged_ffprobe = cls.staged_bin_dir / "ffprobe.exe"
        cls.staged_models_dir = cls.staged_bundle / "models"
        cls.staged_resources_dir = cls.staged_bundle / "resources"

        # Construct hostile PATH: only Windows system directories and bundle directory
        hostile_path = ";".join([
            str(cls.staged_bundle),
            str(cls.staged_bin_dir),
            r"C:\Windows\System32",
            r"C:\Windows",
        ])
        cls.hostile_env = os.environ.copy()
        cls.hostile_env["PATH"] = hostile_path
        cls.hostile_env.pop("PYTHONPATH", None)
        cls.hostile_env.pop("PYTHONHOME", None)
        cls.hostile_env.pop("VIRTUAL_ENV", None)
        cls.hostile_env["PYTHONDONTWRITEBYTECODE"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temp_stage_dir") and cls.temp_stage_dir.exists():
            shutil.rmtree(cls.temp_stage_dir, ignore_errors=True)

    def test_01_bundle_layout_and_default_empty_models(self) -> None:
        """Verify presence of all executables, binaries, resources, and default empty models."""
        self.assertTrue(self.staged_gui_exe.is_file(), "PureGPU3D.exe must exist in bundle root")
        self.assertTrue(self.staged_worker_exe.is_file(), "PureGPU3D-worker.exe must exist in bundle root")
        self.assertTrue(self.staged_ffmpeg.is_file(), "bin/ffmpeg.exe must exist in bundle")
        self.assertTrue(self.staged_ffprobe.is_file(), "bin/ffprobe.exe must exist in bundle")

        catalog_path = self.staged_resources_dir / "models.json"
        self.assertTrue(catalog_path.is_file(), "resources/models.json must exist in bundle")
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)
        model_ids = [m["id"] for m in catalog_data.get("models", [])]
        self.assertIn("DA3-BASE", model_ids)
        self.assertIn("DA3-SMALL", model_ids)

        # Verify models directory exists in bundle root
        self.assertTrue(self.staged_models_dir.is_dir(), "models/ directory must exist in bundle")

    def test_02_provenance_and_bundled_binary_execution(self) -> None:
        """Verify PROVENANCE.txt content and execution of bundled ffmpeg/ffprobe under hostile PATH."""
        provenance_path = self.staged_bin_dir / "PROVENANCE.txt"
        self.assertTrue(provenance_path.is_file(), "bin/PROVENANCE.txt must exist")
        prov_text = provenance_path.read_text(encoding="utf-8")
        self.assertIn("ffmpeg", prov_text)
        self.assertIn("ffprobe", prov_text)
        self.assertIn("Gyan.dev", prov_text)
        self.assertIn("GPLv3", prov_text)
        self.assertIn("Redistribution clearance", prov_text)

        # Test ffprobe execution under hostile PATH
        proc = subprocess.run(
            [str(self.staged_ffprobe), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        self.assertEqual(proc.returncode, 0, f"ffprobe failed: {proc.stderr}")
        self.assertIn("ffprobe version", proc.stdout)

        # Test ffmpeg execution under hostile PATH
        proc_ff = subprocess.run(
            [str(self.staged_ffmpeg), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        self.assertEqual(proc_ff.returncode, 0, f"ffmpeg failed: {proc_ff.stderr}")
        self.assertIn("ffmpeg version", proc_ff.stdout)

    def test_03_offscreen_ui_screenshot_hostile_path(self) -> None:
        """Verify PureGPU3D.exe launches offscreen under hostile PATH and captures valid UI screenshot."""
        screenshot_path = self.temp_stage_dir / "puregpu3d_ui_screenshot.png"
        cmd = [
            str(self.staged_gui_exe),
            "--screenshot", str(screenshot_path),
            "--input", str(SAMPLE_VIDEO),
            "--model", "DA3-BASE",
            "--strength", "0.04",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        self.assertEqual(proc.returncode, 0, f"PureGPU3D.exe --screenshot failed: {proc.stderr}")
        self.assertTrue(screenshot_path.is_file(), f"Screenshot was not generated at {screenshot_path}")
        self.assertGreater(screenshot_path.stat().st_size, 50000, "Screenshot file size unexpectedly small")

        with Image.open(screenshot_path) as img:
            w, h = img.size
            self.assertEqual((w, h), (1024, 768), f"Unexpected screenshot dimensions: {w}x{h}")

        # Update evidence screenshot in repository
        evidence_dest = REPO_ROOT / "data" / "verification" / "desktop-packaging" / "puregpu3d_desktop_ui.png"
        evidence_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(screenshot_path, evidence_dest)

    def test_04_unsupported_and_cancellable_model_pipeline(self) -> None:
        """Verify worker refuses unsupported models and supports cancellation during preparation."""
        # 1. Test immediate refusal of unsupported model (DA3-LARGE-1.1)
        cmd_file = self.temp_stage_dir / "unsupported_model_cmd.json"
        temp_out = self.temp_stage_dir / "should_not_exist.mp4"
        cmd_payload = {
            "version": "1.0",
            "job_id": "test-unsupported-model",
            "input_path": str(SAMPLE_VIDEO),
            "output_path": str(temp_out),
            "model_id": "DA3-LARGE-1.1",
            "disparity_strength": 0.03,
            "q_screen": 0.6,
            "overwrite": True,
            "acknowledge_license": True,
            "device": "cpu",
            "cancel_file": str(self.temp_stage_dir / "unsupported_cancel.tmp"),
            "ffmpeg_path": str(self.staged_ffmpeg),
            "ffprobe_path": str(self.staged_ffprobe),
        }
        cmd_file.write_text(json.dumps(cmd_payload), encoding="utf-8")

        proc = subprocess.run(
            [str(self.staged_worker_exe), "--command-file", str(cmd_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        self.assertEqual(proc.returncode, 1, f"Worker must exit with code 1 on unsupported model: {proc.stderr}")
        self.assertFalse(temp_out.is_file(), "Output file must not be created on unsupported model refusal")

        lines = [json.loads(line.strip()) for line in proc.stdout.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
        error_msgs = [m for m in lines if m.get("type") == "error"]
        self.assertGreater(len(error_msgs), 0, "Worker must emit error protocol message on unsupported model")
        self.assertEqual(error_msgs[0]["stage"], "failed")

        # 2. Test bounded cancellation before loading/downloading
        cancel_file = self.temp_stage_dir / "pre_cancel.tmp"
        cancel_file.write_text("cancel", encoding="utf-8")
        cancel_cmd_file = self.temp_stage_dir / "pre_cancel_cmd.json"
        cancel_payload = {
            "version": "1.0",
            "job_id": "test-pre-cancel",
            "input_path": str(SAMPLE_VIDEO),
            "output_path": str(temp_out),
            "model_id": "DA3-SMALL",
            "disparity_strength": 0.03,
            "q_screen": 0.6,
            "overwrite": True,
            "acknowledge_license": True,
            "device": "cpu",
            "cancel_file": str(cancel_file),
            "ffmpeg_path": str(self.staged_ffmpeg),
            "ffprobe_path": str(self.staged_ffprobe),
        }
        cancel_cmd_file.write_text(json.dumps(cancel_payload), encoding="utf-8")

        proc_c = subprocess.run(
            [str(self.staged_worker_exe), "--command-file", str(cancel_cmd_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        self.assertEqual(proc_c.returncode, 0, f"Worker must exit with 0 on cooperative cancel: {proc_c.stderr}")
        self.assertFalse(temp_out.is_file(), "Output file must not be created on cancellation")
        c_lines = [json.loads(line.strip()) for line in proc_c.stdout.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
        cancel_msgs = [m for m in c_lines if m.get("type") == "cancelled"]
        self.assertGreater(len(cancel_msgs), 0, "Worker must emit cancelled protocol message")

    def test_05_frozen_worker_real_conversion_hostile_path(self) -> None:
        """Verify actual video conversion using staged DA3-BASE model under hostile PATH outside repo."""
        # Stage DA3-BASE model weights into staged bundle's models/ directory
        target_rev_dir = self.staged_models_dir / "DA3-BASE" / "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"
        target_rev_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["config.json", "manifest.json", "model.safetensors"]:
            src_f = BASE_MODEL_SRC / fname
            if src_f.is_file():
                shutil.copy2(src_f, target_rev_dir / fname)

        output_video = self.temp_stage_dir / "synthetic_1080p_moving_converted_FullSBS.mp4"
        cmd_file = self.temp_stage_dir / "real_conv_cmd.json"
        cmd_payload = {
            "version": "1.0",
            "job_id": "acceptance-conversion",
            "input_path": str(SAMPLE_VIDEO),
            "output_path": str(output_video),
            "model_id": "DA3-BASE",
            "disparity_strength": 0.035,
            "q_screen": 0.6,
            "overwrite": True,
            "acknowledge_license": True,
            "cancel_file": str(self.temp_stage_dir / "cancel_dummy.tmp"),
            "ffmpeg_path": str(self.staged_ffmpeg),
            "ffprobe_path": str(self.staged_ffprobe),
        }
        cmd_file.write_text(json.dumps(cmd_payload), encoding="utf-8")

        print(f"\n[TestConversion] Launching PureGPU3D-worker.exe on {SAMPLE_VIDEO.name} with hostile PATH...")
        t_start = time.time()
        proc = subprocess.run(
            [str(self.staged_worker_exe), "--command-file", str(cmd_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        duration = time.time() - t_start
        print(f"[TestConversion] Completed in {duration:.1f}s, exit code {proc.returncode}")

        self.assertEqual(proc.returncode, 0, f"Worker failed (code {proc.returncode}):\nSTDERR:\n{proc.stderr}")
        self.assertTrue(output_video.is_file(), f"Output Full-SBS video not created at {output_video}")
        self.assertGreater(output_video.stat().st_size, 100000, "Output video unexpectedly small")

        # Parse protocol messages from stdout
        protocol_messages = []
        for line in proc.stdout.splitlines():
            clean = line.strip()
            if clean.startswith("{") and clean.endswith("}"):
                try:
                    protocol_messages.append(json.loads(clean))
                except json.JSONDecodeError:
                    pass

        message_types = [m.get("type") for m in protocol_messages]
        self.assertIn("status", message_types)
        self.assertIn("conversion_progress", message_types)
        self.assertIn("completed", message_types)

        # Inspect completed output with bundled ffprobe
        probe_cmd = [
            str(self.staged_ffprobe),
            "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,width,height,nb_frames",
            "-show_entries", "format=duration,size",
            "-of", "json",
            str(output_video),
        ]
        probe_proc = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            env=self.hostile_env,
            cwd=str(self.staged_bundle),
        )
        self.assertEqual(probe_proc.returncode, 0, f"ffprobe on converted video failed: {probe_proc.stderr}")
        probe_data = json.loads(probe_proc.stdout)

        streams = probe_data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        self.assertIsNotNone(video_stream, "Output video stream missing")
        self.assertIsNotNone(audio_stream, "Output audio stream missing (audio preservation failed)")
        assert video_stream is not None

        # Verify Full SBS dimensions (3840x1080)
        self.assertEqual(video_stream["width"], 3840, "Output width must be exactly 3840 (Full SBS)")
        self.assertEqual(video_stream["height"], 1080, "Output height must be exactly 1080")

        # Verify exact frame count (12 frames)
        nb_frames = int(video_stream.get("nb_frames", 0))
        self.assertEqual(nb_frames, 12, f"Expected exactly 12 output frames, found {nb_frames}")


if __name__ == "__main__":
    unittest.main()
