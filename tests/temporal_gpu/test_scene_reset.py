"""Unit tests for scene cut reset, fade detection, resolution changes, and O(1) memory bounds."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.stereo.temporal_gpu import TemporalGPUConfig, TemporalGPUStabilizer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU temporal tests")
class TestGPUSceneReset(unittest.TestCase):
    """Verify reset semantics across scene transitions and strictly bounded O(1) state."""

    def setUp(self) -> None:
        self.device = torch.device("cuda:0")

    def test_hard_cut_resets_history_and_bounds(self) -> None:
        """Verify passing is_cut=True completely resets temporal state and avoids cross-scene leakage."""
        h, w = 120, 160
        f1 = torch.full((3, h, w), 100.0, device=self.device, dtype=torch.float32)
        d1 = torch.full((h, w), 1.0, device=self.device, dtype=torch.float32)

        stabilizer = TemporalGPUStabilizer()
        r1 = stabilizer.process_frame(f1, d1)
        self.assertTrue(r1.is_cut)
        self.assertTrue(r1.cut_flag.item())

        # Frame 2: same scene
        r2 = stabilizer.process_frame(f1, d1)
        self.assertFalse(r2.is_cut)
        self.assertFalse(r2.cut_flag.item())

        # Frame 3: cut to completely new scene with different depth
        f3 = torch.full((3, h, w), 180.0, device=self.device, dtype=torch.float32)
        d3 = torch.full((h, w), 5.0, device=self.device, dtype=torch.float32)
        r3 = stabilizer.process_frame(f3, d3, is_cut=True)

        self.assertTrue(r3.is_cut)
        self.assertTrue(r3.cut_flag.item())
        # Frame 3 depth must equal d3 exactly (no leakage from d1)
        torch.testing.assert_close(r3.depth, d3, rtol=1e-4, atol=1e-4)

    def test_automatic_shot_cut_from_rgb_without_caller_flag(self) -> None:
        """Verify automatic shot cut detection resets temporal filter and bounds without caller is_cut=True."""
        h, w = 120, 160
        f1 = torch.full((3, h, w), 100.0, device=self.device, dtype=torch.float32)
        d1 = torch.full((h, w), 1.0, device=self.device, dtype=torch.float32)

        stabilizer = TemporalGPUStabilizer(TemporalGPUConfig(cut_threshold=0.25))
        r1 = stabilizer.process_frame(f1, d1)
        self.assertTrue(r1.is_cut)

        # Frame 2: same scene (no cut)
        r2 = stabilizer.process_frame(f1, d1)
        self.assertFalse(r2.is_cut)

        # Frame 3: cut to completely new scene, caller does NOT pass is_cut (defaults to False)
        f3 = torch.full((3, h, w), 180.0, device=self.device, dtype=torch.float32)
        d3 = torch.full((h, w), 5.0, device=self.device, dtype=torch.float32)
        r3 = stabilizer.process_frame(f3, d3, is_cut=False)

        # Must automatically detect cut from incoming RGB delta
        self.assertTrue(r3.is_cut)
        self.assertTrue(r3.cut_flag.item())
        torch.testing.assert_close(r3.depth, d3, rtol=1e-4, atol=1e-4)

        # Shot normalization bounds must NOT be contaminated by frame 1/2 inverse depth (1.0)
        # 1.0 / 5.0 is 0.2; bounds must be centered at 0.2, not mixed with 1.0 via EMA
        q_low, q_high = r3.normalization_bounds_float
        self.assertAlmostEqual(q_low, 0.2, places=3)
        self.assertAlmostEqual(q_high, 0.2, places=3)

    def test_fade_to_black_resets_temporal_filter(self) -> None:
        """Verify that black/fade frames automatically reset temporal filter to prevent ghosting."""
        h, w = 120, 160
        f1 = torch.full((3, h, w), 120.0, device=self.device, dtype=torch.float32)
        d1 = torch.full((h, w), 2.0, device=self.device, dtype=torch.float32)

        stabilizer = TemporalGPUStabilizer(TemporalGPUConfig(fade_threshold=10.0))
        stabilizer.process_frame(f1, d1)
        stabilizer.process_frame(f1, d1)

        # Fade frame (near black)
        f_fade = torch.full((3, h, w), 2.0, device=self.device, dtype=torch.float32)
        d_fade = torch.full((h, w), 10.0, device=self.device, dtype=torch.float32)
        r_fade = stabilizer.process_frame(f_fade, d_fade)

        self.assertTrue(r_fade.is_cut)

    def test_resolution_change_resets_state(self) -> None:
        """Verify dynamic resolution changes automatically trigger reset."""
        f1 = torch.full((3, 100, 120), 100.0, device=self.device, dtype=torch.float32)
        d1 = torch.full((100, 120), 2.0, device=self.device, dtype=torch.float32)

        stabilizer = TemporalGPUStabilizer()
        stabilizer.process_frame(f1, d1)

        f2 = torch.full((3, 140, 160), 100.0, device=self.device, dtype=torch.float32)
        d2 = torch.full((140, 160), 2.0, device=self.device, dtype=torch.float32)
        r2 = stabilizer.process_frame(f2, d2)

        self.assertTrue(r2.is_cut)
        self.assertEqual(r2.depth.shape, (140, 160))

    def test_bounded_o1_memory(self) -> None:
        """Verify that running over many frames does not leak GPU memory (strictly O(1) state)."""
        h, w = 180, 240
        stabilizer = TemporalGPUStabilizer()

        # Warm up
        f = torch.rand((3, h, w), device=self.device, dtype=torch.float32)
        d = torch.rand((h, w), device=self.device, dtype=torch.float32) + 0.5
        for _ in range(5):
            stabilizer.process_frame(f, d)

        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated(self.device)

        # Process 50 frames
        for _ in range(50):
            stabilizer.process_frame(f, d)

        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated(self.device)

        # Memory should be bounded within a negligible buffer
        self.assertLessEqual(mem_end - mem_start, 1024 * 1024, "GPU memory leaked across streaming frames")


if __name__ == "__main__":
    unittest.main()
