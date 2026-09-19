import sys
import tempfile
import time
from pathlib import Path
import torch

REPO_ROOT = Path("E:/Hermes/PureGPU3D")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "depth_anything_3" / "src"))

from puregpu3d.models.da3_adapter import DA3DepthAdapter
from puregpu3d.video.gpu_convert import convert_video_gpu

SAMPLE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"
CHECKPOINT_DIR_SMALL = REPO_ROOT / "models" / "DA3-SMALL" / "e08cab65ca0ec38e7826075418411ab90cab4da3"

def main():
    device = "cuda:0"
    print("Loading adapter...")
    adapter = DA3DepthAdapter(CHECKPOINT_DIR_SMALL, identifier="DA3-SMALL", device=device, verify_hashes=False)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        warmup_out = Path(tmp_dir) / "warmup.mp4"
        print("Running warmup...")
        res_warmup = convert_video_gpu(
            input_path=SAMPLE_VIDEO,
            output_path=warmup_out,
            model=adapter,
            depth_scale="1/4",
            codec="hevc",
            enable_temporal_stabilization=True,
            overwrite=True,
        )
        print(f"Warmup done: {res_warmup.effective_fps:.2f} FPS, total wall: {res_warmup.wall_clock_seconds:.3f}s")
        
        timed_out = Path(tmp_dir) / "timed.mp4"
        print("Running timed run 1...")
        t0 = time.perf_counter()
        res1 = convert_video_gpu(
            input_path=SAMPLE_VIDEO,
            output_path=timed_out,
            model=adapter,
            depth_scale="1/4",
            codec="hevc",
            enable_temporal_stabilization=True,
            overwrite=True,
        )
        t1 = time.perf_counter()
        print(f"Run 1: {res1.effective_fps:.2f} FPS, total wall: {res1.wall_clock_seconds:.3f}s, outer wall: {t1-t0:.3f}s")
        print(f"Stages: depth={res1.mean_depth_ms:.2f}ms, temporal={res1.mean_temporal_ms:.2f}ms, stereo={res1.mean_stereo_ms:.2f}ms, decode={res1.mean_decode_ms:.2f}ms, encode={res1.mean_encode_ms:.2f}ms")

        print("Running timed run 2...")
        t0 = time.perf_counter()
        res2 = convert_video_gpu(
            input_path=SAMPLE_VIDEO,
            output_path=timed_out,
            model=adapter,
            depth_scale="1/4",
            codec="hevc",
            enable_temporal_stabilization=True,
            overwrite=True,
        )
        t1 = time.perf_counter()
        print(f"Run 2: {res2.effective_fps:.2f} FPS, total wall: {res2.wall_clock_seconds:.3f}s, outer wall: {t1-t0:.3f}s")
        print(f"Stages: depth={res2.mean_depth_ms:.2f}ms, temporal={res2.mean_temporal_ms:.2f}ms, stereo={res2.mean_stereo_ms:.2f}ms, decode={res2.mean_decode_ms:.2f}ms, encode={res2.mean_encode_ms:.2f}ms")

if __name__ == "__main__":
    main()
