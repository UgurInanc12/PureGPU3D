"""Tests for PyNvVideoCodec NVDEC -> PyTorch -> NVENC interop (Phase P3)."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import torch
import PyNvVideoCodec as nvc

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"


def ffprobe_streams(file_path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_name,codec_type,width,height,nb_frames,r_frame_rate",
        "-of", "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout).get("streams", [])


class TestNvCodecInterop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA device not available")
        if not SAMPLE_VIDEO.exists():
            raise unittest.SkipTest(f"Sample video not found at {SAMPLE_VIDEO}")

    def test_01_driver_and_version(self):
        """Verify PyNvVideoCodec package imports and detects compatible NVENC driver version."""
        self.assertTrue(hasattr(nvc, "__version__"))
        self.assertTrue(hasattr(nvc, "supportedNvEncVersion"))
        # Check driver version is NVENC 12.1+ or 13.0+
        self.assertGreaterEqual(nvc.supportedNvEncVersion, nvc.NVENC_VER_12)

    def test_02_dlpack_zero_copy_pointer_contract(self):
        """Verify that torch.from_dlpack extracts device tensor matching NVDEC plane pointer."""
        dec = nvc.SimpleDecoder(
            str(SAMPLE_VIDEO),
            gpu_id=0,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )
        self.assertGreater(len(dec), 0)
        frame0 = dec[0]

        # Check DLPack interface
        self.assertTrue(hasattr(frame0, "__dlpack__"))
        plane_ptr = frame0.GetPtrToPlane(0)
        self.assertGreater(plane_ptr, 0)

        # Convert to PyTorch tensor via DLPack
        tensor = torch.from_dlpack(frame0)
        self.assertEqual(tensor.device.type, "cuda")
        self.assertEqual(tensor.device.index, 0)
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertEqual(tensor.shape, (1080, 1920, 3))
        # Exact device pointer match proves zero-copy pointer handover
        self.assertEqual(tensor.data_ptr(), plane_ptr)

    def test_03_surface_reuse_hazard_and_safe_cloning(self):
        """
        Verify decoder surface reuse hazard and prove that cloning isolates tensor values.

        The NVDEC SimpleDecoder reuses internal device buffers across consecutive dec[i]
        lookups. Unowned DLPack tensor views will have their underlying memory overwritten
        by subsequent decode operations.
        """
        dec = nvc.SimpleDecoder(
            str(SAMPLE_VIDEO),
            gpu_id=0,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )
        self.assertGreaterEqual(len(dec), 2)

        # Frame 0 view and safe clone
        f0 = dec[0]
        t0_view = torch.from_dlpack(f0)
        t0_cloned = t0_view.clone()

        # Frame 1 decode overwrites decoder surface
        f1 = dec[1]
        t1_view = torch.from_dlpack(f1)

        # Evidence: both unowned views point to the exact same device address
        self.assertEqual(t0_view.data_ptr(), t1_view.data_ptr())
        # The unowned view t0_view now equals t1_view because underlying memory was overwritten
        self.assertTrue(torch.equal(t0_view, t1_view))
        # But the cloned tensor safely preserved original Frame 0 data
        self.assertNotEqual(t0_cloned.data_ptr(), t1_view.data_ptr())
        self.assertFalse(torch.equal(t0_cloned, t1_view))

    def test_04_stream_synchronization(self):
        """Verify decoder and encoder interop on a custom CUDA stream."""
        stream = torch.cuda.Stream()
        dec = nvc.SimpleDecoder(
            str(SAMPLE_VIDEO),
            gpu_id=0,
            cuda_stream=stream.cuda_stream,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )

        with torch.cuda.stream(stream):
            frame0 = dec[0]
            tensor = torch.from_dlpack(frame0)
            tensor_sq = tensor.float().pow(2)

        stream.synchronize()
        self.assertEqual(tensor_sq.shape, (1080, 1920, 3))
        self.assertEqual(tensor_sq.device.type, "cuda")

    def test_05_end_to_end_sbs_nvenc_probe(self):
        """Verify full GPU-resident NVDEC -> PyTorch SBS 3840x1080 -> NVENC HEVC pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "test_sbs_out.mp4"

            stream = torch.cuda.Stream()
            dec = nvc.SimpleDecoder(
                str(SAMPLE_VIDEO),
                gpu_id=0,
                cuda_stream=stream.cuda_stream,
                use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
            )
            total_frames = len(dec)
            meta = dec.get_stream_metadata()
            fps = int(round(meta.average_fps)) if meta.average_fps > 0 else 24

            out_w, out_h = 3840, 1080
            enc = nvc.CreateEncoder(
                out_w,
                out_h,
                "ABGR",
                False,
                codec="hevc",
                cudastream=stream.cuda_stream,
                fps=str(fps),
            )
            extradata = enc.GetSequenceParams()

            muxer = nvc.FFmpegMuxer(
                str(out_path),
                nvc.MEDIA_FORMAT.MP4,
                "hevc",
                out_w,
                out_h,
                fps,
                1,
                1,
                90000,
                extradata,
            )
            muxer.SetUniformPtsIncrement(90000 // fps)

            with torch.cuda.stream(stream):
                for i in range(total_frames):
                    f = dec[i]
                    t_in = torch.from_dlpack(f)
                    # Create SBS tensor on GPU
                    t_right = torch.roll(t_in, shifts=16, dims=1)
                    sbs_rgb = torch.cat([t_in, t_right], dim=1)
                    alpha = torch.full((out_h, out_w, 1), 255, dtype=torch.uint8, device="cuda:0")
                    t_abgr = torch.cat([sbs_rgb, alpha], dim=2)
                    pkts = enc.Encode(t_abgr)
                    for p in pkts:
                        muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])

                pkts = enc.EndEncode()
                for p in pkts:
                    muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])

            stream.synchronize()
            muxer.Finalize()
            del muxer

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)

            # Probe with ffprobe
            streams = ffprobe_streams(out_path)
            v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            self.assertIsNotNone(v_stream)
            self.assertEqual(v_stream.get("codec_name"), "hevc")
            self.assertEqual(int(v_stream.get("width")), 3840)
            self.assertEqual(int(v_stream.get("height")), 1080)
            self.assertEqual(int(v_stream.get("nb_frames")), total_frames)

            # Read back frames with OpenCV
            cap = cv2.VideoCapture(str(out_path))
            read_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                read_count += 1
                self.assertEqual(frame.shape, (1080, 3840, 3))
            cap.release()
            self.assertEqual(read_count, total_frames)

    def test_06_audio_remux_capability(self):
        """Verify that audio track from source can be remuxed with hardware-encoded SBS video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_video = Path(tmp_dir) / "video_only.mp4"
            remuxed = Path(tmp_dir) / "remuxed.mp4"

            # Create quick 3840x1080 HEVC video
            enc = nvc.CreateEncoder(3840, 1080, "ABGR", False, codec="hevc", fps="12")
            extradata = enc.GetSequenceParams()
            muxer = nvc.FFmpegMuxer(str(raw_video), nvc.MEDIA_FORMAT.MP4, "hevc", 3840, 1080, 12, 1, 1, 90000, extradata)
            muxer.SetUniformPtsIncrement(90000 // 12)

            t = torch.zeros((1080, 3840, 4), dtype=torch.uint8, device="cuda:0")
            for _ in range(4):
                for p in enc.Encode(t):
                    muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
            for p in enc.EndEncode():
                muxer.MuxVideoPacket(p["data"], p["picture_type"], p["timestamp"])
            muxer.Finalize()
            del muxer

            # Remux audio from SAMPLE_VIDEO
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(raw_video),
                "-i", str(SAMPLE_VIDEO),
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                str(remuxed),
            ]
            subprocess.run(cmd, check=True)

            streams = ffprobe_streams(remuxed)
            types = [s.get("codec_type") for s in streams]
            self.assertIn("video", types)
            self.assertIn("audio", types)


if __name__ == "__main__":
    unittest.main()
