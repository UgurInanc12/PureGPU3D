"""GPU remux must retain source audio/video start offsets."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from puregpu3d.models.da3_adapter import DA3DepthAdapter
from puregpu3d.video.gpu_convert import convert_video_gpu
from puregpu3d.video.probe import probe_video, find_ffmpeg

ROOT = Path(__file__).resolve().parents[2]


class TestGpuAudioOffset(unittest.TestCase):
    def test_delayed_video_preserves_audio_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'source.mp4', Path(tmp) / 'stereo.mp4'
            subprocess.run([str(find_ffmpeg()), '-v', 'error', '-y',
                '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=24:duration=0.5',
                '-f', 'lavfi', '-i', 'sine=duration=0.583333',
                '-vf', 'settb=1/90000,setpts=PTS+5231', '-fps_mode', 'passthrough',
                '-enc_time_base', '1:90000',
                '-c:v', 'libx264', '-c:a', 'aac', str(src)], check=True, capture_output=True)
            model = DA3DepthAdapter(ROOT / 'models/DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3', device='cuda:0')
            result = convert_video_gpu(src, dst, model=model, depth_scale='1/2', enable_temporal_stabilization=True)
            self.assertEqual(result.total_frames_processed, 12)
            def offset(path):
                streams = probe_video(path).raw_info['streams']
                return (float(next(s for s in streams if s['codec_type'] == 'video')['start_time'])
                        - float(next(s for s in streams if s['codec_type'] == 'audio')['start_time']))
            self.assertAlmostEqual(offset(dst), offset(src), delta=0.002)
