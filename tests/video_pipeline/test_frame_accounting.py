"""Regression for FFmpeg duplicating delayed video frames before RGB export."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.video_pipeline.test_convert_mock import MockDepthAdapter
from puregpu3d.video.convert import convert_video
from puregpu3d.video.probe import find_ffmpeg, probe_video


class TestFrameAccounting(unittest.TestCase):
    def test_delayed_video_preserves_decoded_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'delayed.mp4'
            dst = Path(tmp) / 'stereo.mp4'
            subprocess.run([
                str(find_ffmpeg()), '-v', 'error', '-y',
                '-f', 'lavfi', '-i', 'testsrc2=size=160x90:rate=24:duration=1',
                '-f', 'lavfi', '-i', 'sine=duration=1.083333',
                '-vf', 'settb=1/90000,setpts=PTS+5231', '-fps_mode', 'passthrough',
                '-enc_time_base', '1:90000',
                '-c:v', 'libx264', '-c:a', 'aac', str(src),
            ], check=True, capture_output=True)
            self.assertEqual(probe_video(src).frame_count, 24)
            result = convert_video(src, dst, model=MockDepthAdapter(), device='cpu',
                                   enable_temporal_stabilization=False)
            self.assertEqual(result.total_frames_processed, 24)
            output = probe_video(dst)
            self.assertEqual(output.frame_count, 24)
            self.assertTrue(output.has_audio)
            def av_offset(probe):
                streams = probe.raw_info['streams']
                video = next(s for s in streams if s['codec_type'] == 'video')
                audio = next(s for s in streams if s['codec_type'] == 'audio')
                return float(video['start_time']) - float(audio['start_time'])
            self.assertAlmostEqual(av_offset(output), av_offset(probe_video(src)), delta=0.002)
