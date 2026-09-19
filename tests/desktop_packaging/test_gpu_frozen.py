"""Explicit frozen GPU acceptance: Auto fallback must not mask missing codec DLLs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get('PUREGPU3D_TEST_BUNDLE', str(ROOT / 'dist/PureGPU3D')))


class TestFrozenGpu(unittest.TestCase):
    def test_explicit_gpu_converts_without_host_python_or_cuda_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job.json'
            out = Path(tmp) / 'stereo.mp4'
            job.write_text(json.dumps(dict(
                job_id='frozen-explicit-gpu',
                input_path=str(ROOT / 'data/verification/video/synthetic_1080p_moving.mp4'),
                output_path=str(out), model_id='DA3-SMALL', device='cuda:0',
                depth_scale='1/4', pipeline_route='gpu', enable_temporal_stabilization=True,
                ffmpeg_path=str(BUNDLE / 'bin/ffmpeg.exe'),
                ffprobe_path=str(BUNDLE / 'bin/ffprobe.exe'),
            )), encoding='utf-8')
            env = {k: v for k, v in os.environ.items()
                   if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV') and not k.startswith('CUDA_PATH')}
            env['PATH'] = r'C:\Windows\System32;C:\Windows'
            proc = subprocess.run([str(BUNDLE / 'PureGPU3D-worker.exe'), '--command-file', str(job)],
                                  cwd=tmp, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            messages = []
            for line in proc.stdout.splitlines():
                try:
                    messages.append(json.loads(line))
                except ValueError:
                    pass
            result = next(m['result'] for m in messages if m.get('type') == 'completed')
            self.assertEqual(result['pipeline_route'], 'gpu')
            self.assertEqual(result['total_frames_processed'], 12)
            self.assertEqual((result['output_width'], result['output_height']), (3840, 1080))
            self.assertTrue(result['has_audio'])
            self.assertGreater(result['mean_temporal_ms'], 0)
            self.assertTrue(out.is_file())
