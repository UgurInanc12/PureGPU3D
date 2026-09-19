# Development setup

The release archive is the recommended end-user installation. Source development uses Python 3.11 on Windows x64.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-probe-lock.txt
.venv\Scripts\python -m pip install -r requirements-nvcodec-probe.txt PySide6==6.7.3 pytest
.venv\Scripts\python -m pip install -e .
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 third_party/depth_anything_3
git -C third_party/depth_anything_3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
git -C third_party/depth_anything_3 apply ../../packaging/da3-device-cache.patch
.venv\Scripts\python scripts/run_desktop.py
```

The vendor patch includes the device in the positional cache key, preventing stale tensors across devices. Upstream source and model weights retain their original licenses.

For packaging, install FFmpeg and ffprobe on the build machine and expose them through PATH. The builder also recognizes its original Windows build location. Run `scripts/build_desktop.py` with the virtual environment's Python. It produces `dist/PureGPU3D`.

**Build safety:** the builder replaces its destination. Preserve downloaded models and user data before building; never rebuild over an active application. End users do not need build tools.

Tests use unittest or pytest. Some integration suites require CUDA, verified local models or generated media fixtures and are not a hardware-independent CI gate. `scripts/video_probe.py` generates synthetic conversion evidence. Packaging acceptance must include `tests/desktop_packaging/test_gpu_frozen.py`, which explicitly requests GPU rather than accepting Auto fallback.
