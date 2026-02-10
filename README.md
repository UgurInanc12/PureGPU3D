# PureGPU3D

PureGPU3D is a Windows-first 2D-to-3D video converter that generates **SBS Full** output (`2W x H`) for VR playback.

It includes:
- FFmpeg-based decode/encode/remux pipeline
- CuPy CUDA depth kernel path (with safe CPU fallback)
- Audio passthrough or AAC re-encode
- Container selection (`mp4`, `mkv`, `mov`)
- Encoder and bitrate selection
- Color metadata passthrough
- Gradio web UI + CLI

## Requirements
- Windows 10/11
- Miniconda or Anaconda installed
- NVIDIA GPU recommended (CUDA path), but CPU fallback is supported

## One-Click Start (Recommended)
1. Download/clone this repository.
2. Double-click `open_project.bat`.

`open_project.bat` will:
- Create the `puregpu3d` Conda environment on first run
- Launch the web UI

On later runs, it skips heavy environment updates and starts faster.

## First-Time Setup Only (Optional)
If you want to prepare the environment manually:
- Double-click `setup_project.bat`

Then launch UI:
- Double-click `start_ui.bat`

## CLI Usage
Run from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_cli.ps1 --help
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_cli.ps1 transcode `
  --input "D:\video\input.mp4" `
  --output "D:\video\output_sbs.mp4" `
  --codec h265 `
  --video-encoder auto `
  --video-bitrate-mbps 35 `
  --audio-codec copy
```

## Common Notes
- If NVENC/NVDEC is not available, FFmpeg software paths are used automatically.
- `PyNvVideoCodec` is optional in this release; FFmpeg backend is the default stable path.
- Cancel in UI performs immediate abort and cleans temporary output.

## Project Layout
- `src/puregpu3d`: main application code
- `configs/profiles`: runtime profiles
- `scripts`: setup and launch scripts
- `environment/environment.yml`: Conda environment definition

## Troubleshooting
- If Conda is not detected, install Miniconda and reopen the terminal/session.
- If GPU codec bindings fail with a DLL import error, the app still works with FFmpeg backend.
- If `h264_nvenc` fails on very wide SBS output (for example 7680 width), switch to `h265/hevc_nvenc` or `libx264`.
