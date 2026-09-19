# PureGPU3D

Convert ordinary 2D videos into **Full Side-by-Side stereoscopic video** for VR headsets and compatible 3D players. Depth Anything 3 estimates scene depth; separate left-eye and right-eye views provide a natural sense of depth.

## Download and run

Download the Windows portable archive from [Releases](https://github.com/UgurInanc12/PureGPU3D/releases/latest), extract the **entire folder**, and launch `PureGPU3D.exe`. Keep the accompanying files together.

No separate Python, Conda, FFmpeg or CUDA Toolkit installation is needed. A compatible NVIDIA driver is required for GPU acceleration. Internet access is needed when downloading a model for the first time.

The archive exceeds GitHub's per-file upload limit. Download both `.zip.001` and `.zip.002` assets into the same folder. Open `.001` with 7-Zip, or join them in Windows Command Prompt and extract the resulting ZIP:

```bat
copy /b PureGPU3D-v1.0.1-windows-x64.zip.001+PureGPU3D-v1.0.1-windows-x64.zip.002 PureGPU3D-v1.0.1-windows-x64.zip
```

1. Select a source video and output location.
2. Choose a model and depth processing scale.
3. Leave the pipeline on **Auto**, or explicitly select **GPU** or **Compatible**.
4. Start conversion. Open the exported video in your headset player using **SBS / left-right** mode.

This produces stereoscopic flat-screen video, not a 180-degree or 360-degree scene.

## Features

- **Full SBS:** 1920x1080 input becomes 3840x1080 output, preserving each eye's source resolution.
- **Model selection:** DA3 Small, Base, Mono Large and Metric Large inference support. Missing weights download into the app's `models` folder. Model-specific license acknowledgments apply.
- **Independent depth scale:** 1/4, 1/2 or 1/1. Padding accommodates model patch dimensions without stretching the image. Export resolution stays unchanged.
- **GPU pipeline:** NVDEC decoding, CUDA depth estimation and stereo processing, GPU temporal stabilization, and NVENC HEVC encoding.
- **Compatible pipeline:** FFmpeg decoding with PyTorch processing and hardware encoding when available. Auto reports its selected route; explicitly requesting GPU does not silently fall back.
- **Safe export:** source/output collision protection, temporary staging, exact output frame validation, audio stream copying and cancellation cleanup.
- **Desktop interface:** dark theme, scrollable settings, visible progress and restart-free retry after errors.

## Requirements and limits

Windows x64 is the verified release platform. GPU testing used an RTX 3090; speed and memory needs depend on model, scale, resolution and hardware.

The current desktop path targets **8-bit SDR, constant-frame-rate video**, even dimensions and unrotated square pixels. HDR and VFR inputs are rejected. Audio copy support includes AAC, MP3, AC-3 and E-AC-3. MP4 is the verified export container.

Depth is estimated, not recovered ground truth. Occluded backgrounds, fine edges and fast motion can produce artifacts. Higher depth resolution is not a guarantee of better results. Start with Small, 1/2 scale and the default depth strength; reduce strength if viewing feels uncomfortable. Headset comfort and feature-length reliability are not universally certified.

## v1.0.1

Adds the DA3 desktop workflow, selectable depth scales and GPU video processing. Fixes missing packaged GPU codec modules, cramped controls, retry after failure, and duplicate frames caused by FFmpeg synchronization. Audio/video start offsets are preserved.

A verified 755-frame 1080p clip produced 755-frame, 3840x1080, 25 FPS output with audio through both frozen GPU and Compatible routes. This is a tested example, not a throughput guarantee.

## Development

- `src/puregpu3d/desktop`: PySide6 interface and controller
- `src/puregpu3d/runtime`: worker process and protocol
- `src/puregpu3d/models`: model catalog, downloads and DA3 adapters
- `src/puregpu3d/video` and `stereo`: conversion and stereo processing
- `packaging/PureGPU3D.spec`: portable application packaging
- `scripts/build_desktop.py`: Windows build entry point
- `tests`: model, desktop, media, GPU and packaging checks

The older Gradio/CLI launchers remain in the repository but are not the packaged desktop workflow. Building from source requires the development dependencies and the pinned DA3 source checkout; see [development setup](docs/DEVELOPMENT.md).

## Third-party components

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), PyTorch, PySide6, FFmpeg and NVIDIA PyNvVideoCodec retain their respective licenses. Model weights are downloaded separately and are not included in the portable archive. Bundled FFmpeg build provenance is in `bin/PROVENANCE.txt`. Review third-party and model terms before redistribution or commercial use.
