"""Native PySide6 Main Window for PureGPU3D Desktop Application."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent, QFont, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from puregpu3d.desktop.controller import DesktopController, DesktopState
from puregpu3d.models.catalog import ModelCatalogEntry
from puregpu3d.runtime.protocol import Stage
from puregpu3d.video.probe import VideoProbeResult


class MainWindow(QMainWindow):
    """Main application window for PureGPU3D stereoscopic video conversion."""

    def __init__(self, controller: Optional[DesktopController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or DesktopController(self)

        self.setWindowTitle("PureGPU3D - Stereoscopic 3D Video Converter")
        # Screen-aware startup size
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(880, max(750, int(avail.width() * 0.75)))
            h = min(820, max(650, int(avail.height() * 0.85)))
            w = min(w, avail.width())
            h = min(h, avail.height())
            self.resize(w, h)
        else:
            self.resize(850, 780)
        self.setMinimumSize(700, 520)
        self.setAcceptDrops(True)

        self._init_ui()
        self._connect_signals()
        self._populate_models()
        self._update_ui_state(self.controller.state.value)

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # -------------------------------------------------------------------
        # 1. Header (persistent top)
        # -------------------------------------------------------------------
        header_layout = QHBoxLayout()
        title_label = QLabel("PureGPU3D")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        subtitle_label = QLabel("2D to Full-SBS Stereoscopic Video Converter (Depth Anything 3)")
        subtitle_label.setStyleSheet("color: #b9c8da;")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # -------------------------------------------------------------------
        # Settings Scroll Area (scrollable middle)
        # -------------------------------------------------------------------
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background: transparent; border: none; } "
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(0, 0, 6, 0)

        # -------------------------------------------------------------------
        # 2. Input File Group
        # -------------------------------------------------------------------
        input_group = QGroupBox("Input Video")
        input_layout = QVBoxLayout(input_group)

        input_path_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setMinimumHeight(28)
        self.input_edit.setPlaceholderText("Select source video file or drag and drop here (.mp4, .mkv, .mov)...")
        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.setMinimumHeight(28)
        self.input_browse_btn.clicked.connect(self._on_browse_input)
        input_path_layout.addWidget(self.input_edit)
        input_path_layout.addWidget(self.input_browse_btn)
        input_layout.addLayout(input_path_layout)

        # Media info summary
        self.input_info_label = QLabel("No media file selected.")
        self.input_info_label.setStyleSheet(
            "color: #e4edf8; padding: 6px 8px; background: #263346; border: 1px solid #526780; border-radius: 4px;"
        )
        self.input_info_label.setWordWrap(True)
        self.input_info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        input_layout.addWidget(self.input_info_label)

        # Warning banner for unsupported media (HDR / VFR / Odd geometry)
        self.media_warning_label = QLabel()
        self.media_warning_label.setStyleSheet(
            "color: #b02a37; background: #f8d7da; border: 1px solid #f5c2c7; padding: 6px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.media_warning_label.setWordWrap(True)
        self.media_warning_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.media_warning_label.setVisible(False)
        input_layout.addWidget(self.media_warning_label)

        scroll_layout.addWidget(input_group)

        # -------------------------------------------------------------------
        # 3. Output File Group
        # -------------------------------------------------------------------
        output_group = QGroupBox("Output Video (Full-SBS)")
        output_layout = QVBoxLayout(output_group)

        output_path_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setMinimumHeight(28)
        self.output_edit.setPlaceholderText("Destination Full-SBS video path (*.mp4)...")
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.setMinimumHeight(28)
        self.output_browse_btn.clicked.connect(self._on_browse_output)
        output_path_layout.addWidget(self.output_edit)
        output_path_layout.addWidget(self.output_browse_btn)
        output_layout.addLayout(output_path_layout)

        self.output_geometry_label = QLabel("Output Geometry: Full SBS (2W x H)")
        self.output_geometry_label.setStyleSheet("color: #83c4ff; font-weight: bold;")
        output_layout.addWidget(self.output_geometry_label)

        scroll_layout.addWidget(output_group)

        # -------------------------------------------------------------------
        # 4. Model Selection Group
        # -------------------------------------------------------------------
        model_group = QGroupBox("Depth Anything 3 Model Selection")
        model_layout = QVBoxLayout(model_group)

        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model Checkpoint:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(28)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_select_layout.addWidget(self.model_combo, stretch=1)
        model_layout.addLayout(model_select_layout)

        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet(
            "color: #e4edf8; padding: 6px 8px; background: #263346; border: 1px solid #526780; border-radius: 4px;"
        )
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        model_layout.addWidget(self.model_info_label)

        # Model status and integration notice
        self.model_notice_label = QLabel()
        self.model_notice_label.setStyleSheet("padding: 6px 8px; border-radius: 4px;")
        self.model_notice_label.setWordWrap(True)
        self.model_notice_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        model_layout.addWidget(self.model_notice_label)

        # License acknowledgment checkbox
        self.license_ack_checkbox = QCheckBox("I acknowledge and accept the non-commercial / license terms for this model.")
        self.license_ack_checkbox.setVisible(False)
        self.license_ack_checkbox.toggled.connect(self._on_license_ack_toggled)
        model_layout.addWidget(self.license_ack_checkbox)

        # Depth Processing Scale Selection
        scale_select_layout = QHBoxLayout()
        scale_select_layout.addWidget(QLabel("Depth Processing Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.setMinimumHeight(28)
        self.scale_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.scale_combo.addItem("1/4 (Fast / Low VRAM)", "1/4")
        self.scale_combo.addItem("1/2 (Balanced / Default)", "1/2")
        self.scale_combo.addItem("1/1 (Full Resolution)", "1/1")
        default_scale_idx = self.scale_combo.findData(self.controller.depth_scale)
        if default_scale_idx >= 0:
            self.scale_combo.setCurrentIndex(default_scale_idx)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_combo_changed)
        scale_select_layout.addWidget(self.scale_combo, stretch=1)
        model_layout.addLayout(scale_select_layout)

        self.scale_geometry_label = QLabel(
            f"Depth Processing Geometry: Scale {self.controller.depth_scale} selected. "
            "Select input video to preview requested and padded model tensor dimensions."
        )
        self.scale_geometry_label.setStyleSheet(
            "color: #83c4ff; padding: 6px 8px; background: #263346; border: 1px solid #526780; border-radius: 4px;"
        )
        self.scale_geometry_label.setWordWrap(True)
        self.scale_geometry_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        model_layout.addWidget(self.scale_geometry_label)

        # Pipeline Execution Route Selection
        route_select_layout = QHBoxLayout()
        route_select_layout.addWidget(QLabel("Execution Pipeline Route:"))
        self.route_combo = QComboBox()
        self.route_combo.setMinimumHeight(28)
        self.route_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.route_combo.addItem("Auto (NVIDIA GPU if available, else Compatible)", "auto")
        self.route_combo.addItem("GPU Pipeline (NVDEC -> CUDA -> NVENC)", "gpu")
        self.route_combo.addItem("Compatible Pipeline (FFmpeg Decode/Encode)", "compatible")
        default_route_idx = self.route_combo.findData(self.controller.pipeline_route)
        if default_route_idx >= 0:
            self.route_combo.setCurrentIndex(default_route_idx)
        self.route_combo.currentIndexChanged.connect(self._on_route_combo_changed)
        route_select_layout.addWidget(self.route_combo, stretch=1)
        model_layout.addLayout(route_select_layout)

        self.backend_status_label = QLabel(
            "Resolved Backend: Auto (preflights GPU, falls back to Compatible if unsupported)"
        )
        self.backend_status_label.setStyleSheet(
            "color: #83c4ff; padding: 6px 8px; background: #263346; border: 1px solid #526780; border-radius: 4px;"
        )
        self.backend_status_label.setWordWrap(True)
        self.backend_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        model_layout.addWidget(self.backend_status_label)

        # Batch Size Selection (GPU route only)
        batch_select_layout = QHBoxLayout()
        batch_select_layout.addWidget(QLabel("Depth Batch Size (Frames):"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimumHeight(28)
        self.batch_spin.setRange(1, 20)
        self.batch_spin.setSingleStep(1)
        self.batch_spin.setValue(self.controller.batch_size)
        self.batch_spin.valueChanged.connect(self._on_batch_spin_changed)
        batch_select_layout.addWidget(self.batch_spin, stretch=1)
        model_layout.addLayout(batch_select_layout)

        self.batch_hint_label = QLabel(
            "Default: 1 (sequential). Higher values (up to 20) process independent frames concurrently on GPU, "
            "increasing throughput but substantially increasing GPU VRAM usage. "
            "Batching applies only to the GPU pipeline route (not supported in Compatible mode)."
        )
        self.batch_hint_label.setStyleSheet("color: #b9c8da; font-size: 11px;")
        self.batch_hint_label.setWordWrap(True)
        self.batch_hint_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        model_layout.addWidget(self.batch_hint_label)

        scroll_layout.addWidget(model_group)

        # -------------------------------------------------------------------
        # 5. Stereoscopic Settings Group
        # -------------------------------------------------------------------
        stereo_group = QGroupBox("Stereoscopic Depth Tuning")
        stereo_layout = QGridLayout(stereo_group)

        stereo_layout.addWidget(QLabel("Natural Depth Strength:"), 0, 0)
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setMinimumHeight(28)
        self.strength_spin.setRange(0.000, 0.010)
        self.strength_spin.setSingleStep(0.001)
        self.strength_spin.setDecimals(3)
        self.strength_spin.setValue(self.controller.disparity_strength)
        self.strength_spin.valueChanged.connect(self._on_strength_spin_changed)
        stereo_layout.addWidget(self.strength_spin, 0, 1)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setMinimumHeight(24)
        self.strength_slider.setRange(0, 10)
        self.strength_slider.setValue(int(self.controller.disparity_strength * 1000))
        self.strength_slider.valueChanged.connect(self._on_strength_slider_changed)
        stereo_layout.addWidget(self.strength_slider, 0, 2)

        depth_hint = QLabel("Default: 0.001. Maximum: 0.010. Reduce strength if the stereo effect feels uncomfortable.")
        depth_hint.setStyleSheet("color: #b9c8da; font-size: 11px;")
        stereo_layout.addWidget(depth_hint, 1, 0, 1, 3)

        scroll_layout.addWidget(stereo_group)
        scroll_layout.addStretch()

        self.scroll_area.setWidget(scroll_content)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # -------------------------------------------------------------------
        # 6. Progress and Controls Group
        # -------------------------------------------------------------------
        progress_group = QGroupBox("Conversion Status")
        progress_layout = QVBoxLayout(progress_group)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.details_label = QLabel("Idle")
        self.details_label.setStyleSheet("color: #b9c8da;")
        progress_layout.addWidget(self.details_label)

        btn_layout = QHBoxLayout()
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.setStyleSheet("background-color: #198754; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px;")
        self.convert_btn.clicked.connect(self._on_start_convert)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("QPushButton:enabled { background-color: #b42332; color: white; padding: 8px 16px; border-radius: 4px; } QPushButton:disabled { background-color: #e2e7ed; color: #64748b; border: 1px solid #bdc8d5; padding: 8px 16px; }")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_convert)

        btn_layout.addWidget(self.convert_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()

        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_folder)

        self.open_video_btn = QPushButton("Play Video")
        self.open_video_btn.setEnabled(False)
        self.open_video_btn.clicked.connect(self._on_open_video)

        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addWidget(self.open_video_btn)

        progress_layout.addLayout(btn_layout)
        main_layout.addWidget(progress_group)

        main_layout.addStretch()

    def _connect_signals(self) -> None:
        self.controller.depth_scale_changed.connect(self._on_depth_scale_changed)
        self.controller.pipeline_route_changed.connect(self._on_pipeline_route_changed)
        self.controller.batch_size_changed.connect(self._on_batch_size_changed)
        self.controller.state_changed.connect(self._update_ui_state)
        self.controller.input_probed.connect(self._on_input_probed)
        self.controller.model_changed.connect(self._on_model_changed)
        self.controller.status_updated.connect(self._on_status_updated)
        self.controller.download_progress.connect(self._on_download_progress)
        self.controller.conversion_progress.connect(self._on_conversion_progress)
        self.controller.conversion_completed.connect(self._on_conversion_completed)
        self.controller.conversion_failed.connect(self._on_conversion_failed)
        self.controller.conversion_cancelled.connect(self._on_conversion_cancelled)
        self.controller.log_received.connect(self._on_log_received)

        self.input_edit.textChanged.connect(self._on_input_text_changed)
        self.output_edit.textChanged.connect(self._on_output_text_changed)

    def _on_route_combo_changed(self, index: int) -> None:
        route = self.route_combo.itemData(index)
        if route:
            self.controller.set_pipeline_route(route)

    def _on_pipeline_route_changed(self, route: str) -> None:
        index = self.route_combo.findData(route)
        if index >= 0:
            self.route_combo.setCurrentIndex(index)

    def _on_scale_combo_changed(self, index: int) -> None:
        self.controller.set_depth_scale(self.scale_combo.itemData(index))

    def _on_depth_scale_changed(self, scale: str) -> None:
        index = self.scale_combo.findData(scale)
        self.scale_combo.setCurrentIndex(index)
        self._update_depth_geometry_label()

    def _update_depth_geometry_label(self) -> None:
        from puregpu3d.models.geometry import compute_depth_geometry
        probe = self.controller.input_probe
        if probe is None:
            self.scale_geometry_label.setText(f"Depth scale: {self.controller.depth_scale}. Select input to see dimensions.")
            return
        geometry = compute_depth_geometry(probe.width, probe.height, self.controller.depth_scale)
        self.scale_geometry_label.setText(
            f"Depth content: {geometry.req_width}x{geometry.req_height} | "
            f"Model tensor (padded): {geometry.padded_width}x{geometry.padded_height} | "
            f"Full SBS export: {probe.width * 2}x{probe.height}"
        )

    def _populate_models(self) -> None:
        """Populate model dropdown with all 7 official catalog choices."""
        self.model_combo.clear()
        entries = self.controller.get_catalog_entries()

        # Place general size tiers first, specialists next
        for entry in entries:
            info = self.controller.get_model_info(entry.id)
            integration_tag = "[Supported]" if info["is_supported"] else "[Not Yet Integrated]"
            display_text = f"{entry.ui_name} ({entry.parameters}) - {entry.repo_id} {integration_tag}"
            self.model_combo.addItem(display_text, entry.id)

        # Select default Small model
        default_index = self.model_combo.findData("DA3-SMALL")
        if default_index >= 0:
            self.model_combo.setCurrentIndex(default_index)

    def _on_model_combo_changed(self, index: int) -> None:
        model_id = self.model_combo.itemData(index)
        if model_id:
            self.controller.set_selected_model(model_id)

    def _on_model_changed(self, model_id: str, info: Dict[str, Any]) -> None:
        size_mb = info["weight_bytes"] / (1024 * 1024)
        lic_txt = f"{info['license']} ({info['license_type']})"
        if info["license_conflict"]:
            lic_txt += f" [CONFLICT: {info['conflict_details']}]"

        self.model_info_label.setText(
            f"Parameters: {info['parameters']} | On-Disk Status: {info['status'].upper()} | "
            f"Weight Size: {size_mb:.1f} MB | License: {lic_txt}"
        )

        if info["is_supported"]:
            if info["status"] == "ready":
                weights_notice = "weights present and verified on disk"
            elif info["status"] == "missing":
                weights_notice = "weights missing on disk (will download on first conversion)"
            else:
                weights_notice = f"weights status: {info['status']} on disk"
            self.model_notice_label.setText(
                f"Inference Adapter: Depth Anything 3 {info['ui_name']} supported ({weights_notice})."
            )
            self.model_notice_label.setStyleSheet(
                "color: #0f5132; background: #d1e7dd; border: 1px solid #badbcc; padding: 6px; border-radius: 4px;"
            )
        else:
            self.model_notice_label.setText(
                f"Not yet integrated: {info['ui_name']} model is not supported for conversion in this version. "
                "Conversion cannot be started with this model."
            )
            self.model_notice_label.setStyleSheet(
                "color: #842029; background: #f8d7da; border: 1px solid #f5c2c7; padding: 6px; border-radius: 4px; font-weight: bold;"
            )

        # License acknowledgment checkbox
        if info["ack_needed"] and not info["ack_recorded"]:
            self.license_ack_checkbox.setVisible(True)
            self.license_ack_checkbox.setChecked(False)
        else:
            self.license_ack_checkbox.setVisible(False)

        self._validate_inputs_and_update_buttons()

    def _on_license_ack_toggled(self, checked: bool) -> None:
        if checked:
            model_id = self.controller.selected_model_id
            self.controller.record_license_acknowledgment(model_id)

    def _on_strength_spin_changed(self, val: float) -> None:
        self.strength_slider.blockSignals(True)
        self.strength_slider.setValue(int(val * 1000))
        self.strength_slider.blockSignals(False)
        self.controller.set_disparity_strength(val)

    def _on_strength_slider_changed(self, int_val: int) -> None:
        val = int_val / 1000.0
        self.strength_spin.blockSignals(True)
        self.strength_spin.setValue(val)
        self.strength_spin.blockSignals(False)
        self.controller.set_disparity_strength(val)

    def _on_batch_spin_changed(self, val: int) -> None:
        self.controller.set_batch_size(val)

    def _on_batch_size_changed(self, val: int) -> None:
        if self.batch_spin.value() != val:
            self.batch_spin.blockSignals(True)
            self.batch_spin.setValue(val)
            self.batch_spin.blockSignals(False)

    def _on_browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*)",
        )
        if path:
            self.input_edit.setText(path)

    def _on_browse_output(self) -> None:
        current = self.output_edit.text()
        default_dir = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Full-SBS Video",
            default_dir,
            "MP4 Video (*.mp4);;MKV Video (*.mkv);;All Files (*)",
        )
        if path:
            self.output_edit.setText(path)

    def _on_input_text_changed(self, text: str) -> None:
        text = text.strip()
        if text:
            success, msg = self.controller.set_input_path(text)
            if not success:
                self.input_info_label.setText(f"Error: {msg}")
                self.media_warning_label.setVisible(False)
        else:
            self.input_info_label.setText("No media file selected.")
            self.media_warning_label.setVisible(False)
        self._validate_inputs_and_update_buttons()

    def _on_output_text_changed(self, text: str) -> None:
        text = text.strip()
        if text:
            self.controller.set_output_path(text)
        self._validate_inputs_and_update_buttons()

    def _on_input_probed(self, probe: Optional[VideoProbeResult]) -> None:
        self._update_depth_geometry_label()
        if probe is None:
            self.output_geometry_label.setText("Output Geometry: Full SBS (2W x H)")
            self.media_warning_label.setVisible(False)
            return

        out_w = probe.width * 2
        out_h = probe.height
        self.output_geometry_label.setText(
            f"Output Geometry: Full SBS {out_w}x{out_h} (Left eye: {probe.width}x{probe.height}, Right eye: {probe.width}x{probe.height})"
        )

        audio_str = "No audio"
        if probe.has_audio:
            codecs = ", ".join(a.codec_name for a in probe.audio_streams)
            audio_str = f"{len(probe.audio_streams)} track(s) [{codecs}]"

        hdr_str = "HDR (smpte2084/HLG) - UNSUPPORTED" if probe.is_hdr else "SDR BT.709 (Supported)"
        vfr_str = "VFR - UNSUPPORTED" if probe.is_vfr else f"CFR {probe.fps:.2f} fps"

        info_text = (
            f"Resolution: {probe.width}x{probe.height} | Frame Rate: {vfr_str} | "
            f"Duration: {probe.duration:.2f}s ({probe.frame_count} frames) | "
            f"Color: {hdr_str} | Audio: {audio_str}"
        )
        self.input_info_label.setText(info_text)

        # Update default output field if empty
        if self.controller.output_path and not self.output_edit.text():
            self.output_edit.setText(str(self.controller.output_path))

        # Check for refusal conditions and show prominent warning
        warnings = []
        if probe.is_hdr:
            warnings.append("High Dynamic Range (HDR) input is not supported. Please supply SDR BT.709 video.")
        if probe.is_vfr:
            warnings.append("Variable Frame Rate (VFR) input is not supported. Constant Frame Rate (CFR) is required.")
        if probe.rotation != 0:
            warnings.append(f"Non-zero rotation metadata ({probe.rotation}°) is not supported.")
        if probe.width % 2 != 0 or probe.height % 2 != 0:
            warnings.append("Odd dimensions cannot be converted to standard Full-SBS 4:2:0 format.")

        if warnings:
            self.media_warning_label.setText("\n".join(warnings))
            self.media_warning_label.setVisible(True)
        else:
            self.media_warning_label.setVisible(False)

        self._validate_inputs_and_update_buttons()

    def _validate_inputs_and_update_buttons(self) -> None:
        """Enable or disable conversion button based on strict readiness rules."""
        ready, reason = self.controller.validate_for_conversion()
        is_idle_or_terminal = self.controller.state in (
            DesktopState.IDLE,
            DesktopState.COMPLETED,
            DesktopState.FAILED,
            DesktopState.CANCELLED,
        )
        if is_idle_or_terminal:
            self.convert_btn.setEnabled(ready)
            if not ready:
                self.convert_btn.setToolTip(reason)
            else:
                self.convert_btn.setToolTip("Click to start stereoscopic conversion.")

    def _on_start_convert(self) -> None:
        """Handle convert action with explicit overwrite guard."""
        out_path = self.controller.output_path
        overwrite_confirmed = False

        if out_path and out_path.exists():
            res = QMessageBox.question(
                self,
                "Confirm Overwrite",
                f"The destination output file already exists:\n\n{out_path}\n\n"
                "Do you want to overwrite this file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if res != QMessageBox.StandardButton.Yes:
                return
            overwrite_confirmed = True

        self.open_folder_btn.setEnabled(False)
        self.open_video_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        started, err = self.controller.start_conversion(overwrite_confirmed=overwrite_confirmed)
        if not started:
            QMessageBox.warning(self, "Cannot Start Conversion", err)

    def _on_cancel_convert(self) -> None:
        self.controller.cancel_conversion()

    def _update_ui_state(self, state_str: str) -> None:
        state = DesktopState(state_str)
        is_running = state in (
            DesktopState.VALIDATING,
            DesktopState.PREPARING_MODEL,
            DesktopState.CONVERTING,
            DesktopState.CANCELLING,
        )

        self.input_browse_btn.setEnabled(not is_running)
        self.output_browse_btn.setEnabled(not is_running)
        self.model_combo.setEnabled(not is_running)
        self.scale_combo.setEnabled(not is_running)
        self.route_combo.setEnabled(not is_running)
        self.strength_spin.setEnabled(not is_running)
        self.strength_slider.setEnabled(not is_running)
        self.batch_spin.setEnabled(not is_running)

        self.convert_btn.setEnabled(not is_running and self.controller.validate_for_conversion()[0])
        self.cancel_btn.setEnabled(is_running and state != DesktopState.CANCELLING)

        if state == DesktopState.IDLE:
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: #eef3fa; font-weight: bold;")
        elif state == DesktopState.CANCELLING:
            self.status_label.setText("Cancelling conversion...")
            self.status_label.setStyleSheet("color: #d63384; font-weight: bold;")
        elif state == DesktopState.COMPLETED:
            self.status_label.setText("Conversion Completed Successfully!")
            self.status_label.setStyleSheet("color: #198754; font-weight: bold;")
            self.open_folder_btn.setEnabled(True)
            self.open_video_btn.setEnabled(True)
        elif state == DesktopState.FAILED:
            self.status_label.setText("Conversion Failed")
            self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        elif state == DesktopState.CANCELLED:
            self.status_label.setText("Conversion Cancelled")
            self.status_label.setStyleSheet("color: #6c757d; font-weight: bold;")

    _on_state_changed = _update_ui_state

    def _on_status_updated(self, stage: str, msg: str) -> None:
        self.status_label.setText(f"[{stage.upper()}] {msg}")
        self.details_label.setText(msg)
        if "Resolved route:" in msg or "falling back upfront" in msg or "GPU pipeline" in msg:
            self.backend_status_label.setText(f"Active Backend: {msg}")

    def _on_download_progress(self, dl_bytes: int, total_bytes: int, percent: float, filename: str) -> None:
        self.progress_bar.setValue(int(percent))
        dl_mb = dl_bytes / (1024 * 1024)
        tot_mb = total_bytes / (1024 * 1024)
        self.details_label.setText(f"Downloading model: {dl_mb:.1f} MB / {tot_mb:.1f} MB ({percent:.1f}%)")

    def _on_conversion_progress(self, frame: int, total: int, percent: float, fps: float, eta: Any) -> None:
        self.progress_bar.setValue(int(percent))
        eta_str = f"{eta:.1f}s" if eta is not None else "--"
        self.details_label.setText(
            f"Frame {frame}/{total} ({percent:.1f}%) | Processing Speed: {fps:.1f} fps | ETA: {eta_str}"
        )

    def _on_conversion_completed(self, result: Dict[str, Any]) -> None:
        self.progress_bar.setValue(100)
        wall_time = result.get("wall_clock_seconds", 0.0)
        fps = result.get("effective_fps", 0.0)
        frames = result.get("total_frames_processed", 0)
        backend = result.get("resolved_backend") or result.get("backend") or "Completed"
        self.backend_status_label.setText(f"Resolved Backend: {backend}")
        self.details_label.setText(
            f"Done: {frames} frames processed in {wall_time:.2f}s (Average: {fps:.1f} fps). Backend: {backend}. Output verified."
        )

    def _on_conversion_failed(self, error: str, stage: str) -> None:
        self.details_label.setText(f"Failed at {stage}: {error}")
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.critical(self, "Conversion Failed", f"Stage: {stage}\n\nError: {error}")

    def _on_conversion_cancelled(self) -> None:
        self.details_label.setText("Operation cancelled by user.")

    def _on_log_received(self, line: str) -> None:
        # Diagnostic worker log line
        pass

    def _on_open_folder(self) -> None:
        """Open destination directory in explorer."""
        out_path = self.controller.output_path
        if out_path and out_path.exists():
            folder = out_path.parent
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(out_path)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_open_video(self) -> None:
        """Open completed video with default player."""
        out_path = self.controller.output_path
        if out_path and out_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_path)))

    # Drag and drop support
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            local_path = urls[0].toLocalFile()
            self.input_edit.setText(local_path)
            event.acceptProposedAction()

    def closeEvent(self, event: Any) -> None:
        """Ensure active conversion is safely cancelled before window closes."""
        if self.controller.state in (
            DesktopState.VALIDATING,
            DesktopState.PREPARING_MODEL,
            DesktopState.CONVERTING,
        ):
            res = QMessageBox.question(
                self,
                "Conversion in Progress",
                "A video conversion is currently in progress.\nDo you want to cancel the job and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if res == QMessageBox.StandardButton.Yes:
                self.controller.cancel_conversion()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
