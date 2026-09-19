"""Application entry point for PureGPU3D desktop user interface."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QPalette, QColor
from PySide6.QtWidgets import QApplication

from puregpu3d.desktop.window import MainWindow


def configure_application_fonts() -> str:
    """Ensure legible font discovery under both native and offscreen QPA platforms.

    Diagnosis:
      - Native Windows QPA (qwindows) links with DirectWrite / GDI and enumerates
        140+ OS font families (defaulting to Segoe UI 9pt).
      - Offscreen QPA (qoffscreen) does not enumerate OS font directories by default,
        leaving QFontDatabase.families() empty (0 families) and rendering characters
        as hollow boxes / tofu symbols unless fallback fonts are explicitly registered.

    This function:
      1. Inspects application resources/fonts/ for any bundled fonts.
      2. If QFontDatabase lacks font families or Segoe UI, enumerates system font
         candidates from %WINDIR%\\Fonts (segoeui.ttf, arial.ttf, tahoma.ttf).
      3. Selects the best available font family and sets the default application font.
    """
    loaded_family: Optional[str] = None

    # 1. Check for bundled fonts in resources/fonts (source tree, PyInstaller bundle root, cwd)
    base_dir = Path(__file__).resolve().parent
    candidate_font_dirs = [
        base_dir / "resources" / "fonts",
        base_dir.parent / "resources" / "fonts",
        Path(sys.executable).parent / "resources" / "fonts",
        Path.cwd() / "resources" / "fonts",
    ]
    for fdir in candidate_font_dirs:
        if fdir.is_dir():
            for ttf in sorted(fdir.glob("*.ttf")):
                font_id = QFontDatabase.addApplicationFont(str(ttf))
                if font_id >= 0:
                    fams = QFontDatabase.applicationFontFamilies(font_id)
                    if fams and not loaded_family:
                        loaded_family = fams[0]

    # 2. If no font loaded or QFontDatabase is empty (e.g. offscreen mode on Windows),
    # register standard Windows system fonts
    existing_families = QFontDatabase.families()
    if not loaded_family and (len(existing_families) == 0 or "Segoe UI" not in existing_families):
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        system_fonts_dir = windir / "Fonts"
        if system_fonts_dir.is_dir():
            font_candidates = [
                "segoeui.ttf",
                "segoeuib.ttf",
                "segoeuii.ttf",
                "segoeuiz.ttf",
                "arial.ttf",
                "arialbd.ttf",
                "tahoma.ttf",
            ]
            for font_name in font_candidates:
                font_path = system_fonts_dir / font_name
                if font_path.is_file():
                    font_id = QFontDatabase.addApplicationFont(str(font_path))
                    if font_id >= 0 and not loaded_family:
                        fams = QFontDatabase.applicationFontFamilies(font_id)
                        if fams:
                            loaded_family = fams[0]

    # 3. Select best available family
    available_families = QFontDatabase.families()
    target_family: Optional[str] = "Segoe UI" if "Segoe UI" in available_families else None
    if not target_family:
        for pref in ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans", "Helvetica", "Sans Serif"]:
            if pref in available_families:
                target_family = pref
                break
    if not target_family and available_families:
        target_family = available_families[0]

    if target_family:
        font = QFont(target_family, 9)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        QApplication.setFont(font)

    return target_family or "Sans Serif"


def create_application(argv: Optional[List[str]] = None) -> QApplication:
    """Create and configure QApplication instance."""
    app_args = sys.argv if argv is None else argv
    app = QApplication(app_args)
    app.setApplicationName("PureGPU3D")
    app.setOrganizationName("PureGPU3D")
    # Explicit palette prevents native dark-mode colors mixing with light panels.
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#171d26",
        QPalette.ColorRole.WindowText: "#eef3fa",
        QPalette.ColorRole.Base: "#222c3a",
        QPalette.ColorRole.AlternateBase: "#283548",
        QPalette.ColorRole.Text: "#eef3fa",
        QPalette.ColorRole.Button: "#33445b",
        QPalette.ColorRole.ButtonText: "#ffffff",
        QPalette.ColorRole.Highlight: "#1764b4",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: "#b0bfd2",
    }.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#64748b"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QPushButton { border: 1px solid #8193a8; border-radius: 4px;
                      padding: 7px 12px; font-weight: 600; }
        QPushButton:hover { border: 2px solid #1764b4; }
        QPushButton:focus { border: 2px solid #1764b4; }
        QPushButton:disabled { background: #242c38; color: #95a3b6; border: 1px solid #48566a; }
        QLineEdit, QComboBox, QDoubleSpinBox { padding: 5px; }
        QGroupBox { font-weight: 600; }
    """)
    configure_application_fonts()
    return app


def run_desktop(argv: Optional[List[str]] = None) -> int:
    """Run PureGPU3D desktop application."""
    effective_argv = sys.argv[1:] if argv is None else argv
    if "--worker" in effective_argv:
        from puregpu3d.runtime.worker import main as worker_main
        return worker_main()

    parser = argparse.ArgumentParser(description="PureGPU3D Stereoscopic Video Converter")
    parser.add_argument("--input", type=str, help="Initial input video path.")
    parser.add_argument("--output", type=str, help="Initial output Full-SBS video path.")
    parser.add_argument("--model", type=str, default="DA3-SMALL", help="Initial model selection.")
    parser.add_argument("--strength", type=float, default=0.03, help="Initial depth strength.")
    parser.add_argument("--offscreen", action="store_true", help="Run Qt using offscreen platform.")
    parser.add_argument("--screenshot", type=str, help="Save offscreen window screenshot to specified image path and exit.")

    args, unknown = parser.parse_known_args(argv)

    if args.screenshot or args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication.instance()
    if app is None:
        app = create_application(unknown)
    else:
        configure_application_fonts()

    window = MainWindow()

    if args.model:
        window.controller.set_selected_model(args.model)
    if args.strength is not None:
        window.controller.set_disparity_strength(args.strength)
        window.strength_spin.setValue(args.strength)
    if args.input:
        window.input_edit.setText(str(Path(args.input).resolve()))
    if args.output:
        window.output_edit.setText(str(Path(args.output).resolve()))

    if args.screenshot:
        window.resize(1024, 768)
        window.show()
        app.processEvents()
        window.updateGeometry()
        app.processEvents()
        screenshot_path = Path(args.screenshot).resolve()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = window.grab()
        saved = pixmap.save(str(screenshot_path))
        if not saved:
            sys.stderr.write(f"Failed to save screenshot to {screenshot_path}\n")
            return 1
        return 0

    if not args.offscreen:
        window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_desktop())
