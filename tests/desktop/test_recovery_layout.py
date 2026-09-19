"""Regression tests for Desktop layout compression and post-failure retry lifecycle.

Verifies:
1. Layout uses QScrollArea for settings while keeping progress/actions persistent.
2. Control geometry: model, scale, and route controls do not overlap or compress at default and small window sizes.
3. Long error messages do not overlap or corrupt settings geometry.
4. Screen-aware startup window size logic.
5. Worker lifecycle: failed worker retains terminal error, refuses retry while process is alive,
   and enables restart-free retry once process has fully exited.
6. Successful retry after completion and cancellation.
7. Abnormal clean exit lacking terminal protocol message safely transitions to FAILED and enables retry.
8. Failed worker start cleans up and allows retry.
9. Stale cancellation timers cannot kill subsequent new jobs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QPoint, QProcess, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QMainWindow, QMessageBox, QPushButton, QScrollArea, QWidget

# Ensure headless Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance() or QApplication(["-platform", "offscreen"])

# Auto-dismiss modal dialogs during tests
QMessageBox.critical = lambda *args, **kwargs: QMessageBox.StandardButton.Ok  # type: ignore
QMessageBox.warning = lambda *args, **kwargs: QMessageBox.StandardButton.Ok  # type: ignore
QMessageBox.question = lambda *args, **kwargs: QMessageBox.StandardButton.Yes  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from puregpu3d.desktop.controller import DesktopController, DesktopState
from puregpu3d.desktop.window import MainWindow
from puregpu3d.runtime.protocol import MessageType, Stage

FIXTURE_VIDEO = REPO_ROOT / "data" / "verification" / "video" / "synthetic_1080p_moving.mp4"


def process_events(iterations: int = 10, delay_ms: int = 20) -> None:
    """Flush Qt event queue thoroughly."""
    for _ in range(iterations):
        app.processEvents()
        if delay_ms > 0:
            loop = QEventLoop()
            QTimer.singleShot(delay_ms, loop.quit)
            loop.exec()


class TestDesktopLayoutCompression:
    """Tests asserting non-overlapping controls, QScrollArea settings, and screen-aware sizing."""

    def test_settings_content_in_scroll_area_and_actions_persistent(self) -> None:
        """MainWindow must embed settings in a QScrollArea, keeping progress/action buttons persistent."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.show()
        process_events()

        # Window must have a QScrollArea
        scroll_areas = list(window.findChildren(QScrollArea))
        assert len(scroll_areas) >= 1, "MainWindow must contain a QScrollArea for settings"
        scroll_area = scroll_areas[0]
        scroll_widget = scroll_area.widget()
        assert scroll_widget is not None, "QScrollArea must have a content widget"

        # Model, scale, route, and stereo groups must be inside scroll_widget
        assert window.model_combo.window() == window
        assert scroll_widget.isAncestorOf(window.model_combo), "model_combo must be inside settings scroll area"
        assert scroll_widget.isAncestorOf(window.scale_combo), "scale_combo must be inside settings scroll area"
        assert scroll_widget.isAncestorOf(window.route_combo), "route_combo must be inside settings scroll area"
        assert scroll_widget.isAncestorOf(window.strength_spin), "strength_spin must be inside settings scroll area"

        # Action and progress controls must be persistent (OUTSIDE the scroll area)
        assert not scroll_widget.isAncestorOf(window.convert_btn), "convert_btn must be persistent outside scroll area"
        assert not scroll_widget.isAncestorOf(window.cancel_btn), "cancel_btn must be persistent outside scroll area"
        assert not scroll_widget.isAncestorOf(window.progress_bar), "progress_bar must be persistent outside scroll area"
        assert not scroll_widget.isAncestorOf(window.status_label), "status_label must be persistent outside scroll area"
        window.close()

    def test_controls_do_not_overlap_at_default_and_minimum_sizes(self) -> None:
        """Dropdowns and labels must not overlap vertically or have degenerate sizes at default and min sizes."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.show()

        for test_size in [QSize(850, 780), QSize(750, 650), QSize(700, 550)]:
            window.resize(test_size)
            process_events()

            # Check critical settings controls
            controls = [
                ("model_combo", window.model_combo),
                ("scale_combo", window.scale_combo),
                ("route_combo", window.route_combo),
                ("strength_spin", window.strength_spin),
            ]

            for name, widget in controls:
                assert widget.isVisible(), f"{name} should be visible"
                rect = widget.geometry()
                assert rect.height() >= 20, f"{name} height too small ({rect.height()}px), compressed"
                assert rect.width() >= 50, f"{name} width too small ({rect.width()}px), compressed"

            # Check that model, scale, and route combos do not overlap each other
            model_rect_in_window = window.model_combo.mapTo(window, QPoint(0, 0))
            scale_rect_in_window = window.scale_combo.mapTo(window, QPoint(0, 0))
            route_rect_in_window = window.route_combo.mapTo(window, QPoint(0, 0))

            # Scale must be distinctly below model combo
            assert scale_rect_in_window.y() >= model_rect_in_window.y() + window.model_combo.height(), (
                f"scale_combo (y={scale_rect_in_window.y()}) overlaps model_combo (y={model_rect_in_window.y()}, h={window.model_combo.height()})"
            )
            # Route must be distinctly below scale combo
            assert route_rect_in_window.y() >= scale_rect_in_window.y() + window.scale_combo.height(), (
                f"route_combo (y={route_rect_in_window.y()}) overlaps scale_combo (y={scale_rect_in_window.y()}, h={window.scale_combo.height()})"
            )

        window.close()

    def test_long_error_does_not_compress_or_overlap_settings_controls(self) -> None:
        """A long error message in status/details does not compress or distort model/scale/route rows."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.resize(850, 780)
        window.show()
        process_events()

        long_error = (
            "CUDA Out Of Memory: Tried to allocate 4.50 GiB (GPU 0; 8.00 GiB total capacity; "
            "6.20 GiB already allocated; 1.10 GiB free; 6.30 GiB reserved in total by PyTorch) "
            "If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation. "
            "See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF. "
            "Traceback (most recent call last): File 'puregpu3d/video/gpu_convert.py', line 342, in convert_frames"
        )
        window._on_conversion_failed(long_error, "converting")
        process_events()

        # Check that scale_combo and route_combo still have healthy heights and do not overlap
        scale_pos = window.scale_combo.mapTo(window, QPoint(0, 0))
        route_pos = window.route_combo.mapTo(window, QPoint(0, 0))
        assert window.scale_combo.height() >= 20
        assert window.route_combo.height() >= 20
        assert route_pos.y() >= scale_pos.y() + window.scale_combo.height()
        window.close()

    def test_screen_aware_startup_size(self) -> None:
        """Window startup size must not exceed primary screen available geometry."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            assert window.width() <= avail.width(), f"Window width {window.width()} exceeds screen available width {avail.width()}"
            assert window.height() <= avail.height(), f"Window height {window.height()} exceeds screen available height {avail.height()}"
        assert window.minimumWidth() <= 700
        assert window.minimumHeight() <= 550
        window.close()

    def test_dark_theme_preserved_on_scroll_area(self) -> None:
        """Scroll area must be borderless and transparent to preserve Fusion dark theme."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        scroll_areas = list(window.findChildren(QScrollArea))
        assert len(scroll_areas) >= 1
        sa = scroll_areas[0]
        assert sa.frameShape() == QScrollArea.Shape.NoFrame or "border: none" in sa.styleSheet()
        assert "transparent" in sa.styleSheet()
        window.close()


class TestRetryLifecycle:
    """Tests for restart-free retry lifecycle, error retention, and process isolation."""

    @pytest.fixture
    def setup_files(self, tmp_path: Path):
        in_file = FIXTURE_VIDEO if FIXTURE_VIDEO.is_file() else tmp_path / "mock_in.mp4"
        if not in_file.exists():
            in_file.write_bytes(b"dummy")
        out_file = tmp_path / "mock_out.mp4"
        return in_file, out_file

    def test_failed_worker_retains_error_and_enables_restart_free_retry(self, tmp_path: Path) -> None:
        """Simulate real QProcess worker error:
        1. Emits error message and exits code 1.
        2. While running, retry is refused.
        3. Once worker fully exits, last_error is preserved and convert_btn is re-enabled.
        4. Second conversion run starts cleanly and completes without app restart.
        """
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.show()
        process_events()

        # Prepare dummy fake worker script that fails first, succeeds second
        worker_script = tmp_path / "fake_worker.py"
        flag_file = tmp_path / "run_flag.txt"

        worker_code = f"""
import sys, time, json
flag = r"{flag_file}"
if not sys.platform.startswith("win"):
    import os
if not __import__("pathlib").Path(flag).exists():
    __import__("pathlib").Path(flag).write_text("failed_once")
    # First run: emit status then error then exit 1
    print(json.dumps({{"type": "status", "stage": "converting", "message": "Starting conversion..."}}), flush=True)
    time.sleep(0.15)
    print(json.dumps({{"type": "error", "stage": "converting", "error": "Simulated GPU Memory Failure"}}), flush=True)
    time.sleep(0.2)
    sys.exit(1)
else:
    # Second run: emit status then completed then exit 0
    print(json.dumps({{"type": "status", "stage": "converting", "message": "Retrying conversion..."}}), flush=True)
    time.sleep(0.1)
    print(json.dumps({{"type": "completed", "result": {{"wall_clock_seconds": 0.5, "effective_fps": 30.0, "total_frames_processed": 15, "resolved_backend": "GPU"}}}}), flush=True)
    sys.exit(0)
"""
        worker_script.write_text(worker_code, encoding="utf-8")

        # Setup paths
        window.input_edit.setText(str(FIXTURE_VIDEO))
        out_file = tmp_path / "output_sbs.mp4"
        window.output_edit.setText(str(out_file))
        process_events()

        # Monkeypatch worker launch command to use our fake worker
        def fake_launch_spec():
            return sys.executable, ["-B", str(worker_script), "--command-file", str(controller._command_file)]

        controller._get_worker_launch_spec = fake_launch_spec  # type: ignore

        # Run 1: Launch conversion that will fail
        started, err = controller.start_conversion(overwrite_confirmed=True)
        assert started, f"Failed to start conversion: {err}"
        assert controller.state in (DesktopState.VALIDATING, DesktopState.PREPARING_MODEL, DesktopState.CONVERTING)

        # Wait until error received and process finishes
        start_t = time.time()
        while controller._worker_process is not None and time.time() - start_t < 5.0:
            process_events(1, 20)

        # Worker must have exited
        assert controller._worker_process is None, "Worker process should be cleaned up after exit"
        assert controller.state == DesktopState.FAILED, "Controller state must be FAILED"
        assert controller.last_error == "Simulated GPU Memory Failure", "Terminal error message must be preserved"

        # UI must reflect failed state, but Start/Retry button MUST BE ENABLED!
        process_events(5, 10)
        assert window.convert_btn.isEnabled(), "Convert button MUST be re-enabled for retry after worker finishes"

        # Run 2: Retry without restart
        started2, err2 = controller.start_conversion(overwrite_confirmed=True)
        assert started2, f"Failed to start retry conversion: {err2}"

        start_t2 = time.time()
        while controller._worker_process is not None and time.time() - start_t2 < 5.0:
            process_events(1, 20)

        assert controller._worker_process is None
        assert controller.state == DesktopState.COMPLETED, "Retry conversion should succeed with COMPLETED"
        assert window.convert_btn.isEnabled(), "Convert button should be enabled after completion for subsequent runs"
        window.close()

    def test_abnormal_clean_exit_without_terminal_protocol_transitions_to_failed(self, tmp_path: Path) -> None:
        """If a worker process exits with 0 without emitting completed or error, state must become FAILED."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.show()
        process_events()

        worker_script = tmp_path / "premature_clean_exit.py"
        worker_code = """
import sys, time, json
print(json.dumps({"type": "status", "stage": "converting", "message": "Running..."}), flush=True)
time.sleep(0.1)
# Exits cleanly with code 0 without completed protocol message
sys.exit(0)
"""
        worker_script.write_text(worker_code, encoding="utf-8")

        window.input_edit.setText(str(FIXTURE_VIDEO))
        window.output_edit.setText(str(tmp_path / "out_clean.mp4"))
        process_events()

        controller._get_worker_launch_spec = lambda: (sys.executable, ["-B", str(worker_script), "--command-file", str(controller._command_file)])  # type: ignore

        started, err = controller.start_conversion(overwrite_confirmed=True)
        assert started

        start_t = time.time()
        while controller._worker_process is not None and time.time() - start_t < 5.0:
            process_events(1, 20)

        # Must not hang in converting/validating
        assert controller.state == DesktopState.FAILED
        assert "premature" in str(controller.last_error).lower() or "without completion" in str(controller.last_error).lower()
        assert window.convert_btn.isEnabled()
        window.close()

    def test_stale_cancellation_timer_bound_safely_to_old_process(self, tmp_path: Path) -> None:
        """Cancellation escalation timer for Job 1 must not terminate or kill Job 2."""
        controller = DesktopController()

        # Job 1 script: slow to exit on cancel
        job1_script = tmp_path / "slow_cancel.py"
        job1_script.write_text("""
import sys, time
time.sleep(10)
sys.exit(0)
""", encoding="utf-8")

        # Job 2 script: fast success
        job2_script = tmp_path / "fast_job2.py"
        job2_script.write_text("""
import sys, time, json
print(json.dumps({"type": "status", "stage": "converting", "message": "Job 2 running"}), flush=True)
time.sleep(0.5)
print(json.dumps({"type": "completed", "result": {"done": True}}), flush=True)
sys.exit(0)
""", encoding="utf-8")

        # Setup probe & paths
        controller.set_input_path(FIXTURE_VIDEO)
        controller.set_output_path(tmp_path / "out1.mp4")

        # Start Job 1
        controller._get_worker_launch_spec = lambda: (sys.executable, ["-B", str(job1_script), "--command-file", str(controller._command_file)])  # type: ignore
        started, _ = controller.start_conversion(overwrite_confirmed=True)
        assert started
        old_process = controller._worker_process

        # Cancel Job 1
        controller.cancel_conversion()
        # Force escalation timer to be short for test
        if controller._cancel_timer:
            controller._cancel_timer.setInterval(100)
            controller._cancel_timer.start()

        # Wait for Job 1 to finish
        t0 = time.time()
        while controller._worker_process is not None and time.time() - t0 < 3.0:
            process_events(1, 20)

        # Immediately start Job 2
        controller.set_output_path(tmp_path / "out2.mp4")
        controller._get_worker_launch_spec = lambda: (sys.executable, ["-B", str(job2_script), "--command-file", str(controller._command_file)])  # type: ignore
        started2, _ = controller.start_conversion(overwrite_confirmed=True)
        assert started2
        job2_process = controller._worker_process
        assert job2_process is not None
        assert job2_process is not old_process

        # Wait across the cancellation escalation timeout window
        time.sleep(0.3)
        process_events(5, 20)

        # Job 2 must STILL be alive or have completed cleanly, NOT killed
        assert job2_process.state() != QProcess.ProcessState.NotRunning or controller.state == DesktopState.COMPLETED

        # Wait for Job 2 to finish
        t1 = time.time()
        while controller._worker_process is not None and time.time() - t1 < 3.0:
            process_events(1, 20)

        assert controller.state == DesktopState.COMPLETED

    def test_completed_and_cancelled_allow_retry(self, tmp_path: Path) -> None:
        """Controller and Window must allow restarting conversion after completion and after cancellation."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.show()
        process_events()

        window.input_edit.setText(str(FIXTURE_VIDEO))
        window.output_edit.setText(str(tmp_path / "out_completed.mp4"))
        process_events()

        # Simulate completion
        controller._set_state(DesktopState.COMPLETED)
        window._update_ui_state(DesktopState.COMPLETED.value)
        process_events()
        assert window.convert_btn.isEnabled(), "Convert button must be enabled after completion"

        # Simulate cancellation
        controller._set_state(DesktopState.CANCELLED)
        window._update_ui_state(DesktopState.CANCELLED.value)
        process_events()
        assert window.convert_btn.isEnabled(), "Convert button must be enabled after cancellation"
        window.close()

    def test_failed_start_handles_error_cleanly_and_allows_retry(self, tmp_path: Path) -> None:
        """If worker executable fails to launch, error is handled, state is FAILED, and retry is enabled."""
        controller = DesktopController()
        window = MainWindow(controller=controller)
        window.show()
        process_events()

        window.input_edit.setText(str(FIXTURE_VIDEO))
        window.output_edit.setText(str(tmp_path / "out_fail_start.mp4"))
        process_events()

        # Launch invalid program that cannot start
        controller._get_worker_launch_spec = lambda: ("nonexistent_program_xyz_123.exe", ["--command-file", "dummy"])  # type: ignore

        started, err = controller.start_conversion(overwrite_confirmed=True)
        assert not started
        assert "failed to start" in err.lower()
        assert controller.state == DesktopState.FAILED
        assert controller.last_error is not None
        assert controller._worker_process is None, "Failed process reference must be cleaned up"

        process_events()
        # Start button should be enabled so user can re-try once configuration is fixed
        assert window.convert_btn.isEnabled()
        window.close()

    def test_refuse_retry_while_worker_alive(self, tmp_path: Path) -> None:
        """validate_for_conversion and start_conversion must refuse retry while worker process is alive."""
        controller = DesktopController()

        slow_script = tmp_path / "slow_worker.py"
        slow_script.write_text("import time; time.sleep(2)", encoding="utf-8")

        controller.set_input_path(FIXTURE_VIDEO)
        controller.set_output_path(tmp_path / "out_refuse.mp4")
        controller._get_worker_launch_spec = lambda: (sys.executable, ["-B", str(slow_script), "--command-file", str(controller._command_file)])  # type: ignore

        started, _ = controller.start_conversion(overwrite_confirmed=True)
        assert started
        assert controller._worker_process is not None

        # While worker is alive, validation must refuse
        valid, msg = controller.validate_for_conversion()
        assert not valid
        assert "already running" in msg.lower()

        # Calling start_conversion again must fail
        started_again, err = controller.start_conversion(overwrite_confirmed=True)
        assert not started_again
        assert "already running" in err.lower()

        # Clean up by cancelling
        controller.cancel_conversion()
        t0 = time.time()
        while controller._worker_process is not None and time.time() - t0 < 3.0:
            process_events(1, 20)
