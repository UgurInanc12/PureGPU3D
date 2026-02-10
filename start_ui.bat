@echo off
setlocal
cd /d "%~dp0"

echo [start_ui] Ensuring conda environment...
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_conda.ps1 -SkipUpdateIfExists
if errorlevel 1 (
  echo [start_ui] Setup failed.
  pause
  exit /b 1
)

echo [start_ui] Launching web UI...
powershell -ExecutionPolicy Bypass -File scripts\run_ui.ps1
if errorlevel 1 (
  echo [start_ui] UI start failed.
  pause
  exit /b 1
)

endlocal
