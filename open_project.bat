@echo off
setlocal
cd /d "%~dp0"

echo [open_project] Step 1/2: Ensuring conda environment...
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_conda.ps1 -SkipUpdateIfExists
if errorlevel 1 (
  echo [open_project] Setup failed.
  pause
  exit /b 1
)

echo [open_project] Step 2/2: Launching web UI...
powershell -ExecutionPolicy Bypass -File scripts\run_ui.ps1
if errorlevel 1 (
  echo [open_project] UI start failed.
  pause
  exit /b 1
)

endlocal
