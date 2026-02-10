@echo off
setlocal
cd /d "%~dp0"

echo [setup] Preparing conda environment...
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_conda.ps1
if errorlevel 1 (
  echo [setup] Setup failed.
  pause
  exit /b 1
)

echo [setup] Setup complete.
endlocal
