@echo off
REM ===========================================================================
REM  MIRA launcher.  All launch logic lives here so the desktop shortcut that
REM  points at this file NEVER needs to be updated -- move/rename the project
REM  folder, bump Streamlit, change ports: this script adapts on its own.
REM ===========================================================================
title MIRA - Multi-source Ingest ^& Records Analytics

REM Run from this script's own folder, wherever that ends up being.
cd /d "%~dp0"

REM Pick an available Python without hardcoding a path or version.
set "PY=python"
where python >nul 2>&1 || set "PY=py"

REM Prefer a project-local virtual environment if one ever gets created.
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0venv\Scripts\python.exe"  set "PY=%~dp0venv\Scripts\python.exe"

echo Starting MIRA...  (keep this window open; close it to stop the app)
echo.

REM The app runs headless (see .streamlit\config.toml), so open the browser
REM ourselves a few seconds after launch -- long enough for the server to bind.
start "MIRA-open" cmd /c "timeout /t 4 >nul & start http://localhost:8501"

"%PY%" -m streamlit run app.py

if errorlevel 1 (
  echo.
  echo MIRA exited with an error. Press any key to close.
  pause >nul
)
