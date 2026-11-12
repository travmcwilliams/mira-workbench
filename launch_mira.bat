@echo off
REM ===========================================================================
REM  MIRA launcher.  Double-click it, or run it from a Command Prompt.
REM
REM  First run: creates .venv and installs MIRA into it.  Later runs:
REM  reinstalls only when pyproject.toml has changed (e.g. after git pull).
REM
REM    launch_mira.bat               set up if needed, then start the app
REM    launch_mira.bat --setup-only  set up if needed, then exit
REM ===========================================================================
setlocal
title MIRA - Multi-source Ingest ^& Records Analytics

REM Run from this script's own folder, wherever that ends up being.
cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.mira-installed-pyproject.toml"

if exist "%PY%" goto :have_venv

REM --- find a system Python to build the venv with ------------------------
set "SYSPY="
call :try_py py -3.12
if not defined SYSPY call :try_py py -3.11
if not defined SYSPY call :try_py python
if not defined SYSPY goto :no_python

echo Creating virtual environment in .venv ...
%SYSPY% -m venv "%VENV%"
if not exist "%PY%" goto :fail

:have_venv
REM --- install / refresh dependencies when needed --------------------------
"%PY%" -c "import streamlit, pandas, duckdb" >nul 2>&1
if errorlevel 1 goto :install
if not exist "%STAMP%" goto :install
fc /b "%STAMP%" pyproject.toml >nul 2>&1
if errorlevel 1 goto :install
goto :ready

:install
echo.
echo Installing MIRA and its dependencies. The first time takes a few minutes...
echo.
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -e .
if errorlevel 1 goto :fail
copy /y pyproject.toml "%STAMP%" >nul

:ready
if /i "%~1"=="--setup-only" (
  echo MIRA is set up.
  goto :eof
)

echo Starting MIRA...  (keep this window open; close it to stop the app)
echo.

REM The app runs headless (see .streamlit\config.toml), so open the browser
REM ourselves a few seconds after launch -- long enough for the server to bind.
start "MIRA-open" cmd /c "timeout /t 4 >nul & start http://localhost:8501"

"%PY%" -m streamlit run app.py
if errorlevel 1 goto :fail
goto :eof

:try_py
REM Accept the given interpreter only if it really runs and is 3.11 or 3.12.
REM (Check printed output, not the exit code: the py launcher can exit 0
REM even when the interpreter it points at is broken.)
%* -c "import sys; print('MIRA_PY_OK' if sys.version_info[:2] in ((3,11),(3,12)) else 'NO')" 2>nul | findstr /x MIRA_PY_OK >nul
if not errorlevel 1 set "SYSPY=%*"
exit /b 0

:no_python
echo.
echo MIRA needs Python 3.11 or 3.12, and neither was found.
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH", then run this launcher again.
goto :pause

:fail
echo.
echo MIRA hit an error -- see the messages above.

:pause
echo Press any key to close.
pause >nul
exit /b 1
