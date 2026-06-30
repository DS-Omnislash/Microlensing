@echo off
REM ============================================================
REM  Launch the Microlensing web app (self-bootstrapping).
REM
REM  On each run this script will, only when needed:
REM    1. create the project virtual environment (venv\) if missing,
REM    2. install / update dependencies from requirements.txt,
REM    3. start the web app at http://127.0.0.1:8000
REM
REM  Dependencies are only (re)installed on the first run or when
REM  requirements.txt changes, so normal launches are fast.
REM  Just double-click this file, or run "run" from the project root.
REM ============================================================
setlocal
cd /d "%~dp0"

set "VENV=%~dp0venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.deps_hash"

REM --- 1) Create the virtual environment if it does not exist ---
if not exist "%PY%" (
    echo [setup] Creating virtual environment in venv\ ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [error] Could not create the virtual environment.
        echo         Make sure Python 3 is installed and available on your PATH.
        exit /b 1
    )
)

REM --- 2) Hash requirements.txt so we can detect changes ---
set "REQHASH="
for /f "skip=1 delims=" %%h in ('certutil -hashfile "%~dp0requirements.txt" MD5 2^>nul') do (
    if not defined REQHASH set "REQHASH=%%h"
)
set "REQHASH=%REQHASH: =%"

set "OLDHASH="
if exist "%STAMP%" set /p OLDHASH=<"%STAMP%"

REM --- 3) Install / update dependencies only when needed ---
if not "%REQHASH%"=="%OLDHASH%" (
    echo [setup] Installing / updating dependencies...
    echo         The first run downloads PyTorch and can take a few minutes.
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [error] Dependency installation failed. See the messages above.
        exit /b 1
    )
    > "%STAMP%" echo %REQHASH%
    echo [setup] Dependencies are up to date.
)

REM --- 4) Launch the app ---
echo [run] Starting the Microlensing web app at http://127.0.0.1:8000
echo [run] Press Ctrl+C to stop.
"%PY%" -m uvicorn app.main:app --reload --app-dir "%~dp0webapp"
