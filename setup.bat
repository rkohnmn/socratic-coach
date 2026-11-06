@echo off
setlocal EnableExtensions EnableDelayedExpansion
title TA Phase 1 - Offline Local TA Environment Setup

rem ===========================================================================
rem  setup.bat - one-command wrapper for verify_env.py on Windows 11
rem
rem    1. Finds a compatible Python 3.10-3.12 interpreter (py launcher first).
rem    2. Creates an isolated .venv next to this script on first run.
rem    3. Launches verify_env.py, forwarding any extra arguments, e.g.
rem         setup.bat                          full provisioning + verification
rem         setup.bat --skip-models            fast structural pass (no downloads)
rem         setup.bat --skip-install --skip-playback
rem
rem  Exit codes: 0 = fully operational | 1 = failures (see ACTION REQUIRED)
rem ===========================================================================

cd /d "%~dp0"

set "PYTHON_CMD="

rem -- Prefer the py launcher, newest compatible minor version first ----------
for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%V -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PYTHON_CMD=py -%%V"
    )
)

rem -- Fall back to whatever python.exe is on PATH -----------------------------
if not defined PYTHON_CMD (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo.
    echo [X] ERROR: no compatible Python 3.10-3.12 interpreter was found.
    echo     FIX: install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo          and tick "Add python.exe to PATH", then re-run setup.bat.
    echo.
    pause
    exit /b 1
)

echo [i] Interpreter : !PYTHON_CMD!
for /f "delims=" %%P in ('!PYTHON_CMD! -c "import sys; print(sys.executable)"') do echo [i] Executable : %%P

rem -- Create the isolated virtual environment on first run --------------------
if not exist ".venv\Scripts\python.exe" (
    echo [i] Creating isolated virtual environment at "%~dp0.venv" ...
    !PYTHON_CMD! -m venv .venv
    if errorlevel 1 (
        echo [X] ERROR: failed to create the .venv virtual environment.
        pause
        exit /b 1
    )
)

set "VPY=%~dp0.venv\Scripts\python.exe"

echo [i] Upgrading pip / setuptools / wheel inside .venv ...
"%VPY%" -m pip install --upgrade pip setuptools wheel --disable-pip-version-check >nul 2>&1

echo [i] Launching verify_env.py %*
echo.
"%VPY%" verify_env.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [OK] Phase 1 complete: environment is 100 percent operational.
) else (
    echo [X] Phase 1 finished WITH FAILURES ^(exit code %RC%^).
    echo     Read the ACTION REQUIRED block above, apply the listed fixes,
    echo     then re-run setup.bat.
)
echo.
pause
exit /b %RC%