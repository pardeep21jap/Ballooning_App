@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  BalloonApp - Windows EXE build
echo ============================================================
echo.

rem Move to the folder this script lives in, regardless of caller's cwd.
cd /d "%~dp0"

rem ------------------------------------------------------------------
rem 1. Pick a Python interpreter: prefer the local .venv if present.
rem ------------------------------------------------------------------
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" (
    echo [1/5] Using existing virtual environment: .venv
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [1/5] No .venv found next to this script.
    echo       Creating one now with: python -m venv .venv
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create a virtual environment. Is Python 3.11+ installed and on PATH?
        exit /b 1
    )
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

echo       Interpreter: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

rem ------------------------------------------------------------------
rem 2. Ensure core dependencies are installed.
rem ------------------------------------------------------------------
echo [2/5] Installing/verifying core dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies from requirements.txt.
    exit /b 1
)
echo.

rem ------------------------------------------------------------------
rem 3. Ensure PyInstaller is installed.
rem ------------------------------------------------------------------
echo [3/5] Checking for PyInstaller...
"%PYTHON_EXE%" -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo       PyInstaller not found. Installing...
    "%PYTHON_EXE%" -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller.
        exit /b 1
    )
) else (
    echo       PyInstaller already installed.
)
echo.

rem ------------------------------------------------------------------
rem 4. Clean previous build output.
rem ------------------------------------------------------------------
echo [4/5] Cleaning previous build output (build\, dist\, *.spec)...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "BalloonApp.spec" del /q "BalloonApp.spec"
echo.

rem ------------------------------------------------------------------
rem 5. Build with PyInstaller (one-folder build; bundles the package and
rem    resources, but NOT projects/datasets/models -- those stay external
rem    so user data is never locked inside the build).
rem ------------------------------------------------------------------
echo [5/5] Building with PyInstaller (this can take a few minutes)...
"%PYTHON_EXE%" -m PyInstaller ^
    --name "BalloonApp" ^
    --windowed ^
    --icon "balloon_app\resources\balloonapp.ico" ^
    --noconfirm ^
    --add-data "balloon_app\resources;balloon_app\resources" ^
    --collect-submodules balloon_app ^
    main.py

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed. See output above for details.
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete.
echo  Executable: dist\BalloonApp\BalloonApp.exe
echo.
echo  Note: projects\, datasets\, and models\ are NOT bundled.
echo  Copy them next to BalloonApp.exe (or let the app recreate
echo  empty ones on first run) before distributing.
echo ============================================================

endlocal
