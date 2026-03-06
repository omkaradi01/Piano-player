@echo off
:: start.bat — One-click launch for Piano Transcriber (Windows)
:: Double-click this file to run the app

cd /d "%~dp0"

echo.
echo 🎹  Piano Transcriber — Starting up...
echo ----------------------------------------

:: ── Check Python ─────────────────────────────────────────────────────────
where python >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found.
    echo         Download it from: https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found

:: ── Check ffmpeg ──────────────────────────────────────────────────────────
where ffmpeg >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] ffmpeg is not installed.
    echo         1. Download from: https://ffmpeg.org/download.html
    echo            (get the "Windows builds" from gyan.dev)
    echo         2. Extract the zip
    echo         3. Copy ffmpeg.exe into this folder, OR add it to your PATH
    echo.
    pause
    exit /b 1
)
echo [OK] ffmpeg found

:: ── Create virtual environment (first time only) ──────────────────────────
IF NOT EXIST "venv\" (
    echo [..] Creating virtual environment (first time only)...
    python -m venv venv
)

:: ── Activate ──────────────────────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Install packages (first time only) ────────────────────────────────────
IF NOT EXIST "venv\.packages_installed" (
    echo [..] Installing packages (2-5 min the first time)...
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo. > venv\.packages_installed
    echo [OK] All packages installed
) ELSE (
    echo [OK] Packages already installed
)

:: ── Open browser after short delay ────────────────────────────────────────
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:5050"

echo.
echo ----------------------------------------
echo  Launching!  ^>  http://localhost:5050
echo  (Close this window to stop the server)
echo ----------------------------------------
echo.

:: ── Start the server ──────────────────────────────────────────────────────
python app.py

pause
