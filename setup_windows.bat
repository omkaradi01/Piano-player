@echo off
:: setup_windows.bat — One-time setup for Piano Transcriber (Windows)
:: Usage: Double-click, or run from Command Prompt

echo.
echo 🎹  Piano Transcriber — Windows Setup
echo ----------------------------------------

:: Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Download from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

:: Check ffmpeg
where ffmpeg >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo [NOTE] ffmpeg not found.
    echo Please install it from https://ffmpeg.org/download.html
    echo and add it to your PATH, then re-run this script.
    pause
    exit /b 1
)
echo [OK] ffmpeg found

:: Create venv
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate and install
call venv\Scripts\activate.bat
echo Installing Python packages ^(this may take a few minutes^)...
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] Packages installed

echo.
echo ----------------------------------------
echo Setup complete! To start the app:
echo.
echo   venv\Scripts\activate
echo   python app.py
echo.
echo Then open  http://localhost:5050  in your browser.
echo.
pause
