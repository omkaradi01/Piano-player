#!/usr/bin/env bash
# setup.sh — One-time setup for Piano Transcriber (macOS / Linux)
# Usage: bash setup.sh

set -e
echo ""
echo "🎹  Piano Transcriber — Setup"
echo "────────────────────────────────────────"

# 1. Check Python 3.9+
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Install from https://python.org"
  exit 1
fi
PY_VER=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_VER" -lt 9 ]; then
  echo "❌  Python 3.9+ required. Found 3.$PY_VER"
  exit 1
fi
echo "✓  Python 3.$PY_VER found"

# 2. Check / install ffmpeg
if command -v ffmpeg &>/dev/null; then
  echo "✓  ffmpeg found"
else
  echo "⚙  Installing ffmpeg…"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
  else
    sudo apt-get install -y ffmpeg
  fi
fi

# 3. Create virtual environment
if [ ! -d "venv" ]; then
  echo "⚙  Creating virtual environment…"
  python3 -m venv venv
fi
source venv/bin/activate

# 4. Install Python packages
echo "⚙  Installing Python packages (this may take a few minutes)…"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓  All packages installed"

echo ""
echo "────────────────────────────────────────"
echo "✅  Setup complete! To start the app:"
echo ""
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "   Then open  http://localhost:5050  in your browser."
echo ""
