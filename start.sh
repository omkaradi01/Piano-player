#!/usr/bin/env bash
# start.sh — One-click launch for Piano Transcriber (macOS / Linux)
# Just run:  bash start.sh

set -e
cd "$(dirname "$0")"

echo ""
echo "🎹  Piano Transcriber — Starting up…"
echo "────────────────────────────────────────"

# ── Check ffmpeg ─────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
  echo "❌  ffmpeg not found. Run: brew install ffmpeg"
  exit 1
fi
echo "✓  ffmpeg found"

# ── Prefer venv311 (Python 3.11 with all packages) ───────────────────────
VENV311="$(dirname "$0")/venv311"
if [ -d "$VENV311" ]; then
  source "$VENV311/bin/activate"
  echo "✓  Using venv311 (Python 3.11)"
else
  echo "⚠   venv311 not found — using system Python"
  echo "    (If you see import errors, run: python3.11 -m venv venv311)"
fi

# ── Quick dependency check ────────────────────────────────────────────────
python3 -c "import flask, yt_dlp, basic_pitch, pretty_midi, librosa" 2>/dev/null || {
  echo "⚙   Installing missing packages…"
  pip3 install flask yt-dlp "basic-pitch[onnx]" pretty_midi librosa soundfile demucs
}
echo "✓  Packages OK"

# ── Open browser ──────────────────────────────────────────────────────────
(sleep 3 && open http://localhost:5050) &

echo ""
echo "────────────────────────────────────────"
echo "✅  Open:  http://localhost:5050"
echo "    Press Ctrl+C to stop"
echo "────────────────────────────────────────"
echo ""

python3 app.py
