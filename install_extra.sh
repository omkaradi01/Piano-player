#!/bin/bash
# install_extra.sh — Run ONCE to install torchcrepe + fix scikit-learn version
set -e
cd "$(dirname "$0")"

VENV="$(dirname "$0")/venv311"
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
  echo "✓  Using venv311"
fi

echo ""
echo "🔧  Fixing scikit-learn version (basic-pitch needs ≤1.5.1)…"
pip install "scikit-learn==1.5.1" --quiet

echo "🔧  Installing torchcrepe (neural pitch tracker)…"
pip install torchcrepe --quiet

echo "🔧  Installing torchcodec (required by torchaudio to save stems)…"
pip install torchcodec --quiet

echo ""
echo "✅  Done! Run: bash start.sh"
echo ""
