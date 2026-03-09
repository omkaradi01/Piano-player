# Piano Transcriber — Project Guide

## Mission
Turn any YouTube song (especially Tamil film music) into a high-quality, playable piano cover MIDI — directly in the browser. The output should sound like a skilled pianist arranged and played the song, not like a raw transcription.

## Vision
Build the best open-source YouTube-to-piano-cover tool, with special emphasis on Tamil/Indian film music (Carnatic raga-based melodies, gamakas, complex rhythms). The tool should handle any genre but excel at Tamil music.

## Core Architecture
```
YouTube URL
  → yt-dlp (download audio)
  → Source Separation (isolate vocals, bass, drums, other)
  → Pitch Tracking (extract vocal melody as note events)
  → Key/Raga Detection (identify scale)
  → Beat Tracking (find tempo + beat positions)
  → Chord Detection (harmonic analysis)
  → Piano Arrangement (melody → right hand, chords → left hand)
  → MIDI output
  → Browser playback (Tone.js + Salamander Grand Piano)
```

## Tech Stack
- **Runtime**: Python 3.12, Flask web server
- **Frontend**: Vanilla HTML/JS, Tone.js for audio, on-screen piano visualization
- **Venv**: `venv/` directory (Python 3.12 from /opt/homebrew/bin/python3.12)

## Key Principles
1. **Quality over speed** — A 5-minute processing time that produces a great piano cover beats a 30-second one that sounds wrong
2. **Tamil music first** — Carnatic ragas, gamakas (ornaments), and Indian rhythmic patterns should be handled correctly, not forced into Western major/minor boxes
3. **Graceful degradation** — Each pipeline stage should have fallbacks. If the best model fails, fall back to the next best, never crash
4. **Real-time feedback** — The UI must show progress with meaningful stage descriptions and percentages so the user knows it's working
5. **Minimal dependencies** — Prefer pip-installable packages over manual model downloads where possible

## File Structure
- `app.py` — Flask server, job management, API endpoints
- `pipeline.py` — Core transcription pipeline (download → separate → transcribe → arrange → MIDI)
- `templates/index.html` — Web UI with piano visualization
- `training_data/` — 62 Tamil song pairs (original + piano cover) with extracted features
- `songs.csv` — Training song catalog
- `requirements.txt` — Python dependencies

## What NOT to Do
- Do not add unnecessary abstractions or over-engineer
- Do not break the web UI — it must always show progress and play results
- Do not remove training data support — the 62 Tamil song pairs are valuable for style matching
- Do not add features unrelated to the core mission (no lyrics display, no karaoke mode, no multi-instrument output)
- Do not use deprecated or unmaintained libraries when better alternatives exist
- Do not ignore Indian music theory — Tamil film music uses Carnatic ragas, not just Western scales

## Current State (v16)
The pipeline works end-to-end but has quality issues in every stage:
- Source separation (htdemucs_ft) is decent but not SOTA
- Pitch tracking (torchcrepe) struggles with ornamental singing
- Key detection (Krumhansl-Schmuckler) only knows major/minor, not ragas
- Beat tracking (librosa) is basic, single global tempo
- Chord detection (chroma templates) is ~55-60% accuracy
- Left hand is a simple root-fifth pattern with no voice leading
- Quantization is single-grid (16th notes only)
- Processing is slow (demucs shifts=3 takes 15+ min on CPU)

See TODO.md for the full upgrade plan.
