# 🎹 Piano Transcriber

Turn any YouTube song into a professional piano version — directly in your browser.

## How It Works

```
YouTube URL
    ↓  yt-dlp          — downloads the audio
    ↓  ffmpeg           — converts to WAV
    ↓  basic-pitch      — Spotify's AI model: audio → polyphonic MIDI
    ↓  Arranger         — splits into right/left hand, quantises, adds dynamics
    ↓  MIDI file
         ↓  Browser (Tone.js + Salamander Grand Piano)
         → Plays as a real piano 🎹
```

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.9+ | https://python.org |
| ffmpeg | any | https://ffmpeg.org |

---

## Quick Start

### macOS / Linux

```bash
bash setup.sh
source venv/bin/activate
python app.py
```

### Windows

1. Double-click **`setup_windows.bat`** (first time only)
2. Open Command Prompt in this folder and run:
   ```
   venv\Scripts\activate
   python app.py
   ```

Then open **http://localhost:5050** in your browser.

---

## Usage

1. Copy any YouTube song URL (e.g. `https://www.youtube.com/watch?v=…`)
2. Paste it into the app and click **Transcribe →**
3. Wait 1–4 minutes (depends on song length and your CPU)
4. The piano version plays automatically with the on-screen keyboard visualization
5. Adjust **Tempo** and **Volume** sliders to taste
6. Click **⬇ Download MIDI** to save the MIDI file (open in GarageBand, Logic, Ableton, etc.)

---

## Processing Time (rough guide)

| Song length | Approx. time |
|-------------|--------------|
| 3 min       | ~1–2 min     |
| 5 min       | ~2–3 min     |
| 10 min      | ~4–6 min     |

First run downloads the basic-pitch AI model (~80 MB) — this only happens once.

---

## Tips for Best Results

- **Songs with clear melody** (pop, classical, film scores) transcribe best
- **Live recordings or heavily compressed audio** may give rougher results
- Use the **Tempo slider** to slow down complex passages
- Open the downloaded MIDI in a DAW (GarageBand / Logic / Ableton) for further editing

---

## Legal Note

This app is for personal and educational use only. Downloading copyrighted YouTube
content may violate YouTube's Terms of Service and local copyright laws. Only use
it with content you own or have rights to use.

---

## Tech Stack

- **[basic-pitch](https://github.com/spotify/basic-pitch)** by Spotify Research — AI pitch transcription
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — YouTube audio download
- **[Flask](https://flask.palletsprojects.com/)** — web server
- **[pretty_midi](https://github.com/craffel/pretty-midi)** — MIDI manipulation
- **[Tone.js](https://tonejs.github.io/)** — browser audio engine
- **[Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)** — piano samples
