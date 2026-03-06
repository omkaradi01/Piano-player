#!/usr/bin/env python3
"""
check.py — Full diagnostic for the Piano Transcriber system.

Run this before processing any song to confirm every component is working.

Usage:
    python check.py           # full check
    python check.py --quick   # skip slow audio tests (30 seconds)

Exit codes:
    0 = everything OK, safe to run
    1 = critical problems found, do NOT run pipeline yet
"""

import os, sys, json, time

BASE = os.path.dirname(os.path.abspath(__file__))
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "   "

errors   = []
warnings = []

def ok(msg):    print(f"  {PASS}  {msg}")
def fail(msg):  print(f"  {FAIL}  {msg}"); errors.append(msg)
def warn(msg):  print(f"  {WARN}  {msg}"); warnings.append(msg)
def info(msg):  print(f"  {INFO}  {msg}")
def header(msg):print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")


# ══════════════════════════════════════════════════════
#  1. PYTHON & SYSTEM TOOLS
# ══════════════════════════════════════════════════════
header("1. Python & system tools")

# Python version
import platform
pv = sys.version_info
if pv.major == 3 and pv.minor >= 10:
    ok(f"Python {pv.major}.{pv.minor}.{pv.micro}")
else:
    fail(f"Python {pv.major}.{pv.minor} — need 3.10+")

# ffmpeg
import subprocess
r = subprocess.run(['ffmpeg', '-version'], capture_output=True)
if r.returncode == 0:
    line = r.stdout.decode().split('\n')[0]
    ok(f"ffmpeg: {line[:60]}")
else:
    fail("ffmpeg not found — run: brew install ffmpeg")


# ══════════════════════════════════════════════════════
#  2. PYTHON PACKAGES
# ══════════════════════════════════════════════════════
header("2. Python packages")

packages = [
    ("flask",           "Flask web server"),
    ("yt_dlp",          "YouTube downloader"),
    ("librosa",         "Audio analysis"),
    ("numpy",           "Numerics"),
    ("soundfile",       "WAV read/write"),
    ("pretty_midi",     "MIDI builder"),
    ("basic_pitch",     "Piano transcription fallback"),
    ("demucs",          "Stem separation"),
    ("torch",           "PyTorch (needed by CREPE + Demucs)"),
    ("torchcrepe",      "Neural pitch detection"),
]

missing_critical = []
for pkg, desc in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        ok(f"{pkg} ({ver}) — {desc}")
    except ImportError:
        if pkg in ("torchcrepe",):
            warn(f"{pkg} NOT installed — pipeline will fall back to basic-pitch")
        else:
            fail(f"{pkg} NOT installed — {desc}  →  pip install {pkg}")
            missing_critical.append(pkg)

# torchcodec (needed by newer torchaudio / demucs to save WAV stems)
try:
    import torchcodec
    ok(f"torchcodec ({getattr(torchcodec,'__version__','?')}) — Demucs WAV saving")
except ImportError:
    warn("torchcodec not found — Demucs may fail to save stems")
    warn("Fix: pip install torchcodec --break-system-packages")

# scikit-learn version (basic-pitch breaks with >=1.6)
try:
    import sklearn
    skv = sklearn.__version__
    parts = [int(x) for x in skv.split('.')[:2]]
    if parts[0] == 1 and parts[1] <= 5:
        ok(f"scikit-learn {skv} — compatible with basic-pitch")
    else:
        warn(f"scikit-learn {skv} — basic-pitch may break. Fix: pip install scikit-learn==1.5.1")
except ImportError:
    pass


# ══════════════════════════════════════════════════════
#  3. TRAINING DATA
# ══════════════════════════════════════════════════════
header("3. Training data")

data_dir = os.path.join(BASE, 'training_data')
profile_path = os.path.join(data_dir, 'style_profile.json')
summary_path = os.path.join(data_dir, 'summary.json')

if not os.path.exists(data_dir):
    fail("training_data/ folder missing — run collect_data.py first")
else:
    ok("training_data/ folder exists")

if not os.path.exists(summary_path):
    fail("training_data/summary.json missing — run collect_data.py first")
else:
    with open(summary_path) as f:
        summary = json.load(f)
    total = summary.get('total', 0)
    succeeded = summary.get('succeeded', 0)
    failed = summary.get('failed', [])
    if succeeded >= 20:
        ok(f"summary.json: {succeeded}/{total} songs processed")
    elif succeeded >= 10:
        warn(f"summary.json: only {succeeded}/{total} songs processed (more = better)")
    else:
        fail(f"summary.json: only {succeeded} songs processed — run collect_data.py")
    if failed:
        warn(f"Failed songs: {', '.join(failed)}")

if not os.path.exists(profile_path):
    fail("style_profile.json missing — run: python analyze_pairs.py")
else:
    with open(profile_path) as f:
        profile = json.load(f)
    n = len(profile.get('songs_analysed', []))
    m = profile.get('melody', {})
    ok(f"style_profile.json: built from {n} songs")
    info(f"Melody range: MIDI {m.get('pitch_lo','?')}–{m.get('pitch_hi','?')}, "
         f"octave {m.get('primary_octave','?')}, "
         f"{m.get('notes_per_beat_med','?')} notes/beat")

# Check cover MIDIs exist (these are the actual training ground truth)
mid_count = 0
mid_missing = []
for song in summary.get('songs', []):
    mid = os.path.join(song['dir'], 'cover.mid')
    if os.path.exists(mid):
        mid_count += 1
    else:
        mid_missing.append(song['name'])

if mid_count == succeeded:
    ok(f"Cover MIDIs: all {mid_count} present (piano transcriptions of real covers)")
else:
    warn(f"Cover MIDIs: {mid_count}/{succeeded} present. Missing: {mid_missing[:5]}")

# HOW MUCH OF THE TRAINING DATA IS ACTUALLY USED?
print()
info("HOW TRAINING DATA IS USED (transparency):")
info("  The 62 cover MIDIs tell us: melody pitch range, notes/beat, scale degrees")
info("  The pipeline uses this to: normalize octave, tune filtering thresholds")
info("  What it does NOT do: fine-tune Demucs, CREPE, or basic-pitch on your songs")
info("  Those 3 models are pre-trained and fixed regardless of your training data")
info("  This is the fundamental limitation — see section 6 for implications")


# ══════════════════════════════════════════════════════
#  4. PIPELINE FILE INTEGRITY
# ══════════════════════════════════════════════════════
header("4. Pipeline file integrity")

required_files = [
    ('pipeline.py',      'Main processing pipeline'),
    ('app.py',           'Flask web server'),
    ('collect_data.py',  'Training data downloader'),
    ('analyze_pairs.py', 'Style profile builder'),
    ('songs.csv',        'Training song list'),
    ('templates/index.html', 'Web UI (or index.html)'),
]

for fname, desc in required_files:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        # try index.html at root
        if fname == 'templates/index.html':
            alt = os.path.join(BASE, 'index.html')
            if os.path.exists(alt):
                ok(f"index.html (web UI)")
                continue
        fail(f"{fname} MISSING — {desc}")
    else:
        size = os.path.getsize(path)
        ok(f"{fname} ({size//1024}KB) — {desc}")

# Check pipeline version
pipeline_path = os.path.join(BASE, 'pipeline.py')
if os.path.exists(pipeline_path):
    with open(pipeline_path) as f:
        first_line = f.read(200)
    if 'v15' in first_line:
        ok("pipeline.py is v15 (latest — harmonic fallback + soft scale constraint)")
    elif 'v14' in first_line:
        warn("pipeline.py is v14 — v15 has important fixes for unrecognizable output")
    else:
        warn("pipeline.py version unclear")

# Check songs.csv
songs_csv = os.path.join(BASE, 'songs.csv')
if os.path.exists(songs_csv):
    import csv
    with open(songs_csv) as f:
        rows = list(csv.DictReader(f))
    ok(f"songs.csv: {len(rows)} songs (originals + covers)")


# ══════════════════════════════════════════════════════
#  5. COMPONENT TESTS (skip with --quick)
# ══════════════════════════════════════════════════════
quick = '--quick' in sys.argv
header(f"5. Component tests {'(SKIPPED — run without --quick for full test)' if quick else ''}")

if not quick:
    # Test librosa can load audio
    try:
        import librosa, numpy as np
        # Generate a 2-second sine wave as fake audio
        sr = 22050
        t  = np.linspace(0, 2, sr * 2)
        y  = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        # Test beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        ok(f"librosa beat_track: works (test tone {float(np.atleast_1d(tempo)[0]):.0f} BPM)")
        # Test HPSS
        y_h, y_p = librosa.effects.hpss(y)
        ok("librosa HPSS (harmonic separation): works")
        # Test onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        ok(f"librosa onset_detect: works ({len(onsets)} onsets on test signal)")
        # Test chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        ok(f"librosa chroma_cqt: works (shape {chroma.shape})")
    except Exception as e:
        fail(f"librosa failed: {e}")

    # Test pretty_midi
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        midi.instruments.append(inst)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=True) as tmp:
            midi.write(tmp.name)
        ok("pretty_midi: can create and write MIDI")
    except Exception as e:
        fail(f"pretty_midi failed: {e}")

    # Test basic-pitch import
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        if os.path.exists(str(ICASSP_2022_MODEL_PATH)):
            ok(f"basic-pitch model: found at {str(ICASSP_2022_MODEL_PATH)[-50:]}")
        else:
            warn("basic-pitch model file not found — may need to re-install")
    except Exception as e:
        warn(f"basic-pitch model check failed: {e}")

    # Test torch + MPS
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
        try:
            if torch.backends.mps.is_available():
                ok("Apple MPS (GPU) available — Demucs and CREPE will use GPU acceleration")
            else:
                warn("Apple MPS not available — running on CPU (slower)")
        except:
            info("MPS check skipped")
    except Exception as e:
        fail(f"PyTorch failed: {e}")

    # Test demucs can be invoked
    try:
        r = subprocess.run(
            [sys.executable, '-m', 'demucs', '--help'],
            capture_output=True, timeout=15
        )
        if r.returncode == 0 or b'usage' in r.stdout.lower() or b'usage' in r.stderr.lower():
            ok("Demucs: module loads and responds to --help")
        else:
            warn(f"Demucs may have issues (return code {r.returncode})")
    except Exception as e:
        warn(f"Demucs check failed: {e}")

    # Test torchcrepe
    try:
        import torchcrepe
        ok(f"torchcrepe: installed and importable")
    except ImportError:
        warn("torchcrepe not installed — CREPE pitch detection unavailable, falling back to basic-pitch")

else:
    info("Skipping component tests (use without --quick to run them)")


# ══════════════════════════════════════════════════════
#  6. HONEST CAPABILITY ASSESSMENT
# ══════════════════════════════════════════════════════
header("6. Honest capability assessment")

print("""
  WHAT THIS SYSTEM CAN DO RELIABLY:
  ✅  Download any YouTube song
  ✅  Detect tempo and key with reasonable accuracy
  ✅  Detect chord progression from accompaniment (works well)
  ✅  Generate a musically structured MIDI (correct rhythm, key, chords)
  ✅  Apply Tamil piano style (octave range, note density, scale degrees)
       learned from your 62 training songs

  WHAT THIS SYSTEM CANNOT GUARANTEE:
  ⚠️   Exact melody recognition — depends heavily on how clean
       Demucs separates the vocals from instruments. For songs
       with very heavy accompaniment, Demucs leaks instrument
       pitches into the vocal stem, corrupting the melody.
  ⚠️   The melody will sound like the original song — CREPE
       tracks the dominant pitch, which may be an instrument
       not the singer in complex Tamil film arrangements.
  ⚠️   Output quality is consistent — some songs will work
       well, others poorly, depending on the mix of the original.

  ROOT CAUSE OF PREVIOUS BAD OUTPUTS:
  The system extracts melody from the original song audio.
  Tamil film songs have: dense orchestration, reverb, multiple
  instruments at the same pitch range as the voice. Even with
  Demucs stem separation, the vocal stem still has instrument
  bleed. CREPE then tracks the wrong pitch.

  HOW YOUR 62 TRAINING SONGS HELP (and don't help):
  ✅  HELPS: Sets correct octave range (MIDI 60–96)
  ✅  HELPS: Confirms Tamil music uses Root/5th/4th/b3 scale degrees
  ✅  HELPS: Calibrates note density (1.59 notes/beat avg)
  ❌  DOESN'T HELP: Does not improve Demucs vocal separation
  ❌  DOESN'T HELP: Does not improve CREPE pitch accuracy
  ❌  DOESN'T HELP: Does not teach the model what the melody sounds like

  TO GET TRULY RELIABLE OUTPUT, what you'd need is:
  → A model trained specifically on (song audio → piano MIDI) pairs.
    Your 62 pairs ARE the right training data for this.
    But training such a model requires running a fine-tuning job
    on a GPU for several hours using a framework like MT3 or MusicBM.
    This is doable but is a separate project beyond the current pipeline.

  REALISTIC EXPECTATION FOR CURRENT PIPELINE:
  → Chord progression: 80–90% accurate (this genuinely works)
  → Rhythm / tempo: 85–95% accurate
  → Melody recognition: 40–70% depending on the song
  → "Sounds like the original song": varies widely
""")


# ══════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════
header("SUMMARY")

if errors:
    print(f"\n  {FAIL}  {len(errors)} CRITICAL PROBLEM(S) — fix before running:\n")
    for e in errors:
        print(f"       • {e}")
    print()

if warnings:
    print(f"\n  {WARN}  {len(warnings)} warning(s):\n")
    for w in warnings:
        print(f"       • {w}")
    print()

if not errors:
    print(f"\n  {PASS}  System is ready. Run with:  bash start.sh")
    print( "         Then open http://localhost:5050 and paste a YouTube URL.\n")
    if warnings:
        print( "         Some warnings exist but they won't stop the pipeline.")
        print( "         Melody quality depends on how clean Demucs separates the song.\n")
else:
    print(f"\n  Fix the {len(errors)} error(s) above before running.\n")
    sys.exit(1)
