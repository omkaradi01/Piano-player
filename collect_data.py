#!/usr/bin/env python3
"""
collect_data.py — Download and transcribe Tamil song + piano cover pairs.

What this does for each pair:
  1. Downloads the original song audio from YouTube
  2. Downloads the piano cover audio from YouTube
  3. Transcribes the piano cover → MIDI  (very accurate on solo piano)
  4. Extracts musical features from the original (key, tempo, chroma)
  5. Saves everything neatly to training_data/

Usage:
  cd ~/Desktop/"Piano player"
  source venv311/bin/activate
  python collect_data.py songs.csv

songs.csv format:
  name,original_url,cover_url
  kannazhaga,https://youtube.com/...,https://youtube.com/...
"""

import os, sys, json, csv
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


# ─────────────────────────────────────────────
#  Download
# ─────────────────────────────────────────────

def download_audio(url, out_wav):
    """Download audio from YouTube and convert to WAV."""
    import yt_dlp
    opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_wav.replace('.wav', '.%(ext)s'),
        'quiet': True, 'no_warnings': True, 'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.exists(out_wav):
        raise FileNotFoundError(f"Download failed: {out_wav} not found after yt_dlp")


# ─────────────────────────────────────────────
#  Trim non-musical intro/outro
# ─────────────────────────────────────────────

def trim_to_music(wav_path, out_path, is_piano_cover=False):
    """
    Detect and trim talking/silence/advertisement at the start and end.

    For original songs: finds where the music starts by detecting when
    sustained harmonic energy begins (talking voices have a very different
    spectral shape from music).

    For piano covers: finds where piano notes begin by detecting the first
    significant onset, and trims the last few seconds of decay/talking.

    Uses a conservative approach — only trims if there's clear evidence of
    non-musical content. Never trims more than 30s from start or 10s from end.
    """
    import librosa, soundfile as sf

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    duration = len(y) / sr

    # Compute short-time energy in harmonic component
    y_harm = librosa.effects.harmonic(y, margin=4)
    frame_len = 2048
    hop_len   = 512
    rms = librosa.feature.rms(y=y_harm, frame_length=frame_len, hop_length=hop_len)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    # Threshold: 15% of the median RMS of the whole track
    rms_threshold = float(np.median(rms)) * 0.15

    # Find first frame where harmonic energy exceeds threshold
    start_trim = 0.0
    for i, (t, r) in enumerate(zip(times, rms)):
        if r > rms_threshold:
            # Confirm it stays above threshold for 1s (not a transient)
            end_confirm = min(i + int(sr / hop_len), len(rms))
            if np.mean(rms[i:end_confirm]) > rms_threshold:
                start_trim = max(0.0, t - 0.2)  # keep 0.2s before onset
                break

    # Find last frame above threshold
    end_trim = duration
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] > rms_threshold:
            end_trim = min(duration, times[i] + 0.5)
            break

    # Safety limits: never cut more than 30s from start, 10s from end
    start_trim = min(start_trim, 30.0)
    end_trim   = max(end_trim, duration - 10.0)

    if start_trim > 0.5 or end_trim < duration - 0.5:
        print(f"    Trimmed: {start_trim:.1f}s – {end_trim:.1f}s "
              f"(removed {start_trim:.1f}s intro, "
              f"{duration - end_trim:.1f}s outro)")
    else:
        print(f"    No trim needed ({duration:.1f}s)")

    start_sample = int(start_trim * sr)
    end_sample   = int(end_trim   * sr)
    y_trimmed    = y[start_sample:end_sample]

    sf.write(out_path, y_trimmed, sr)
    return float(end_trim - start_trim)


# ─────────────────────────────────────────────
#  Extract piano from mixed cover (Demucs)
# ─────────────────────────────────────────────

def extract_piano_from_mixed_cover(cover_wav, out_wav, song_dir):
    """
    When a piano cover also has drums, bass, or other instruments,
    run Demucs to isolate just the piano/melodic content.

    Demucs separates audio into: vocals, drums, bass, other.
    Piano falls into the 'other' stem. Removing drums and bass
    gives us a much cleaner piano signal for transcription.

    This is the same Demucs we use in the main pipeline, so it
    will already be installed and cached.
    """
    import subprocess, sys

    stems_dir = os.path.join(song_dir, 'cover_stems')
    os.makedirs(stems_dir, exist_ok=True)

    print(f"    Running Demucs on mixed cover to isolate piano…")
    cmd = [
        sys.executable, '-m', 'demucs',
        '-n', 'htdemucs',          # 4-stem model
        '-o', stems_dir,
        '--two-stems', 'other',    # extract 'other' (piano/keys) vs rest
        cover_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ⚠  Demucs failed — using full cover audio instead")
        print(f"       {result.stderr[-300:]}")
        import shutil
        shutil.copy(cover_wav, out_wav)
        return False

    # Find the 'other' stem
    other_path = None
    for root, _, files in os.walk(stems_dir):
        for f in files:
            if 'other' in f.lower() and f.endswith('.wav') and 'no_' not in f.lower():
                other_path = os.path.join(root, f)
                break

    if not other_path:
        print(f"    ⚠  Could not find 'other' stem — using full cover")
        import shutil
        shutil.copy(cover_wav, out_wav)
        return False

    # Convert to mono 22kHz for consistency
    import subprocess as sp
    sp.run(['ffmpeg', '-y', '-i', other_path,
            '-ac', '1', '-ar', '22050', '-sample_fmt', 's16', out_wav],
           capture_output=True)

    print(f"    ✓ Piano stem extracted → {os.path.basename(out_wav)}")
    return True


# ─────────────────────────────────────────────
#  Transcribe piano cover → MIDI
# ─────────────────────────────────────────────

def transcribe_piano_cover(cover_wav, output_mid, bpm=120.0):
    """
    Transcribe solo piano audio to MIDI using basic-pitch.

    Solo piano is fundamentally easier to transcribe than mixed music:
    - Single instrument, no source separation needed
    - Clear attack transients (easy onset detection)
    - No overlapping harmonics from multiple instruments

    basic-pitch achieves 85-95% note accuracy on solo piano, which is
    excellent quality for training data.

    Key difference from how we use basic-pitch in the main pipeline:
    - melodia_trick=False  → piano is POLYPHONIC (we want all notes, not just melody)
    - onset_threshold=0.5  → higher threshold = fewer false positives on piano
    - full frequency range → piano goes A0 (27.5Hz) to C8 (4186Hz)
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    print(f"    Transcribing piano cover ({os.path.basename(cover_wav)})…")
    _, midi_data, note_events = predict(
        cover_wav,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.50,
        frame_threshold=0.30,
        minimum_note_length=50,    # 50ms — piano has clear short notes
        midi_tempo=bpm,
        minimum_frequency=27.5,    # A0 (lowest piano key)
        maximum_frequency=4186.0,  # C8 (highest piano key)
        melodia_trick=False,       # keep polyphonic — piano plays chords too
    )

    midi_data.write(output_mid)
    total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    rh_notes = [n for inst in midi_data.instruments for n in inst.notes if n.pitch >= 60]
    lh_notes = [n for inst in midi_data.instruments for n in inst.notes if n.pitch < 60]
    print(f"    → {total_notes} notes  (RH ≥ C4: {len(rh_notes)},  LH < C4: {len(lh_notes)})")
    return total_notes


# ─────────────────────────────────────────────
#  Extract features from original song
# ─────────────────────────────────────────────

_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def extract_song_features(original_wav):
    """
    Extract musical features from the original song that will be used
    later to align it with the piano cover MIDI for pattern learning.
    """
    import librosa

    print(f"    Extracting features from original…")
    y, sr = librosa.load(original_wav, sr=22050)
    duration = len(y) / sr

    # Tempo + beats
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
    bpm = float(np.atleast_1d(np.asarray(tempo, dtype=float)).ravel()[0])
    if not (40.0 <= bpm <= 220.0):
        bpm = 100.0
    beat_times = np.asarray(beat_times, dtype=float)
    beat_times = beat_times[beat_times < duration]

    # Chroma (pitch class distribution across whole song)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    chroma_mean = chroma.mean(axis=1)

    # Key detection (Krumhansl-Schmuckler)
    best, best_root, best_mode = -999.0, 0, 'minor'
    for r in range(12):
        mj = float(np.corrcoef(chroma_mean, np.roll(_KS_MAJOR, r))[0, 1])
        mn = float(np.corrcoef(chroma_mean, np.roll(_KS_MINOR, r))[0, 1])
        if mj > best: best, best_root, best_mode = mj, r, 'major'
        if mn > best: best, best_root, best_mode = mn, r, 'minor'
    key_name = NOTE_NAMES[best_root] + (' major' if best_mode == 'major' else ' minor')

    # Beat-level chroma (for chord analysis later)
    beat_chroma = []
    times_arr = librosa.times_like(chroma, sr=sr)
    for i, bt in enumerate(beat_times[:-1]):
        next_bt = beat_times[i + 1]
        mask = (times_arr >= bt) & (times_arr < next_bt)
        if mask.sum() > 0:
            bc = chroma[:, mask].mean(axis=1)
            beat_chroma.append([round(float(x), 4) for x in bc])

    features = {
        'bpm':         round(bpm, 2),
        'duration':    round(duration, 2),
        'beat_count':  int(len(beat_times)),
        'key':         key_name,
        'key_root':    int(best_root),
        'key_mode':    best_mode,
        'key_confidence': round(float(best), 3),
        'chroma_mean': [round(float(x), 4) for x in chroma_mean],
        'beat_chroma': beat_chroma,
    }

    print(f"    → Key: {key_name} (conf={best:.3f})  BPM: {bpm:.1f}  "
          f"Duration: {duration:.1f}s  Beats: {len(beat_times)}")
    return features


# ─────────────────────────────────────────────
#  Analyse MIDI: extract style patterns
# ─────────────────────────────────────────────

def analyse_midi(cover_mid, features):
    """
    Extract style patterns from the transcribed piano cover MIDI.
    These patterns capture how Tamil piano cover pianists actually play:
    - Which scale degrees they emphasise
    - Typical note durations
    - Left hand patterns and voicings
    - Note density per beat
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(cover_mid)
    all_notes = sorted(
        [n for inst in pm.instruments for n in inst.notes],
        key=lambda n: n.start
    )
    if not all_notes:
        return {}

    key_root = features.get('key_root', 0)
    bpm      = features.get('bpm', 120.0)
    beat_dur = 60.0 / bpm

    # Separate into RH (melody, ≥ C4 = MIDI 60) and LH (< C4)
    rh = [n for n in all_notes if n.pitch >= 60]
    lh = [n for n in all_notes if n.pitch < 60]

    # Scale degree usage in RH melody (relative to key root)
    scale_degrees = [(n.pitch - key_root) % 12 for n in rh]
    degree_counts = {i: 0 for i in range(12)}
    for d in scale_degrees:
        degree_counts[d] += 1
    total = max(sum(degree_counts.values()), 1)
    degree_dist = {str(k): round(v / total, 4) for k, v in degree_counts.items()}

    # Note duration distribution (RH)
    rh_durs = [(n.end - n.start) for n in rh]
    rh_dur_beats = [d / beat_dur for d in rh_durs]

    # Note density: average RH notes per beat
    total_dur = features.get('duration', 1.0)
    total_beats = max(features.get('beat_count', 1), 1)
    notes_per_beat = round(len(rh) / total_beats, 3)

    # LH interval patterns (most common intervals)
    lh_pitches = [n.pitch for n in lh]
    lh_intervals = []
    for i in range(len(lh_pitches) - 1):
        lh_intervals.append(abs(lh_pitches[i + 1] - lh_pitches[i]))

    # RH octave range
    rh_pitches = [n.pitch for n in rh]
    octave_dist = {str(p // 12 - 1): 0 for p in range(21, 109)}
    for p in rh_pitches:
        oct_key = str(p // 12 - 1)
        if oct_key in octave_dist:
            octave_dist[oct_key] += 1

    # Most common melody octave
    if rh_pitches:
        median_pitch = int(np.median(rh_pitches))
        primary_octave = median_pitch // 12 - 1
    else:
        primary_octave = 4

    patterns = {
        'rh_note_count':      len(rh),
        'lh_note_count':      len(lh),
        'notes_per_beat_rh':  notes_per_beat,
        'scale_degree_dist':  degree_dist,
        'rh_dur_mean_beats':  round(float(np.mean(rh_dur_beats)) if rh_dur_beats else 0.5, 3),
        'rh_dur_median_beats':round(float(np.median(rh_dur_beats)) if rh_dur_beats else 0.5, 3),
        'primary_octave':     primary_octave,
        'rh_pitch_min':       int(min(rh_pitches)) if rh_pitches else 60,
        'rh_pitch_max':       int(max(rh_pitches)) if rh_pitches else 84,
        'rh_pitch_median':    int(np.median(rh_pitches)) if rh_pitches else 72,
        'lh_common_intervals':sorted(set(lh_intervals))[:10] if lh_intervals else [],
    }
    return patterns


# ─────────────────────────────────────────────
#  Process one song-cover pair
# ─────────────────────────────────────────────

def process_pair(name, original_url, cover_url, song_dir, cover_type='piano'):
    os.makedirs(song_dir, exist_ok=True)

    original_wav = os.path.join(song_dir, 'original.wav')
    cover_wav    = os.path.join(song_dir, 'cover.wav')
    cover_mid    = os.path.join(song_dir, 'cover.mid')
    features_f   = os.path.join(song_dir, 'features.json')
    patterns_f   = os.path.join(song_dir, 'patterns.json')

    original_raw = os.path.join(song_dir, 'original_raw.wav')
    cover_raw    = os.path.join(song_dir, 'cover_raw.wav')

    # 1. Download original
    if not os.path.exists(original_raw):
        print(f"  ↓ Downloading original song…")
        download_audio(original_url, original_raw)
    else:
        print(f"  ✓ Original already downloaded")

    # 2. Trim original (removes ads/talking intro)
    if not os.path.exists(original_wav):
        print(f"  ✂  Trimming original (removing non-musical intro/outro)…")
        trim_to_music(original_raw, original_wav, is_piano_cover=False)
    else:
        print(f"  ✓ Original already trimmed")

    # 3. Download cover
    if not os.path.exists(cover_raw):
        print(f"  ↓ Downloading piano cover…")
        download_audio(cover_url, cover_raw)
    else:
        print(f"  ✓ Cover already downloaded")

    # 4. Process cover audio
    cover_trimmed = os.path.join(song_dir, 'cover_trimmed.wav')
    if cover_type == 'mixed':
        # Mixed cover: first trim talking, then extract piano via Demucs
        if not os.path.exists(cover_trimmed):
            print(f"  ✂  Trimming mixed cover…")
            trim_to_music(cover_raw, cover_trimmed, is_piano_cover=True)
        else:
            print(f"  ✓ Cover already trimmed")
        if not os.path.exists(cover_wav):
            print(f"  🎹 Extracting piano stem from mixed cover (Demucs)…")
            extract_piano_from_mixed_cover(cover_trimmed, cover_wav, song_dir)
        else:
            print(f"  ✓ Piano stem already extracted")
    else:
        # Clean piano cover: just trim talking intro/outro
        if not os.path.exists(cover_wav):
            print(f"  ✂  Trimming cover (removing talking intro/outro)…")
            trim_to_music(cover_raw, cover_wav, is_piano_cover=True)
        else:
            print(f"  ✓ Cover already trimmed")

    # 3. Extract features from original
    if not os.path.exists(features_f):
        features = extract_song_features(original_wav)
        with open(features_f, 'w') as f:
            json.dump(features, f, indent=2)
    else:
        with open(features_f) as f:
            features = json.load(f)
        print(f"  ✓ Features already extracted  (key={features['key']}, bpm={features['bpm']})")

    # 4. Transcribe piano cover → MIDI
    if not os.path.exists(cover_mid):
        transcribe_piano_cover(cover_wav, cover_mid, features.get('bpm', 120.0))
    else:
        print(f"  ✓ Cover MIDI already transcribed")

    # 5. Analyse style patterns from MIDI
    if not os.path.exists(patterns_f):
        patterns = analyse_midi(cover_mid, features)
        with open(patterns_f, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"  ✓ Patterns extracted  "
              f"(RH notes/beat={patterns.get('notes_per_beat_rh','?')}, "
              f"octave={patterns.get('primary_octave','?')})")
    else:
        print(f"  ✓ Patterns already analysed")

    return features


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'songs.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        pairs = [r for r in reader
                 if r.get('original_url','').strip() and r.get('cover_url','').strip()]

    if not pairs:
        print("No rows with URLs found in songs.csv — fill in the original_url and cover_url columns.")
        sys.exit(1)

    mixed = [r['name'] for r in pairs if r.get('cover_type','piano').strip() == 'mixed']
    print(f"\n🎹  Processing {len(pairs)} song-cover pairs")
    if mixed:
        print(f"    Mixed covers (Demucs piano extraction): {', '.join(mixed)}")
    print(f"    Data will be saved to: {DATA_DIR}\n")

    results = []
    failed  = []

    for i, row in enumerate(pairs, 1):
        name         = row['name'].strip().replace(' ', '_').lower()
        original_url = row['original_url'].strip()
        cover_url    = row['cover_url'].strip()
        cover_type   = row.get('cover_type', 'piano').strip() or 'piano'
        song_dir     = os.path.join(DATA_DIR, f"{i:02d}_{name}")

        label = f"[mixed → Demucs]" if cover_type == 'mixed' else ""
        print(f"[{i}/{len(pairs)}]  {name}  {label}")
        try:
            features = process_pair(name, original_url, cover_url, song_dir, cover_type)
            results.append({'name': name, 'dir': song_dir, 'cover_type': cover_type, **features})
            print(f"  ✅ Done\n")
        except Exception as e:
            import traceback
            print(f"  ❌ Failed: {e}\n")
            traceback.print_exc()
            failed.append(name)

    # Save summary
    summary = {
        'total':      len(pairs),
        'succeeded':  len(results),
        'failed':     failed,
        'songs':      results,
    }
    summary_path = os.path.join(DATA_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"{'─'*50}")
    print(f"✅  {len(results)}/{len(pairs)} pairs processed successfully")
    if failed:
        print(f"❌  Failed: {', '.join(failed)}")
    print(f"    Summary saved to: {summary_path}")
    print(f"\nNext step:  python analyze_pairs.py")


if __name__ == '__main__':
    main()
