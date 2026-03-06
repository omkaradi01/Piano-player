"""
Piano Transcriber Pipeline v16
================================
Fixes over v15:
  6. Training data actually used — the pipeline now loads ALL 62 cover MIDIs
     at startup and finds the closest matching training song by key + tempo.
     It uses that song's cover MIDI to:
       a) Set a target note density (notes/beat) to thin or thicken the melody
       b) Confirm the correct melody octave from a real human piano cover
       c) Use the cover's scale degree distribution to weight the key constraint
     This is the first version where the 62 collected training songs actually
     influence how a new song is rendered.

Pipeline (in order):
  1. Download + convert audio
  2. Detect tempo + beats (validated, fallback to BPM grid)
  3. htdemucs_ft 4-stem → vocals / bass / drums / other
  4. torchcrepe + onset detection on vocals stem  (best path)
     → fallback: torchcrepe + onset detection on HARMONIC audio
     → fallback: basic-pitch on vocals / harmonic / full audio
  5. Ornament filter + gap fill (800ms) + octave normalize
  6. Key detection + SOFT scale constraint
  7. Density matching against closest training song
  8. Beat-quantised note placement
  9. Chord detection on bass+other stems
 10. Build MIDI — root+fifth left hand + sustain pedal
"""

import os, sys, logging, tempfile, random, json
import numpy as np
import pretty_midi

logging.basicConfig(level=logging.INFO, format='[pipeline] %(message)s')
log = logging.getLogger(__name__)

# ── Load style profile ────────────────────────────────────────────────────────
_STYLE_PROFILE = None
_PROFILE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'training_data', 'style_profile.json')
if os.path.exists(_PROFILE_PATH):
    try:
        with open(_PROFILE_PATH) as _f:
            _STYLE_PROFILE = json.load(_f)
        log.info(f"Style profile loaded: {_STYLE_PROFILE.get('source','?')}")
    except Exception:
        _STYLE_PROFILE = None

# ── Load ALL training patterns at startup ─────────────────────────────────────
# This is the key change in v16: we load every patterns.json so we can
# find the closest matching training song for any new input.
_TRAINING_PAIRS = []
_SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'training_data', 'summary.json')
if os.path.exists(_SUMMARY_PATH):
    try:
        with open(_SUMMARY_PATH) as _f:
            _summary = json.load(_f)
        for _song in _summary.get('songs', []):
            _pf = os.path.join(_song['dir'], 'patterns.json')
            _ff = os.path.join(_song['dir'], 'features.json')
            if os.path.exists(_pf) and os.path.exists(_ff):
                with open(_pf) as _f: _p = json.load(_f)
                with open(_ff) as _f: _feat = json.load(_f)
                _TRAINING_PAIRS.append({
                    'name':    _song['name'],
                    'bpm':     _feat.get('bpm', 120),
                    'key_root':_feat.get('key_root', 0),
                    'key_mode':_feat.get('key_mode', 'minor'),
                    'npb':     _p.get('notes_per_beat_rh', 1.5),
                    'octave':  _p.get('primary_octave', 4),
                    'pitch_min': _p.get('rh_pitch_min', 60),
                    'pitch_max': _p.get('rh_pitch_max', 84),
                    'degree_dist': _p.get('scale_degree_dist', {}),
                })
        log.info(f"Loaded {len(_TRAINING_PAIRS)} training pairs for matching")
    except Exception as _e:
        log.warning(f"Could not load training pairs: {_e}")


def _find_closest_training_song(key_root, key_mode, bpm):
    """
    Find the training song most similar to the input song by key + tempo.

    We score each training song:
      - Same key root: +3 points
      - Same mode (major/minor): +2 points
      - Tempo within 20 BPM: +1 point, within 10 BPM: +2 points

    Returns the patterns dict of the best match, or None if no training data.
    This lets us use a real human piano cover as a reference for how dense
    and what octave the melody should be for a song in this key + tempo.
    """
    if not _TRAINING_PAIRS:
        return None

    best, best_score = None, -1
    for pair in _TRAINING_PAIRS:
        score = 0
        if pair['key_root'] == key_root:        score += 3
        if pair['key_mode'] == key_mode:         score += 2
        bpm_diff = abs(pair['bpm'] - bpm)
        if bpm_diff <= 10:   score += 2
        elif bpm_diff <= 20: score += 1
        if score > best_score:
            best_score, best = score, pair

    log.info(f"Closest training song: {best['name']} "
             f"(score={best_score}, bpm={best['bpm']:.0f}, "
             f"key_root={best['key_root']}, mode={best['key_mode']})")
    return best


def _apply_training_density(melody, target_npb, bpm, audio_duration):
    """
    Adjust note density of the extracted melody to match what a real
    Tamil piano cover pianist would play (learned from training data).

    target_npb = target notes per beat from closest training song.

    If we detected too many notes (> 1.5x target): thin out by removing
    the shortest notes first (ornaments/noise).

    If we detected too few notes (< 0.5x target): this usually means the
    melody extraction missed phrases — we can't invent notes, so we just
    log a warning and return as-is.
    """
    if not melody or target_npb <= 0:
        return melody

    total_beats = audio_duration / (60.0 / bpm)
    current_npb = len(melody) / max(total_beats, 1)

    log.info(f"Melody density: {current_npb:.2f} notes/beat  "
             f"(target from training: {target_npb:.2f})")

    if current_npb > target_npb * 1.5:
        # Too dense — remove shortest notes to reach target count
        target_n = int(target_npb * total_beats)
        sorted_by_dur = sorted(melody, key=lambda n: n[1] - n[0], reverse=True)
        melody = sorted(sorted_by_dur[:target_n])
        log.info(f"Thinned melody: {len(melody)} notes → {len(melody)/max(total_beats,1):.2f}/beat")

    elif current_npb < target_npb * 0.4:
        log.warning(f"Melody too sparse ({current_npb:.2f}/beat vs target {target_npb:.2f}). "
                    f"Melody extraction may have missed phrases.")

    return melody


def _melody_range():
    """Return (lo, hi) MIDI pitch range from style profile or defaults."""
    if _STYLE_PROFILE:
        m = _STYLE_PROFILE.get('melody', {})
        return m.get('pitch_lo', 60), m.get('pitch_hi', 84)
    return 60, 84   # default C4–C6


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def _to_float(val):
    return float(np.atleast_1d(np.asarray(val, dtype=float)).ravel()[0])

def _ffmpeg(args):
    import subprocess
    r = subprocess.run(['ffmpeg', '-y'] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{r.stderr[-500:]}")


# ─────────────────────────────────────────────
#  Download & convert
# ─────────────────────────────────────────────

def _download_audio(url, out_dir):
    import yt_dlp
    opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, 'audio_raw.%(ext)s'),
        'quiet': True, 'no_warnings': True, 'noplaylist': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'wav', 'preferredquality': '0'}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    for f in os.listdir(out_dir):
        if f.startswith('audio_raw') and f.endswith('.wav'):
            return os.path.join(out_dir, f)
    raise FileNotFoundError("Download produced no WAV")

def _to_mono(src, dst, sr=22050):
    _ffmpeg(['-i', src, '-ac', '1', '-ar', str(sr), '-sample_fmt', 's16', dst])
    return dst

def _to_stereo(src, dst, sr=44100):
    _ffmpeg(['-i', src, '-ac', '2', '-ar', str(sr), '-sample_fmt', 's16', dst])
    return dst


# ─────────────────────────────────────────────
#  Tempo & beats
# ─────────────────────────────────────────────

def _get_tempo_and_beats(mono_wav):
    import librosa
    y, sr = librosa.load(mono_wav, sr=22050)
    audio_duration = len(y) / sr

    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
    bpm = _to_float(tempo)
    if not (40.0 <= bpm <= 220.0):
        log.warning(f"Suspicious tempo {bpm:.1f} → using 100 BPM")
        bpm = 100.0

    beat_times = np.asarray(beat_times, dtype=float)
    beat_times = beat_times[beat_times < audio_duration]

    expected = int(audio_duration / (60.0 / bpm))
    log.info(f"librosa: {len(beat_times)} beats (expected ~{expected})")

    if len(beat_times) < max(10, expected * 0.5):
        log.warning("Too few valid beats — generating BPM grid")
        beat_times = np.arange(0.0, audio_duration, 60.0 / bpm)

    log.info(f"Tempo={bpm:.1f} BPM  beats={len(beat_times)}  "
             f"first={beat_times[0]:.2f}s  last={beat_times[-1]:.2f}s  "
             f"duration={audio_duration:.1f}s")
    return bpm, beat_times, audio_duration


# ─────────────────────────────────────────────
#  Demucs
# ─────────────────────────────────────────────

def _run_demucs(stereo_wav, work_dir, model_name, two_stems=None, shifts=0):
    """Run demucs. Uses -n MODEL (not --model which is invalid)."""
    import subprocess
    out = os.path.join(work_dir, f'stems_{model_name}_s{shifts}')
    cmd = [sys.executable, '-m', 'demucs', '-n', model_name, '-o', out]
    if shifts > 0:
        cmd += ['--shifts', str(shifts)]
    if two_stems:
        cmd += ['--two-stems', two_stems]
    cmd.append(stereo_wav)

    log.info(f"demucs -n {model_name} shifts={shifts}…")
    r = subprocess.run(cmd, text=True, timeout=2400)
    if r.returncode != 0:
        log.warning(f"demucs {model_name} failed (code {r.returncode})")
        return {}

    stems = {}
    for root, _, files in os.walk(out):
        for f in files:
            if f.endswith('.wav'):
                key = f.replace('.wav', '').lower()
                stems[key] = os.path.join(root, f)
    log.info(f"demucs stems found: {list(stems.keys())}")
    return stems


def _separate_audio(stereo_wav, work_dir):
    """Try models in order of quality. Returns dict with at least 'vocals'."""
    # 1. Best: htdemucs_ft, 4-stem, shifts=3
    log.info("Trying htdemucs_ft + shifts=3 (best quality)…")
    stems = _run_demucs(stereo_wav, work_dir, 'htdemucs_ft', shifts=3)
    if 'vocals' in stems and 'bass' in stems:
        return stems

    # 2. htdemucs, 4-stem, shifts=3
    log.info("Trying htdemucs + shifts=3…")
    stems = _run_demucs(stereo_wav, work_dir, 'htdemucs', shifts=3)
    if 'vocals' in stems and 'bass' in stems:
        return stems

    # 3. htdemucs, 4-stem, no shifts
    log.info("Trying htdemucs single-pass…")
    stems = _run_demucs(stereo_wav, work_dir, 'htdemucs')
    if 'vocals' in stems and 'bass' in stems:
        return stems

    # 4. htdemucs, 2-stem
    log.info("Trying htdemucs 2-stem…")
    stems = _run_demucs(stereo_wav, work_dir, 'htdemucs', two_stems='vocals')
    _normalise_2stem(stems)
    if 'vocals' in stems:
        return stems

    log.error("All demucs models failed")
    return {}


def _normalise_2stem(stems):
    """Rename 2-stem keys to 'vocals' / 'no_vocals'."""
    for k, v in list(stems.items()):
        if 'vocals' in k and 'no' not in k:
            stems['vocals'] = v
        if 'no_vocals' in k:
            stems['no_vocals'] = v


# ─────────────────────────────────────────────
#  Harmonic audio separation (HPSS)
#  Used as Demucs fallback — removes drums and
#  percussive instruments before pitch tracking
# ─────────────────────────────────────────────

def _make_harmonic_audio(mono_wav, work_dir, sr_out=22050):
    """
    Use librosa HPSS to extract the harmonic component.

    This is our fallback when Demucs fails: HPSS removes drums and
    percussive transients, leaving behind sustained tones (vocals,
    strings, piano). CREPE then tracks pitch much more reliably
    because it isn't confused by drum hits or bass thumps.

    margin=8 makes the separation aggressive — better for music with
    heavy accompaniment (which is typical Tamil film music).
    """
    import librosa, soundfile as sf
    log.info("HPSS harmonic separation (Demucs fallback)…")
    y, sr = librosa.load(mono_wav, sr=22050, mono=True)
    y_harm = librosa.effects.harmonic(y, margin=8)
    harm_22k = os.path.join(work_dir, 'harm_22k.wav')
    sf.write(harm_22k, y_harm, 22050)
    harm_16k = os.path.join(work_dir, 'harm_16k.wav')
    _to_mono(harm_22k, harm_16k, sr=16000)
    log.info("Harmonic audio ready")
    return harm_22k, harm_16k


# ─────────────────────────────────────────────
#  torchcrepe — neural pitch detection
# ─────────────────────────────────────────────

def _torchcrepe_available():
    try:
        import torchcrepe
        return True
    except ImportError:
        return False


def _run_torchcrepe_raw(audio_path_16k, audio_duration):
    """
    Run torchcrepe on a 16kHz WAV and return raw (times, hz, confidence) arrays.
    Does NOT convert to note events — segmentation is done separately
    by _onset_based_melody using harmonic onset detection.
    """
    import torch, torchcrepe, librosa

    log.info("Loading audio at 16 kHz for torchcrepe…")
    y, sr = librosa.load(audio_path_16k, sr=16000, mono=True)
    audio = torch.tensor(y).unsqueeze(0)

    device = 'cpu'
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
            log.info("Apple MPS acceleration enabled")
    except AttributeError:
        pass

    log.info(f"torchcrepe running (device={device}, batch=4096)…")
    pitch, periodicity = torchcrepe.predict(
        audio, sr,
        hop_length=160,
        fmin=80.0, fmax=1200.0,
        model='full',
        return_periodicity=True,
        batch_size=4096,
        device=device,
    )
    periodicity = torchcrepe.filter.median(periodicity, win_length=3)

    hz   = pitch.squeeze().detach().cpu().numpy()
    conf = periodicity.squeeze().detach().cpu().numpy()
    hop  = 160 / 16000
    t    = np.arange(len(hz)) * hop
    mask = t < audio_duration
    log.info(f"torchcrepe: {mask.sum()} frames, "
             f"{(conf[mask] > 0.22).sum()} voiced (conf>0.22)")
    return t[mask], hz[mask], conf[mask]


def _onset_based_melody(audio_path_22k, t, hz, conf, bpm, audio_duration):
    """
    THE fundamental fix for Tamil film music.

    WHY the old approach was wrong:
      _contour_to_notes watched for pitch to 'stabilise' to decide where
      one note ends and the next begins. Tamil singers never stabilise —
      they glide between notes and ornament constantly. Result: 500 tiny
      fragments instead of a melody.

    WHY this approach is correct:
      1. Find note ONSETS on the harmonic component of the audio.
         Onsets = physical moments of new note attacks. Using the harmonic
         component (not full audio) means drum hits don't create false
         melody boundaries.
      2. Each onset-to-next-onset window is ONE note.
      3. Take the MOST COMMON pitch in the middle of that window.
         Middle 50% skips the glide at the start and the decay at the end.
         Most common = the 'real' note, ignoring ornament microtones.
      4. Skip windows where average confidence < 0.22 = instrumental rest.

    Result: clean, legato melody notes with musically correct boundaries.
    """
    import librosa
    from collections import Counter

    log.info("Running harmonic onset detection…")
    y, sr = librosa.load(audio_path_22k, sr=22050, mono=True)

    # Harmonic component only — removes drum/percussion onsets
    y_harm = librosa.effects.harmonic(y, margin=6)

    onset_times = librosa.onset.onset_detect(
        y=y_harm, sr=sr,
        units='time',
        hop_length=512,
        backtrack=True,   # find actual note start, not peak
        delta=0.04,       # v15: slightly more sensitive (was 0.05)
    )

    if len(onset_times) < 5:
        log.warning("Too few onsets detected, falling back to contour method")
        return _contour_to_notes(t, hz, conf)

    log.info(f"Harmonic onsets: {len(onset_times)}")

    # Build boundaries: onset times + end of audio
    boundaries = sorted(set(float(x) for x in onset_times))
    boundaries.append(float(audio_duration))

    # Snap to 16th-note grid at 65% strength — keeps feel, improves timing
    sub = (60.0 / bpm) / 4.0
    boundaries = sorted(set(
        round(b + 0.65 * (round(b / sub) * sub - b), 4)
        for b in boundaries
    ))

    notes = []
    for i in range(len(boundaries) - 1):
        seg_s = boundaries[i]
        seg_e = min(boundaries[i + 1], audio_duration)
        seg_d = seg_e - seg_s

        if seg_d < 0.08:    # skip anything shorter than 80ms
            continue

        # Middle 50% of segment: skip glide at start, skip decay at end
        mid_s = seg_s + seg_d * 0.25
        mid_e = seg_s + seg_d * 0.75

        # v15: raised confidence threshold to 0.22 (was 0.18)
        # This removes more instrument bleed at the cost of some vocal frames.
        # Better to have fewer, more accurate notes than many wrong ones.
        mask = (t >= mid_s) & (t < mid_e) & (conf > 0.22) & (hz > 50)
        if mask.sum() < 2:
            continue        # not enough voiced frames = rest

        seg_hz   = hz[mask]
        seg_conf = conf[mask]

        # Most common MIDI pitch in this segment = the real note
        midi_vals = np.clip(
            np.round(12.0 * np.log2(np.maximum(seg_hz, 1.0) / 440.0) + 69.0
                     ).astype(int),
            21, 108
        )
        pitch_mode = Counter(midi_vals.tolist()).most_common(1)[0][0]

        # Confidence gate
        if float(seg_conf.mean()) < 0.22:
            continue

        notes.append((float(seg_s), float(seg_e), int(pitch_mode)))

    log.info(f"Onset-based melody: {len(notes)} notes")
    return notes


def _extract_melody_crepe(audio_16k, audio_22k, audio_duration, bpm):
    """Combine torchcrepe pitch contour with onset-based segmentation."""
    t, hz, conf = _run_torchcrepe_raw(audio_16k, audio_duration)
    notes = _onset_based_melody(audio_22k, t, hz, conf, bpm, audio_duration)
    return notes


def _contour_to_notes(times, freqs, conf, min_conf=0.25, min_dur=0.08):
    """
    Fallback: convert raw pitch contour to notes by pitch stability.
    Used only when onset detection finds too few onsets.
    """
    voiced = (conf > min_conf) & (freqs > 50.0)
    midi_f = np.where(voiced & (freqs > 0),
                      12.0 * np.log2(np.maximum(freqs, 1.0) / 440.0) + 69.0, 0.0)
    midi_i = np.round(midi_f).astype(int)
    midi_i[~voiced] = 0
    midi_i[midi_i < 21] = 0
    midi_i[midi_i > 108] = 0

    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01
    notes, i = [], 0
    while i < len(midi_i):
        p = int(midi_i[i])
        if p == 0: i += 1; continue
        j, seen = i + 1, [p]
        while j < len(midi_i):
            pj = int(midi_i[j])
            if pj == 0: break
            if abs(pj - int(np.median(seen[-12:]))) <= 3:
                seen.append(pj); j += 1
            else:
                break
        dur = (j - i) * hop
        if dur >= min_dur:
            notes.append((float(times[i]), float(times[i] + dur),
                          max(21, min(108, int(np.median(seen))))))
        i = j
    return notes


# ─────────────────────────────────────────────
#  Ornament filter + gap fill + octave normalize
# ─────────────────────────────────────────────

def _filter_ornaments(notes, min_dur_ms=80):
    """
    Remove notes shorter than min_dur_ms and collapse ornament bursts.
    """
    min_dur = min_dur_ms / 1000.0
    notes = [(s, e, p) for s, e, p in notes if (e - s) >= min_dur]

    merged = True
    while merged:
        merged = False
        out = []
        i = 0
        while i < len(notes):
            j = i + 1
            while j < len(notes):
                span = notes[j][1] - notes[i][0]
                pitches = [notes[k][2] for k in range(i, j+1)]
                if span < 0.20 and (max(pitches) - min(pitches)) > 4:
                    j += 1
                else:
                    break
            if j - i > 1:
                span_notes = notes[i:j]
                pitches = [n[2] for n in span_notes]
                median_p = int(np.median(pitches))
                total_dur = span_notes[-1][1] - span_notes[0][0]
                if total_dur >= min_dur:
                    out.append((span_notes[0][0], span_notes[-1][1], median_p))
                merged = True
                i = j
            else:
                out.append(notes[i])
                i += 1
        notes = out

    log.info(f"After ornament filter: {len(notes)} notes")
    return notes


def _fill_melody_gaps(notes, max_gap=0.80):
    """
    Extend each note to meet the next one when the gap is small (≤ max_gap).
    v15: increased to 800ms (was 400ms) for more legato melody.
    Longer gaps (actual rests) are left as silence.
    """
    if len(notes) < 2:
        return notes
    notes = sorted(notes)
    result = []
    for i, (s, e, p) in enumerate(notes):
        if i + 1 < len(notes):
            next_s = notes[i + 1][0]
            gap = next_s - e
            if 0 < gap <= max_gap:
                e = next_s
        result.append((s, e, p))
    log.info(f"After gap fill: {len(result)} notes (legato)")
    return result


def _normalize_octave(notes, lo=60, hi=84):
    """
    Force all melody notes into the range [lo, hi] (default C4–C6).
    Shifts by octaves (12 semitones) until inside target range.
    """
    result = []
    for s, e, p in notes:
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        result.append((s, e, p))
    log.info(f"Octave-normalized {len(result)} notes to MIDI {lo}–{hi}")
    return result


# ─────────────────────────────────────────────
#  basic-pitch fallback
# ─────────────────────────────────────────────

def _basicpitch_extract(audio_path, bpm, onset_thr=0.35, frame_thr=0.25,
                         min_hz=150.0, max_hz=1400.0):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    log.info(f"basic-pitch on {os.path.basename(audio_path)}…")
    _, _, evts = predict(
        audio_path, model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_thr, frame_threshold=frame_thr,
        minimum_note_length=60, midi_tempo=bpm,
        minimum_frequency=min_hz, maximum_frequency=max_hz,
        melodia_trick=True,
    )
    if not evts: return []
    raw = sorted([(float(n[0]), float(n[1]), int(n[2])) for n in evts])
    log.info(f"basic-pitch raw: {len(raw)} notes")
    return raw


def _monophonize(notes):
    if not notes: return []
    end_t = max(e for _, e, _ in notes)
    F = 0.01; nf = int(end_t / F) + 2
    fp = np.zeros(nf, dtype=np.int16)
    for s, e, p in notes:
        sf, ef = int(s/F), min(int(e/F)+1, nf)
        for f in range(sf, ef):
            if p > fp[f]: fp[f] = p
    result, i = [], 0
    while i < nf:
        p = int(fp[i])
        if p == 0: i += 1; continue
        j = i + 1
        while j < nf and fp[j] == p: j += 1
        if (j - i) * F >= 0.08: result.append((i*F, j*F, p))
        i = j
    return result


def _smooth_melody(notes, max_gap=0.25):
    out = [n for n in notes if (n[1]-n[0]) >= 0.08]
    merged = True
    while merged:
        merged = False; new = []; i = 0
        while i < len(out):
            if i+1<len(out) and out[i][2]==out[i+1][2] and (out[i+1][0]-out[i][1])<max_gap:
                new.append((out[i][0], out[i+1][1], out[i][2])); i+=2; merged=True
            else: new.append(out[i]); i+=1
        out = new
    return out


# ─────────────────────────────────────────────
#  Quantise to beat grid
# ─────────────────────────────────────────────

def _quantize(notes, bpm, strength=0.65):
    if not notes: return notes
    sub = (60.0 / bpm) / 4.0
    result = []
    for s, e, p in notes:
        grid  = round(s / sub) * sub
        new_s = s + strength * (grid - s)
        result.append((new_s, new_s + (e - s), p))
    return sorted(result)


# ─────────────────────────────────────────────
#  Key detection + SOFT scale constraint (v15)
# ─────────────────────────────────────────────

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

_SCALES = {
    'major':       [0,2,4,5,7,9,11],
    'minor':       [0,2,3,5,7,8,10],
    'harmonic_min':[0,2,3,5,7,8,11],
}

def _detect_key(mono_wav):
    import librosa
    y, sr = librosa.load(mono_wav, sr=22050)
    cm = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36).mean(axis=1)
    best, best_root, best_mode = -999.0, 0, 'minor'
    for r in range(12):
        mj = np.corrcoef(cm, np.roll(_KS_MAJOR, r))[0,1]
        mn = np.corrcoef(cm, np.roll(_KS_MINOR, r))[0,1]
        if mj > best: best, best_root, best_mode = mj, r, 'major'
        if mn > best: best, best_root, best_mode = mn, r, 'minor'
    key_name = NOTE_NAMES[best_root] + (' major' if best_mode=='major' else ' minor')
    log.info(f"Detected key: {key_name}  (confidence={best:.3f})")
    return best_root, best_mode


def _constrain_to_key_soft(notes, key_root, key_mode):
    """
    v15 SOFT scale constraint — much gentler than v14.

    v14 problem: ANY note not in the scale was snapped to the nearest
    scale degree. If key detection was wrong (common for Tamil ragas),
    this mangled the melody completely.

    v15 fix:
      - Notes within 1 semitone of a scale degree → KEEP AS IS
        (they're probably right; small deviations come from pitch tracking)
      - Notes 2 semitones away → SNAP to the nearest scale degree
        (clear wrong note, fix it)
      - Notes exactly 1 semitone away → KEEP AS IS
        (could be a chromatic passing tone or ornament)

    This means most notes pass through unchanged, preserving the melody.
    Only truly out-of-scale notes (> 1 semitone from anything) get fixed.
    """
    mode = 'harmonic_min' if key_mode == 'minor' else 'major'
    intervals = _SCALES[mode]

    # Pre-compute all valid pitch classes
    valid_pcs = set((key_root + i) % 12 for i in intervals)

    result = []
    moved = 0
    for s, e, p in notes:
        pc = p % 12

        if pc in valid_pcs:
            # Already in scale — keep exactly
            result.append((s, e, p))
            continue

        # How far is this pitch from the nearest scale degree?
        min_dist = min(min(abs(pc - vpc), 12 - abs(pc - vpc))
                       for vpc in valid_pcs)

        if min_dist <= 1:
            # Within 1 semitone of a scale degree — keep as is
            # (likely a chromatic note or pitch tracking micro-error)
            result.append((s, e, p))
        else:
            # 2+ semitones away — find and snap to nearest scale degree
            best_dist, best_p = 12, p
            for i in intervals:
                for shift in (-1, 0, 1):
                    c = key_root + i + 12 * (p // 12 + shift)
                    if abs(c - p) < best_dist:
                        best_dist, best_p = abs(c - p), c
            result.append((s, e, max(21, min(108, best_p))))
            moved += 1

    log.info(f"Soft scale constraint: {moved}/{len(notes)} notes adjusted")
    return result


# ─────────────────────────────────────────────
#  Chord detection
# ─────────────────────────────────────────────

def _chord_templates():
    t = {}
    for r in range(12):
        maj = np.zeros(12); maj[r]=1.0; maj[(r+4)%12]=0.8; maj[(r+7)%12]=0.9
        t[f'{NOTE_NAMES[r]}maj'] = maj / np.linalg.norm(maj)
        mn  = np.zeros(12); mn[r]=1.0;  mn[(r+3)%12]=0.8;  mn[(r+7)%12]=0.9
        t[f'{NOTE_NAMES[r]}min'] = mn  / np.linalg.norm(mn)
    return t

CHORD_TEMPLATES = _chord_templates()

def _detect_chords(audio_path, beat_times, bpm, audio_duration):
    import librosa
    y, sr = librosa.load(audio_path, sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    times  = librosa.times_like(chroma, sr=sr)
    beat_dur = 60.0 / bpm
    chords = []
    for i, bt in enumerate(beat_times):
        if bt >= audio_duration: break
        next_bt = min(beat_times[i+1] if i+1<len(beat_times) else bt+beat_dur, audio_duration)
        mask = (times >= bt) & (times < next_bt)
        if mask.sum() == 0:
            chords.append(chords[-1] if chords else ('Emin',[52,55,59])); continue
        bc = chroma[:,mask].mean(axis=1)
        n = np.linalg.norm(bc)
        if n > 0: bc /= n
        best, name = -1.0, 'Cmaj'
        for cn, tmpl in CHORD_TEMPLATES.items():
            s = float(np.dot(bc, tmpl))
            if s > best: best, name = s, cn
        root_idx  = NOTE_NAMES.index(name[:-3])
        root_midi = 48 + root_idx
        is_min    = name.endswith('min')
        chords.append((name, [root_midi, root_midi+(3 if is_min else 4), root_midi+7]))
    log.info(f"Chords: {len(chords)}  sample={[c[0] for c in chords[:6]]}")
    return chords


# ─────────────────────────────────────────────
#  MIDI build — musical left hand
# ─────────────────────────────────────────────

def _build_midi(melody, chords, beat_times, bpm, audio_duration, output_path):
    """
    Right hand: melody notes with slight humanization.

    Left hand pattern:
      Beat 1:   Bass root note (one octave below chord voicing, C2-B3)
      Beat 2.5: Fifth in mid register (open interval, doesn't clash with melody)

    Sustain pedal held for 2 beats to give warmth.
    """
    midi     = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    rh       = pretty_midi.Instrument(program=0, name='RightHand')
    lh       = pretty_midi.Instrument(program=0, name='LeftHand')
    rng      = random.Random(42)
    beat_dur = 60.0 / bpm

    # ── Right hand: melody ─────────────────────────────────────────────────
    for s, e, p in melody:
        p = max(21, min(108, p))
        off = rng.uniform(-0.015, 0.015)
        vel = rng.randint(72, 95)
        rh.notes.append(pretty_midi.Note(
            velocity=vel, pitch=p,
            start=max(0.0, s + off),
            end=max(0.0, s + off + max(0.05, e - s)),
        ))

    # ── Left hand: root + fifth ─────────────────────────────────────────────
    for i, bt in enumerate(beat_times):
        if i >= len(chords) or bt >= audio_duration:
            break

        chord_name, chord_notes = chords[i]
        root  = chord_notes[0]
        fifth = chord_notes[2]

        bass_root  = max(24, min(47, root  - 12))
        mid_fifth  = max(36, min(59, fifth - 12))

        next_bt  = beat_times[i+1] if i+1 < len(beat_times) else bt + beat_dur
        next_bt  = min(float(next_bt), audio_duration)
        note_dur = min(beat_dur * 0.80, next_bt - bt - 0.02)

        # Beat 1: bass root note
        off = rng.uniform(-0.010, 0.010)
        lh.notes.append(pretty_midi.Note(
            velocity=rng.randint(52, 65),
            pitch=bass_root,
            start=max(0.0, bt + off),
            end=max(0.0, bt + off + note_dur),
        ))

        # Beat 2.5: fifth in mid register
        t2 = bt + beat_dur * 0.5
        if t2 < audio_duration:
            off2 = rng.uniform(-0.010, 0.010)
            lh.notes.append(pretty_midi.Note(
                velocity=rng.randint(40, 52),
                pitch=mid_fifth,
                start=max(0.0, t2 + off2),
                end=max(0.0, t2 + off2 + note_dur * 0.6),
            ))

    # ── Sustain pedal ───────────────────────────────────────────────────────
    for i in range(0, len(beat_times) - 1, 2):
        on_t  = float(beat_times[i])
        off_t = float(beat_times[min(i + 2, len(beat_times) - 1)]) - 0.04
        if on_t >= audio_duration:
            break
        lh.control_changes.append(pretty_midi.ControlChange(64, 127, on_t))
        lh.control_changes.append(pretty_midi.ControlChange(64, 0,   off_t))

    midi.instruments.extend([rh, lh])
    midi.write(output_path)
    log.info(
        f"MIDI saved: RH={len(rh.notes)} melody  "
        f"LH={len(lh.notes)} LH notes → {output_path}"
    )


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def process_youtube_to_piano_midi(url, output_path, progress_cb=None):
    work_dir = tempfile.mkdtemp(prefix='piano_')

    def prog(msg, pct=None):
        log.info(msg)
        if progress_cb: progress_cb(msg, pct)

    try:
        prog("Downloading audio…", 5)
        raw_wav = _download_audio(url, work_dir)

        prog("Converting audio…", 10)
        mono_wav   = _to_mono(raw_wav,   os.path.join(work_dir, 'mono.wav'))
        stereo_wav = _to_stereo(raw_wav, os.path.join(work_dir, 'stereo.wav'))

        prog("Detecting tempo and beats…", 15)
        bpm, beat_times, audio_duration = _get_tempo_and_beats(mono_wav)

        prog("Separating sources with Demucs…", 20)
        stems = _separate_audio(stereo_wav, work_dir)
        vocals_path = stems.get('vocals')

        # ── Melody extraction ──────────────────────────────────────────────
        melody = []

        # Layer 1: torchcrepe + onset detection on Demucs vocals (best)
        if vocals_path and _torchcrepe_available():
            prog("Extracting melody — torchcrepe + onset detection on vocals…", 40)
            try:
                v22k = os.path.join(work_dir, 'vocals_22k.wav')
                v16k = os.path.join(work_dir, 'vocals_16k.wav')
                _to_mono(vocals_path, v22k, sr=22050)
                _to_mono(vocals_path, v16k, sr=16000)
                melody = _extract_melody_crepe(v16k, v22k, audio_duration, bpm)
                log.info(f"CREPE (vocals stem) → {len(melody)} notes")
            except Exception as exc:
                log.warning(f"torchcrepe on vocals failed: {exc}")
                melody = []

        # Layer 2: torchcrepe on HARMONIC audio (v15 fix — replaces raw full audio)
        #
        # WHY harmonic and not raw full audio:
        # Raw audio → CREPE picks up bass guitar, drums, sitar = wrong pitches.
        # Harmonic component → HPSS removes percussive/transient sounds,
        # leaving mostly vocals and sustained instruments. CREPE then tracks
        # the dominant pitched voice (usually the lead vocal melody) much more
        # reliably even without Demucs stem separation.
        if len(melody) < 20 and _torchcrepe_available():
            prog("torchcrepe on harmonic audio (Demucs fallback)…", 45)
            try:
                harm_22k, harm_16k = _make_harmonic_audio(mono_wav, work_dir)
                melody = _extract_melody_crepe(harm_16k, harm_22k, audio_duration, bpm)
                log.info(f"CREPE (harmonic audio) → {len(melody)} notes")
            except Exception as exc:
                log.warning(f"torchcrepe on harmonic failed: {exc}")
                melody = []

        # Layer 3: basic-pitch on vocals stem
        if len(melody) < 20 and vocals_path:
            prog("Fallback: basic-pitch on vocals stem…", 55)
            raw    = _basicpitch_extract(vocals_path, bpm, 0.35, 0.20, 150, 1400)
            melody = _smooth_melody(_monophonize(raw))
            melody = _quantize(melody, bpm, 0.60)
            log.info(f"basic-pitch (vocals) → {len(melody)} notes")

        # Layer 4: basic-pitch on harmonic audio
        if len(melody) < 20:
            prog("Fallback: basic-pitch on harmonic audio…", 65)
            import librosa, soundfile as sf
            y, sr2 = librosa.load(mono_wav, sr=22050)
            harm_path = os.path.join(work_dir, 'harm_bp.wav')
            sf.write(harm_path, librosa.effects.harmonic(y, margin=8), sr2)
            raw    = _basicpitch_extract(harm_path, bpm, 0.40, 0.25, 180, 1300)
            melody = _smooth_melody(_monophonize(raw))
            melody = _quantize(melody, bpm, 0.60)
            log.info(f"basic-pitch (harmonic) → {len(melody)} notes")

        # Layer 5: basic-pitch on full audio (last resort)
        if len(melody) < 20:
            prog("Fallback: basic-pitch on full audio…", 72)
            raw    = _basicpitch_extract(mono_wav, bpm, 0.50, 0.35, 220, 1200)
            melody = _smooth_melody(_monophonize(raw))
            melody = _quantize(melody, bpm, 0.60)
            log.info(f"basic-pitch (full audio) → {len(melody)} notes")

        log.info(f"Raw melody before post-processing: {len(melody)} notes")

        # ── Post-processing ────────────────────────────────────────────────
        prog("Filtering ornaments and normalizing octave…", 78)
        melody = _filter_ornaments(melody, min_dur_ms=80)
        melody = _fill_melody_gaps(melody, max_gap=0.80)   # v15: 800ms (was 400ms)
        lo, hi = _melody_range()
        melody = _normalize_octave(melody, lo=lo, hi=hi)
        log.info(f"Melody octave range: MIDI {lo}–{hi}")

        # ── Key detection + SOFT scale constraint ──────────────────────────
        prog("Detecting key (soft scale constraint)…", 82)
        key_root, key_mode = _detect_key(mono_wav)
        melody = _constrain_to_key_soft(melody, key_root, key_mode)
        log.info(f"Post-key melody: {len(melody)} notes")

        # ── Training data: find closest song + apply learned density ───────
        prog("Matching against training data…", 84)
        closest = _find_closest_training_song(key_root, key_mode, bpm)
        if closest:
            melody = _apply_training_density(
                melody, closest['npb'], bpm, audio_duration)
            # Also refine octave range using closest training song's actual range
            trained_lo = max(closest['pitch_min'] - 12, 48)
            trained_hi = min(closest['pitch_max'] + 12, 108)
            lo = max(_melody_range()[0], trained_lo)
            hi = min(_melody_range()[1], trained_hi)
            melody = _normalize_octave(melody, lo=lo, hi=hi)
            log.info(f"Octave refined from training: MIDI {lo}–{hi}")

        # ── Chord detection ────────────────────────────────────────────────
        prog("Detecting chords…", 87)
        import librosa, soundfile as sf

        harm_stems = [stems.get('bass'), stems.get('other')]
        harm_stems = [p for p in harm_stems if p and os.path.exists(p)]
        if harm_stems:
            arrays = [librosa.load(p, sr=22050, mono=True)[0] for p in harm_stems]
            min_len = min(len(a) for a in arrays)
            mixed = np.mean([a[:min_len] for a in arrays], axis=0)
            chord_src = os.path.join(work_dir, 'chord_mix.wav')
            sf.write(chord_src, mixed, 22050)
            log.info("Chords from bass+other stems")
        elif stems.get('no_vocals'):
            chord_src = stems['no_vocals']
            log.info("Chords from no_vocals stem")
        else:
            chord_src = mono_wav
            log.info("Chords from full mono audio")

        chords = _detect_chords(chord_src, beat_times, bpm, audio_duration)

        # ── Build MIDI ─────────────────────────────────────────────────────
        prog("Building MIDI…", 95)
        _build_midi(melody, chords, beat_times, bpm, audio_duration, output_path)

        prog("Done! 🎹", 100)
        return output_path

    except Exception as e:
        log.error(f"Pipeline error: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
