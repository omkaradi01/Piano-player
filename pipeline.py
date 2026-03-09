"""
Piano Transcriber Pipeline v18
================================
MAJOR REWRITE: Switch from monophonic pitch tracking to polyphonic transcription.

Key change: basic-pitch (Spotify, ONNX) for polyphonic note transcription
replaces the RMVPE → onset → pitch-contour chain that produced flat,
repetitive results.

Pipeline:
  1. Download + convert audio
  2. Detect tempo + beats (beat_this → librosa)
  3. Source separation (Mel-Band RoFormer → htdemucs_ft)
  4. basic-pitch on vocal stem → RH melody (polyphonic!)
  5. basic-pitch on accompaniment stem → LH harmony notes
  6. Raga/Key detection + soft scale constraint
  7. Beat-quantized note placement
  8. Build MIDI — RH melody + LH from actual transcribed harmony
"""

import os, sys, logging, tempfile, random, json
import numpy as np
import pretty_midi

# Fix scipy.signal.gaussian removed in scipy >= 1.14
try:
    import scipy.signal
    if not hasattr(scipy.signal, 'gaussian'):
        from scipy.signal.windows import gaussian as _gaussian
        scipy.signal.gaussian = _gaussian
except Exception:
    pass

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
    if best:
        log.info(f"Closest training song: {best['name']} "
                 f"(score={best_score}, bpm={best['bpm']:.0f}, "
                 f"key_root={best['key_root']}, mode={best['key_mode']})")
    return best


def _apply_training_density(melody, target_npb, bpm, audio_duration):
    if not melody or target_npb <= 0:
        return melody
    total_beats = audio_duration / (60.0 / bpm)
    current_npb = len(melody) / max(total_beats, 1)
    log.info(f"Melody density: {current_npb:.2f} notes/beat  "
             f"(target from training: {target_npb:.2f})")
    if current_npb > target_npb * 1.5:
        target_n = int(target_npb * total_beats)
        sorted_by_dur = sorted(melody, key=lambda n: n[1] - n[0], reverse=True)
        melody = sorted(sorted_by_dur[:target_n])
        log.info(f"Thinned melody: {len(melody)} notes")
    elif current_npb < target_npb * 0.4:
        log.warning(f"Melody too sparse ({current_npb:.2f}/beat vs target {target_npb:.2f})")
    return melody


def _melody_range():
    if _STYLE_PROFILE:
        m = _STYLE_PROFILE.get('melody', {})
        return m.get('pitch_lo', 48), m.get('pitch_hi', 96)
    return 48, 96


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
#  Tempo & beats  (beat_this → librosa fallback)
# ─────────────────────────────────────────────

def _beat_this_available():
    try:
        import beat_this
        return True
    except ImportError:
        return False


def _get_tempo_and_beats(mono_wav):
    """Detect tempo and beat positions. Uses beat_this (SOTA) with librosa fallback."""
    import librosa
    y, sr = librosa.load(mono_wav, sr=22050)
    audio_duration = len(y) / sr

    # Try beat_this first (ISMIR 2024 SOTA — handles tempo changes, odd meters)
    if _beat_this_available():
        try:
            return _get_beats_beat_this(mono_wav, audio_duration)
        except Exception as exc:
            log.warning(f"beat_this failed: {exc}, falling back to librosa")

    # Fallback: librosa
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
    bpm = _to_float(tempo)
    if not (40.0 <= bpm <= 220.0):
        log.warning(f"Suspicious tempo {bpm:.1f} → using 100 BPM")
        bpm = 100.0

    beat_times = np.asarray(beat_times, dtype=float)
    beat_times = beat_times[beat_times < audio_duration]

    expected = int(audio_duration / (60.0 / bpm))
    if len(beat_times) < max(10, expected * 0.5):
        log.warning("Too few valid beats — generating BPM grid")
        beat_times = np.arange(0.0, audio_duration, 60.0 / bpm)

    log.info(f"librosa: Tempo={bpm:.1f} BPM  beats={len(beat_times)}  "
             f"duration={audio_duration:.1f}s")
    return bpm, beat_times, audio_duration


def _get_beats_beat_this(mono_wav, audio_duration):
    """Use beat_this for SOTA beat tracking."""
    import torch
    from beat_this.inference import File2Beats

    device = 'cpu'
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
    except AttributeError:
        pass

    log.info(f"beat_this running (device={device})…")
    file2beats = File2Beats(device=device, dbn=False)
    beats, downbeats = file2beats(mono_wav)

    beat_times = np.asarray(beats, dtype=float)
    beat_times = beat_times[beat_times < audio_duration]

    if len(beat_times) < 5:
        raise ValueError("beat_this returned too few beats")

    # Estimate BPM from median inter-beat interval
    ibis = np.diff(beat_times)
    median_ibi = float(np.median(ibis))
    bpm = 60.0 / median_ibi if median_ibi > 0 else 120.0
    bpm = max(40.0, min(220.0, bpm))

    log.info(f"beat_this: Tempo≈{bpm:.1f} BPM  beats={len(beat_times)}  "
             f"duration={audio_duration:.1f}s")
    return bpm, beat_times, audio_duration


# ─────────────────────────────────────────────
#  Source separation (htdemucs_ft, with future
#  Mel-Band RoFormer slot)
# ─────────────────────────────────────────────

def _run_demucs(stereo_wav, work_dir, model_name, two_stems=None, shifts=0):
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


def _audio_separator_available():
    try:
        from audio_separator.separator import Separator
        return True
    except ImportError:
        return False


def _separate_audio_roformer(stereo_wav, work_dir):
    """Use Mel-Band RoFormer via audio-separator (SDR ~12.6, SOTA)."""
    from audio_separator.separator import Separator

    out_dir = os.path.join(work_dir, 'roformer_stems')
    os.makedirs(out_dir, exist_ok=True)

    log.info("Mel-Band RoFormer separation (SOTA)…")
    separator = Separator(output_dir=out_dir, output_format='wav')
    separator.load_model(model_filename="mel_band_roformer_kim_ft_unwa.ckpt")
    output_files = separator.separate(stereo_wav)

    stems = {}
    for f in output_files:
        # Ensure full path
        if not os.path.isabs(f):
            f = os.path.join(out_dir, f)
        fl = os.path.basename(f).lower()
        if 'vocal' in fl and 'no_vocal' not in fl and 'instrument' not in fl:
            stems['vocals'] = f
        elif 'instrument' in fl or 'no_vocal' in fl or 'other' in fl:
            stems['no_vocals'] = f

    log.info(f"RoFormer stems: {list(stems.keys())} paths={list(stems.values())}")
    return stems


def _separate_audio(stereo_wav, work_dir):
    """Source separation. Mel-Band RoFormer (best) → htdemucs_ft fallback."""
    # 1. Try Mel-Band RoFormer (SDR ~12.6 — SOTA vocal separation)
    if _audio_separator_available():
        try:
            stems = _separate_audio_roformer(stereo_wav, work_dir)
            if 'vocals' in stems:
                return stems
        except Exception as exc:
            log.warning(f"Mel-Band RoFormer failed: {exc}, falling back to demucs")

    # 2. htdemucs_ft, 4-stem, no shifts
    log.info("Trying htdemucs_ft single-pass…")
    stems = _run_demucs(stereo_wav, work_dir, 'htdemucs_ft')
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

    log.error("All separation models failed")
    return {}


def _normalise_2stem(stems):
    for k, v in list(stems.items()):
        if 'vocals' in k and 'no' not in k:
            stems['vocals'] = v
        if 'no_vocals' in k:
            stems['no_vocals'] = v


# ─────────────────────────────────────────────
#  Harmonic audio separation (HPSS fallback)
# ─────────────────────────────────────────────

def _make_harmonic_audio(mono_wav, work_dir):
    import librosa, soundfile as sf
    log.info("HPSS harmonic separation…")
    y, sr = librosa.load(mono_wav, sr=22050, mono=True)
    y_harm = librosa.effects.harmonic(y, margin=8)
    harm_22k = os.path.join(work_dir, 'harm_22k.wav')
    sf.write(harm_22k, y_harm, 22050)
    harm_16k = os.path.join(work_dir, 'harm_16k.wav')
    _to_mono(harm_22k, harm_16k, sr=16000)
    return harm_22k, harm_16k


# ─────────────────────────────────────────────
#  Pitch tracking (RMVPE → torchcrepe fallback)
# ─────────────────────────────────────────────

def _rmvpe_available():
    try:
        from rmvpe import RMVPE
        return True
    except ImportError:
        return False

def _torchcrepe_available():
    try:
        import torchcrepe
        return True
    except ImportError:
        return False


def _run_rmvpe_raw(audio_path_16k, audio_duration):
    """Run RMVPE — SOTA vocal pitch tracker for polyphonic music."""
    import torch
    from rmvpe import RMVPE

    device = 'cpu'
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
    except AttributeError:
        pass

    log.info(f"RMVPE running (device={device})…")
    model = RMVPE(device=device)
    f0 = model.infer_from_audio(audio_path_16k, sample_rate=16000)
    # f0 is numpy array of Hz values, 0 = unvoiced

    hop = 160 / 16000  # standard RMVPE hop
    t = np.arange(len(f0)) * hop
    mask = t < audio_duration
    t, f0 = t[mask], f0[mask]
    conf = np.where(f0 > 0, 0.9, 0.0)  # RMVPE doesn't output confidence, use binary

    log.info(f"RMVPE: {mask.sum()} frames, {(f0 > 0).sum()} voiced")
    return t, f0, conf


def _run_torchcrepe_raw(audio_path_16k, audio_duration):
    """Run torchcrepe on a 16kHz WAV."""
    import torch, torchcrepe, librosa

    y, sr = librosa.load(audio_path_16k, sr=16000, mono=True)
    audio = torch.tensor(y).unsqueeze(0)

    device = 'cpu'
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
    except AttributeError:
        pass

    log.info(f"torchcrepe running (device={device})…")
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
             f"{(conf[mask] > 0.22).sum()} voiced")
    return t[mask], hz[mask], conf[mask]


# ─────────────────────────────────────────────
#  Polyphonic transcription (basic-pitch)
# ─────────────────────────────────────────────

def _basic_pitch_available():
    try:
        from basic_pitch.inference import predict
        return True
    except ImportError:
        return False


def _transcribe_basic_pitch(audio_path, onset_thresh=0.45, frame_thresh=0.25,
                            min_note_ms=100, min_freq=None, max_freq=None):
    """
    v18: Use basic-pitch (Spotify) for polyphonic transcription.
    Returns list of (start, end, pitch, velocity) tuples.
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    log.info(f"basic-pitch transcribing: {audio_path}")
    model_output, midi_data, note_events = predict(
        audio_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_note_ms,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        melodia_trick=True,
    )

    # note_events: [(start_s, end_s, pitch_midi, amplitude, pitch_bend)]
    notes = []
    for ev in note_events:
        start, end, pitch, amp = ev[0], ev[1], ev[2], ev[3]
        vel = int(np.clip(amp * 127, 30, 127))
        if end - start >= 0.05:  # minimum 50ms
            notes.append((float(start), float(end), int(pitch), vel))

    log.info(f"basic-pitch: {len(notes)} notes transcribed")
    return notes


def _clean_melody_smart(notes, min_dur=0.08, min_pitch=48, dedup_window=0.20,
                        max_jump=19):
    """
    Context-aware melody cleanup — preserves musicality.
    1. Pitch floor + duration floor (relaxed to keep important notes)
    2. De-duplicate rapid repeated pitches (keep longest within window)
    3. Smart monophonic: prefer melodic continuity (smallest pitch jump)
       instead of blindly picking highest note
    4. Merge very short gaps between same pitch (legato smoothing)
    """
    if not notes:
        return []

    # Step 1: pitch floor + duration floor (relaxed)
    notes = [(s, e, p, v) for s, e, p, v in notes
             if p >= min_pitch and (e - s) >= min_dur]
    log.info(f"  After pitch/dur filter: {len(notes)}")

    # Step 2: de-duplicate rapid repeated pitches
    notes = sorted(notes, key=lambda n: n[0])
    deduped = []
    for s, e, p, v in notes:
        dup = False
        for ds, de, dp, dv in reversed(deduped[-5:]):
            if dp == p and abs(s - ds) < dedup_window:
                if (e - s) > (de - ds):
                    deduped.remove((ds, de, dp, dv))
                    deduped.append((s, e, p, v))
                dup = True
                break
        if not dup:
            deduped.append((s, e, p, v))
    notes = deduped
    log.info(f"  After dedup: {len(notes)}")

    # Step 3: smart monophonic — melodic continuity
    # Group overlapping notes, then pick the one closest to previous pitch
    notes = sorted(notes, key=lambda n: n[0])
    mono = []
    prev_pitch = None

    i = 0
    while i < len(notes):
        s, e, p, v = notes[i]
        # Collect all notes that overlap with this one
        cluster = [(s, e, p, v)]
        j = i + 1
        while j < len(notes) and notes[j][0] < e:
            cluster.append(notes[j])
            # Extend the overlap window
            e = max(e, notes[j][1])
            j += 1

        if len(cluster) == 1:
            chosen = cluster[0]
        else:
            # Pick the note with best melodic continuity
            if prev_pitch is not None:
                # Score: prefer small interval, but also prefer louder & longer
                def score(n):
                    interval = abs(n[2] - prev_pitch)
                    duration = n[1] - n[0]
                    loudness = n[3]
                    # Penalize big jumps, reward duration and velocity
                    return -interval * 2 + duration * 10 + loudness * 0.1
                chosen = max(cluster, key=score)
            else:
                # First note: pick loudest and longest
                chosen = max(cluster, key=lambda n: (n[1] - n[0]) * n[3])

        # Filter out wild jumps (> max_jump semitones from prev)
        if prev_pitch is not None and abs(chosen[2] - prev_pitch) > max_jump:
            # Try to find a better candidate in the cluster
            close = [n for n in cluster if abs(n[2] - prev_pitch) <= max_jump]
            if close:
                chosen = max(close, key=lambda n: (n[1] - n[0]) * n[3])
            # else keep the chosen one anyway — might be a legitimate octave jump

        # Truncate previous note if it overlaps
        if mono and mono[-1][1] > chosen[0]:
            ps, pe, pp, pv = mono[-1]
            mono[-1] = (ps, chosen[0], pp, pv)

        mono.append(chosen)
        prev_pitch = chosen[2]
        i = j if j > i else i + 1

    # Step 4: merge same-pitch notes with tiny gaps (legato smoothing)
    merged = [mono[0]] if mono else []
    for s, e, p, v in mono[1:]:
        ps, pe, pp, pv = merged[-1]
        if pp == p and (s - pe) < 0.08:  # tiny gap, same pitch → merge
            merged[-1] = (ps, e, pp, max(pv, v))
        else:
            merged.append((s, e, p, v))

    # Remove notes that got truncated too short
    merged = [(s, e, p, v) for s, e, p, v in merged if (e - s) >= min_dur]
    log.info(f"  After smart monophonic: {len(merged)}")
    return sorted(merged, key=lambda n: n[0])


def _fill_gaps_from_full(melody, full_notes, min_gap=1.5, min_pitch=55):
    """Fill vocal silence gaps with melody from full audio transcription."""
    if not full_notes or not melody:
        return melody

    # Clean full audio notes too
    full_clean = _clean_melody_smart(full_notes, min_dur=0.15, min_pitch=min_pitch)

    melody_sorted = sorted(melody, key=lambda n: n[0])
    gaps = []

    # Start gap
    if melody_sorted[0][0] > min_gap:
        gaps.append((0.0, melody_sorted[0][0]))

    # Interior gaps
    for i in range(len(melody_sorted) - 1):
        gap_s = melody_sorted[i][1]
        gap_e = melody_sorted[i + 1][0]
        if gap_e - gap_s > min_gap:
            gaps.append((gap_s, gap_e))

    if not gaps:
        return melody

    fill = []
    for gs, ge in gaps:
        fill.extend([n for n in full_clean if n[0] >= gs and n[1] <= ge])

    merged = sorted(melody + fill, key=lambda n: n[0])
    log.info(f"Gap fill: {len(melody)} melody + {len(fill)} fill = {len(merged)}")
    return merged


def _notes_to_melody_tuples(notes_4):
    """Convert 4-tuples (s, e, p, vel) to 3-tuples (s, e, p) for legacy functions."""
    return [(s, e, p) for s, e, p, v in notes_4]


def _notes_to_lh_chords(harmony_notes, beat_times, bpm, audio_duration):
    """
    Convert transcribed harmony notes into per-beat chord voicings for LH.
    Groups notes by beat window and finds the most common pitches.
    """
    if not harmony_notes or len(beat_times) == 0:
        return []

    beat_dur = 60.0 / bpm
    chords = []

    for i, bt in enumerate(beat_times):
        if bt >= audio_duration:
            break
        next_bt = float(beat_times[i+1]) if i+1 < len(beat_times) else bt + beat_dur
        next_bt = min(next_bt, audio_duration)

        # Find notes active during this beat
        active = []
        for s, e, p, v in harmony_notes:
            # Note overlaps with this beat window
            if s < next_bt and e > bt:
                active.append(p)

        if not active:
            # Use previous chord or default
            if chords:
                chords.append(chords[-1])
            else:
                chords.append(('N', [48, 52, 55]))
            continue

        # Find the most common pitch classes
        from collections import Counter
        pcs = Counter([p % 12 for p in active])
        top_pcs = [pc for pc, _ in pcs.most_common(3)]

        # Build chord from actual transcribed pitches, centered around bass range
        bass_oct = 3  # octave 3 = MIDI 36-47
        chord_notes = sorted([pc + 12 * bass_oct for pc in top_pcs])
        # Ensure bass is in range
        while chord_notes[0] < 36:
            chord_notes = [p + 12 for p in chord_notes]
        while chord_notes[0] > 55:
            chord_notes = [p - 12 for p in chord_notes]

        # Name it
        root_pc = top_pcs[0]
        name = NOTE_NAMES[root_pc] + ('min' if len(top_pcs) >= 2 and
               (top_pcs[1] - top_pcs[0]) % 12 == 3 else 'maj')
        chords.append((name, chord_notes))

    log.info(f"Harmony → {len(chords)} beat chords from transcription")
    return chords


# ─────────────────────────────────────────────
#  Note segmentation (pitch-contour + onset) [fallback]
# ─────────────────────────────────────────────

def _pitch_contour_to_notes(t, hz, conf, bpm, audio_duration, min_conf=0.22):
    """
    v17: Pitch-contour-based note segmentation.

    Instead of relying solely on energy onsets (which fail for Tamil gamakas),
    segment the pitch contour into stable-pitch regions:
    1. Find frames where pitch is voiced (conf > threshold, hz > 50)
    2. Group consecutive frames with similar pitch (within 1 semitone)
    3. Each group = one note, pitch = median of the group
    4. Merge adjacent notes with same pitch
    """
    from collections import Counter

    # Convert hz to MIDI
    voiced = (conf > min_conf) & (hz > 50)
    midi_f = np.where(voiced, 12.0 * np.log2(np.maximum(hz, 1.0) / 440.0) + 69.0, 0.0)
    midi_i = np.round(midi_f).astype(int)
    midi_i[~voiced] = 0
    midi_i[(midi_i < 21) | (midi_i > 108)] = 0

    hop = float(t[1] - t[0]) if len(t) > 1 else 0.01
    notes = []
    i = 0
    while i < len(midi_i):
        p = int(midi_i[i])
        if p == 0:
            i += 1
            continue
        # Collect contiguous frames within ±1 semitone
        j = i + 1
        pitches = [p]
        while j < len(midi_i):
            pj = int(midi_i[j])
            if pj == 0:
                break
            # Allow ±1 semitone from running median (gamaka tolerance)
            running_med = int(np.median(pitches[-20:]))
            if abs(pj - running_med) <= 1:
                pitches.append(pj)
                j += 1
            else:
                break

        dur = (j - i) * hop
        if dur >= 0.08:
            # Use mode (most common pitch) — stable target note
            final_pitch = Counter(pitches).most_common(1)[0][0]
            notes.append((float(t[i]), float(t[i]) + dur, int(final_pitch)))
        i = j

    log.info(f"Pitch-contour segmentation: {len(notes)} notes")
    return notes


def _onset_based_melody(audio_path_22k, t, hz, conf, bpm, audio_duration):
    """Onset-based segmentation with gamaka awareness."""
    import librosa
    from collections import Counter

    log.info("Running harmonic onset detection…")
    y, sr = librosa.load(audio_path_22k, sr=22050, mono=True)
    y_harm = librosa.effects.harmonic(y, margin=6)

    onset_times = librosa.onset.onset_detect(
        y=y_harm, sr=sr, units='time', hop_length=512,
        backtrack=True, delta=0.04,
    )

    if len(onset_times) < 5:
        log.warning("Too few onsets, using pitch-contour segmentation")
        return _pitch_contour_to_notes(t, hz, conf, bpm, audio_duration)

    log.info(f"Harmonic onsets: {len(onset_times)}")

    boundaries = sorted(set(float(x) for x in onset_times))
    boundaries.append(float(audio_duration))

    notes = []
    for i in range(len(boundaries) - 1):
        seg_s = boundaries[i]
        seg_e = min(boundaries[i + 1], audio_duration)
        seg_d = seg_e - seg_s

        if seg_d < 0.08:
            continue

        mid_s = seg_s + seg_d * 0.25
        mid_e = seg_s + seg_d * 0.75

        mask = (t >= mid_s) & (t < mid_e) & (conf > 0.22) & (hz > 50)
        if mask.sum() < 2:
            continue

        seg_hz   = hz[mask]
        seg_conf = conf[mask]

        midi_vals = np.clip(
            np.round(12.0 * np.log2(np.maximum(seg_hz, 1.0) / 440.0) + 69.0).astype(int),
            21, 108
        )
        pitch_mode = Counter(midi_vals.tolist()).most_common(1)[0][0]

        if float(seg_conf.mean()) < 0.22:
            continue

        notes.append((float(seg_s), float(seg_e), int(pitch_mode)))

    log.info(f"Onset-based melody: {len(notes)} notes")
    return notes


def _extract_melody(audio_16k, audio_22k, audio_duration, bpm):
    """Extract melody using best available pitch tracker + onset segmentation."""
    # Try RMVPE first (SOTA for vocal pitch in polyphonic music)
    if _rmvpe_available():
        try:
            log.info("Using RMVPE for pitch tracking…")
            t, hz, conf = _run_rmvpe_raw(audio_16k, audio_duration)
            notes = _onset_based_melody(audio_22k, t, hz, conf, bpm, audio_duration)
            if len(notes) >= 10:
                return notes
            # Try pitch-contour segmentation if onset-based gave too few
            notes = _pitch_contour_to_notes(t, hz, conf, bpm, audio_duration)
            if len(notes) >= 10:
                return notes
        except Exception as exc:
            log.warning(f"RMVPE failed: {exc}")

    # Fallback: torchcrepe
    if _torchcrepe_available():
        try:
            log.info("Using torchcrepe for pitch tracking…")
            t, hz, conf = _run_torchcrepe_raw(audio_16k, audio_duration)
            notes = _onset_based_melody(audio_22k, t, hz, conf, bpm, audio_duration)
            if len(notes) >= 10:
                return notes
            notes = _pitch_contour_to_notes(t, hz, conf, bpm, audio_duration)
            if len(notes) >= 10:
                return notes
        except Exception as exc:
            log.warning(f"torchcrepe failed: {exc}")

    log.warning("No pitch tracker produced enough notes")
    return []


# ─────────────────────────────────────────────
#  Gamaka-aware ornament filter + gap fill
# ─────────────────────────────────────────────

def _filter_ornaments_gamaka(notes, min_dur_ms=80):
    """
    v17: Gamaka-aware ornament filter.
    Detects oscillatory patterns (pitch going back and forth around a central note)
    and collapses them to the central pitch. This is critical for Tamil vocal music
    where gamakas (ornamental oscillations) are pervasive.
    """
    min_dur = min_dur_ms / 1000.0
    notes = [(s, e, p) for s, e, p in notes if (e - s) >= min_dur]

    if len(notes) < 3:
        return notes

    # Pass 1: Detect oscillatory patterns (A-B-A or A-B-A-B around central pitch)
    merged = True
    while merged:
        merged = False
        out = []
        i = 0
        while i < len(notes):
            # Look ahead for oscillation pattern
            if i + 2 < len(notes):
                span = notes[i + 2][1] - notes[i][0]
                p0, p1, p2 = notes[i][2], notes[i+1][2], notes[i+2][2]
                # A-B-A pattern where B is within 3 semitones of A
                if span < 0.5 and abs(p0 - p2) <= 1 and abs(p1 - p0) <= 3:
                    # Collapse to the target note (A)
                    central = p0
                    end_idx = i + 2
                    # Extend: keep absorbing if pattern continues
                    while end_idx + 1 < len(notes):
                        nxt = notes[end_idx + 1]
                        if (nxt[1] - notes[i][0]) > 0.8:
                            break
                        if abs(nxt[2] - central) <= 3:
                            end_idx += 1
                        else:
                            break
                    out.append((notes[i][0], notes[end_idx][1], central))
                    i = end_idx + 1
                    merged = True
                    continue

            # Check for short ornament bursts (rapid notes spanning small pitch range)
            j = i + 1
            while j < len(notes):
                span = notes[j][1] - notes[i][0]
                pitches = [notes[k][2] for k in range(i, j+1)]
                if span < 0.25 and (max(pitches) - min(pitches)) > 4:
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

    log.info(f"After gamaka filter: {len(notes)} notes")
    return notes


def _fill_melody_gaps(notes, max_gap=0.80):
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
    return result


def _normalize_octave(notes, lo=60, hi=84):
    result = []
    for s, e, p in notes:
        while p < lo: p += 12
        while p > hi: p -= 12
        result.append((s, e, p))
    return result


# ─────────────────────────────────────────────
#  Multi-resolution quantization (v17)
# ─────────────────────────────────────────────

def _quantize_multi(notes, bpm, beat_times):
    """
    v17: Multi-resolution quantization.
    Try 8th notes, 16th notes, AND triplet 8ths — snap each note to whichever
    grid point is closest. Adaptive strength: stronger near beats, weaker between.
    """
    if not notes:
        return notes

    beat_dur = 60.0 / bpm
    # Build three grids
    grids = {
        'eighth':   beat_dur / 2.0,
        'sixteenth': beat_dur / 4.0,
        'triplet':  beat_dur / 3.0,
    }

    # Create a set of beat positions for proximity check
    beat_set = set(float(b) for b in beat_times)
    beat_arr = np.array(sorted(beat_set)) if beat_set else np.array([0.0])

    result = []
    for s, e, p in notes:
        dur = e - s

        # Find closest grid point across all three grids
        best_grid_s = s
        best_dist = float('inf')
        for grid_name, sub in grids.items():
            snapped = round(s / sub) * sub
            dist = abs(snapped - s)
            if dist < best_dist:
                best_dist = dist
                best_grid_s = snapped

        # Adaptive strength: stronger near beats
        nearest_beat_dist = float(np.min(np.abs(beat_arr - s)))
        if nearest_beat_dist < beat_dur * 0.15:
            strength = 0.80  # near a beat — strong quantization
        elif nearest_beat_dist < beat_dur * 0.35:
            strength = 0.55  # moderate
        else:
            strength = 0.30  # between beats — light quantization

        # Very short notes (grace notes, gamakas) — don't quantize
        if dur < 0.10:
            strength = 0.0

        new_s = s + strength * (best_grid_s - s)
        result.append((new_s, new_s + dur, p))

    return sorted(result)


# ─────────────────────────────────────────────
#  Key detection — Raga-aware (v17)
# ─────────────────────────────────────────────

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# v17: Extended scale dictionary with Carnatic ragas
_SCALES = {
    # Western
    'major':            [0,2,4,5,7,9,11],
    'minor':            [0,2,3,5,7,8,10],
    'harmonic_min':     [0,2,3,5,7,8,11],
    # Carnatic ragas (most common in Tamil film music)
    'shankarabharanam':  [0,2,4,5,7,9,11],  # = Ionian/major
    'kharaharapriya':    [0,2,3,5,7,9,10],  # = Dorian
    'kalyani':           [0,2,4,6,7,9,11],  # = Lydian
    'mohanam':           [0,2,4,7,9],        # pentatonic
    'mayamalavagowla':   [0,1,4,5,7,8,11],  # exotic
    'harikambhoji':      [0,2,4,5,7,9,10],  # = Mixolydian
    'thodi':             [0,1,3,5,7,8,10],  # phrygian-like
    'hindolam':          [0,3,5,8,10],       # pentatonic minor
    'natabhairavi':      [0,2,3,5,7,8,10],  # = Aeolian/natural minor
    'charukesi':         [0,2,4,5,7,8,10],
    'hamsadhwani':       [0,2,4,7,11],       # pentatonic
    'abhogi':            [0,2,3,5,9],        # pentatonic
    'valaji':            [0,2,7,9,11],       # pentatonic
    'bilahari':          [0,2,4,7,9],        # = Mohanam ascending
}

# Build raga chroma profiles for correlation-based detection
_RAGA_PROFILES = {}
for _raga_name, _intervals in _SCALES.items():
    _prof = np.zeros(12)
    for _iv in _intervals:
        _prof[_iv % 12] = 1.0
    # Weight: root strongest, fifth next
    if len(_intervals) >= 5:
        _prof[_intervals[0]] = 2.0   # root (Sa)
        _prof[_intervals[4] % 12] = 1.5  # fifth (Pa) if present
    _RAGA_PROFILES[_raga_name] = _prof / np.linalg.norm(_prof)


def _detect_key_raga(mono_wav):
    """
    v17: Raga-aware key detection.
    Correlates chroma against ALL profiles (Western + Carnatic ragas).
    Returns (root, mode_name, scale_intervals).
    """
    import librosa
    y, sr = librosa.load(mono_wav, sr=22050)
    cm = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36).mean(axis=1)

    best_corr, best_root, best_raga = -999.0, 0, 'minor'
    for r in range(12):
        rotated_cm = np.roll(cm, -r)
        for raga_name, profile in _RAGA_PROFILES.items():
            corr = float(np.corrcoef(rotated_cm, profile)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_root = r
                best_raga = raga_name

    # Also try K-S major/minor as sanity check
    for r in range(12):
        mj = float(np.corrcoef(cm, np.roll(_KS_MAJOR, r))[0, 1])
        mn = float(np.corrcoef(cm, np.roll(_KS_MINOR, r))[0, 1])
        if mj > best_corr:
            best_corr, best_root, best_raga = mj, r, 'major'
        if mn > best_corr:
            best_corr, best_root, best_raga = mn, r, 'minor'

    intervals = _SCALES.get(best_raga, [0,2,3,5,7,8,10])
    key_name = NOTE_NAMES[best_root] + ' ' + best_raga
    log.info(f"Detected key/raga: {key_name}  (confidence={best_corr:.3f})")

    # Map to simple mode for training data compatibility
    simple_mode = 'major' if best_raga in (
        'major', 'shankarabharanam', 'kalyani', 'harikambhoji',
        'mohanam', 'hamsadhwani', 'bilahari', 'charukesi'
    ) else 'minor'

    return best_root, simple_mode, best_raga, intervals


def _constrain_to_scale_soft(notes, key_root, intervals):
    """
    v17: Soft scale constraint using detected raga intervals.
    Same logic as v15 but uses the actual raga scale, not just major/minor.
    """
    valid_pcs = set((key_root + i) % 12 for i in intervals)

    result = []
    moved = 0
    for s, e, p in notes:
        pc = p % 12
        if pc in valid_pcs:
            result.append((s, e, p))
            continue

        min_dist = min(min(abs(pc - vpc), 12 - abs(pc - vpc)) for vpc in valid_pcs)
        if min_dist <= 1:
            result.append((s, e, p))
        else:
            best_dist, best_p = 12, p
            for iv in intervals:
                for shift in (-1, 0, 1):
                    c = key_root + iv + 12 * (p // 12 + shift)
                    if abs(c - p) < best_dist:
                        best_dist, best_p = abs(c - p), c
            result.append((s, e, max(21, min(108, best_p))))
            moved += 1

    log.info(f"Soft scale constraint ({len(intervals)} tones): {moved}/{len(notes)} adjusted")
    return result


# ─────────────────────────────────────────────
#  Chord detection (madmom → chroma fallback)
# ─────────────────────────────────────────────

def _madmom_chords_available():
    try:
        import madmom
        return True
    except ImportError:
        return False


def _detect_chords_madmom(audio_path, beat_times, bpm, audio_duration):
    """Use madmom's CNN chord recognition (~75-80% accuracy)."""
    import madmom

    log.info("madmom chord recognition…")
    proc = madmom.features.chords.CNNChordFeatureProcessor()
    feats = proc(audio_path)
    decode = madmom.features.chords.CRFChordRecognitionProcessor()
    raw_chords = decode(feats)  # list of (start, end, chord_label)

    # Map madmom chord labels to our format
    chords = []
    for i, bt in enumerate(beat_times):
        if bt >= audio_duration:
            break
        next_bt = float(beat_times[i+1]) if i+1 < len(beat_times) else bt + 60.0/bpm
        next_bt = min(next_bt, audio_duration)
        mid = (bt + next_bt) / 2.0

        # Find which madmom chord covers this beat
        chord_label = 'N'
        for cs, ce, cl in raw_chords:
            if cs <= mid < ce:
                chord_label = cl
                break

        name, midi_notes = _parse_chord_label(chord_label)
        chords.append((name, midi_notes))

    log.info(f"madmom chords: {len(chords)}  sample={[c[0] for c in chords[:6]]}")
    return chords


def _parse_chord_label(label):
    """Parse a chord label like 'C:maj', 'F#:min' into (name, [midi_notes])."""
    if label == 'N' or not label:
        return ('Cmaj', [48, 52, 55])

    parts = label.split(':')
    root_name = parts[0]
    quality = parts[1] if len(parts) > 1 else 'maj'

    # Normalize root name
    root_name = root_name.replace('b', '#')  # handle flats loosely
    if root_name not in NOTE_NAMES:
        # Try common substitutions
        flat_map = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#'}
        root_name = flat_map.get(root_name, 'C')

    if root_name in NOTE_NAMES:
        root_idx = NOTE_NAMES.index(root_name)
    else:
        root_idx = 0

    root_midi = 48 + root_idx
    is_min = 'min' in quality

    if is_min:
        return (f'{root_name}min', [root_midi, root_midi + 3, root_midi + 7])
    else:
        return (f'{root_name}maj', [root_midi, root_midi + 4, root_midi + 7])


def _chord_templates():
    t = {}
    for r in range(12):
        maj = np.zeros(12); maj[r]=1.0; maj[(r+4)%12]=0.8; maj[(r+7)%12]=0.9
        t[f'{NOTE_NAMES[r]}maj'] = maj / np.linalg.norm(maj)
        mn  = np.zeros(12); mn[r]=1.0;  mn[(r+3)%12]=0.8;  mn[(r+7)%12]=0.9
        t[f'{NOTE_NAMES[r]}min'] = mn  / np.linalg.norm(mn)
    return t

CHORD_TEMPLATES = _chord_templates()


def _detect_chords_chroma(audio_path, beat_times, bpm, audio_duration):
    """Fallback: chroma template matching (~55-60% accuracy)."""
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
            chords.append(chords[-1] if chords else ('Cmaj',[48,52,55])); continue
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
    log.info(f"Chroma chords: {len(chords)}  sample={[c[0] for c in chords[:6]]}")
    return chords


def _detect_chords(audio_path, beat_times, bpm, audio_duration):
    """Detect chords. Uses madmom CNN when available, otherwise chroma templates."""
    if _madmom_chords_available():
        try:
            return _detect_chords_madmom(audio_path, beat_times, bpm, audio_duration)
        except Exception as exc:
            log.warning(f"madmom chords failed: {exc}, falling back to chroma")
    return _detect_chords_chroma(audio_path, beat_times, bpm, audio_duration)


# ─────────────────────────────────────────────
#  MIDI build — voice-led LH + adaptive pedal
# ─────────────────────────────────────────────

def _choose_voicing(chord_notes, prev_voicing, bass_range=(36, 52), mid_range=(48, 67)):
    """
    v17: Voice leading — choose inversion that minimizes total movement
    from previous chord voicing.
    """
    root, third, fifth = chord_notes[0], chord_notes[1], chord_notes[2]

    # Generate candidate voicings (root position + inversions across octaves)
    candidates = []
    for oct_shift in (-12, 0, 12):
        r = root + oct_shift
        t = third + oct_shift
        f = fifth + oct_shift

        # Root position
        candidates.append([r, t, f])
        # 1st inversion
        candidates.append([t, f, r + 12])
        # 2nd inversion
        candidates.append([f, r + 12, t + 12])

    # Filter: bass note must be in bass_range, upper notes in mid_range
    valid = []
    for cand in candidates:
        cand_sorted = sorted(cand)
        bass = cand_sorted[0]
        if bass_range[0] <= bass <= bass_range[1]:
            if all(mid_range[0] <= n <= mid_range[1] for n in cand_sorted[1:]):
                valid.append(cand_sorted)

    if not valid:
        # Fallback: just use root position in range
        bass = root
        while bass < bass_range[0]: bass += 12
        while bass > bass_range[1]: bass -= 12
        mid = third
        while mid < mid_range[0]: mid += 12
        while mid > mid_range[1]: mid -= 12
        top = fifth
        while top < mid_range[0]: top += 12
        while top > mid_range[1]: top -= 12
        return sorted([bass, mid, top])

    if prev_voicing is None:
        return valid[0]

    # Pick voicing with minimum total voice movement from previous
    best_v, best_cost = valid[0], float('inf')
    for v in valid:
        cost = sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev_voicing)))
        if cost < best_cost:
            best_cost, best_v = cost, v

    return sorted(best_v)


def _select_lh_pattern(bpm):
    """Select left-hand pattern based on tempo."""
    if bpm < 80:
        return 'arpeggiated'
    elif bpm < 130:
        return 'root_fifth'
    else:
        return 'alberti'


def _build_midi(melody, chords, beat_times, bpm, audio_duration, output_path):
    """
    v17: Musical MIDI builder with voice leading, adaptive LH patterns,
    and chord-change-aware sustain pedal.
    """
    midi     = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    rh       = pretty_midi.Instrument(program=0, name='RightHand')
    lh       = pretty_midi.Instrument(program=0, name='LeftHand')
    rng      = random.Random(42)
    beat_dur = 60.0 / bpm
    pattern  = _select_lh_pattern(bpm)
    log.info(f"LH pattern: {pattern} (bpm={bpm:.0f})")

    # ── Right hand: melody with humanization ────────────────────────────
    for s, e, p in melody:
        p = max(21, min(108, p))
        off = rng.uniform(-0.012, 0.012)
        vel = rng.randint(72, 95)
        rh.notes.append(pretty_midi.Note(
            velocity=vel, pitch=p,
            start=max(0.0, s + off),
            end=max(0.0, s + off + max(0.05, e - s)),
        ))

    # ── Left hand with voice leading ────────────────────────────────────
    prev_voicing = None
    prev_chord_name = None

    for i, bt in enumerate(beat_times):
        if i >= len(chords) or bt >= audio_duration:
            break

        chord_name, chord_notes = chords[i]
        next_bt = float(beat_times[i+1]) if i+1 < len(beat_times) else bt + beat_dur
        next_bt = min(next_bt, audio_duration)
        avail = next_bt - bt

        if avail < 0.05:
            continue

        # Voice leading: choose smoothest voicing
        voicing = _choose_voicing(chord_notes, prev_voicing)
        prev_voicing = voicing
        bass_note = voicing[0]
        mid_notes = voicing[1:]

        note_dur = min(beat_dur * 0.80, avail - 0.02)

        if pattern == 'arpeggiated':
            # Arpeggiated: bass → mid1 → mid2, each offset by 1/3 beat
            step = beat_dur / 3.0
            for j, p in enumerate(voicing):
                t_start = bt + j * step
                if t_start >= audio_duration:
                    break
                off = rng.uniform(-0.008, 0.008)
                lh.notes.append(pretty_midi.Note(
                    velocity=rng.randint(45, 60),
                    pitch=max(24, min(72, p)),
                    start=max(0.0, t_start + off),
                    end=max(0.0, t_start + off + note_dur * 0.7),
                ))

        elif pattern == 'alberti':
            # Alberti: low-high-mid-high within one beat
            if len(voicing) >= 3:
                alberti = [voicing[0], voicing[2], voicing[1], voicing[2]]
            else:
                alberti = [voicing[0], voicing[-1]] * 2
            step = beat_dur / 4.0
            for j, p in enumerate(alberti):
                t_start = bt + j * step
                if t_start >= audio_duration:
                    break
                off = rng.uniform(-0.006, 0.006)
                lh.notes.append(pretty_midi.Note(
                    velocity=rng.randint(40, 55),
                    pitch=max(24, min(72, p)),
                    start=max(0.0, t_start + off),
                    end=max(0.0, t_start + off + step * 0.85),
                ))

        else:  # root_fifth
            # Beat 1: bass root
            off = rng.uniform(-0.010, 0.010)
            lh.notes.append(pretty_midi.Note(
                velocity=rng.randint(52, 65),
                pitch=max(24, min(60, bass_note)),
                start=max(0.0, bt + off),
                end=max(0.0, bt + off + note_dur),
            ))
            # Beat 2.5: fifth or upper voicing note
            t2 = bt + beat_dur * 0.5
            if t2 < audio_duration and len(mid_notes) > 0:
                off2 = rng.uniform(-0.010, 0.010)
                lh.notes.append(pretty_midi.Note(
                    velocity=rng.randint(40, 52),
                    pitch=max(36, min(67, mid_notes[-1])),
                    start=max(0.0, t2 + off2),
                    end=max(0.0, t2 + off2 + note_dur * 0.6),
                ))

        # ── Chord-change-aware sustain pedal ────────────────────────────
        chord_changed = (chord_name != prev_chord_name)
        prev_chord_name = chord_name

        if chord_changed or i == 0:
            # Release old pedal slightly before, press new one
            if i > 0:
                lh.control_changes.append(
                    pretty_midi.ControlChange(64, 0, max(0.0, bt - 0.05)))
            lh.control_changes.append(
                pretty_midi.ControlChange(64, 127, bt))

    # Final pedal release
    if len(beat_times) > 0:
        final = min(float(beat_times[-1]) + beat_dur, audio_duration)
        lh.control_changes.append(pretty_midi.ControlChange(64, 0, final))

    midi.instruments.extend([rh, lh])
    midi.write(output_path)
    log.info(
        f"MIDI saved: RH={len(rh.notes)} melody  "
        f"LH={len(lh.notes)} accomp → {output_path}"
    )


# ─────────────────────────────────────────────
#  Energy contour — context-aware dynamics
# ─────────────────────────────────────────────

def _compute_energy_contour(audio_path, hop_sec=0.5):
    """Compute RMS energy contour — returns array of (time, energy_0_to_1)."""
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop = int(hop_sec * sr)
    energies = []
    for i in range(0, len(y), hop):
        chunk = y[i:i+hop]
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
        energies.append((i / sr, rms))
    # Normalize to 0-1
    peak = max(e for _, e in energies) if energies else 1.0
    if peak > 0:
        energies = [(t, e / peak) for t, e in energies]
    return energies


def _get_energy_at_time(energy_contour, t):
    """Lookup energy level (0-1) at a given time."""
    if not energy_contour:
        return 0.6
    # Binary search would be better but this is fine for ~500 entries
    best = energy_contour[0][1]
    for ct, ce in energy_contour:
        if ct > t:
            break
        best = ce
    return best


def _apply_energy_dynamics(rh_notes, energy_contour):
    """Scale velocity of RH notes based on local energy — quiet sections softer."""
    if not energy_contour:
        return rh_notes
    result = []
    for s, e, p, v in rh_notes:
        energy = _get_energy_at_time(energy_contour, s)
        # Map energy 0→0.5 to velocity scale 0.55→1.0
        # So quiet sections play at ~55% velocity, loud at 100%
        vel_scale = 0.55 + 0.45 * min(1.0, energy / 0.5)
        new_v = int(max(35, min(110, v * vel_scale)))
        result.append((s, e, p, new_v))
    return result


# ─────────────────────────────────────────────
#  Phase 9: Context-Aware Intelligence
# ─────────────────────────────────────────────

def _detect_song_sections(audio_path, beat_times, vocals_path=None):
    """
    Detect song sections (intro, verse, chorus, instrumental, outro)
    using energy contour + vocal presence analysis.

    Returns list of dicts:
      [{'start': float, 'end': float, 'type': str, 'energy': float}, ...]
    """
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        audio_duration = len(y) / sr

        # Compute RMS energy in ~2-second windows
        window_sec = 2.0
        hop = int(window_sec * sr)
        energies = []
        for i in range(0, len(y), hop):
            chunk = y[i:i + hop]
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
            t = i / sr
            energies.append((t, rms))

        # Normalize energy to 0-1
        peak_energy = max(e for _, e in energies) if energies else 1.0
        if peak_energy > 0:
            energies = [(t, e / peak_energy) for t, e in energies]

        # Compute vocal energy if vocal stem available
        vocal_energies = []
        if vocals_path and os.path.exists(vocals_path):
            try:
                yv, srv = librosa.load(vocals_path, sr=22050, mono=True)
                vocal_peak = float(np.sqrt(np.mean(yv ** 2))) if len(yv) > 0 else 0.0
                for i in range(0, len(yv), hop):
                    chunk = yv[i:i + hop]
                    rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
                    t = i / sr
                    # Vocal presence: vocal RMS > 0.05 of vocal peak
                    has_vocal = rms > (vocal_peak * 0.05) if vocal_peak > 0 else False
                    vocal_energies.append((t, rms, has_vocal))
            except Exception as exc:
                log.warning(f"Vocal energy analysis failed: {exc}")

        # Build sections from energy + vocal presence
        sections = []
        for idx, (t, energy) in enumerate(energies):
            t_end = t + window_sec
            if idx + 1 < len(energies):
                t_end = energies[idx + 1][0]
            else:
                t_end = min(t + window_sec, audio_duration)

            # Determine vocal presence for this window
            has_vocal = False
            if vocal_energies:
                for vt, vrms, vp in vocal_energies:
                    if abs(vt - t) < window_sec:
                        has_vocal = vp
                        break

            # Classify section type
            if has_vocal and energy > 0.55:
                section_type = 'chorus'
            elif has_vocal and energy > 0.15:
                section_type = 'verse'
            elif not has_vocal and energy > 0.15:
                section_type = 'instrumental'
            else:
                section_type = 'instrumental'  # low energy, no vocals

            sections.append({
                'start': t,
                'end': t_end,
                'type': section_type,
                'energy': energy,
            })

        # Post-process: mark intro and outro
        if sections:
            # First section(s) with low energy → intro
            for s in sections:
                if s['energy'] < 0.25:
                    s['type'] = 'intro'
                else:
                    break

            # Last section(s) with declining energy → outro
            for s in reversed(sections):
                if s['energy'] < 0.25:
                    s['type'] = 'outro'
                else:
                    break

        # Merge consecutive sections of the same type
        merged = []
        for s in sections:
            if merged and merged[-1]['type'] == s['type']:
                merged[-1]['end'] = s['end']
                merged[-1]['energy'] = max(merged[-1]['energy'], s['energy'])
            else:
                merged.append(dict(s))

        log.info(f"Song sections: {len(merged)} sections detected: "
                 f"{[s['type'] for s in merged]}")
        return merged

    except Exception as exc:
        log.warning(f"Song section detection failed: {exc}")
        return []


def _get_section_at_time(sections, t):
    """Return the section dict at time t, or None."""
    for s in sections:
        if s['start'] <= t < s['end']:
            return s
    return None


def _detect_phrase_boundaries(melody_notes, min_gap=0.3):
    """
    Find phrase boundaries: gaps > min_gap seconds between consecutive melody notes.
    Returns list of boundary times.
    """
    if not melody_notes or len(melody_notes) < 2:
        return []

    sorted_notes = sorted(melody_notes, key=lambda n: n[0])
    boundaries = []
    for i in range(len(sorted_notes) - 1):
        note_end = sorted_notes[i][1]
        next_start = sorted_notes[i + 1][0]
        gap = next_start - note_end
        if gap > min_gap:
            boundaries.append(next_start)

    log.info(f"Phrase boundaries: {len(boundaries)} found (min_gap={min_gap}s)")
    return boundaries


def _add_breathing_room(melody_notes, phrase_boundaries, min_silence_ms=80):
    """
    At each phrase boundary, ensure at least min_silence_ms of silence.
    Trims the note before the boundary if needed.
    """
    if not melody_notes or not phrase_boundaries:
        return melody_notes

    min_silence = min_silence_ms / 1000.0
    trim_amount = 0.04  # 40ms breathing room trim

    sorted_notes = sorted(melody_notes, key=lambda n: n[0])
    boundary_set = set(phrase_boundaries)

    result = []
    for i, (s, e, p, v) in enumerate(sorted_notes):
        # Check if next note starts at a phrase boundary
        if i + 1 < len(sorted_notes):
            next_start = sorted_notes[i + 1][0]
            if next_start in boundary_set:
                # Ensure gap of at least min_silence before boundary
                max_end = next_start - min_silence
                if e > max_end:
                    e = max(s + 0.05, max_end)  # don't make note too short
            else:
                # Not a phrase boundary, but still trim slightly for breathing
                # Only if note is very close to next
                gap = next_start - e
                if 0 < gap < trim_amount:
                    e = e - trim_amount
                    e = max(s + 0.05, e)

        result.append((s, e, p, v))

    log.info(f"Breathing room applied at {len(phrase_boundaries)} boundaries")
    return result


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def _build_midi_v18(rh_notes, lh_chords, beat_times, bpm, audio_duration, output_path,
                    energy_contour=None, sections=None):
    """
    v18 + Phase 9: Build MIDI from basic-pitch transcribed notes.
    LH: Section-adaptive chords with improved humanization.
    - intro/outro: LH off (melody only)
    - verse: LH on chord changes only, low velocity
    - chorus: full LH pattern, moderate velocity
    - instrumental: root-fifth pattern, moderate velocity
    Fixes chord overlap and robotic feel.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    rh = pretty_midi.Instrument(program=0, name='RightHand')
    lh = pretty_midi.Instrument(program=0, name='LeftHand')
    rng = random.Random(42)
    beat_dur = 60.0 / bpm
    log.info(f"v18 MIDI build: RH={len(rh_notes)} notes, bpm={bpm:.0f}, "
             f"sections={'yes' if sections else 'no'}")

    # ── Compute note density for staccato variation ──
    # Count notes in local 2-second windows
    def _local_density(t, window=2.0):
        count = 0
        for ns, ne, np_, nv in rh_notes:
            if abs(ns - t) < window:
                count += 1
        return count

    # ── Right hand: improved humanization (Phase 9) ──────────────────────
    for idx, note_data in enumerate(rh_notes):
        s, e, p, v = note_data
        p = max(21, min(108, p))
        dur = max(0.05, e - s)

        # Phase 9: Enhanced humanization
        # Velocity variation ±12 (was ±8)
        vel = max(40, min(110, v + rng.randint(-12, 12)))

        # Timing variation ±15ms (was ±8ms)
        off = rng.uniform(-0.015, 0.015)

        # Tempo rubato: strong beats slightly early, weak beats slightly late
        if len(beat_times) > 0:
            beat_arr = np.asarray(beat_times, dtype=float)
            nearest_idx = int(np.argmin(np.abs(beat_arr - s)))
            dist_to_beat = abs(s - float(beat_arr[nearest_idx]))
            if dist_to_beat < beat_dur * 0.1:
                # On a strong beat: slightly early
                off += -0.010
            elif dist_to_beat > beat_dur * 0.35:
                # Off-beat / weak beat: slightly late
                off += 0.010

        # Staccato variation in dense passages
        density = _local_density(s)
        if density > 8:
            # High density: shorten some notes for staccato feel
            dur *= rng.uniform(0.70, 0.90)
        elif density > 5:
            dur *= rng.uniform(0.85, 1.0)

        rh.notes.append(pretty_midi.Note(
            velocity=vel, pitch=p,
            start=max(0.0, s + off),
            end=max(0.0, s + off + dur),
        ))

    # ── Build a quick lookup: RH notes active at a given time ──
    def _min_rh_pitch_at(t, window=0.1):
        """Find the lowest RH pitch active near time t."""
        min_p = 999
        for ns, ne, np_, nv in rh_notes:
            if ns - window <= t <= ne + window:
                min_p = min(min_p, np_)
        return min_p if min_p < 999 else None

    def _rh_active_at(t, window=0.1):
        """Check if any RH note is active near time t."""
        for ns, ne, np_, nv in rh_notes:
            if ns - window <= t <= ne + window:
                return True
        return False

    # ── Left hand: section-adaptive chords (Phase 9) ─────────────────────
    prev_voicing = None
    prev_chord_name = None

    # Default restrike / style based on tempo
    if bpm < 80:
        default_restrike = 4
        default_style = 'sustained'
    elif bpm < 130:
        default_restrike = 2
        default_style = 'bass_chord'
    else:
        default_restrike = 2
        default_style = 'broken'

    log.info(f"LH default style: {default_style}, restrike every {default_restrike} beats")

    i = 0
    while i < len(beat_times) and i < len(lh_chords):
        bt = float(beat_times[i])
        if bt >= audio_duration:
            break

        chord_name, chord_notes = lh_chords[i]
        chord_changed = (chord_name != prev_chord_name)

        # ── Phase 9: Section-aware LH behavior ──
        section = _get_section_at_time(sections, bt) if sections else None
        section_type = section['type'] if section else 'chorus'  # default to chorus behavior

        # intro/outro: LH off entirely
        if section_type in ('intro', 'outro'):
            prev_chord_name = chord_name
            i += 1
            continue

        # Determine style and velocity based on section
        if section_type == 'verse':
            lh_style = 'sustained'  # chord changes only, no restrike
            restrike_interval = 999  # effectively no restrike
            vel_lo, vel_hi = 35, 42
            play_on_change_only = True
        elif section_type == 'chorus':
            lh_style = default_style
            restrike_interval = default_restrike
            vel_lo, vel_hi = 42, 55
            play_on_change_only = False
        elif section_type == 'instrumental':
            lh_style = 'root_fifth'
            restrike_interval = default_restrike
            vel_lo, vel_hi = 38, 50
            play_on_change_only = False
        else:
            lh_style = default_style
            restrike_interval = default_restrike
            vel_lo, vel_hi = 42, 55
            play_on_change_only = False

        # Energy-aware: skip LH in very quiet sections
        local_energy = _get_energy_at_time(energy_contour, bt) if energy_contour else 0.6
        if local_energy < 0.15 and not chord_changed:
            prev_chord_name = chord_name
            i += 1
            continue

        # Only play on chord changes OR at restrike intervals
        if play_on_change_only and not chord_changed:
            prev_chord_name = chord_name
            i += 1
            continue
        if not chord_changed and (i % restrike_interval != 0):
            prev_chord_name = chord_name
            i += 1
            continue

        prev_chord_name = chord_name

        # Ensure 3 notes for voicing
        while len(chord_notes) < 3:
            chord_notes = chord_notes + [chord_notes[-1] + 7]
        chord_notes = chord_notes[:3]

        voicing = _choose_voicing(chord_notes, prev_voicing)
        prev_voicing = voicing

        # ── Phase 9: Fix chord overlap — ensure LH max pitch is at least
        #    7 semitones below lowest concurrent RH note ──
        min_rh = _min_rh_pitch_at(bt)
        if min_rh is not None:
            max_lh_allowed = min_rh - 7
            voicing = [p for p in voicing if p <= max_lh_allowed]
            if not voicing:
                # All voicing notes too high — transpose down an octave
                voicing = _choose_voicing(chord_notes, prev_voicing)
                voicing = [max(24, p - 12) for p in voicing]
                voicing = [p for p in voicing if p <= max_lh_allowed] or [max(24, max_lh_allowed - 7)]

        if not voicing:
            i += 1
            continue

        # ── Phase 9: Reduce LH velocity when RH is active ──
        rh_playing = _rh_active_at(bt)
        vel_reduction = 0.85 if rh_playing else 1.0  # 15% reduction when RH active

        # Find how long to hold
        hold_end = bt + beat_dur * restrike_interval
        for j in range(i + 1, min(i + restrike_interval + 1, len(lh_chords))):
            if j < len(beat_times):
                if j < len(lh_chords) and lh_chords[j][0] != chord_name:
                    hold_end = float(beat_times[j])
                    break
        hold_end = min(hold_end, audio_duration)
        hold_dur = hold_end - bt

        if hold_dur < 0.1:
            i += 1
            continue

        if lh_style == 'sustained':
            for p in voicing:
                p = max(36, min(67, p))
                off = rng.uniform(-0.006, 0.006)
                raw_vel = rng.randint(vel_lo, vel_hi)
                final_vel = max(25, int(raw_vel * vel_reduction))
                lh.notes.append(pretty_midi.Note(
                    velocity=final_vel, pitch=p,
                    start=max(0.0, bt + off),
                    end=max(0.0, bt + hold_dur - 0.03),
                ))

        elif lh_style == 'bass_chord':
            bass = max(36, min(55, voicing[0]))
            off = rng.uniform(-0.008, 0.008)
            raw_vel = rng.randint(vel_lo, vel_hi)
            final_vel = max(25, int(raw_vel * vel_reduction))
            lh.notes.append(pretty_midi.Note(
                velocity=final_vel, pitch=bass,
                start=max(0.0, bt + off),
                end=max(0.0, bt + hold_dur - 0.03),
            ))
            chord_time = bt + beat_dur * 0.95
            if chord_time < hold_end - 0.1 and len(voicing) > 1:
                for p in voicing[1:]:
                    p = max(48, min(67, p))
                    off2 = rng.uniform(-0.005, 0.005)
                    softer_vel = max(25, int(rng.randint(max(25, vel_lo - 5), vel_hi - 5) * vel_reduction))
                    lh.notes.append(pretty_midi.Note(
                        velocity=softer_vel, pitch=p,
                        start=max(0.0, chord_time + off2),
                        end=max(0.0, chord_time + (hold_dur - beat_dur) - 0.03),
                    ))

        elif lh_style == 'root_fifth':
            # Root-fifth pattern for instrumental sections
            bass = max(36, min(55, voicing[0]))
            off = rng.uniform(-0.010, 0.010)
            raw_vel = rng.randint(vel_lo, vel_hi)
            final_vel = max(25, int(raw_vel * vel_reduction))
            note_dur = min(beat_dur * 0.80, hold_dur - 0.02)
            lh.notes.append(pretty_midi.Note(
                velocity=final_vel, pitch=bass,
                start=max(0.0, bt + off),
                end=max(0.0, bt + off + note_dur),
            ))
            # Fifth on the and-beat
            t2 = bt + beat_dur * 0.5
            fifth = bass + 7
            if fifth > 60:
                fifth -= 12
            if t2 < audio_duration and hold_dur > beat_dur * 0.6:
                off2 = rng.uniform(-0.010, 0.010)
                softer_vel = max(25, int(rng.randint(max(25, vel_lo - 3), vel_hi - 3) * vel_reduction))
                lh.notes.append(pretty_midi.Note(
                    velocity=softer_vel,
                    pitch=max(36, min(60, fifth)),
                    start=max(0.0, t2 + off2),
                    end=max(0.0, t2 + off2 + note_dur * 0.6),
                ))

        else:  # broken — gentle arpeggio over 1 beat, then sustain
            step = beat_dur / len(voicing) if voicing else beat_dur
            for j, p in enumerate(voicing):
                p = max(36, min(67, p))
                t_start = bt + j * step * 0.3
                off = rng.uniform(-0.005, 0.005)
                raw_vel = rng.randint(vel_lo, vel_hi)
                final_vel = max(25, int(raw_vel * vel_reduction))
                lh.notes.append(pretty_midi.Note(
                    velocity=final_vel, pitch=p,
                    start=max(0.0, t_start + off),
                    end=max(0.0, bt + hold_dur - 0.03),
                ))

        # Sustain pedal
        lh.control_changes.append(pretty_midi.ControlChange(64, 127, max(0.0, bt)))
        lh.control_changes.append(
            pretty_midi.ControlChange(64, 0, max(0.0, hold_end - 0.06)))

        i += 1

    # Final pedal off
    if len(beat_times) > 0:
        final = min(float(beat_times[-1]) + beat_dur, audio_duration)
        lh.control_changes.append(pretty_midi.ControlChange(64, 0, final))

    midi.instruments.extend([rh, lh])
    midi.write(output_path)
    log.info(f"MIDI saved: RH={len(rh.notes)} LH={len(lh.notes)} → {output_path}")


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

        prog("Separating sources…", 25)
        stems = _separate_audio(stereo_wav, work_dir)
        vocals_path = stems.get('vocals')
        accomp_path = stems.get('no_vocals') or stems.get('other')

        # ══════════════════════════════════════════════════════════════════
        #  v18.1: LAYER-BY-LAYER transcription
        #  Layer 1: Clean monophonic melody from vocals
        #  Layer 2: Sparse chord accompaniment from instrumental stem
        #  Layer 3: Gap-fill from full audio for instrumental sections
        # ══════════════════════════════════════════════════════════════════
        use_basic_pitch = _basic_pitch_available()
        rh_notes_4 = []  # (start, end, pitch, velocity)
        lh_chords = []

        if use_basic_pitch:
            # ── LAYER 1: Vocal melody (strict monophonic) ────────────────
            vocal_notes = []
            full_notes = []

            if vocals_path:
                prog("Layer 1: Transcribing vocal melody…", 35)
                try:
                    v22k = os.path.join(work_dir, 'vocals_22k.wav')
                    _to_mono(vocals_path, v22k, sr=22050)
                    vocal_raw = _transcribe_basic_pitch(
                        v22k, onset_thresh=0.4, frame_thresh=0.28,
                        min_note_ms=80, min_freq=130, max_freq=1400
                    )
                    log.info(f"Vocal raw: {len(vocal_raw)} notes")
                    # Smart cleanup → monophonic melody with continuity
                    vocal_notes = _clean_melody_smart(
                        vocal_raw, min_dur=0.08, min_pitch=48, dedup_window=0.20
                    )
                    log.info(f"Vocal clean melody: {len(vocal_notes)} notes")
                except Exception as exc:
                    log.warning(f"basic-pitch on vocals failed: {exc}")

            # ── LAYER 3: Full audio for instrumental gap-fill ────────────
            prog("Layer 3: Transcribing full audio for gaps…", 50)
            try:
                full_raw = _transcribe_basic_pitch(
                    mono_wav, onset_thresh=0.4, frame_thresh=0.25,
                    min_note_ms=80, min_freq=130, max_freq=1400
                )
                log.info(f"Full audio raw: {len(full_raw)} notes")
                full_notes = full_raw
            except Exception as exc:
                log.warning(f"basic-pitch on full audio failed: {exc}")

            # Merge layers: vocal melody + gap-fill from full audio
            if vocal_notes:
                rh_notes_4 = _fill_gaps_from_full(vocal_notes, full_notes)
            elif full_notes:
                rh_notes_4 = _clean_melody_smart(
                    full_notes, min_dur=0.12, min_pitch=55
                )
            log.info(f"Final RH melody: {len(rh_notes_4)} notes")

            # ── LAYER 2: Accompaniment → sparse chords ───────────────────
            if accomp_path:
                prog("Layer 2: Transcribing accompaniment…", 60)
                try:
                    acc22k = os.path.join(work_dir, 'accomp_22k.wav')
                    _to_mono(accomp_path, acc22k, sr=22050)
                    accomp_notes = _transcribe_basic_pitch(
                        acc22k, onset_thresh=0.55, frame_thresh=0.35,
                        min_note_ms=200, max_freq=700
                    )
                    lh_chords = _notes_to_lh_chords(
                        accomp_notes, beat_times, bpm, audio_duration
                    )
                    log.info(f"Accompaniment → {len(lh_chords)} beat chords")
                except Exception as exc:
                    log.warning(f"basic-pitch on accompaniment failed: {exc}")

        # ══════════════════════════════════════════════════════════════════
        #  FALLBACK: old RMVPE/torchcrepe chain if basic-pitch unavailable
        # ══════════════════════════════════════════════════════════════════
        if not rh_notes_4:
            prog("Fallback: extracting melody (RMVPE/torchcrepe)…", 50)
            melody_3 = []

            if vocals_path:
                try:
                    v22k = os.path.join(work_dir, 'vocals_22k.wav')
                    v16k = os.path.join(work_dir, 'vocals_16k.wav')
                    _to_mono(vocals_path, v22k, sr=22050)
                    _to_mono(vocals_path, v16k, sr=16000)
                    melody_3 = _extract_melody(v16k, v22k, audio_duration, bpm)
                except Exception as exc:
                    log.warning(f"Vocal melody fallback failed: {exc}")

            if len(melody_3) < 20:
                try:
                    harm_22k, harm_16k = _make_harmonic_audio(mono_wav, work_dir)
                    melody_3 = _extract_melody(harm_16k, harm_22k, audio_duration, bpm)
                except Exception as exc:
                    log.warning(f"Harmonic melody fallback failed: {exc}")

            if melody_3:
                melody_3 = _filter_ornaments_gamaka(melody_3, min_dur_ms=80)
                melody_3 = _fill_melody_gaps(melody_3, max_gap=0.80)
                rh_notes_4 = [(s, e, p, 80) for s, e, p in melody_3]

        log.info(f"RH notes: {len(rh_notes_4)}")

        # ── Key/raga detection ───────────────────────────────────────────
        prog("Detecting key/raga…", 75)
        key_root, key_mode, raga_name, intervals = _detect_key_raga(mono_wav)

        # Soft scale constraint on RH
        rh_melody_3 = [(s, e, p) for s, e, p, v in rh_notes_4]
        rh_melody_3 = _constrain_to_scale_soft(rh_melody_3, key_root, intervals)
        rh_notes_4 = [(s, e, p, rh_notes_4[i][3])
                      for i, (s, e, p) in enumerate(rh_melody_3)
                      if i < len(rh_notes_4)]

        # ── Quantize to beat grid ────────────────────────────────────────
        prog("Quantizing to beat grid…", 82)
        rh_3 = [(s, e, p) for s, e, p, v in rh_notes_4]
        rh_3 = _quantize_multi(rh_3, bpm, beat_times)
        rh_notes_4 = [(s, e, p, rh_notes_4[i][3] if i < len(rh_notes_4) else 80)
                      for i, (s, e, p) in enumerate(rh_3)]

        # ── Chord detection fallback if no transcribed harmony ───────────
        if not lh_chords:
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
            elif accomp_path:
                chord_src = accomp_path
            else:
                chord_src = mono_wav

            lh_chords = _detect_chords(chord_src, beat_times, bpm, audio_duration)

        # ── Energy contour for context-aware dynamics ──────────────────────
        prog("Analyzing dynamics…", 88)
        try:
            energy_contour = _compute_energy_contour(mono_wav, hop_sec=0.5)
            rh_notes_4 = _apply_energy_dynamics(rh_notes_4, energy_contour)
            log.info(f"Energy contour: {len(energy_contour)} frames")
        except Exception as exc:
            log.warning(f"Energy contour failed: {exc}")
            energy_contour = None

        # ── Phase 9: Song section detection ──────────────────────────────
        prog("Detecting song sections…", 90)
        sections = []
        try:
            sections = _detect_song_sections(mono_wav, beat_times, vocals_path)
        except Exception as exc:
            log.warning(f"Section detection failed (non-fatal): {exc}")
            sections = []

        # ── Phase 9: Phrase boundary detection + breathing room ──────────
        prog("Adding musical phrasing…", 92)
        try:
            boundaries = _detect_phrase_boundaries(rh_notes_4, min_gap=0.3)
            rh_notes_4 = _add_breathing_room(rh_notes_4, boundaries)
        except Exception as exc:
            log.warning(f"Phrase boundary processing failed (non-fatal): {exc}")

        # ── Build MIDI ─────────────────────────────────────────────────────
        prog("Building piano arrangement…", 95)
        _build_midi_v18(rh_notes_4, lh_chords, beat_times, bpm, audio_duration, output_path,
                        energy_contour=energy_contour, sections=sections)

        prog("Done!", 100)
        return {
            'midi_path': output_path,
            'sections': sections,
            'bpm': bpm,
            'duration': audio_duration,
            'key': f"{NOTE_NAMES[key_root]} {key_mode}",
        }

    except Exception as e:
        log.error(f"Pipeline error: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
