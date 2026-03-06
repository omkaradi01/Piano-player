#!/usr/bin/env python3
"""
analyze_pairs.py — Learn style patterns from collected song-cover pairs.

Run this after collect_data.py has processed all pairs.

What this produces:
  training_data/style_profile.json — a learned style profile encoding
  how Tamil piano cover pianists actually play, distilled from all pairs.

This profile is then used by the main pipeline to:
  - Set the correct melody octave range
  - Choose note durations that match real Tamil piano covers
  - Select left-hand patterns that human pianists actually use
  - Understand which scale degrees are emphasised in Tamil music

Usage:
  python analyze_pairs.py
"""

import os, sys, json
import numpy as np

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
PROFILE_OUT = os.path.join(DATA_DIR, 'style_profile.json')
NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def load_all_pairs():
    summary_path = os.path.join(DATA_DIR, 'summary.json')
    if not os.path.exists(summary_path):
        print("ERROR: training_data/summary.json not found.")
        print("Run collect_data.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    pairs = []
    for song in summary.get('songs', []):
        song_dir    = song['dir']
        features_f  = os.path.join(song_dir, 'features.json')
        patterns_f  = os.path.join(song_dir, 'patterns.json')
        cover_mid   = os.path.join(song_dir, 'cover.mid')

        if not all(os.path.exists(p) for p in [features_f, patterns_f, cover_mid]):
            print(f"  ⚠  Skipping {song['name']} — missing files")
            continue

        with open(features_f) as f:  features = json.load(f)
        with open(patterns_f) as f:  patterns = json.load(f)

        pairs.append({
            'name':     song['name'],
            'dir':      song_dir,
            'features': features,
            'patterns': patterns,
            'mid':      cover_mid,
        })

    print(f"Loaded {len(pairs)} pairs from training_data/")
    return pairs


def aggregate_style(pairs):
    """
    Aggregate style patterns across all pairs into a single profile.
    This is what the pipeline will use when generating piano covers
    for new songs it hasn't seen before.
    """

    # ── Melody octave ─────────────────────────────────────────────────────
    primary_octaves = [p['patterns'].get('primary_octave', 4) for p in pairs]
    octave_counter  = {}
    for o in primary_octaves:
        octave_counter[o] = octave_counter.get(o, 0) + 1
    most_common_octave = max(octave_counter, key=octave_counter.get)

    rh_mins     = [p['patterns'].get('rh_pitch_min', 60)    for p in pairs]
    rh_maxs     = [p['patterns'].get('rh_pitch_max', 84)    for p in pairs]
    rh_medians  = [p['patterns'].get('rh_pitch_median', 72) for p in pairs]

    melody_lo = int(np.percentile(rh_mins,    25))   # 25th percentile of minimums
    melody_hi = int(np.percentile(rh_maxs,    75))   # 75th percentile of maximums
    melody_centre = int(np.median(rh_medians))

    # ── Note density ──────────────────────────────────────────────────────
    notes_per_beat = [p['patterns'].get('notes_per_beat_rh', 1.0) for p in pairs]
    avg_notes_per_beat    = round(float(np.mean(notes_per_beat)), 3)
    median_notes_per_beat = round(float(np.median(notes_per_beat)), 3)

    # ── Note duration ─────────────────────────────────────────────────────
    dur_means   = [p['patterns'].get('rh_dur_mean_beats',   0.5) for p in pairs]
    dur_medians = [p['patterns'].get('rh_dur_median_beats', 0.5) for p in pairs]
    avg_dur_beats    = round(float(np.mean(dur_means)),   3)
    median_dur_beats = round(float(np.median(dur_medians)), 3)

    # ── Scale degree distribution ─────────────────────────────────────────
    # Average degree distribution across all songs
    # (normalized to key root so songs in different keys are comparable)
    degree_sum = {str(i): 0.0 for i in range(12)}
    for p in pairs:
        dd = p['patterns'].get('scale_degree_dist', {})
        for k, v in dd.items():
            if k in degree_sum:
                degree_sum[k] += float(v)
    n = max(len(pairs), 1)
    avg_degree_dist = {k: round(v / n, 4) for k, v in degree_sum.items()}

    # Top 5 most used scale degrees
    sorted_degrees = sorted(avg_degree_dist.items(), key=lambda x: x[1], reverse=True)
    top_degrees = [int(k) for k, _ in sorted_degrees[:5]]

    # ── Key distribution ──────────────────────────────────────────────────
    keys = [p['features'].get('key', '') for p in pairs]
    key_counts = {}
    for k in keys:
        key_counts[k] = key_counts.get(k, 0) + 1

    modes = [p['features'].get('key_mode', 'minor') for p in pairs]
    pct_minor = round(sum(1 for m in modes if m == 'minor') / max(len(modes), 1), 3)

    # ── BPM distribution ──────────────────────────────────────────────────
    bpms = [p['features'].get('bpm', 100) for p in pairs]
    avg_bpm    = round(float(np.mean(bpms)), 1)
    median_bpm = round(float(np.median(bpms)), 1)

    # ── LH interval patterns ──────────────────────────────────────────────
    all_lh_intervals = []
    for p in pairs:
        all_lh_intervals.extend(p['patterns'].get('lh_common_intervals', []))
    lh_interval_counts = {}
    for iv in all_lh_intervals:
        lh_interval_counts[str(iv)] = lh_interval_counts.get(str(iv), 0) + 1
    top_lh_intervals = sorted(lh_interval_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    top_lh_intervals = [int(k) for k, _ in top_lh_intervals]

    profile = {
        'source':          f'{len(pairs)} Tamil song-cover pairs',
        'songs_analysed':  [p['name'] for p in pairs],

        'melody': {
            'pitch_lo':           melody_lo,
            'pitch_hi':           melody_hi,
            'pitch_centre':       melody_centre,
            'primary_octave':     most_common_octave,
            'notes_per_beat_avg': avg_notes_per_beat,
            'notes_per_beat_med': median_notes_per_beat,
            'dur_mean_beats':     avg_dur_beats,
            'dur_median_beats':   median_dur_beats,
        },

        'scale': {
            'pct_minor':          pct_minor,
            'top_scale_degrees':  top_degrees,
            'degree_distribution':avg_degree_dist,
        },

        'harmony': {
            'key_distribution':   key_counts,
            'top_lh_intervals':   top_lh_intervals,
        },

        'tempo': {
            'bpm_avg':    avg_bpm,
            'bpm_median': median_bpm,
            'bpm_min':    round(float(min(bpms)), 1),
            'bpm_max':    round(float(max(bpms)), 1),
        },
    }

    return profile


def print_profile(profile):
    m = profile['melody']
    s = profile['scale']
    t = profile['tempo']

    print(f"\n{'═'*55}")
    print(f"  Tamil Piano Cover Style Profile")
    print(f"  Based on: {profile['source']}")
    print(f"{'═'*55}")
    print(f"\n  MELODY")
    print(f"    Pitch range:      MIDI {m['pitch_lo']}–{m['pitch_hi']}"
          f"  (centre: {m['pitch_centre']})")
    print(f"    Primary octave:   Octave {m['primary_octave']}")
    print(f"    Notes per beat:   {m['notes_per_beat_med']} (median)")
    print(f"    Note duration:    {m['dur_median_beats']} beats (median)")

    print(f"\n  SCALE / HARMONY")
    print(f"    Minor songs:      {s['pct_minor']*100:.0f}%")
    degree_names = ['Root','b2','2','b3','3','4','b5','5','b6','6','b7','7']
    top = [degree_names[d] for d in s['top_scale_degrees'][:5]]
    print(f"    Top scale degrees:{', '.join(top)}")
    iv_names = {0:'unison',1:'m2',2:'M2',3:'m3',4:'M3',5:'P4',
                7:'P5',8:'m6',9:'M6',10:'m7',11:'M7',12:'octave'}
    top_iv = [iv_names.get(i, str(i)) for i in profile['harmony']['top_lh_intervals'][:5]]
    print(f"    LH intervals:     {', '.join(top_iv)}")

    print(f"\n  TEMPO")
    print(f"    BPM range:        {t['bpm_min']}–{t['bpm_max']}")
    print(f"    BPM median:       {t['bpm_median']}")

    print(f"\n  SONGS ANALYSED")
    for name in profile['songs_analysed']:
        print(f"    • {name}")
    print(f"{'═'*55}\n")


def main():
    pairs = load_all_pairs()
    if not pairs:
        print("No pairs to analyse.")
        sys.exit(1)

    print(f"\nAnalysing {len(pairs)} song-cover pairs…\n")
    profile = aggregate_style(pairs)

    with open(PROFILE_OUT, 'w') as f:
        json.dump(profile, f, indent=2)

    print_profile(profile)
    print(f"✅  Style profile saved to: {PROFILE_OUT}")
    print(f"\n    The pipeline will automatically use this profile")
    print(f"    the next time you run bash start.sh\n")


if __name__ == '__main__':
    main()
