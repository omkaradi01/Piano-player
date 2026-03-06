#!/usr/bin/env python3
"""
auto_collect.py — Automatically search YouTube for Tamil song + piano cover pairs
and append them to songs.csv so collect_data.py can process them.

USAGE
─────
  python auto_collect.py                # search for 30 new pairs, title-filter only  (fast ~4 min)
  python auto_collect.py 50            # search for 50 new pairs
  python auto_collect.py 30 --verify   # also download 45s of each cover and check for drums (slow ~45 min)

HOW IT WORKS
────────────
  1. For each Tamil song in the built-in list, searches YouTube for:
       "<song name> official audio"            → original song
       "<song name> piano cover Tamil"         → piano cover
  2. Filters covers by title:
       MUST contain: piano / keyboard / piano cover / etc.
       MUST NOT contain: drum / guitar / violin / flute / orchestra / etc.
  3. With --verify: downloads 45 seconds of the cover, runs librosa HPSS
     to measure how much is percussive (drums). If >25% → skip or mark mixed.
  4. Appends valid pairs to songs.csv.

TIME ESTIMATES (per 30 pairs)
─────────────────────────────
  auto_collect.py alone (no --verify) : ~4 minutes
  auto_collect.py --verify            : ~40-50 minutes (downloads audio clips)
  collect_data.py on 30 new songs     : ~3-5 hours  (downloads full audio + Demucs + transcription)
  analyze_pairs.py (re-runs each time): ~5 minutes
"""

import os, sys, csv, time, re, tempfile, subprocess

# ─── List of Tamil film songs to search for ───────────────────────────────────
TAMIL_SONGS = [
    "Nenjukkul Peidhidum Vinnaithaandi Varuvaaya",
    "Munbe Vaa Sillunu Oru Kaadhal",
    "Uyirin Uyire Kaakha Kaakha",
    "Kannazhaga Moonu",
    "Kadhal Rojave Roja",
    "Thendral Vanthu Theendum Pothu",
    "Oru Maalai Ghajini",
    "Pudhu Vellai Mazhai Roja",
    "Omana Penne Kireedam",
    "Roja Jaaneman Roja",
    "En Iniya Pon Nilave Ninaithale Inikkum",
    "Snehithane Alaipayuthey",
    "Mannipaaya Vinnaithaandi Varuvaaya",
    "Adiye 180",
    "Yennamo Yedho Kadal",
    "Kannaana Kanney Viswasam",
    "Oh Penne Vaalee",
    "Nenjame Sathuranga Vettai",
    "Poove Sempoove Muthu",
    "Kaadhal Mannan Kaadhal Mannan",
    "Thanga Magan Thanga Magan",
    "Nee Paartha Vizhigal",
    "Inji Iduppazhagi 3",
    "Chikku Bukku Rayile",
    "Putham Pudhu Kaalai",
    "En Aasai Machan Subramaniapuram",
    "Kangal Nanaindha Azhagiya Tamil Magan",
    "Aalaporaan Tamizhan Mersal",
    "Kuyil Paatu Mani Ratnam",
    "Mudhal Murai Paarthein",
    "Rajavin Parvaiyile Rajinikanth",
    "Ninaithale Inikkum classic Tamil",
    "Keladi Kanmani Keladi Kanmani",
    "Saayndhu Saayndhu Vaa",
    "Vandaan Vandaan Kadhalar Dhinam",
    "Andangkaka Kaaki Sattai",
    "Venmegam Kanden Kadhalai",
    "Irandaam Ulagam title song",
    "Yen Intha Mayakkam Yaaradi Nee Mohini",
    "Kadhal Sadugudu Alaipayuthey",
    "Oh Maname Minsara Kanavu",
    "Unnai Kaanadhu Naan",
    "Vaa Vaa Anbe Anbe Karthik",
    "Poo Malai Vaangi Vandha",
    "Kannamoochi Yenada Kandukonden",
    "Enna Solla Pogirai Kandukondein",
    "Nenjodu Kalandha Mouna Ragam",
    "Panivizhum Malarvanam Mouna Ragam",
    "Mazhai Kuruvi Azhagan",
    "Nee Kobapattal Goa Tamil",
]

SONGS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'songs.csv')

# ── Title keywords ─────────────────────────────────────────────────────────────
# A cover must have at least one of these in its title
PIANO_MUST_HAVE = [
    'piano', 'keyboard', 'piano cover', 'piano version',
    'piano instrumental', 'solo piano', 'piano tutorial',
    'piano rendition', 'piano arrangement', 'piano only',
]
# A cover must NOT have any of these — these indicate other instruments
PIANO_MUST_NOT = [
    'drum', 'guitar', 'violin', 'flute', 'bass', 'veena',
    'mridangam', 'tabla', 'carnatic', 'orchestra', 'band',
    'ensemble', 'karaoke', 'lyrics', 'lyric video',
    'official video', 'trailer', 'making of', 'mashup',
    'bgm collection', 'medley', 'acapella', 'full movie',
    'saxophone', 'saxophone', 'sitar', 'sarod', 'trumpet',
    'cello', 'viola', 'harp', 'with strings', 'with orchestra',
    'vocal cover', 'voice cover', 'cover song',   # covers by a singer, not pianist
]
# Titles containing these suggest it MIGHT have other instruments too (flag as mixed)
MIXED_HINTS = [
    'with drums', 'full band', 'orchestral', 'with beats',
    'arrangement', 'string quartet', 'jazz',
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_name(s):
    s = re.sub(r'[^a-z0-9\s]', '', s.lower())
    s = re.sub(r'\s+', '_', s.strip())
    return re.sub(r'_+', '_', s)[:40]


def load_existing(csv_path):
    if not os.path.exists(csv_path):
        return set(), 0
    names, count = set(), 0
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            names.add(row.get('name', '').lower())
            count += 1
    return names, count


def yt_search(query, n=6):
    """Search YouTube via yt-dlp. Returns list of {url, title, duration}."""
    import yt_dlp
    opts = {
        'quiet': True, 'no_warnings': True,
        'extract_flat': True, 'playlistend': n,
    }
    results = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f'ytsearch{n}:{query}', download=False)
            for entry in (info or {}).get('entries', []):
                if entry and entry.get('id'):
                    results.append({
                        'url':      f"https://www.youtube.com/watch?v={entry['id']}",
                        'title':    entry.get('title', ''),
                        'duration': int(entry.get('duration') or 0),
                    })
    except Exception as e:
        print(f"      [search error] {e}")
    return results


def passes_piano_title_filter(title):
    """
    Return (passes: bool, cover_type: str).

    Two-level check:
      1. MUST have a piano keyword → otherwise it's not a piano cover
      2. MUST NOT have instrument/non-piano keywords → rejects mixed/wrong covers
    """
    t = title.lower()
    has_piano = any(k in t for k in PIANO_MUST_HAVE)
    has_bad   = any(k in t for k in PIANO_MUST_NOT)
    if not has_piano or has_bad:
        return False, 'reject'
    mixed = any(k in t for k in MIXED_HINTS)
    return True, ('mixed' if mixed else 'piano')


def pick_original(results):
    """Pick best result for the original song."""
    bad = ['cover', 'piano', 'karaoke', 'instrumental', 'remix', 'guitar',
           'violin', 'flute', 'live', 'concert', 'reaction']
    for r in results:
        t = r['title'].lower()
        if any(k in t for k in bad):
            continue
        if r['duration'] < 90 or r['duration'] > 650:
            continue
        return r
    # Relax duration
    for r in results:
        t = r['title'].lower()
        if any(k in t for k in bad):
            continue
        if 30 < r['duration'] < 700:
            return r
    return None


def pick_cover(results, orig_duration):
    """Pick best piano-only cover from search results."""
    candidates = []
    for r in results:
        ok, ctype = passes_piano_title_filter(r['title'])
        if not ok:
            continue
        # Duration check: cover should be 30%–200% of original
        if orig_duration and r['duration']:
            ratio = r['duration'] / orig_duration
            if ratio < 0.30 or ratio > 2.0:
                continue
        candidates.append((r, ctype))

    if not candidates:
        return None, None

    # Rank: "piano cover" in title > "piano version" > other piano keywords
    def score(rc):
        r, _ = rc
        t = r['title'].lower()
        if 'piano cover' in t:      return 4
        if 'solo piano' in t:       return 3
        if 'piano version' in t:    return 3
        if 'piano instrumental' in t: return 2
        return 1

    candidates.sort(key=score, reverse=True)
    best_r, best_ctype = candidates[0]
    return best_r, best_ctype


# ─── Audio verification (--verify flag) ───────────────────────────────────────

def verify_cover_audio(url, threshold_percussive=0.25):
    """
    Download 45 seconds of the cover audio and use librosa HPSS to estimate
    how much of the audio is percussive (drums/transients) vs harmonic (piano).

    Returns: ('piano', confidence) | ('mixed', confidence) | ('skip', reason)

    threshold_percussive=0.25 means: if >25% of the audio energy is in the
    percussive component, it's classified as 'mixed' (has drums).

    This catches covers that have "piano" in the title but also have drums
    underneath — a common trick in YouTube cover videos.
    """
    import yt_dlp, librosa, soundfile as sf
    import numpy as np

    tmpdir = tempfile.mkdtemp(prefix='verify_')
    tmp_audio = os.path.join(tmpdir, 'sample.wav')

    try:
        # Download just the first 45 seconds (much faster than full download)
        opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tmpdir, 'sample.%(ext)s'),
            'quiet': True, 'no_warnings': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'wav', 'preferredquality': '0'}],
            'download_ranges': lambda info, _: [{'start_time': 10, 'end_time': 55}],
            'force_keyframes_at_cuts': False,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        # Find the downloaded file
        wav = None
        for f in os.listdir(tmpdir):
            if f.endswith('.wav'):
                wav = os.path.join(tmpdir, f)
                break
        if not wav:
            return 'skip', 'download failed'

        # Load and run HPSS
        y, sr = librosa.load(wav, sr=22050, mono=True, duration=40.0)
        if len(y) < sr * 5:
            return 'skip', 'audio too short'

        y_harm, y_perc = librosa.effects.hpss(y, margin=4)

        # Compare RMS energy in harmonic vs percussive components
        rms_harm = float(np.sqrt(np.mean(y_harm ** 2)))
        rms_perc = float(np.sqrt(np.mean(y_perc ** 2)))
        total    = rms_harm + rms_perc
        if total < 1e-6:
            return 'skip', 'silent audio'

        perc_ratio = rms_perc / total
        harm_ratio = rms_harm / total

        if perc_ratio > threshold_percussive:
            return 'mixed', perc_ratio    # has significant drums
        else:
            return 'piano', perc_ratio   # mostly harmonic = clean piano

    except Exception as e:
        return 'skip', str(e)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = sys.argv[1:]
    target  = 30
    verify  = False

    for a in args:
        if a == '--verify':
            verify = True
        elif a.isdigit():
            target = int(a)

    print(f"\n🎹  Auto-collecting Tamil song + piano cover pairs from YouTube")
    print(f"    Target:   {target} new pairs")
    print(f"    Verify:   {'YES — downloading audio clips to check for drums' if verify else 'NO  — title filtering only (run with --verify to enable)'}")
    print(f"    CSV:      {SONGS_CSV}")
    if verify:
        print(f"\n    ⏱  Estimated time: ~{target * 1.5:.0f}–{target * 2:.0f} minutes (audio download + HPSS analysis per cover)")
    else:
        print(f"\n    ⏱  Estimated time: ~{target // 8 + 1}–{target // 6 + 1} minutes (title search only)")
    print()

    existing_names, existing_count = load_existing(SONGS_CSV)
    print(f"    Already in CSV: {existing_count} songs\n")

    new_pairs = []
    tried     = 0
    skipped_title   = 0
    skipped_audio   = 0

    for song_query in TAMIL_SONGS:
        if len(new_pairs) >= target:
            break

        words = song_query.split()
        short = ' '.join(words[:4])
        safe  = _safe_name(short)
        tried += 1

        # Skip if name already exists in CSV
        if any(safe in n for n in existing_names):
            print(f"  ⟳  {short} — already in CSV, skipping")
            continue

        print(f"  [{tried}]  {short}")

        # ── Search: original ──────────────────────────────────────────────
        orig_results = yt_search(f"{song_query} official audio Tamil", n=6)
        orig = pick_original(orig_results)
        if not orig:
            print(f"         ⚠  No suitable original found — skipping")
            time.sleep(1.0)
            continue
        print(f"         Original : {orig['title'][:65]}  ({orig['duration']}s)")

        # ── Search: piano cover ───────────────────────────────────────────
        cover_r, ctype = None, None
        for query_suffix in ["piano cover Tamil", "piano cover", "piano instrumental Tamil"]:
            results = yt_search(f"{short} {query_suffix}", n=8)
            cover_r, ctype = pick_cover(results, orig['duration'])
            if cover_r:
                break

        if not cover_r:
            print(f"         ⚠  No piano cover passed title filter — skipping")
            skipped_title += 1
            time.sleep(1.0)
            continue

        print(f"         Cover    : {cover_r['title'][:65]}  ({cover_r['duration']}s)  [{ctype}]")

        # ── Optional: audio verification ──────────────────────────────────
        if verify:
            print(f"         Verifying audio (downloading 45s)…", end='', flush=True)
            result, confidence = verify_cover_audio(cover_r['url'])
            if result == 'skip':
                print(f" ✗  skipped ({confidence})")
                skipped_audio += 1
                time.sleep(1.0)
                continue
            elif result == 'mixed':
                # Has drums but piano is still there — use Demucs to extract piano
                ctype = 'mixed'
                print(f" ⚠  drums detected (percussive ratio={confidence:.1%}) → marked as mixed")
            else:
                ctype = 'piano'
                print(f" ✓  clean piano (percussive ratio={confidence:.1%})")

        # ── Record the pair ───────────────────────────────────────────────
        pair_n    = existing_count + len(new_pairs) + 1
        pair_name = f"auto_{pair_n:03d}_{safe}"

        new_pairs.append({
            'name':         pair_name,
            'original_url': orig['url'],
            'cover_url':    cover_r['url'],
            'cover_type':   ctype,
        })
        print(f"         ✓  Saved as: {pair_name}")

        time.sleep(2.0)   # be respectful to YouTube

    # ── Write to songs.csv ─────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Found:          {len(new_pairs)} new pairs")
    print(f"  Rejected (title): {skipped_title}  Rejected (audio): {skipped_audio}")

    if not new_pairs:
        print("  Nothing new to add.")
        return

    fieldnames   = ['name', 'original_url', 'cover_url', 'cover_type']
    write_header = not os.path.exists(SONGS_CSV)
    with open(SONGS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for p in new_pairs:
            w.writerow(p)

    print(f"\n✅  Added {len(new_pairs)} pairs to songs.csv")
    print(f"    Total in CSV now: {existing_count + len(new_pairs)}")

    print(f"""
┌─────────────────────────────────────────────────────┐
│  NEXT STEPS                                         │
│                                                     │
│  cd ~/Desktop/"Piano player"                        │
│  source venv311/bin/activate                        │
│                                                     │
│  # Step 2: Download + transcribe new songs          │
│  python collect_data.py songs.csv                   │
│  (takes ~{len(new_pairs)*6} min – {len(new_pairs)*10} min depending on song length) │
│                                                     │
│  # Step 3: Rebuild style profile                    │
│  python analyze_pairs.py                            │
│  (takes ~5 minutes)                                 │
│                                                     │
│  # Step 4: Test the converter                       │
│  bash start.sh                                      │
└─────────────────────────────────────────────────────┘
""")


if __name__ == '__main__':
    main()
