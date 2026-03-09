# Piano Transcriber — Implementation TODO

## Phase 0: Fix Critical Issues (App Must Work First)
- [x] **P0-1**: Fix demucs speed — reduce shifts from 3 to 0 for default
- [x] **P0-2**: Fix progress reporting — backend/frontend field name mismatch (message→step, pct→progress)
- [x] **P0-3**: Fix frontend polling — progress bar and step labels now update correctly
- [ ] **P0-4**: Add error handling for yt-dlp failures (age-restricted, geo-blocked, invalid URLs) with clear user messages
- [ ] **P0-5**: Test end-to-end flow: paste URL → see progress → hear playback → download MIDI

## Phase 1: Add Pop2Piano as Fast Alternative Pipeline
- [x] **P1-1**: Install Pop2Piano via HuggingFace Transformers
- [x] **P1-2**: Create `pipeline_pop2piano.py` — end-to-end audio-to-piano
- [x] **P1-3**: Add "Quick (Pop2Piano)" vs "Detailed (Custom Pipeline)" toggle in the UI
- [x] **P1-4**: Pop2Piano pipeline: download audio → load model → generate piano MIDI → done
- [ ] **P1-5**: Test Pop2Piano on Tamil songs — evaluate quality vs current pipeline
- [ ] **P1-6**: Evaluate Music2MIDI (github.com/ytinyui/music2midi) as potential upgrade over Pop2Piano

## Phase 2: Upgrade Source Separation
- [x] **P2-1**: Install Mel-Band RoFormer via audio-separator pip package
- [x] **P2-2**: Model weights auto-download (mel_band_roformer_kim_ft_unwa.ckpt)
- [x] **P2-3**: Create `_separate_audio_roformer()` function in pipeline.py
- [ ] **P2-4**: Benchmark: compare Mel-Band RoFormer vs htdemucs_ft vocal isolation quality on 3 test songs
- [x] **P2-5**: Make Mel-Band RoFormer the default, htdemucs_ft as fallback
- [x] **P2-6**: Remove shifts=3 from demucs fallback (use shifts=0 for speed)

## Phase 3: Upgrade Pitch Tracking
- [x] **P3-1**: Create standalone rmvpe.py module + download rmvpe.pt weights (173MB)
- [x] **P3-2**: Create `_run_rmvpe_raw()` function — returns (times, hz, confidence)
- [x] **P3-3**: Implement pitch-contour-based note segmentation (`_pitch_contour_to_notes`)
- [x] **P3-4**: Add gamaka-aware ornament collapsing (`_filter_ornaments_gamaka`)
- [ ] **P3-5**: Benchmark: compare RMVPE vs torchcrepe on 3 Tamil songs with known melodies
- [x] **P3-6**: Make RMVPE primary, torchcrepe fallback. basic-pitch removed from fallback chain

## Phase 4: Upgrade Key/Scale Detection (Raga-Aware)
- [x] **P4-1**: Add Carnatic raga pitch-class profiles to `_SCALES` dict:
  - Shankarabharanam (Ionian): [0,2,4,5,7,9,11]
  - Kharaharapriya (Dorian): [0,2,3,5,7,9,10]
  - Kalyani (Lydian): [0,2,4,6,7,9,11]
  - Mohanam (pentatonic): [0,2,4,7,9]
  - Mayamalavagowla: [0,1,4,5,7,8,11]
  - Harikambhoji: [0,2,4,5,7,9,10]
  - Thodi: [0,1,3,5,7,8,10]
  - Karaharapriya: [0,2,3,5,7,9,10]
  - Hindolam (pentatonic): [0,3,5,8,10]
  - Natabhairavi: [0,2,3,5,7,8,10]
- [x] **P4-2**: Modify `_detect_key()` to correlate chroma against ALL profiles (Western + raga), pick best match
- [ ] **P4-3**: Evaluate compIAM library (`pip install compiam`) for ML-based raga detection
- [ ] **P4-4**: If compIAM works well, use it on vocal stem for raga identification, fall back to chroma correlation
- [x] **P4-5**: Update `_constrain_to_key_soft()` to use the detected raga's scale degrees

## Phase 5: Upgrade Beat Tracking
- [x] **P5-1**: Install beat_this from GitHub — ISMIR 2024 SOTA beat tracker
- [x] **P5-2**: Create `_get_beats_beat_this()` using beat_this
- [x] **P5-3**: Handle tempo changes — beat_this returns per-beat positions
- [ ] **P5-4**: Benchmark: compare beat_this vs librosa on 3 Tamil songs with complex rhythm
- [x] **P5-5**: Make beat_this primary, librosa as fallback

## Phase 6: Upgrade Chord Detection
- [x] **P6-1**: Install madmom CNNChordRecognition from GitHub
- [ ] **P6-2**: Evaluate BTC (github.com/jayg996/BTC-ISMIR19) — clone, test on 3 songs
- [x] **P6-3**: Integrate madmom as primary, chroma templates as fallback
- [ ] **P6-4**: Support 7th chords (maj7, min7, dom7) in addition to triads
- [ ] **P6-5**: Benchmark: compare new chord detection vs current chroma templates

## Phase 7: Upgrade Left Hand Arrangement
- [x] **P7-1**: Implement voice leading algorithm:
  - For each chord, evaluate root position + inversions
  - Score by total voice movement from previous chord
  - Penalize parallel fifths/octaves
  - Pick smoothest voicing
- [x] **P7-2**: Add multiple LH patterns:
  - Arpeggiated (BPM < 80, slow ballads)
  - Root-fifth (BPM 80-120, current pattern)
  - Alberti bass (BPM > 120, upbeat songs)
  - Block chords (verse sections, sparse)
- [x] **P7-3**: Implement chord-change-aware sustain pedal:
  - Release pedal slightly before chord changes
  - Re-press on new chord
  - Shorter pedal for fast tempos
- [ ] **P7-4**: Add dynamics variation — verse quieter, chorus louder
- [ ] **P7-5**: Evaluate AccoMontage2 (github.com/billyblu2000/AccoMontage2) for ML-based piano arrangement

## Phase 8: Upgrade Quantization
- [x] **P8-1**: Implement multi-resolution quantization grid:
  - Try 8th notes, 16th notes, AND triplet 8ths
  - Snap each note to whichever grid point is closest
- [x] **P8-2**: Adaptive quantization strength:
  - Stronger (0.8) for notes near beat positions
  - Weaker (0.3) for notes between beats
  - Zero for very short notes (grace notes, gamakas)
- [x] **P8-3**: Add slight humanization — random timing offset (±12ms) and velocity variation

## Phase 9: Structure-Aware Processing
- [ ] **P9-1**: Install allin1 (`pip install allin1`) for song structure detection
- [ ] **P9-2**: Detect sections: intro, verse, chorus, bridge, instrumental, outro
- [ ] **P9-3**: Use vocal stem for sung sections, "other" stem for instrumental sections
- [ ] **P9-4**: Apply different density targets per section (verse sparse, chorus dense)
- [ ] **P9-5**: Section-aware key detection (handle modulations between sections)

## Phase 10: Performance & UX
- [ ] **P10-1**: Add processing time estimate in UI based on audio duration
- [ ] **P10-2**: Cache downloaded audio — if same URL is submitted again, skip download
- [ ] **P10-3**: Add "Cancel" button for in-progress jobs
- [ ] **P10-4**: Show waveform or spectrogram visualization during processing
- [ ] **P10-5**: Add A/B comparison — let user hear Pop2Piano vs Custom Pipeline results side by side
- [ ] **P10-6**: Mobile-responsive UI improvements

## Notes
- Each phase should be independently testable
- Always keep fallbacks — new model fails? Fall back to previous model
- Test each upgrade on at least 3 songs: 1 Tamil ballad, 1 Tamil upbeat, 1 Western pop
- Track quality improvements in a test log
