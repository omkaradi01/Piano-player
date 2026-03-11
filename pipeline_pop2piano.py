"""
Pop2Piano Pipeline — End-to-end audio-to-piano-MIDI.
Uses the Pop2Piano model from HuggingFace Transformers.
Much simpler than the custom pipeline: audio in → piano MIDI out.
"""

import os, logging, tempfile
import numpy as np

logging.basicConfig(level=logging.INFO, format='[pop2piano] %(message)s')
log = logging.getLogger(__name__)


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


def _load_audio(wav_path, sr=44100):
    """Load audio as numpy array at target sample rate."""
    import librosa
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    return y


# Cache model globally so it's only loaded once
_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor
        import torch

        log.info("Loading Pop2Piano model (first time may download ~1GB)…")
        _model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        _processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

        # Force CPU — Pop2Piano's generate() calls .numpy() which fails on MPS
        device = 'cpu'

        _model = _model.to(device)
        _model.eval()
        log.info("Pop2Piano model loaded!")

    return _model, _processor


def process_youtube_to_piano_midi(url, output_path, progress_cb=None):
    """End-to-end: YouTube URL → piano MIDI file."""
    work_dir = tempfile.mkdtemp(prefix='pop2piano_')

    def prog(msg, pct=None):
        log.info(msg)
        if progress_cb:
            progress_cb(msg, pct)

    try:
        prog("Downloading audio…", 10)
        raw_wav = _download_audio(url, work_dir)

        prog("Loading audio…", 20)
        audio = _load_audio(raw_wav, sr=44100)
        log.info(f"Audio loaded: {len(audio)/44100:.1f}s")

        prog("Loading Pop2Piano model…", 30)
        model, processor = _get_model()

        prog("Generating piano arrangement…", 50)
        import torch

        inputs = processor(
            audio=audio,
            sampling_rate=44100,
            return_tensors="pt"
        )

        # Move inputs to same device as model
        device = next(model.parameters()).device
        input_features = inputs["input_features"].to(device)

        with torch.no_grad():
            model_output = model.generate(
                input_features=input_features,
                composer="composer1",  # default style
            )

        prog("Converting to MIDI…", 85)
        tokenizer_output = processor.batch_decode(
            token_ids=model_output,
            feature_extractor_output=inputs
        )

        midi_obj = tokenizer_output["pretty_midi_objects"][0]
        midi_obj.write(output_path)

        note_count = sum(len(inst.notes) for inst in midi_obj.instruments)
        log.info(f"Pop2Piano MIDI saved: {note_count} notes → {output_path}")

        prog("Done! 🎹", 100)
        return output_path

    except Exception as e:
        log.error(f"Pop2Piano error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
