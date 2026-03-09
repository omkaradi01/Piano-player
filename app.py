"""
Flask web server for Piano Transcriber v18.
Supports two modes: 'detailed' (custom pipeline) and 'quick' (Pop2Piano).
Downloads: MIDI + MP3 (rendered via fluidsynth or pretty_midi).
"""

import os, uuid, threading, subprocess, tempfile
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

JOBS = {}   # job_id -> { status, message, pct, midi_path, mp3_path, error, mode }
MIDI_DIR = os.path.join(os.path.dirname(__file__), 'midi_outputs')
os.makedirs(MIDI_DIR, exist_ok=True)


def _get_pipeline(mode):
    """Import the correct pipeline based on mode."""
    if mode == 'quick':
        from pipeline_pop2piano import process_youtube_to_piano_midi
    else:
        from pipeline import process_youtube_to_piano_midi
    return process_youtube_to_piano_midi


def _midi_to_mp3(midi_path, mp3_path):
    """Convert MIDI to MP3 using pretty_midi + pyfluidsynth + ffmpeg."""
    try:
        import pretty_midi, soundfile as sf, numpy as np
        mid = pretty_midi.PrettyMIDI(midi_path)
        audio = mid.fluidsynth(fs=44100)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        wav_tmp = mp3_path.replace('.mp3', '_synth.wav')
        sf.write(wav_tmp, audio, 44100)
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_tmp, '-b:a', '192k', mp3_path
        ], capture_output=True, timeout=120)
        if os.path.exists(wav_tmp):
            os.remove(wav_tmp)
        if os.path.exists(mp3_path):
            return mp3_path
    except Exception as e:
        print(f"MP3 render failed: {e}")
    return None


def run_job(job_id: str, url: str, mode: str):
    midi_path = os.path.join(MIDI_DIR, f'{job_id}.mid')
    mp3_path = os.path.join(MIDI_DIR, f'{job_id}.mp3')

    def progress_cb(message: str, pct=None):
        JOBS[job_id]['message'] = message
        if pct is not None:
            JOBS[job_id]['pct'] = pct

    JOBS[job_id]['status'] = 'running'
    try:
        pipeline_fn = _get_pipeline(mode)
        pipeline_fn(url, midi_path, progress_cb=progress_cb)

        # Render MP3 from MIDI
        progress_cb("Rendering MP3…", 97)
        mp3_result = _midi_to_mp3(midi_path, mp3_path)

        JOBS[job_id]['status']    = 'done'
        JOBS[job_id]['midi_path'] = midi_path
        JOBS[job_id]['mp3_path']  = mp3_result
        JOBS[job_id]['message']   = 'Ready to play!'
        JOBS[job_id]['pct']       = 100
    except Exception as e:
        JOBS[job_id]['status']  = 'error'
        JOBS[job_id]['message'] = str(e)
        JOBS[job_id]['error']   = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json(force=True)
    url  = (data.get('url') or '').strip()
    mode = (data.get('mode') or 'detailed').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if mode not in ('detailed', 'quick'):
        mode = 'detailed'

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        'status': 'queued',
        'message': f'Queued — {mode} mode',
        'pct': 0,
        'midi_path': None,
        'mp3_path': None,
        'mode': mode,
    }

    t = threading.Thread(target=run_job, args=(job_id, url, mode), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'mode': mode})


@app.route('/api/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify({
        'status':   job['status'],
        'message':  job['message'],
        'pct':      job['pct'],
        'step':     job['message'],
        'progress': job['pct'],
        'error':    job.get('error', ''),
        'mode':     job.get('mode', 'detailed'),
        'has_mp3':  bool(job.get('mp3_path')),
    })


@app.route('/api/midi/<job_id>')
def get_midi(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'MIDI not ready'}), 404
    return send_file(job['midi_path'], mimetype='audio/midi',
                     as_attachment=True,
                     download_name=f'piano_cover_{job_id}.mid')


@app.route('/api/mp3/<job_id>')
def get_mp3(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    mp3 = job.get('mp3_path')
    if not mp3 or not os.path.exists(mp3):
        return jsonify({'error': 'MP3 not available'}), 404
    return send_file(mp3, mimetype='audio/mpeg',
                     as_attachment=True,
                     download_name=f'piano_cover_{job_id}.mp3')


@app.route('/api/info/<job_id>')
def get_info(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    path = job['midi_path']
    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    return jsonify({'midi_size_bytes': size, 'has_mp3': bool(job.get('mp3_path'))})


if __name__ == '__main__':
    print("\n  Piano Transcriber v18 running at http://localhost:5050\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
