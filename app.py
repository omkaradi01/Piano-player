"""
Flask web server for Piano Transcriber.
"""

import os, uuid, threading, time
from flask import Flask, request, jsonify, render_template, send_file
from pipeline import process_youtube_to_piano_midi

app = Flask(__name__)

JOBS = {}   # job_id -> { status, message, pct, midi_path, error }
MIDI_DIR = os.path.join(os.path.dirname(__file__), 'midi_outputs')
os.makedirs(MIDI_DIR, exist_ok=True)


def run_job(job_id: str, url: str):
    midi_path = os.path.join(MIDI_DIR, f'{job_id}.mid')

    def progress_cb(message: str, pct=None):
        JOBS[job_id]['message'] = message
        if pct is not None:
            JOBS[job_id]['pct'] = pct

    JOBS[job_id]['status'] = 'running'
    try:
        process_youtube_to_piano_midi(url, midi_path, progress_cb=progress_cb)
        JOBS[job_id]['status']    = 'done'
        JOBS[job_id]['midi_path'] = midi_path
        JOBS[job_id]['message']   = 'Ready to play!'
        JOBS[job_id]['pct']       = 100
    except Exception as e:
        JOBS[job_id]['status']  = 'error'
        JOBS[job_id]['message'] = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json(force=True)
    url  = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {'status': 'queued', 'message': 'Queued — waiting to start',
                    'pct': 0, 'midi_path': None}

    t = threading.Thread(target=run_job, args=(job_id, url), daemon=True)
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify({
        'status':  job['status'],
        'message': job['message'],
        'pct':     job['pct'],
    })


@app.route('/api/midi/<job_id>')
def get_midi(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'MIDI not ready'}), 404
    return send_file(job['midi_path'], mimetype='audio/midi',
                     as_attachment=False)


@app.route('/api/info/<job_id>')
def get_info(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    path = job['midi_path']
    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    return jsonify({'midi_size_bytes': size, 'midi_path': path})


if __name__ == '__main__':
    print("\n🎹  Piano Transcriber running at http://localhost:5050\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
