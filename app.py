import os
import re
import uuid
import threading
import queue
from io import BytesIO
from flask import Flask, render_template, request, make_response, jsonify
from docx import Document
from transcribe import run_transcription

app = Flask(__name__)

# Global dictionary to store transcription results
transcription_jobs = {}

# Global queue for transcription jobs
transcription_queue = queue.Queue()

# Global list and lock for tracking pending job order
pending_jobs = []
pending_jobs_lock = threading.Lock()

# Variable and lock for the currently processing job
current_job = None
current_job_lock = threading.Lock()

def transcription_worker():
    global current_job
    while True:
        job_id, file_path = transcription_queue.get()
        # Mark this job as currently processing and remove it from pending list
        with pending_jobs_lock:
            if job_id in pending_jobs:
                pending_jobs.remove(job_id)
        with current_job_lock:
            current_job = job_id
        try:
            transcript = run_transcription(file_path)
            transcription_jobs[job_id] = transcript
        except Exception as e:
            transcription_jobs[job_id] = f"Error during transcription: {str(e)}"
        finally:
            with current_job_lock:
                current_job = None
            transcription_queue.task_done()

# Start the worker thread as a daemon
worker_thread = threading.Thread(target=transcription_worker, daemon=True)
worker_thread.start()

def transcribe_audio(audio_file):
    # Ensure the 'audio' folder exists.
    audio_dir = "audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    # Save the uploaded audio file to the server.
    file_path = os.path.join(audio_dir, f"{uuid.uuid4()}_{audio_file.filename}")
    audio_file.save(file_path)
    
    # Generate a unique job id and mark it as pending.
    job_id = uuid.uuid4().hex
    transcription_jobs[job_id] = None  # None indicates pending transcription.
    
    # Add job id to the pending jobs list
    with pending_jobs_lock:
        pending_jobs.append(job_id)
    
    # Enqueue the job rather than starting a new thread
    transcription_queue.put((job_id, file_path))
    return job_id

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return "No audio file provided", 400

    job_id = transcribe_audio(audio_file)
    return render_template("waiting.html", job_id=job_id)

@app.route("/check_status/<job_id>")
def check_status(job_id):
    # Check if the transcription is complete
    transcript = transcription_jobs.get(job_id)
    if transcript is not None:
        return jsonify(status="completed")
    
    # Otherwise, determine the queue position
    with current_job_lock:
        if current_job == job_id:
            # Currently processing job
            return jsonify(status="processing", position=1)
    
    with pending_jobs_lock:
        if job_id in pending_jobs:
            position = pending_jobs.index(job_id) + 2  # 1-based indexing; +1 for current job
            queue_length = len(pending_jobs)+1
            return jsonify(status="pending", position=position, queue_length=queue_length)
    
    # Fallback if not found in either (could be an error)
    return jsonify(status="unknown")

@app.route("/editor")
def editor():
    job_id = request.args.get("job_id")
    transcript = transcription_jobs.get(job_id)
    if not transcript:
        return "Transcription is still in progress or not found.", 404

    speakers = re.findall(r'(SPEAKER_\d+|UNKNOWN)', transcript)
    speakers = list(set(speakers))
    return render_template("editor.html", transcript=transcript, speakers=speakers)

@app.route("/download", methods=["POST"])
def download():
    transcript = request.form.get("transcript", "")
    format_type = request.form.get("format", "markup")
    
    if format_type == "word":
        document = Document()
        document.add_paragraph(transcript)
        file_stream = BytesIO()
        document.save(file_stream)
        file_stream.seek(0)
        response = make_response(file_stream.read())
        response.headers["Content-Disposition"] = "attachment; filename=transcript.docx"
        response.mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        filename = "transcript.md"
        response = make_response(transcript)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.mimetype = "text/markdown"
    
    return response

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
