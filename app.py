import os
import re
import uuid
import threading
from io import BytesIO
from flask import Flask, render_template, request, make_response, jsonify
from docx import Document

from transcribe import run_transcription  


app = Flask(__name__)

# Global dictionary to store transcription results
transcription_jobs = {}

# Updated transcribe_audio function that saves the file, then runs transcription asynchronously.
def transcribe_audio(audio_file):
    # Ensure the 'audio' folder exists.
    audio_dir = "audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    # Save the uploaded audio file to the server.
    file_path = os.path.join(audio_dir, audio_file.filename)
    audio_file.save(file_path)
    
    # Generate a unique job id and mark the job as pending.
    job_id = uuid.uuid4().hex
    transcription_jobs[job_id] = None  # None indicates that transcription is pending.
    
    # Run transcription asynchronously.
    def async_transcription(job_id, file_path):
        # Import run_transcription from your module (to be provided later).
        transcript = run_transcription(file_path)
        transcription_jobs[job_id] = transcript  # Save the transcript once done.
    
    threading.Thread(target=async_transcription, args=(job_id, file_path)).start()
    return job_id

# Home page remains unchanged.
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# The /transcribe route now calls transcribe_audio and then renders a waiting page.
@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return "No audio file provided", 400

    job_id = transcribe_audio(audio_file)
    return render_template("waiting.html", job_id=job_id)

# Endpoint for checking the status of the transcription job.
@app.route("/check_status/<job_id>")
def check_status(job_id):
    transcript = transcription_jobs.get(job_id)
    if transcript is not None:
        return jsonify(status="completed")
    return jsonify(status="pending")

# Once transcription is complete, the /editor route loads the transcript.
@app.route("/editor")
def editor():
    job_id = request.args.get("job_id")
    transcript = transcription_jobs.get(job_id)
    if not transcript:
        return "Transcription is still in progress or not found.", 404

    # Extract unique speaker labels (e.g., SPEAKER_00 or UNKNOWN)
    speakers = re.findall(r'(SPEAKER_\d+|UNKNOWN)', transcript)
    speakers = list(set(speakers))
    return render_template("editor.html", transcript=transcript, speakers=speakers)

# Download route remains largely unchanged.
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
