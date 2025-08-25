import os
import re
import uuid
import threading
import queue
from io import BytesIO
from flask import Flask, render_template, request, make_response, jsonify
from docx import Document
from transcribe import run_transcription
import html2text  # For HTML-to-Markdown conversion

# Allowed audio file extensions

#ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'aiff', 'avi', 'mp4', 'mov', 'mkv', 'webm', 'mpg'}

ALLOWED_EXTENSIONS = {
    # Audio-only containers:
    'aac', 'aax', 'aa', 'ac3', 'ac4', 'aiff', 'au', 'caf', 'flac',
    'mp2', 'mp3', 'ogg', 'opus', 'wav', 'w64', 'wv', 'tta', 'ape',
    'm4a', 'wma', 'ads',
    # Video containers (if they contain audio):
    'avi', 'mp4', 'mov', 'mpeg', 'mpg', 'ts', 'mkv', 'flv', 'webm',
    '3gp', '3g2', 'mxf', 'vob', 'rm', 'swf', 'm4v', 'ismv', 'nut',
    'asf', 'matroska', 'ogv'
}

def allowed_file(filename):
    """Check if the file has a valid audio extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global structures for job management.
# NOTE: In a multi-process environment these globals won't be shared; consider using a persistent store.
transcription_jobs = {}
transcription_jobs_lock = threading.Lock()  # Protects transcription_jobs
transcription_queue = queue.Queue()
pending_jobs = []
pending_jobs_lock = threading.Lock()
current_job = None
current_job_lock = threading.Lock()

def transcription_worker():
    """Background worker to process transcription jobs."""
    global current_job
    while True:
        job_id, file_path = transcription_queue.get()
        # Remove the job from pending jobs and mark as current.
        with pending_jobs_lock:
            if job_id in pending_jobs:
                pending_jobs.remove(job_id)
        with current_job_lock:
            current_job = job_id
        try:
            transcript = run_transcription(file_path)
            with transcription_jobs_lock:
                transcription_jobs[job_id] = transcript
        except Exception as e:
            with transcription_jobs_lock:
                transcription_jobs[job_id] = f"Error during transcription: {str(e)}"
        finally:
            with current_job_lock:
                current_job = None
            transcription_queue.task_done()

def transcribe_audio(audio_file):
    """Save the audio file and enqueue it for transcription."""
    audio_dir = "audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    file_path = os.path.join(audio_dir, f"{uuid.uuid4()}_{audio_file.filename}")
    audio_file.save(file_path)
    
    job_id = uuid.uuid4().hex
    with transcription_jobs_lock:
        transcription_jobs[job_id] = None  # None indicates pending transcription.
    
    with pending_jobs_lock:
        pending_jobs.append(job_id)
    
    transcription_queue.put((job_id, file_path))
    return job_id

def create_app():
    """Application factory function for creating the Flask app."""
    app = Flask(__name__)
    
    # Start the transcription worker thread (once per application instance)
    if not hasattr(app, 'worker_thread_started'):
        worker_thread = threading.Thread(target=transcription_worker, daemon=True)
        worker_thread.start()
        app.worker_thread_started = True

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        audio_file = request.files.get("audio_file")
        if not audio_file:
            return "No audio file provided", 400
        
        # Validate file type
        if not allowed_file(audio_file.filename):
            return "Invalid file type. Please upload a valid audio file.", 400

        job_id = transcribe_audio(audio_file)
        return render_template("waiting.html", job_id=job_id)

    @app.route("/check_status/<job_id>")
    def check_status(job_id):
        with transcription_jobs_lock:
            transcript = transcription_jobs.get(job_id)
        if transcript is not None:
            return jsonify(status="completed")
        
        with current_job_lock:
            if current_job == job_id:
                return jsonify(status="processing", position=1)
        
        with pending_jobs_lock:
            if job_id in pending_jobs:
                position = pending_jobs.index(job_id) + 2  # 1-based indexing; +1 for current job
                queue_length = len(pending_jobs) + 1
                return jsonify(status="pending", position=position, queue_length=queue_length)
        
        return jsonify(status="unknown")

    @app.route("/editor")
    def editor():
        job_id = request.args.get("job_id")
        with transcription_jobs_lock:
            transcript = transcription_jobs.get(job_id)
        if not transcript:
            return "Transcription is still in progress or not found.", 404

        def preformat_transcript(text):
            import re
            # Collapse multiple newline characters into one.
            text = re.sub(r'\n+', '\n', text)
            
            # Pattern to match a speaker followed by a timestamp (and an optional colon)
            pattern = re.compile(
                r'(?P<speaker>[^\[\]\n:]+?)\s*'
                r'(?P<time>\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\])'
                r'(?P<colon>:?)'
            )
            def replacer(match):
                speaker = match.group('speaker').strip()
                time = match.group('time').strip()
                colon = match.group('colon')
                return f'<b class="speaker-name">{speaker}</b> <i>{time}</i>{colon}'                
                #return f"<b>{speaker}</b> <i>{time}</i>{colon}"
            formatted_text = pattern.sub(replacer, text)
            
            # Instead of replacing newlines with <br/>, wrap each non-empty line in <p> tags.
            lines = formatted_text.split("\n")
            paragraphs = [f"<p>{line.strip()}</p>" for line in lines if line.strip()]
            return "".join(paragraphs)
        
        formatted_transcript = preformat_transcript(transcript)
        # Extract speakers for renaming (if needed)
        import re
        #speakers = re.findall(r'(SPEAKER_\d+|UNKNOWN)', transcript)
        speakers = re.findall(r'\*\*(.*?)\*\*', transcript)
        speakers = list(set(speakers))
        return render_template("editor.html", transcript=formatted_transcript, speakers=speakers)

    @app.route("/download", methods=["POST"])
    def download():
        # Expecting HTML content from the WYSIWYG editor
        html = request.form.get("html_content", "")
        format_type = request.form.get("format", "markup")
        
        if format_type == "word":
            # Use html2docx for pure Python conversion of HTML to DOCX.
            from html2docx import html2docx
            from io import BytesIO

            # Convert HTML to a python-docx Document object.
            document = html2docx(html, title="Transcript")
            # Save the document to an in-memory bytes buffer.
            #file_stream = BytesIO()
            #document.save(file_stream)
            #file_stream.seek(0)
            docx_data = document.getvalue()
            
            response = make_response(docx_data)
            response.headers["Content-Disposition"] = "attachment; filename=transcript.docx"
            response.mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            import html2text
            # Convert HTML to Markdown while preserving basic formatting
            markdown_text = html2text.html2text(html)
            filename = "transcript.md"
            response = make_response(markdown_text)
            response.headers["Content-Disposition"] = f"attachment; filename={filename}"
            response.mimetype = "text/markdown"
        
        return response

    return app

if __name__ == "__main__":
    import os
    pid = os.getpid()
    with open(".process", "w") as f:
        f.write(f"{pid}")
    app = create_app()
    app.run(debug=False, host="0.0.0.0")
