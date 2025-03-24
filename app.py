from flask import Flask, render_template, request, make_response
import re
from io import BytesIO
import os

# Import for creating a Word document.
from docx import Document


from transcribe import run_transcription  

app = Flask(__name__)

# Dummy transcription function – replace with your actual implementation.

def transcribe_audio(audio_file):
    # Ensure the 'audio' folder exists.
    audio_dir = "audio"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    # Save the uploaded audio file to the server.
    file_path = os.path.join(audio_dir, audio_file.filename)
    audio_file.save(file_path)
    
    # Call the external transcription procedure.
    # Replace 'transcription_module' with the actual module name when available.

    transcript = run_transcription(file_path)
    return transcript

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return "No audio file provided", 400

    transcript = transcribe_audio(audio_file)

    # Extract unique speaker labels
    speakers = re.findall(r'(SPEAKER_\d+|UNKNOWN)', transcript)
    speakers = list(set(speakers))

    return render_template("editor.html", transcript=transcript, speakers=speakers)

@app.route("/download", methods=["POST"])
def download():
    transcript = request.form.get("transcript", "")
    format_type = request.form.get("format", "markup")
    
    if format_type == "word":
        # Convert transcript to a Word document using python-docx
        document = Document()
        document.add_paragraph(transcript)
        file_stream = BytesIO()
        document.save(file_stream)
        file_stream.seek(0)
        response = make_response(file_stream.read())
        response.headers["Content-Disposition"] = "attachment; filename=transcript.docx"
        response.mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        # Download as Markdown file
        filename = "transcript.md"
        response = make_response(transcript)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.mimetype = "text/markdown"
    
    return response

if __name__ == "__main__":
    app.run(debug=True)
