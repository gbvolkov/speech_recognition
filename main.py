# %%
from pathlib import Path
import os

import logging
from transcribe import convert_audio_to_wav, transcription_factory, transcribe


# %%
HF_TOKEN="XXXXXX"

diarization_model="pyannote/speaker-diarization-3.1"
align_model='jonatasgrosman/wav2vec2-large-xlsr-53-russian'
whisper_model="openai/whisper-large-v3-turbo"
logging.basicConfig(level=logging.INFO)

transcriptor = transcription_factory(whisper_model, diarization_model)

folder_path = './audio/'
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.wav', '.mp4'}

# %%
folder = Path(folder_path)
full_path = folder.resolve()
done_folder = os.path.join(str(full_path), 'done/')
os.makedirs(done_folder, exist_ok=True)
        
# Iterate through all files in the directory (non-recursive)
for file in folder.iterdir():
    if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
        full_path = file.resolve()
        audio_type = file.suffix.lower().replace('.', '')  # e.g., 'mp3'
        logging.info(f"Found audio file: {full_path} (Type: {audio_type})")
        # Call the conversion function
        transcribe(str(full_path), transcriptor)
        #move file to done
        done_file = os.path.join(done_folder, f"{os.path.basename(full_path)}")
        os.rename(str(full_path), done_file)
        
logging.info("Complete")

# %%



