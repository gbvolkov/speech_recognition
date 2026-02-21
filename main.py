# %%
from pathlib import Path
import os

import logging
from transcribe import transcription_factory, transcribe

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
WHISPER_MODEL = "openai/whisper-large-v3"
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.wav', '.mp4'}


def main():
    logging.basicConfig(level=logging.INFO)
    transcriptor = transcription_factory(WHISPER_MODEL, DIARIZATION_MODEL)

    folder = Path("./audio/")
    done_folder = os.path.join(str(folder.resolve()), "done")
    os.makedirs(done_folder, exist_ok=True)

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            full_path = file.resolve()
            logging.info(f"Found audio file: {full_path}")
            transcribe(str(full_path), transcriptor)
            done_file = os.path.join(done_folder, os.path.basename(full_path))
            os.rename(str(full_path), done_file)

    logging.info("Complete")


if __name__ == "__main__":
    main()
