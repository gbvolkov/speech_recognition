import math
from transformers.models.whisper import tokenization_whisper
import pandas as pd

# Monkey-patch _find_longest_common_sequence to replace None values with math.nan.
_original_find_lcs = tokenization_whisper._find_longest_common_sequence
with open('hf.txt') as f:
    HF_TOKEN = f.read()

def _patched_find_longest_common_sequence(*args, **kwargs):
    """
    Patched version of _find_longest_common_sequence that sanitizes timestamp lists,
    replacing None with math.nan so that comparisons don't fail.
    """
    print(f"####PATCH: {len(args)}")
    args = list(args)
    if len(args) >= 4:
        # args[2] and args[3] are assumed to be the left and right token timestamp sequences.
        left_timestamps = args[2]
        right_timestamps = args[3]
        args[2] = [ts if ts is not None else math.nan for ts in left_timestamps]
        args[3] = [ts if ts is not None else math.nan for ts in right_timestamps]
    return _original_find_lcs(*args, **kwargs)

# Apply the monkey patch.
#tokenization_whisper._find_longest_common_sequence = _patched_find_longest_common_sequence


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment

import textwrap
import logging

LOCAL_MODEL = False

def merge_chunks_into_sentences(chunks):
    """
    Merge consecutive transcription chunks with the same speaker into complete sentences based on punctuation.
    A sentence is considered complete if it ends with '.', '!' or '?'.
    """
    if not chunks:
        return []
    # Ensure chunks are sorted by start time.
    chunks.sort(key=lambda x: x["start"])
    sentences = []
    current_sentence = None

    for chunk in chunks:
        # Start a new sentence if none exists.
        if current_sentence is None:
            current_sentence = {
                "speaker": chunk["speaker"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"].strip()
            }
        else:
            # If same speaker and gap is small, continue the sentence.
            if chunk["speaker"] == current_sentence["speaker"] and (chunk["start"] - current_sentence["end"]) < 1.0:
                current_sentence["text"] += " " + chunk["text"].strip()
                current_sentence["end"] = chunk["end"]
            else:
                sentences.append(current_sentence)
                current_sentence = {
                    "speaker": chunk["speaker"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"].strip()
                }
        # Finalize the sentence if it ends with a sentence-ending punctuation.
        if current_sentence and current_sentence["text"] and current_sentence["text"][-1] in ".!?":
            sentences.append(current_sentence)
            current_sentence = None

    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def save_speech_to_file_with_indent(segments, filename):
    text = ""
    with open(filename, "w", encoding="utf-8") as file:
        for segment in segments:
            # Format the speaker tag.
            speaker_tag = f"{segment['speaker'].upper()}:\n"
            
            # Wrap the text to 128 characters and indent each line.
            wrapped_text = textwrap.fill(segment["text"], width=128, subsequent_indent="    ")
            
            # Write the formatted text to the file.
            text = text + speaker_tag + wrapped_text + "\n\n"
            file.write(speaker_tag)
            file.write(wrapped_text)
            file.write("\n\n")  # Add a blank line between speakers.
        return text

def convert_audio_to_wav(input_file, output_file, audio_type):
    """
    Converts an M4A (or similar) file to WAV format.
    """
    audio = AudioSegment.from_file(input_file, format=audio_type)
    audio.export(output_file, format='wav')
    
    logging.info(f"Successfully converted '{input_file}' to '{output_file}'")

def deduplicate(chunked_script):
    deduplicated = []
    current_text = ''
    for chunk in chunked_script:
        if chunk['text'].strip() != current_text.strip():
            deduplicated.append(chunk)
            current_text = chunk['text']
        elif chunk['timestamp'][0] < deduplicated[-1]['timestamp'][1]:
            start = min(deduplicated[-1]['timestamp'][0], chunk['timestamp'][0])
            end = max(deduplicated[-1]['timestamp'][1], chunk['timestamp'][1])
            deduplicated[-1]['timestamp'] = (start, end)
    return deduplicated

def transcription_factory(whisper_model_id, diarization_model_id, align_model_id=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(device)

    # Initialize Whisper pipeline.
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.config.forced_decoder_ids = None
    whisper_model.to(device)
    whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=30,  # Process audio in 30-second chunks.
        stride_length_s=10,  # Optional overlap between chunks.    
        torch_dtype=torch_dtype,
        device=device,
    )

    diarization_pipeline = Pipeline.from_pretrained(diarization_model_id, use_auth_token=HF_TOKEN)
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    def transcript(file_name):
        logging.info(f'=============>started with {file_name}')
        script = whisper_pipe(file_name, return_timestamps='word', generate_kwargs={"language": "russian"})
        logging.info(f'Loaded transcript for {file_name}')
        diarized = diarization_pipeline(file_name)
        logging.debug(diarized)
        speaker_transcription = []

        #######
        df = pd.DataFrame(script["chunks"])
        df.to_csv('./data/chunks.csv', index=False)
        #######
        #######
        df = pd.DataFrame(diarized.itertracks(yield_label=True))
        df.to_csv('./data/diarized.csv', index=False)
        #######

        pre_chunks = sorted(script['chunks'], 
            key=lambda x: (
                x['timestamp'][0] if x['timestamp'][0] is not None else float('inf'),
                x['timestamp'][1] if x['timestamp'][1] is not None else float('inf')        
        ))
        #######
        df = pd.DataFrame(pre_chunks)
        df.to_csv('./data/pre_chunks.csv', index=False)
        #######
        
        chunks = deduplicate(pre_chunks)
        
        #######
        df = pd.DataFrame(chunks)
        df.to_csv('./data/deduplicated_chunks.csv', index=False)
        #######
            
        for chunk in chunks:
            start_time = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else float('inf')
            end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else float('inf')
            speaker = "Unknown"
            for turn, _, speaker_label in diarized.itertracks(yield_label=True):
                start = max(start_time, turn.start)
                end = min(end_time, turn.end)
                if start <= end:
                    speaker = speaker_label
                    break
            speaker_transcription.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": chunk["text"]
            })
        logging.debug(speaker_transcription)
        
        #######
        df = pd.DataFrame(speaker_transcription)
        df.to_csv('./data/speaker_transcription.csv', index=False)
        #######    

        transcribed = []
        for segment in speaker_transcription:
            transcribed.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment["speaker"] if 'speaker' in segment else "ND"
            })
        logging.debug(transcribed)
        
        #######
        df = pd.DataFrame(transcribed)
        df.to_csv('./data/transcribed.csv', index=False)
        #######    

        # Use the new sentence-completeness approach.
        merged = merge_chunks_into_sentences(transcribed)
        
        #######
        df = pd.DataFrame(merged)
        df.to_csv('./data/merged.csv', index=False)
        #######    


        trans_folder = os.path.join(os.path.dirname(file_name), 'transcripts/')
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        text = save_speech_to_file_with_indent(merged, trans_file)
        logging.info(f'<=============Done with {file_name}')
        return text

    return transcript

def transcribe(audio_name, transcriptor):
    wav_name = audio_name
    name, ext = os.path.splitext(audio_name)
    ext = ext.replace('.', '')
    btemp = False
    if ext != 'wav':
        wav_name = f"{name}.wav"
        convert_audio_to_wav(audio_name, wav_name, ext)
        btemp = True
    text = transcriptor(wav_name)
    if btemp:
        os.remove(wav_name)
    return text

def move_to_done(filename):
    dest_dir = './audio/done'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_path = os.path.join(dest_dir, os.path.basename(filename))
    os.rename(filename, dest_path)
   
def run_transcription(file_name):
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    diarization_model = "pyannote/speaker-diarization-3.1"
    align_model = 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'
    whisper_model = "openai/whisper-large-v3"

    transcriptor = transcription_factory(whisper_model, diarization_model)
    transcription_text = transcribe(file_name, transcriptor)
    
    move_to_done(file_name)
    return transcription_text

if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    with open('hf.txt') as f:
        HF_TOKEN = f.read()
    
    diarization_model = "pyannote/speaker-diarization-3.1"
    align_model = 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'
    whisper_model = "openai/whisper-large-v3"
    
    transcriptor = transcription_factory(whisper_model, diarization_model)

    folder_path = './audio/'
    AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.wav'}

    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            full_path = file.resolve()
            audio_type = file.suffix.lower().replace('.', '')
            logging.info(f"Found audio file: {full_path} (Type: {audio_type})")
            out_file, _ = os.path.splitext(full_path)
            out_file = f"{out_file}.wav"        
            transcribe(str(full_path), transcriptor)
            print(f"<============= Done with {full_path}")