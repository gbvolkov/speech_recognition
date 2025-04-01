import math
from transformers.models.whisper import tokenization_whisper

with open('hf.txt') as f:
    HF_TOKEN = f.read()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment

import textwrap
import logging

# Import PySBD for rule-based sentence segmentation.
import pysbd

LOCAL_MODEL = False

import gc
import torch

def cleanup_gpu_memory():
    """Clean up GPU memory by running garbage collection and clearing CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_timestamp(seconds):
    """
    Format seconds into HH:MM:SS string.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def merge_chunks_into_sentences(chunks):
    """
    Merge consecutive transcription chunks into longer segments using heuristics.
    This version does not use any speaker information.
    Merging is based solely on temporal proximity and punctuation (i.e. if the current text does not end with a sentence-terminator).
    """
    if not chunks:
        return []
    # Sort chunks by start time.
    chunks.sort(key=lambda x: x["start"])
    merged_segments = []
    current_segment = None

    for chunk in chunks:
        if current_segment is None:
            current_segment = {
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"].strip()
            }
        else:
            #gap = chunk["start"] - current_segment["end"]
            ends_strong = current_segment["text"] and current_segment["text"][-1] in ".!?"
            # Merge if within the allowed gap and the current segment hasn't ended with strong punctuation.
            if not ends_strong:
                current_segment["text"] += " " + chunk["text"].strip()
                current_segment["end"] = chunk["end"]
            else:
                merged_segments.append(current_segment)
                current_segment = {
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"].strip()
                }
            # If the current segment now ends with a terminal punctuation, finalize it.
            if current_segment and current_segment["text"] and current_segment["text"][-1] in ".!?":
                merged_segments.append(current_segment)
                current_segment = None

    if current_segment:
        merged_segments.append(current_segment)
    return merged_segments

def split_segments_with_pysbd(merged_segments):
    """
    Further split each merged segment into final sentences using PySBD.
    We also apply final speaker assignment here: each final sentence gets the speaker that
    holds the majority duration in the merged segment.
    Approximate timings are assigned by distributing the merged segment's duration evenly.
    """
    final_segments = []
    segmenter = pysbd.Segmenter(language="ru", clean=False)
    for seg in merged_segments:
        sentences = segmenter.segment(seg["text"])
        if not sentences:
            final_segments.append({
                "speaker": None,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })
        else:
            duration = seg["end"] - seg["start"]
            num_sentences = len(sentences)
            for i, sentence in enumerate(sentences):
                sent_start = seg["start"] + (duration * i / num_sentences)
                sent_end = seg["start"] + (duration * (i + 1) / num_sentences)
                final_segments.append({
                    "speaker": None,
                    "start": sent_start,
                    "end": sent_end,
                    "text": sentence.strip()
                })
    return final_segments

def save_speech_to_file_with_indent(segments, filename):
    """
    Save transcript to file with speaker tags, time intervals, and wrapped text.
    """
    text = ""
    with open(filename, "w", encoding="utf-8") as file:
        for segment in segments:
            # Format the timestamp as [HH:MM:SS - HH:MM:SS]
            timestamp_tag = f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}]"
            speaker_tag = f"**{segment['speaker'].upper()}** {timestamp_tag}:\n"
            wrapped_text = textwrap.fill(segment["text"], width=128, subsequent_indent="    ")
            text += speaker_tag + wrapped_text + "\n\n"
            file.write(speaker_tag)
            file.write(wrapped_text)
            file.write("\n\n")
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
        else:
            print("Skipping duplicate chunk:", chunk)
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
        chunk_length_s=10,
        stride_length_s=3,
        torch_dtype=torch_dtype,
        device=device,
    )

    diarization_pipeline = Pipeline.from_pretrained(diarization_model_id, use_auth_token=HF_TOKEN)
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    def transcript(file_name):
        logging.info(f'=============>started with {file_name}')
        # Get the transcript from Whisper with word-level timestamps.
        script = whisper_pipe(file_name, return_timestamps='word', generate_kwargs={"language": "russian"})
        logging.info(f'Loaded transcript for {file_name}')
        
        # Run diarization on the audio file.
        diarized = diarization_pipeline(file_name)
        logging.debug(diarized)
        
        # -----------------------------
        # Pre-process: sort and deduplicate Whisper chunks.
        # -----------------------------
        pre_chunks = sorted(
            script['chunks'],
            key=lambda x: (
                x['timestamp'][0] if x['timestamp'][0] is not None else float('inf'),
                x['timestamp'][1] if x['timestamp'][1] is not None else float('inf')
            )
        )
        chunks = pre_chunks
        # chunks = deduplicate(pre_chunks)
        
        # Build a list of transcription chunks.
        # Here we simply set a default speaker value ("Not Defined") as we won't use it later.
        transcribed = []
        for chunk in chunks:
            start_time = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else float('inf')
            end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else float('inf')
            transcribed.append({
                "start": start_time,
                "end": end_time,
                "text": chunk["text"].strip(),
                "speaker": None  # Default; will be replaced later.
            })
        
        # -----------------------------
        # Step 1: Merge Whisper words into sentences.
        # (We keep your existing merging and PySBD splitting as is.)
        # -----------------------------
        merged_segments = merge_chunks_into_sentences(transcribed)
        sentences = split_segments_with_pysbd(merged_segments)
        
        # -----------------------------
        # Step 2 & 3: For each sentence, collect overlapping diarization blocks
        # and assign the speaker with the maximum total intersection duration.
        # -----------------------------
        for sentence in sentences:
            sent_start = sentence["start"]
            sent_end = sentence["end"]
            overlapping_turns = []
            for turn, _, speaker_label in diarized.itertracks(yield_label=True):
                if sent_start < turn.end and sent_end > turn.start:
                    overlapping_turns.append({
                        "speaker": speaker_label,
                        "start": turn.start,
                        "end": turn.end
                    })
            sentence["diarized_segments"] = overlapping_turns

            # Compute intersection durations per speaker.
            speaker_durations = {}
            for turn in overlapping_turns:
                intersection_start = max(sent_start, turn["start"])
                intersection_end = min(sent_end, turn["end"])
                duration = intersection_end - intersection_start
                if duration > 0:
                    speaker = turn["speaker"]
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            # Assign speaker based on maximum intersection duration.
            if speaker_durations:
                sentence["speaker"] = max(speaker_durations.items(), key=lambda item: item[1])[0]
            else:
                sentence["speaker"] = "Unknown"
        
        # -----------------------------
        # Step 4: Merge consecutive sentences with the same speaker into blocks.
        # -----------------------------
        sentences.sort(key=lambda s: s["start"])
        merged_sentences = []
        if sentences:
            current_sentence = sentences[0]
            for s in sentences[1:]:
                if s["speaker"] == current_sentence["speaker"]:
                    # Extend the current sentence block.
                    current_sentence["end"] = s["end"]
                    current_sentence["text"] += " " + s["text"]
                    # Optionally merge diarized segments.
                    current_sentence["diarized_segments"].extend(s.get("diarized_segments", []))
                else:
                    merged_sentences.append(current_sentence)
                    current_sentence = s
            merged_sentences.append(current_sentence)
        else:
            merged_sentences = sentences

        # -----------------------------
        # Save the final transcript to file.
        # -----------------------------
        trans_folder = os.path.join(os.path.dirname(file_name), 'transcripts/')
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        text = save_speech_to_file_with_indent(merged_sentences, trans_file)
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
    # Clean up GPU memory after processing each file.
    cleanup_gpu_memory()
    return text

def move_to_done(filename):
    dest_dir = './audio/done'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_path = os.path.join(dest_dir, os.path.basename(filename))
    os.rename(filename, dest_path)
   
def run_transcription(file_name):
    if os.path.isfile("./audio/transcripts/debug.txt"):
        with open("./audio/transcripts/debug.txt", "r", encoding="utf-8") as f:
            transcription_text=f.read()
        return transcription_text
    
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)

    diarization_model = "pyannote/speaker-diarization-3.1"
    #align_model = 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'
    whisper_model = "openai/whisper-large-v3"

    transcriptor = transcription_factory(whisper_model, diarization_model)
    transcription_text = transcribe(file_name, transcriptor)
    
    move_to_done(file_name)
    return transcription_text

if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)

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
