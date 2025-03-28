import math
from transformers.models.whisper import tokenization_whisper

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

# Import PySBD for rule-based sentence segmentation.
import pysbd

LOCAL_MODEL = False

def format_timestamp(seconds):
    """
    Format seconds into HH:MM:SS string.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def merge_chunks_into_sentences(chunks, allow_cross_speaker_merge=True, cross_speaker_gap_threshold=0.3):
    """
    Merge consecutive transcription chunks into longer segments using heuristics.
    Instead of finalizing speaker assignment here, we accumulate each chunk's duration per speaker.
    The merged segment retains a "speakers" dict that maps speaker -> total duration.
    """
    if not chunks:
        return []
    # Sort chunks by start time.
    chunks.sort(key=lambda x: x["start"])
    merged_segments = []
    current_segment = None

    for chunk in chunks:
        chunk_duration = chunk["end"] - chunk["start"]
        if current_segment is None:
            current_segment = {
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"].strip(),
                # Accumulate durations in a dict: speaker -> total duration
                "speakers": { chunk["speaker"]: chunk_duration }
            }
        else:
            gap = chunk["start"] - current_segment["end"]
            same_speaker = (chunk["speaker"] in current_segment["speakers"])
            ends_strong = current_segment["text"] and current_segment["text"][-1] in ".!?"
            if (same_speaker or (allow_cross_speaker_merge and gap < cross_speaker_gap_threshold)) and not ends_strong:
                current_segment["text"] += " " + chunk["text"].strip()
                current_segment["end"] = chunk["end"]
                current_segment["speakers"][chunk["speaker"]] = current_segment["speakers"].get(chunk["speaker"], 0) + chunk_duration
            else:
                merged_segments.append(current_segment)
                current_segment = {
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"].strip(),
                    "speakers": { chunk["speaker"]: chunk_duration }
                }
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
        if "speakers" in seg and seg["speakers"]:
            final_speaker = max(seg["speakers"].items(), key=lambda item: item[1])[0]
        else:
            final_speaker = seg.get("speaker", "Unknown")
        sentences = segmenter.segment(seg["text"])
        if not sentences:
            final_segments.append({
                "speaker": final_speaker,
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
                    "speaker": final_speaker,
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
            speaker_tag = f"{segment['speaker'].upper()} {timestamp_tag}:\n"
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
        chunk_length_s=30,
        stride_length_s=10,
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

        pre_chunks = sorted(script['chunks'], 
            key=lambda x: (
                x['timestamp'][0] if x['timestamp'][0] is not None else float('inf'),
                x['timestamp'][1] if x['timestamp'][1] is not None else float('inf')        
        ))
        chunks = deduplicate(pre_chunks)
            
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
        transcribed = []
        for segment in speaker_transcription:
            transcribed.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment["speaker"] if 'speaker' in segment else "ND"
            })
        logging.debug(transcribed)

        # Merge chunks using relaxed merging with accumulated speaker durations.
        merged = merge_chunks_into_sentences(transcribed, allow_cross_speaker_merge=True, cross_speaker_gap_threshold=0.3)
        # Further split merged segments into final sentences using PySBD, and then assign final speaker by majority.
        final_segments = split_segments_with_pysbd(merged)
        for seg in final_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            intersecting_turns = []
            # Iterate over diarized segments (using itertracks with labels)
            for turn, _, speaker_label in diarized.itertracks(yield_label=True):
                # Check for intersection: if the final segment's interval overlaps with the diarized turn.
                if seg_start < turn.end and seg_end > turn.start:
                    intersecting_turns.append({
                        "speaker": speaker_label,
                        "start": turn.start,
                        "end": turn.end
                    })
            # Add the list of intersecting diarized segments to the final segment.
            seg["diarized_segments"] = intersecting_turns

            # Compute the intersection duration for each speaker.
            speaker_durations = {}
            for turn in intersecting_turns:
                # Calculate the intersection duration between the final segment and the diarized turn.
                intersection_start = max(seg_start, turn["start"])
                intersection_end = min(seg_end, turn["end"])
                duration = intersection_end - intersection_start
                if duration > 0:
                    speaker = turn["speaker"]
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            # Choose the speaker with the maximum intersection duration.
            if speaker_durations:
                real_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]
            else:
                real_speaker = seg.get("speaker", "Unknown")
            seg["speaker"] = real_speaker
        
        trans_folder = os.path.join(os.path.dirname(file_name), 'transcripts/')
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        text = save_speech_to_file_with_indent(final_segments, trans_file)
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
