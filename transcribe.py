import math
import sys
import os
import torch
import torch.nn.functional as F
import textwrap
import logging

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torchaudio

LOCAL_MODEL = False
with open('hf.txt') as f:
    HF_TOKEN = f.read()

# ---------------------------
# Forced Alignment Functions (for boundary refinement)
# ---------------------------

def load_audio_segment(file_name, start_time, end_time, target_sample_rate=16000):
    """
    Load a segment of audio from file between start_time and end_time (in seconds)
    and resample to target_sample_rate if necessary.
    """
    waveform, sample_rate = torchaudio.load(file_name)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = waveform[:, start_sample:end_sample]
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        segment = resampler(segment)
    return segment, target_sample_rate

def get_trellis(emission, tokens, blank_id=0):
    """
    Build a trellis for forced alignment.
    """
    num_frames = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.full((num_frames, num_tokens + 1), -float('inf'), device=emission.device)
    trellis[0, 0] = 0
    for t in range(1, num_frames):
        trellis[t, 0] = trellis[t-1, 0] + emission[t, blank_id]
    for t in range(1, num_frames):
        for j in range(1, num_tokens + 1):
            trellis[t, j] = max(
                trellis[t-1, j] + emission[t, blank_id],
                trellis[t-1, j-1] + emission[t, tokens[j-1]]
            )
    return trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    """
    Backtrack the trellis to recover an alignment path.
    """
    T, J = trellis.shape
    j = J - 1
    t = T - 1
    path = []
    while t > 0 and j > 0:
        score = trellis[t, j]
        score_blank = trellis[t-1, j] + emission[t, blank_id]
        score_token = trellis[t-1, j-1] + emission[t, tokens[j-1]]
        if score_token > score_blank:
            path.append((t, j))
            t, j = t-1, j-1
        else:
            path.append((t, j))
            t = t-1
    path.reverse()
    return path

def merge_repeats(path, transcript_text):
    """
    Evenly distribute timing across words from transcript_text.
    """
    words = transcript_text.split()
    if not path or len(words) == 0:
        return []
    start_frame = path[0][0]
    end_frame = path[-1][0]
    total_frames = end_frame - start_frame
    word_duration = total_frames / len(words)
    aligned_words = []
    for idx, word in enumerate(words):
        aligned_words.append({
            "start": start_frame + idx * word_duration,
            "end": start_frame + (idx + 1) * word_duration,
            "text": word
        })
    return aligned_words

def forced_align_chunk(chunk, file_name, align_model, align_processor, device):
    """
    Perform forced alignment on an entire Whisper chunk.
    Returns a list of word dictionaries with refined timestamps.
    """
    start_time = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0
    end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0
    if end_time <= start_time:
        return []
    audio_segment, sample_rate = load_audio_segment(file_name, start_time, end_time)
    min_length = 400
    if audio_segment.size(-1) < min_length:
        pad_amount = min_length - audio_segment.size(-1)
        audio_segment = F.pad(audio_segment, (0, pad_amount))
    with torch.inference_mode():
        audio_segment = audio_segment.to(device)
        if audio_segment.ndim == 1:
            audio_segment = audio_segment.unsqueeze(0)
        emissions = align_model(audio_segment).logits  # shape: (1, frames, vocab_size)
        emissions = torch.log_softmax(emissions, dim=-1)[0]
    transcript_text = chunk["text"].lower()
    tokens = align_processor.tokenizer(transcript_text, add_special_tokens=False).input_ids
    blank_id = align_processor.tokenizer.pad_token_id if align_processor.tokenizer.pad_token_id is not None else 0
    trellis = get_trellis(emissions, tokens, blank_id)
    path = backtrack(trellis, emissions, tokens, blank_id)
    aligned_words = merge_repeats(path, transcript_text)
    num_frames = emissions.size(0)
    duration = end_time - start_time
    time_per_frame = duration / num_frames
    for word in aligned_words:
        word["start"] = word["start"] * time_per_frame + start_time
        word["end"] = word["end"] * time_per_frame + start_time
    return aligned_words

def refine_boundary(boundary_time, file_name, delta, text_prev, text_next, align_model, align_processor, device):
    """
    Refine a boundary using forced alignment over a small window.
    The window is [boundary_time - delta, boundary_time + delta].
    The refined boundary is set as the midpoint between the last word of text_prev and the first word of text_next.
    """
    window_start = max(0, boundary_time - delta)
    window_end = boundary_time + delta
    sub_text = text_prev.strip() + " " + text_next.strip()
    sub_chunk = {"timestamp": (window_start, window_end), "text": sub_text}
    aligned_words = forced_align_chunk(sub_chunk, file_name, align_model, align_processor, device)
    if not aligned_words:
        return boundary_time
    words = sub_text.split()
    num_prev = len(text_prev.split())
    if num_prev == 0 or num_prev >= len(aligned_words):
        return boundary_time
    last_prev_end = aligned_words[num_prev - 1]["end"]
    first_next_start = aligned_words[num_prev]["start"]
    return (last_prev_end + first_next_start) / 2.0

# ---------------------------
# Transcript Module
# ---------------------------

def save_speech_to_file_with_indent(segments, filename):
    """
    Save the speaker-attributed transcript to a file with formatted indentation.
    """
    text_out = ""
    with open(filename, "w", encoding="utf-8") as file:
        for segment in segments:
            speaker_tag = f"{segment['speaker'].upper()}:\n"
            wrapped_text = textwrap.fill(segment["text"], width=128, subsequent_indent="    ")
            text_out += speaker_tag + wrapped_text + "\n\n"
            file.write(speaker_tag)
            file.write(wrapped_text)
            file.write("\n\n")
    return text_out

def convert_audio_to_wav(input_file, output_file, audio_type):
    """
    Convert an audio file to WAV format.
    """
    audio = AudioSegment.from_file(input_file, format=audio_type)
    audio.export(output_file, format='wav')
    logging.info(f"Successfully converted '{input_file}' to '{output_file}'")

def deduplicate(chunked_script):
    """
    Deduplicate chunk-level data.
    """
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

def merge_blocks(blocks):
    """
    Merge each block's words into a single segment.
    """
    merged = []
    for block in blocks:
        text = " ".join([w["text"] for w in block["words"]])
        merged.append({
            "speaker": block["speaker"],
            "start": block["start"],
            "end": block["end"],
            "text": text
        })
    return merged

def merge_all_segments_by_speaker(segments):
    """
    Merge adjacent segments with the same speaker into one.
    """
    if not segments:
        return []
    segments.sort(key=lambda s: s["start"])
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"]:
            last["end"] = seg["end"]
            last["text"] += " " + seg["text"]
        else:
            merged.append(seg)
    return merged

def transcription_factory(whisper_model_id, diarization_model_id, align_model_id=None):
    """
    Creates a transcription function that uses Whisper for ASR,
    Pyannote for diarization, and forced alignment via Wav2Vec2 to refine only boundaries.
    
    Process:
      1. Extract word-based segments from Whisper (using provided "words").
      2. Group words into speaker blocks using diarization boundaries.
         (If the very first word(s) fall before any diarization turn, assign them to the first turn's speaker.)
      3. Refine boundaries between adjacent blocks with different speakers using forced alignment on a small window.
      4. Merge blocks into segments, and then rejoin adjacent segments with the same speaker.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(f"Using device: {device}")

    # Initialize Whisper ASR pipeline.
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

    # Initialize Pyannote diarization pipeline.
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_id, use_auth_token=HF_TOKEN)
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    # Load forced alignment model (Wav2Vec2) if provided.
    if align_model_id is not None:
        align_processor = Wav2Vec2Processor.from_pretrained(align_model_id)
        align_model = Wav2Vec2ForCTC.from_pretrained(align_model_id)
        align_model.to(device)
    else:
        align_processor, align_model = None, None

    def transcript(file_name):
        logging.info(f'=============> Started transcription for {file_name}')
        # Obtain Whisper transcript with word-level timestamps.
        script = whisper_pipe(file_name, return_timestamps='word', generate_kwargs={"language": "russian"})
        logging.info(f"Whisper transcription loaded for {file_name}")

        # Run diarization.
        diarized = diarization_pipeline(file_name)
        logging.debug(f"Diarization result: {diarized}")

        # ------------------------------------------
        # Step 1: Build a full word-level transcript using Whisper words.
        # ------------------------------------------
        word_list = []
        for chunk in script['chunks']:
            if "words" in chunk and chunk["words"]:
                for w in chunk["words"]:
                    w_start, w_end = w["timestamp"]
                    word_list.append({
                        "start": w_start,
                        "end": w_end,
                        "text": w["word"].strip()
                    })
            else:
                word_list.append({
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1],
                    "text": chunk["text"].strip()
                })
        word_list.sort(key=lambda w: w["start"])

        # ------------------------------------------
        # Step 2: Form speaker blocks using diarization.
        # For each diarization turn, select all words whose midpoints fall within it.
        # If the very first word(s) occur before any diarization turn, assign them to the first turn's speaker.
        # ------------------------------------------
        blocks = []
        turns = list(diarized.itertracks(yield_label=True))
        if turns:
            first_turn_start = min(turn.start for turn, _, _ in turns)
            first_turn_speaker = turns[0][2]
            if word_list and word_list[0]["start"] < first_turn_start:
                initial_words = [w for w in word_list if w["start"] < first_turn_start]
                if initial_words:
                    blocks.append({
                        "speaker": first_turn_speaker,
                        "start": min(w["start"] for w in initial_words),
                        "end": max(w["end"] for w in initial_words),
                        "words": initial_words
                    })
        for turn, _, speaker_label in turns:
            block_words = [w for w in word_list if turn.start <= (w["start"] + w["end"]) / 2.0 < turn.end]
            if block_words:
                block = {
                    "speaker": speaker_label,
                    "start": min(w["start"] for w in block_words),
                    "end": max(w["end"] for w in block_words),
                    "words": block_words
                }
                blocks.append(block)
        blocks.sort(key=lambda b: b["start"])

        # Merge adjacent blocks from the same speaker if gap is small.
        gap_threshold = 3.0  # seconds
        merged_blocks = []
        if blocks:
            current = blocks[0]
            for block in blocks[1:]:
                if block["speaker"] == current["speaker"] and (block["start"] - current["end"]) < gap_threshold:
                    current["end"] = block["end"]
                    current["words"].extend(block["words"])
                else:
                    merged_blocks.append(current)
                    current = block
            merged_blocks.append(current)
        else:
            merged_blocks = blocks

        # ------------------------------------------
        # Step 3: Refine boundaries between adjacent blocks.
        # For each pair of adjacent blocks with different speakers, refine the boundary.
        # ------------------------------------------
        delta = 0.5  # seconds window
        for i in range(len(merged_blocks) - 1):
            if merged_blocks[i]["speaker"] != merged_blocks[i+1]["speaker"]:
                # Use last two words from block i and first two from block i+1.
                text_prev = " ".join([w["text"] for w in merged_blocks[i]["words"][-2:]])
                text_next = " ".join([w["text"] for w in merged_blocks[i+1]["words"][:2]])
                boundary_time = merged_blocks[i]["end"]
                refined_boundary = refine_boundary(boundary_time, file_name, delta, text_prev, text_next, align_model, align_processor, device)
                merged_blocks[i]["end"] = refined_boundary
                merged_blocks[i+1]["start"] = refined_boundary

        # ------------------------------------------
        # Step 4: Merge each block's words into a segment.
        # ------------------------------------------
        segments = merge_blocks(merged_blocks)
        # Final rejoin: merge adjacent segments with the same speaker.
        final_segments = merge_all_segments_by_speaker(segments)
        logging.debug(f"Merged speaker segments: {final_segments}")

        # ------------------------------------------
        # Step 5: Save final transcript.
        # ------------------------------------------
        trans_folder = os.path.join(os.path.dirname(file_name), 'transcripts/')
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        text_out = save_speech_to_file_with_indent(final_segments, trans_file)
        logging.info(f'<============= Done with {file_name}')
        return text_out

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
    transcriptor = transcription_factory(whisper_model, diarization_model, align_model_id=align_model)
    transcription = transcribe(file_name, transcriptor)
    move_to_done(file_name)
    return transcription

if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    with open('hf.txt') as f:
        HF_TOKEN = f.read()
    diarization_model = "pyannote/speaker-diarization-3.1"
    align_model = 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'
    whisper_model = "openai/whisper-large-v3"
    transcriptor = transcription_factory(whisper_model, diarization_model, align_model_id=align_model)
    folder_path = './audio/'
    AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.wav'}
    from pathlib import Path
    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            full_path = file.resolve()
            audio_type = file.suffix.lower().replace('.', '')
            logging.info(f"Found audio file: {full_path} (Type: {audio_type})")
            transcribe(str(full_path), transcriptor)
            print(f"<============= Done with {full_path}")
