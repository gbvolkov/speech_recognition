import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline

import textwrap
import os
os.environ['CURL_CA_BUNDLE'] = ''


def find_intersections(speakers, texts):
    intersections = []

    for text in texts:
        text_start, text_end = text['start'], text['end']-0.1
        for turn, _, speaker in speakers.itertracks(yield_label=True):
            speaker_start, speaker_end = turn.start, turn.end
            
            # Find the overlap between the speaker's interval and the text's interval
            start = max(text_start, speaker_start)
            end = min(text_end, speaker_end)
            
            if start < end:  # There is an intersection
                if intersections and intersections[-1]['speaker'] == speaker:
                    intersections[-1]['end'] = end
                    intersections[-1]['text'] += ' ' + text['text']
                else:
                    intersections.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker,
                        'text': text['text']
                    })
    return intersections

LOCAL_MODEL = False


def merge_speech_segments(segments):
    merged_segments = []
    for segment in segments:
        if merged_segments and segment["speaker"] == merged_segments[-1]["speaker"]:
            # Extend the end time and append text for the same speaker
            merged_segments[-1]["end"] = segment["end"]
            merged_segments[-1]["text"] += " " + segment["text"]
        else:
            # Add a new segment if the speaker changes
            merged_segments.append(segment)
    return merged_segments

def save_speech_to_file_with_indent(segments, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for segment in segments:
            # Format the speaker tag
            speaker_tag = f"{segment['speaker'].upper()}:\n"
            
            # Wrap the text to 128 characters and indent each line
            wrapped_text = textwrap.fill(segment["text"], width=128, subsequent_indent="    ")
            
            # Write the formatted text to the file
            file.write(speaker_tag)
            file.write(wrapped_text)
            file.write("\n\n")  # Add a blank line between speakers


HF_TOKEN="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"

WHISPER_MODEL="large"
if LOCAL_MODEL:
    DIARIZATION_MODEL="/Projects/AI/models/speaker-diarization-3.1/config.yaml"
    ALIGN_MODEL="/Projects/AI/models/wav2vec2-large-xlsr-53-russian/"
else:
    DIARIZATION_MODEL="pyannote/speaker-diarization-3.1"
    ALIGN_MODEL=None

pipeline = Pipeline.from_pretrained(
    DIARIZATION_MODEL,
    use_auth_token=HF_TOKEN)
# send pipeline to GPU (when available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(torch.device(DEVICE))

from whisperx.diarize import DiarizationPipeline
from whisperx import load_align_model, align
from whisperx.diarize import assign_word_speakers
diarization_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, model_name=DIARIZATION_MODEL, device=DEVICE)
model = whisper.load_model(WHISPER_MODEL, download_root='./models', device=DEVICE)


def transcript(file_name):
    script = model.transcribe(file_name)

    diarized = diarization_pipeline(file_name)
    print(diarized)
    model_a, metadata = load_align_model(language_code=script["language"], device=DEVICE, model_name=ALIGN_MODEL)
    script_aligned = align(script["segments"], model_a, metadata, file_name, DEVICE)
    result_segments, word_seg = list(assign_word_speakers(
        diarized, script_aligned    
    ).values())

    transcribed = []
    for result_segment in result_segments:
        transcribed.append(
            {
                "start": result_segment["start"],
                "end": result_segment["end"],
                "text": result_segment["text"],
                "speaker": result_segment["speaker"] if 'speaker' in result_segment else "ND"
            }
        )

    merged = merge_speech_segments(transcribed)

    out_file, _ = os.path.splitext(file_name)
    out_file = f"{out_file}_transcript.txt"
    save_speech_to_file_with_indent(merged, out_file)


if __name__ == "__main__":
    audios=["./audio/audio1266668284.m4a", "./audio/audio1415011527.m4a", "./audio/audio1499365096.m4a"]
    
    for audio in audios:
        transcript(audio)