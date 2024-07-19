import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
import os
os.environ['CURL_CA_BUNDLE'] = ''


def find_intersections(speakers, texts):
    intersections = []

    for text in texts:
        text_start, text_end = text['start']+0.1, text['end']
        for turn, _, speaker  in speakers.itertracks(yield_label=True):
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

#pipeline = Pipeline.from_pretrained(
#    "pyannote/speaker-diarization-3.1",
#    use_auth_token="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")
TOKEN="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"
AUDIO="audio/2407151757656693.1.0.0.mp3"
WHISPER_MODEL="medium"

pipeline = Pipeline.from_pretrained(
    "/Users/7810155/Documents/Projects/AI/models/speaker-diarization-3.1/config.yaml",
    use_auth_token=TOKEN)

# send pipeline to GPU (when available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(torch.device(DEVICE))

waveform, sample_rate = torchaudio.load(AUDIO)
diarization = pipeline(AUDIO)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


model = whisper.load_model(WHISPER_MODEL)
script = model.transcribe(AUDIO, fp16=False)

for segment in script["segments"]:
    print(f"{segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")

intersections = find_intersections(diarization, script["segments"])
for segment in intersections:
    print(f"{segment['start']:.1f}s - {segment['end']:.1f}s: {segment['speaker']}: {segment['text']}")


from whisperx.diarize import DiarizationPipeline
from whisperx import load_align_model, align
from whisperx.diarize import assign_word_speakers

diarization_pipeline = DiarizationPipeline(use_auth_token=TOKEN, model_name='/Users/7810155/Documents/Projects/AI/models/speaker-diarization-3.1/config.yaml')
diarized = diarization_pipeline(AUDIO)
print(diarized)
model_a, metadata = load_align_model(language_code=script["language"], device=DEVICE, model_name='/Users/7810155/Documents/Projects/AI/models/wav2vec2-large-xlsr-53-russian/pytorch_model.bin')
script_aligned = align(script["segments"], model_a, metadata, AUDIO, DEVICE)
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
            "speaker": result_segment["speaker"],
        }
    )

for start, end, text, speaker in [i.values() for i in transcribed]:
    print(start, end, speaker, text)