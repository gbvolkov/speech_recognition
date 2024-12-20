import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment

import textwrap
import os
import logging

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


def convert_audio_to_wav(input_file, output_file, audio_type):
    """
    Converts an M4A file to MP3 format.

    :param input_file: Path to the input M4A file.
    :param output_file: Path where the output MP3 will be saved.
    :param audio_type: Type of audio file.
    """
    # Load the M4A file
    audio = AudioSegment.from_file(input_file, format=audio_type)
    # Export as MP3
    audio.export(output_file, format='wav')#, bitrate=bitrate)
    
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

    #Initializing up wisper pipeline
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.config.forced_decoder_ids = None
    whisper_model.to(device)
    whisper_processor = AutoProcessor.from_pretrained(
        whisper_model_id
    )
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=30,  # Process audio in 30-second chunks
        stride_length_s=10,  # Optional overlap between chunks    
        torch_dtype=torch_dtype,
        device=device,
    )


    diarization_pipeline = Pipeline.from_pretrained(diarization_model_id)
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    def transcript(file_name):
        logging.info(f'=============>started with {file_name}')
        #audio, sample_rate = torchaudio.load(file_name, backend='soundfile')

        script = whisper_pipe(file_name, return_timestamps='word', generate_kwargs={"language": "russian"})
        #with open('script_2.txt', "w", encoding="utf-8") as f:
        #    f.write(script["text"])    
        logging.info(f'loaded for {file_name}')
        diarized = diarization_pipeline(file_name, min_speakers=1, max_speakers=9)
        logging.debug(diarized)
        # Combine results
        speaker_transcription = []

        pre_chunks = sorted(script['chunks'], key=lambda x: (x['timestamp'][0], x['timestamp'][1]))
        chunks = deduplicate(pre_chunks)
        #with open('audio/chunks.txt', "w", encoding="utf-8") as f:
        #    for chunk in chunks:    
        #        f.write(f'{chunk["timestamp"][0]}-{chunk["timestamp"][1]}: {chunk["text"]}\n')    
            
        for chunk in chunks:
            #start_time, end_time = chunk["timestamp"][0]-delta, chunk["timestamp"][1]-delta
            start_time, end_time = chunk["timestamp"][0], chunk["timestamp"][1]
            speaker = "Unknown"
            for turn, _, speaker_label in diarized.itertracks(yield_label=True):
                # Find the overlap between the speaker's interval and the text's interval
                start = max(start_time, turn.start)
                end = min(end_time, turn.end)
                #logging.debug(f'{chunk['text']}====>{start}({start_time}):{end}({end_time})')
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
            transcribed.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": segment["speaker"] if 'speaker' in segment else "ND"
                }
            )
        logging.debug(transcribed)

        merged = merge_speech_segments(transcribed)

        trans_folder = os.path.join(os.path.dirname(file_name), 'transcripts/')
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        save_speech_to_file_with_indent(merged, trans_file)
        logging.info(f'<=============Done with {file_name}')

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
    transcriptor(wav_name)
    if btemp:
        os.remove(wav_name)

if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    HF_TOKEN="XXXXXX"

    diarization_model="pyannote/speaker-diarization-3.1"
    align_model='jonatasgrosman/wav2vec2-large-xlsr-53-russian'
    whisper_model="openai/whisper-large-v3"
    
    transcriptor = transcription_factory(whisper_model, diarization_model)

    folder_path = './audio/'
    AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.wav'}

    folder = Path(folder_path)
    # Iterate through all files in the directory (non-recursive)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            full_path = file.resolve()
            audio_type = file.suffix.lower().replace('.', '')  # e.g., 'mp3'
            logging.info(f"Found audio file: {full_path} (Type: {audio_type})")
            out_file, _ = os.path.splitext(full_path)
            out_file = f"{out_file}.wav"        
            # Call the conversion function
            transcribe(str(full_path), transcriptor)



