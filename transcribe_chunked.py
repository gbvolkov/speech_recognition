from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from transcribe import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    HF_TOKEN,
    UNKNOWN_SPEAKER,
    TranscriptionConfig,
    _apply_runtime_determinism,
    _assert_runtime_dependencies,
    _load_diarization_pipeline,
    _parse_bool,
    _parse_float,
    _prepare_diarization_turns,
    _prepare_word_chunks,
    _serialize_sentence,
    _serialize_word,
    _write_job_metrics,
    assign_sentence_speakers_v2,
    cleanup_gpu_memory,
    convert_audio_to_wav,
    deduplicate_word_chunks,
    load_config_from_env,
    merge_consecutive_sentences,
    move_to_done,
    normalize_runtime_config,
    save_speech_to_file_with_indent,
    segment_words,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_PARENT_FIELDS = {field.name for field in fields(TranscriptionConfig)}
_TRANSCRIPTOR_CACHE = {}
_TRANSCRIPTOR_LOCK = Lock()


@dataclass(frozen=True)
class ChunkedTranscriptionConfig(TranscriptionConfig):
    chunk_collar_sec: float = 0.35
    chunk_turn_merge_gap_sec: float = 0.25
    chunk_min_duration_sec: float = 0.60
    chunk_decode_max_duration_sec: float = 20.0
    emit_chunk_debug: bool = True


def _to_parent_config(config: ChunkedTranscriptionConfig | TranscriptionConfig) -> TranscriptionConfig:
    kwargs = {name: getattr(config, name) for name in _PARENT_FIELDS}
    return TranscriptionConfig(**kwargs)


def load_chunked_config_from_env() -> ChunkedTranscriptionConfig:
    base = load_config_from_env()
    return ChunkedTranscriptionConfig(
        **asdict(base),
        chunk_collar_sec=_parse_float(os.getenv("SR_CHUNK_COLLAR_SEC"), 0.35),
        chunk_turn_merge_gap_sec=_parse_float(os.getenv("SR_CHUNK_TURN_MERGE_GAP_SEC"), 0.25),
        chunk_min_duration_sec=_parse_float(os.getenv("SR_CHUNK_MIN_DURATION_SEC"), 0.60),
        chunk_decode_max_duration_sec=_parse_float(os.getenv("SR_CHUNK_DECODE_MAX_DURATION_SEC"), 20.0),
        emit_chunk_debug=_parse_bool(os.getenv("SR_CHUNK_EMIT_DEBUG"), True),
    )


def normalize_chunked_config(config: ChunkedTranscriptionConfig | TranscriptionConfig | None) -> ChunkedTranscriptionConfig:
    if config is None:
        config = load_chunked_config_from_env()

    if isinstance(config, ChunkedTranscriptionConfig):
        candidate = config
    else:
        parent = normalize_runtime_config(config)
        candidate = ChunkedTranscriptionConfig(**asdict(parent))

    normalized_parent = normalize_runtime_config(_to_parent_config(candidate))
    return replace(candidate, **asdict(normalized_parent))


def _merge_turns_for_chunk_decode(turns, max_gap_sec):
    if not turns:
        return []

    ordered = sorted(turns, key=lambda item: (item["start"], item["end"]))
    merged = [ordered[0].copy()]
    for turn in ordered[1:]:
        current = turn.copy()
        prev = merged[-1]
        gap = current["start"] - prev["end"]
        if current["speaker"] == prev["speaker"] and gap <= max_gap_sec:
            prev["end"] = max(prev["end"], current["end"])
        else:
            merged.append(current)
    return merged


def _extract_audio_window(input_file, output_file, start_sec, end_sec):
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg is not available in PATH.")

    duration = max(0.01, float(end_sec) - float(start_sec))
    command = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        input_file,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_file,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"FFmpeg chunk extraction failed: {stderr}") from exc


def _decode_turn_words(file_name, turn, idx, tmp_dir: Path, whisper_pipe, generate_kwargs, config):
    core_start = float(turn["start"])
    core_end = float(turn["end"])
    if core_end <= core_start:
        return [], 0, {"turn_index": idx, "skip_reason": "empty_turn"}

    decode_start = max(0.0, core_start - config.chunk_collar_sec)
    decode_end = core_end + config.chunk_collar_sec
    decode_duration = decode_end - decode_start

    if decode_duration < config.chunk_min_duration_sec:
        expand = (config.chunk_min_duration_sec - decode_duration) / 2.0
        decode_start = max(0.0, decode_start - expand)
        decode_end = decode_end + expand

    if (decode_end - decode_start) > config.chunk_decode_max_duration_sec:
        center = (core_start + core_end) / 2.0
        half = config.chunk_decode_max_duration_sec / 2.0
        decode_start = max(0.0, center - half)
        decode_end = decode_start + config.chunk_decode_max_duration_sec
        if decode_end < core_end:
            decode_end = core_end
            decode_start = max(0.0, decode_end - config.chunk_decode_max_duration_sec)

    chunk_file = tmp_dir / f"turn_{idx:04d}_{turn['speaker']}.wav"
    _extract_audio_window(file_name, str(chunk_file), decode_start, decode_end)

    script = whisper_pipe(str(chunk_file), return_timestamps="word", generate_kwargs=generate_kwargs)
    raw_chunks = script.get("chunks", [])
    local_words = _prepare_word_chunks(raw_chunks)

    kept_words = []
    dropped_outside_core = 0
    speaker = turn["speaker"] or UNKNOWN_SPEAKER
    for word in local_words:
        start = float(word["start"]) + decode_start
        end = float(word["end"]) + decode_start
        if end <= start:
            end = start + 0.01
        midpoint = (start + end) / 2.0
        if midpoint < core_start or midpoint > core_end:
            dropped_outside_core += 1
            continue
        duration = max(0.01, end - start)
        kept_words.append(
            {
                "text": word["text"],
                "start": start,
                "end": end,
                "speaker": speaker,
                "speaker_confidence": 1.0,
                "speaker_vote_margin": 1.0,
                "speaker_durations": {speaker: duration},
            }
        )

    debug_payload = {
        "turn_index": idx,
        "speaker": speaker,
        "core_start": round(core_start, 3),
        "core_end": round(core_end, 3),
        "decode_start": round(decode_start, 3),
        "decode_end": round(decode_end, 3),
        "raw_chunk_count": len(raw_chunks),
        "prepared_word_count": len(local_words),
        "kept_word_count": len(kept_words),
        "dropped_outside_core_count": dropped_outside_core,
        "text_preview": (script.get("text") or "")[:220],
    }
    return kept_words, len(raw_chunks), debug_payload


def _write_chunked_debug(
    file_name,
    config,
    diar_turns,
    decode_turns,
    turn_debug,
    words,
    sentences,
    merged_sentences,
):
    if not config.debug_trace:
        return

    debug_dir = Path(file_name).parent / "transcripts" / "debug_chunked"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).stem

    payload = {
        "source_file": str(file_name),
        "pipeline": "chunked_v1",
        "config": asdict(config),
        "counts": {
            "diarization_turn_count": len(diar_turns),
            "decode_turn_count": len(decode_turns),
            "word_count": len(words),
            "segment_count_pre_merge": len(sentences),
            "segment_count_final": len(merged_sentences),
        },
        "diarization_turns": [
            {"speaker": turn["speaker"], "start": round(turn["start"], 3), "end": round(turn["end"], 3)}
            for turn in diar_turns
        ],
        "decode_turns": [
            {"speaker": turn["speaker"], "start": round(turn["start"], 3), "end": round(turn["end"], 3)}
            for turn in decode_turns
        ],
        "turn_debug": turn_debug,
        "words_sample": [_serialize_word(word) for word in words[: min(len(words), config.debug_max_items)]],
        "segments_pre_merge_sample": [
            _serialize_sentence(sentence, include_words=False)
            for sentence in sentences[: min(len(sentences), config.debug_max_items)]
        ],
        "segments_final": [_serialize_sentence(sentence, include_words=False) for sentence in merged_sentences],
    }

    debug_json = debug_dir / f"{stem}.chunked.debug.json"
    debug_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    words_jsonl = debug_dir / f"{stem}.chunked.words.jsonl"
    with open(words_jsonl, "w", encoding="utf-8") as handle:
        for word in words:
            handle.write(json.dumps(_serialize_word(word), ensure_ascii=False))
            handle.write("\n")

    logging.info("Chunked debug artifacts written to %s", debug_dir)


def transcription_factory_chunked(whisper_model_id, diarization_model_id, config=None):
    config = normalize_chunked_config(config)
    _apply_runtime_determinism(config)
    _assert_runtime_dependencies()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(
        "Initializing chunked ASR/diarization pipelines on device=%s | profile=%s",
        device,
        config.profile,
    )

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.config.forced_decoder_ids = None
    whisper_model.to(device)
    whisper_processor = AutoProcessor.from_pretrained(whisper_model_id, token=HF_TOKEN)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=config.chunk_length_s,
        stride_length_s=config.stride_length_s,
        dtype=torch_dtype,
        device=device,
    )

    diarization_pipeline = _load_diarization_pipeline(diarization_model_id, HF_TOKEN)
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    generate_kwargs = {"language": config.force_language, "task": config.force_task}
    if config.deterministic_decoding:
        generate_kwargs.update({"temperature": 0.0, "do_sample": False, "num_beams": 1})

    def transcript(file_name):
        logging.info("=============> chunked started with %s", file_name)
        diarization_kwargs = {}
        if config.num_speakers is not None:
            diarization_kwargs["num_speakers"] = config.num_speakers

        diarized_result = (
            diarization_pipeline(file_name, **diarization_kwargs)
            if diarization_kwargs
            else diarization_pipeline(file_name)
        )
        diar_turns = _prepare_diarization_turns(diarized_result, config.min_turn_duration_sec)
        decode_turns = _merge_turns_for_chunk_decode(diar_turns, config.chunk_turn_merge_gap_sec)

        words = []
        raw_chunk_count = 0
        turn_debug = []
        with tempfile.TemporaryDirectory(prefix="sr_chunked_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for idx, turn in enumerate(decode_turns):
                turn_words, turn_raw_count, debug_payload = _decode_turn_words(
                    file_name=file_name,
                    turn=turn,
                    idx=idx,
                    tmp_dir=tmp_dir,
                    whisper_pipe=whisper_pipe,
                    generate_kwargs=generate_kwargs,
                    config=config,
                )
                raw_chunk_count += turn_raw_count
                words.extend(turn_words)
                if config.emit_chunk_debug:
                    turn_debug.append(debug_payload)

        words.sort(key=lambda item: (item["start"], item["end"]))
        words = deduplicate_word_chunks(words, config.dedup_overlap_ratio)
        usable_word_count = len(words)

        sentences = segment_words(words, config, split_on_speaker_change=True)
        sentences = assign_sentence_speakers_v2(sentences, config)
        merged_sentences = merge_consecutive_sentences(sentences, config)

        trans_folder = os.path.join(os.path.dirname(file_name), "transcripts")
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        text = save_speech_to_file_with_indent(merged_sentences, trans_file)

        if config.emit_metrics:
            _write_job_metrics(
                file_name=file_name,
                segments=merged_sentences,
                config=config,
                raw_chunk_count=raw_chunk_count,
                usable_word_count=usable_word_count,
                turn_count=len(diar_turns),
            )

        _write_chunked_debug(
            file_name=file_name,
            config=config,
            diar_turns=diar_turns,
            decode_turns=decode_turns,
            turn_debug=turn_debug,
            words=words,
            sentences=sentences,
            merged_sentences=merged_sentences,
        )
        logging.info("<============= chunked done with %s", file_name)
        return text

    return transcript


def _get_or_create_transcriptor(config):
    key = (DEFAULT_WHISPER_MODEL, DEFAULT_DIARIZATION_MODEL, config)
    with _TRANSCRIPTOR_LOCK:
        transcriptor = _TRANSCRIPTOR_CACHE.get(key)
        if transcriptor is None:
            transcriptor = transcription_factory_chunked(
                whisper_model_id=DEFAULT_WHISPER_MODEL,
                diarization_model_id=DEFAULT_DIARIZATION_MODEL,
                config=config,
            )
            _TRANSCRIPTOR_CACHE[key] = transcriptor
    return transcriptor


def transcribe(audio_name, transcriptor):
    wav_name = audio_name
    _, ext = os.path.splitext(audio_name)
    ext = ext.lower().replace(".", "")
    temp_wav_created = False

    try:
        if ext != "wav":
            wav_name = f"{os.path.splitext(audio_name)[0]}.wav"
            convert_audio_to_wav(audio_name, wav_name, ext)
            temp_wav_created = True
        return transcriptor(wav_name)
    finally:
        if temp_wav_created and os.path.exists(wav_name):
            os.remove(wav_name)
        cleanup_gpu_memory()


def run_transcription(file_name, config=None, move_processed=True):
    config = normalize_chunked_config(config)
    transcriptor = _get_or_create_transcriptor(config)
    transcription_text = transcribe(file_name, transcriptor)
    if move_processed:
        move_to_done(file_name)
    return transcription_text


if __name__ == "__main__":
    config = load_chunked_config_from_env()
    transcriptor = _get_or_create_transcriptor(config)

    folder = Path("./audio/")
    audio_extensions = {".mp3", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".wav", ".mp4"}
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in audio_extensions:
            full_path = file.resolve()
            logging.info("Found audio file for chunked transcription: %s", full_path)
            transcribe(str(full_path), transcriptor)
            print(f"<============= Chunked done with {full_path}")
