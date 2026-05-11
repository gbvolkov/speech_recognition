from __future__ import annotations

with open('hf.txt', encoding='utf-8') as f:
    HF_TOKEN = f.read().strip()

import json
import os
import random
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE


def _configure_ffmpeg_runtime():
    """Make FFmpeg DLLs discoverable on Windows before torchcodec/pyannote import."""
    if os.name != "nt":
        return

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return

    ffmpeg_bin = os.path.dirname(ffmpeg_path)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_bin not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{ffmpeg_bin}{os.pathsep}{current_path}" if current_path else ffmpeg_bin

    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(ffmpeg_bin)
        except OSError:
            # PATH update above is still useful if this fails.
            pass


_configure_ffmpeg_runtime()

import textwrap
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

import gc

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
#DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
UNKNOWN_SPEAKER = "Unknown"


@dataclass(frozen=True)
class TranscriptionConfig:
    profile: str = "default"

    # Segmentation controls (v2)
    max_word_gap_sec: float = 0.8
    max_segment_duration_sec: float = 12.0
    dedup_overlap_ratio: float = 0.7

    # Speaker assignment controls (v2)
    min_word_overlap_ratio: float = 0.5
    min_speaker_margin: float = 0.12
    max_block_merge_gap_sec: float = 0.6
    min_turn_duration_sec: float = 0.3

    # Decode + diarization controls
    chunk_length_s: int = 30
    stride_length_s: int = 5
    force_language: str = "russian"
    force_task: str = "transcribe"
    deterministic_decoding: bool = True
    num_speakers: int | None = None
    strict_determinism: bool = True
    random_seed: int = 1337

    # Runtime behavior
    emit_metrics: bool = True
    debug_trace: bool = True
    debug_max_items: int = 200


_TRANSCRIPTOR_CACHE = {}
_TRANSCRIPTOR_LOCK = Lock()


def _parse_bool(raw, default):
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value: '{raw}'")


def _parse_float(raw, default):
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"Invalid float value: '{raw}'")


def _parse_int(raw, default):
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Invalid integer value: '{raw}'")


def _parse_optional_int(raw):
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Invalid optional integer value: '{raw}'")


def _normalize_whisper_language(language):
    """
    Normalize language into canonical Whisper language name (e.g. "russian").
    Accepts canonical names, ISO codes (e.g. "ru"), or token form "<|ru|>".
    """
    if language is None:
        raise RuntimeError("Whisper language cannot be empty.")

    raw = str(language).strip().lower()
    if not raw:
        raise RuntimeError("Whisper language cannot be empty.")

    # token form <|ru|>
    if raw.startswith("<|") and raw.endswith("|>"):
        code = raw[2:-2]
        if code in LANGUAGES:
            return LANGUAGES[code]
        raise RuntimeError(
            f"Unsupported Whisper language token '{language}'. "
            "Use a valid token like '<|ru|>' or language name like 'russian'."
        )

    # canonical language name (e.g. "russian")
    if raw in TO_LANGUAGE_CODE:
        return raw

    # ISO code (e.g. "ru")
    if raw in LANGUAGES:
        return LANGUAGES[raw]

    raise RuntimeError(
        f"Unsupported Whisper language '{language}'. "
        "Use a canonical language name (e.g. 'russian')."
    )


def _normalize_whisper_task(task):
    if task is None:
        raise RuntimeError("Whisper task cannot be empty.")
    normalized = str(task).strip().lower()
    if normalized in TASK_IDS:
        return normalized
    raise RuntimeError(f"Unsupported Whisper task '{task}'. Supported: {TASK_IDS}.")


def apply_profile(config):
    profile = (config.profile or "default").strip().lower()
    if profile == "quality":
        return replace(config, chunk_length_s=20, stride_length_s=8, deterministic_decoding=True)
    if profile == "fast":
        return replace(config, chunk_length_s=30, stride_length_s=5, deterministic_decoding=True)
    if profile == "default":
        return replace(config, profile="default")
    raise RuntimeError(
        f"Unsupported SR_PROFILE='{profile}'. Supported values: default, quality, fast."
    )


def _assert_removed_mode_env_vars():
    for name in ("SR_SEGMENTATION_MODE", "SR_SPEAKER_MODE"):
        raw = os.getenv(name)
        if raw is not None:
            raise RuntimeError(
                f"{name} is not supported anymore. Remove it from your environment. "
                "Only one built-in pipeline is available."
            )


def normalize_runtime_config(config):
    if config is None:
        return load_config_from_env()
    normalized = apply_profile(config)
    return replace(
        normalized,
        force_language=_normalize_whisper_language(normalized.force_language),
        force_task=_normalize_whisper_task(normalized.force_task),
    )


def load_config_from_env():
    _assert_removed_mode_env_vars()
    profile = os.getenv("SR_PROFILE", "quality").strip().lower() or "default"
    base = apply_profile(TranscriptionConfig(profile=profile))

    config = TranscriptionConfig(
        profile=base.profile,
        max_word_gap_sec=_parse_float(os.getenv("SR_MAX_WORD_GAP_SEC"), base.max_word_gap_sec),
        max_segment_duration_sec=_parse_float(
            os.getenv("SR_MAX_SEGMENT_DURATION_SEC"), base.max_segment_duration_sec
        ),
        dedup_overlap_ratio=_parse_float(os.getenv("SR_DEDUP_OVERLAP_RATIO"), base.dedup_overlap_ratio),
        min_word_overlap_ratio=_parse_float(
            os.getenv("SR_MIN_WORD_OVERLAP_RATIO"), base.min_word_overlap_ratio
        ),
        min_speaker_margin=_parse_float(os.getenv("SR_MIN_SPEAKER_MARGIN"), base.min_speaker_margin),
        max_block_merge_gap_sec=_parse_float(
            os.getenv("SR_MAX_BLOCK_MERGE_GAP_SEC"), base.max_block_merge_gap_sec
        ),
        min_turn_duration_sec=_parse_float(os.getenv("SR_MIN_TURN_DURATION_SEC"), base.min_turn_duration_sec),
        chunk_length_s=_parse_int(os.getenv("SR_CHUNK_LENGTH_S"), base.chunk_length_s),
        stride_length_s=_parse_int(os.getenv("SR_STRIDE_LENGTH_S"), base.stride_length_s),
        force_language=_normalize_whisper_language(os.getenv("SR_FORCE_LANGUAGE", base.force_language)),
        force_task=_normalize_whisper_task(os.getenv("SR_FORCE_TASK", base.force_task)),
        deterministic_decoding=_parse_bool(
            os.getenv("SR_DETERMINISTIC_DECODING"), base.deterministic_decoding
        ),
        num_speakers=_parse_optional_int(os.getenv("SR_NUM_SPEAKERS")),
        strict_determinism=_parse_bool(
            os.getenv("SR_STRICT_DETERMINISM"), base.strict_determinism
        ),
        random_seed=_parse_int(os.getenv("SR_RANDOM_SEED"), base.random_seed),
        emit_metrics=_parse_bool(os.getenv("SR_EMIT_METRICS"), base.emit_metrics),
        debug_trace=_parse_bool(os.getenv("SR_DEBUG_TRACE"), base.debug_trace),
        debug_max_items=max(10, _parse_int(os.getenv("SR_DEBUG_MAX_ITEMS"), base.debug_max_items)),
    )

    return config


def _apply_runtime_determinism(config):
    seed = int(config.random_seed)
    has_cuda = torch.cuda.is_available()
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        # NumPy is optional for this project runtime.
        pass

    torch.manual_seed(seed)

    if has_cuda:
        # Needed by CUDA for reproducible GEMM kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.strict_determinism:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if has_cuda and hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False

        if has_cuda and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    logging.info(
        "Determinism settings: strict=%s seed=%d cuda=%s",
        config.strict_determinism,
        seed,
        has_cuda,
    )


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

def save_speech_to_file_with_indent(segments, filename):
    """
    Save transcript to file with speaker tags, time intervals, and wrapped text.
    """
    text = ""
    with open(filename, "w", encoding="utf-8") as file:
        for segment in segments:
            # Format the timestamp as [HH:MM:SS - HH:MM:SS]
            timestamp_tag = f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}]"
            speaker = (segment.get("speaker") or UNKNOWN_SPEAKER).upper()
            speaker_tag = f"**{speaker}** {timestamp_tag}:\n"
            wrapped_text = textwrap.fill(segment["text"], width=128, subsequent_indent="    ")
            text += speaker_tag + wrapped_text + "\n\n"
            file.write(speaker_tag)
            file.write(wrapped_text)
            file.write("\n\n")
        return text

def convert_audio_to_wav(input_file, output_file, audio_type=None):
    """Convert an audio file to WAV using FFmpeg CLI (works on Python 3.13+)."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError(
            "ffmpeg is not available in PATH. Install FFmpeg and ensure `ffmpeg` command is resolvable."
        )

    command = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_file,
        output_file,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        logging.error(f"Failed to convert '{input_file}' to WAV with ffmpeg. {stderr}")
        raise RuntimeError(f"FFmpeg conversion failed for '{input_file}': {stderr}") from e

    logging.info(f"Successfully converted '{input_file}' to '{output_file}'")

def _assert_runtime_dependencies():
    """
    Fail fast when required runtime dependencies are not correctly installed.
    No fallback behavior is allowed.
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError(
            "Missing dependency: FFmpeg is not available in PATH.\n"
            "Install instructions:\n"
            "1. Install FFmpeg full-shared build for Windows.\n"
            "2. Add FFmpeg bin directory to PATH.\n"
            "3. Restart terminal/IDE and re-run."
        )

    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    ffmpeg_shared_dlls = [
        name for name in os.listdir(ffmpeg_dir)
        if name.lower().endswith(".dll") and (name.lower().startswith("av") or name.lower().startswith("sw"))
    ]
    if not ffmpeg_shared_dlls:
        raise RuntimeError(
            "Incompatible FFmpeg install detected: static build without shared DLLs.\n"
            f"Detected ffmpeg executable: {ffmpeg_exe}\n"
            "Install instructions:\n"
            "1. Install FFmpeg full-shared build for Windows (not full_build/static).\n"
            "2. Ensure ffmpeg bin folder contains files like avcodec-*.dll / avutil-*.dll.\n"
            "3. Put that bin folder first in PATH.\n"
            "4. Restart terminal/IDE and re-run."
        )

    try:
        import torchcodec  # noqa: F401
    except Exception as e:
        root_error = str(e).splitlines()[0] if str(e) else repr(e)
        raise RuntimeError(
            "Missing or broken dependency: torchcodec could not be loaded.\n"
            "Install instructions:\n"
            "1. Install FFmpeg full-shared build and ensure its DLLs are on PATH.\n"
            "2. Install compatible versions of torch, torchaudio, and torchcodec:\n"
            "   uv add \"torch==2.10.0\" \"torchaudio==2.10.0\" \"torchcodec==0.10.0\"\n"
            "3. Verify compatibility table:\n"
            "   https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec\n"
            "4. Recreate venv and reinstall dependencies if needed.\n"
            f"Original error: {root_error}"
        ) from None


def _load_diarization_pipeline(diarization_model_id, token):
    """Load diarization pipeline using the current pyannote API."""
    from pyannote.audio import Pipeline
    return Pipeline.from_pretrained(diarization_model_id, token=token)


def _normalize_for_dedup(text):
    cleaned = re.sub(r"\s+", " ", text).strip().lower().replace("ё", "е")
    cleaned = re.sub(r"(^[^\w]+|[^\w]+$)", "", cleaned, flags=re.UNICODE)
    return cleaned


def _overlap_ratio(a_start, a_end, b_start, b_end):
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    min_duration = max(min(a_end - a_start, b_end - b_start), 1e-9)
    return intersection / min_duration


def _prepare_word_chunks(raw_chunks):
    words = []
    for chunk in raw_chunks:
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
            continue

        start, end = timestamp
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        words.append({"text": text, "start": start, "end": end})

    prev_end = None
    for word in words:
        if word["start"] is None and prev_end is not None:
            word["start"] = prev_end
        if word["end"] is not None:
            prev_end = float(word["end"])

    next_start = None
    for word in reversed(words):
        if word["end"] is None and next_start is not None:
            word["end"] = next_start
        if word["start"] is not None:
            next_start = float(word["start"])

    cleaned = []
    for word in words:
        if word["start"] is None or word["end"] is None:
            continue

        start = float(word["start"])
        end = float(word["end"])
        if end <= start:
            end = start + 0.01
        cleaned.append({"text": word["text"], "start": start, "end": end})

    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    return cleaned


def deduplicate_word_chunks(words, overlap_threshold, debug_state=None, debug_limit=0):
    if not words:
        if debug_state is not None:
            debug_state["dedup"] = {
                "input_count": 0,
                "output_count": 0,
                "removed_duplicates": 0,
                "sample_removed": [],
            }
        return []

    deduplicated = [words[0].copy()]
    removed_duplicates = 0
    removed_events = []
    for word in words[1:]:
        prev = deduplicated[-1]
        current = word.copy()
        if _normalize_for_dedup(current["text"]) == _normalize_for_dedup(prev["text"]):
            ratio = _overlap_ratio(prev["start"], prev["end"], current["start"], current["end"])
            if ratio >= overlap_threshold:
                removed_duplicates += 1
                if debug_state is not None and len(removed_events) < debug_limit:
                    removed_events.append(
                        {
                            "reason": "text_match_with_overlap",
                            "overlap_ratio": _round_float(ratio),
                            "kept": _serialize_word(prev),
                            "removed": _serialize_word(current),
                        }
                    )
                prev["start"] = min(prev["start"], current["start"])
                prev["end"] = max(prev["end"], current["end"])
                continue
        deduplicated.append(current)

    if debug_state is not None:
        debug_state["dedup"] = {
            "input_count": len(words),
            "output_count": len(deduplicated),
            "removed_duplicates": removed_duplicates,
            "sample_removed": removed_events,
        }
    return deduplicated


def _extract_diarization_turns(diarized_result):
    annotation = getattr(diarized_result, "speaker_diarization", diarized_result)
    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(
            "Unsupported diarization output object: expected annotation.itertracks(yield_label=True)."
        )

    turns = []
    for turn, _, speaker_label in annotation.itertracks(yield_label=True):
        turns.append(
            {"speaker": str(speaker_label), "start": float(turn.start), "end": float(turn.end)}
        )

    turns.sort(key=lambda item: (item["start"], item["end"]))
    return turns


def _merge_adjacent_turns(turns, max_gap=0.05):
    if not turns:
        return []

    merged = [turns[0].copy()]
    for turn in turns[1:]:
        current = turn.copy()
        prev = merged[-1]
        gap = current["start"] - prev["end"]
        if current["speaker"] == prev["speaker"] and gap <= max_gap:
            prev["end"] = max(prev["end"], current["end"])
        else:
            merged.append(current)
    return merged


def _smooth_micro_turns(turns, min_turn_duration):
    if len(turns) < 3:
        return turns

    smoothed = [turn.copy() for turn in turns]
    i = 1
    while i < len(smoothed) - 1:
        prev_turn = smoothed[i - 1]
        turn = smoothed[i]
        next_turn = smoothed[i + 1]
        duration = turn["end"] - turn["start"]
        same_neighbors = prev_turn["speaker"] == next_turn["speaker"]
        bridgeable = (next_turn["start"] - prev_turn["end"]) <= 0.35
        if duration < min_turn_duration and same_neighbors and bridgeable:
            prev_turn["end"] = next_turn["end"]
            smoothed.pop(i + 1)
            smoothed.pop(i)
            continue
        i += 1
    return smoothed


def _prepare_diarization_turns(diarized_result, min_turn_duration_sec):
    turns = _extract_diarization_turns(diarized_result)
    turns = _merge_adjacent_turns(turns)
    turns = _smooth_micro_turns(turns, min_turn_duration_sec)
    return turns


def _span_overlap(start, end, turn_start, turn_end):
    return max(0.0, min(end, turn_end) - max(start, turn_start))


def _speaker_vote(start, end, turns):
    speaker_durations = {}
    overlapping_turns = []
    for turn in turns:
        if start < turn["end"] and end > turn["start"]:
            overlap = _span_overlap(start, end, turn["start"], turn["end"])
            if overlap <= 0:
                continue
            overlapping_turns.append(turn)
            speaker = turn["speaker"]
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + overlap

    total = sum(speaker_durations.values())
    if not speaker_durations:
        return {
            "speaker": UNKNOWN_SPEAKER,
            "speaker_durations": {},
            "total": 0.0,
            "best_ratio": 0.0,
            "margin": 0.0,
            "overlapping_turns": overlapping_turns,
        }

    ranked = sorted(speaker_durations.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_duration = ranked[0]
    second_duration = ranked[1][1] if len(ranked) > 1 else 0.0
    best_ratio = best_duration / total if total > 0 else 0.0
    margin = (best_duration - second_duration) / total if total > 0 else 0.0
    return {
        "speaker": best_speaker,
        "speaker_durations": speaker_durations,
        "total": total,
        "best_ratio": best_ratio,
        "margin": margin,
        "overlapping_turns": overlapping_turns,
    }


def _format_words(words):
    text = " ".join(word["text"].strip() for word in words if word.get("text"))
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\w\)])\s*-\s*([\w\(])", r"\1-\2", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_word_speakers(words, turns, config, debug_state=None):
    assigned = []
    rejected_to_unknown = 0
    unknown_words = 0
    sample_votes = []
    speaker_word_counts = {}
    debug_limit = max(1, int(config.debug_max_items))

    for word in words:
        vote = _speaker_vote(word["start"], word["end"], turns)
        speaker = vote["speaker"]
        confidence = vote["best_ratio"]
        rejected = False

        if speaker != UNKNOWN_SPEAKER and (
            vote["best_ratio"] < config.min_word_overlap_ratio
            or vote["margin"] < config.min_speaker_margin
        ):
            speaker = UNKNOWN_SPEAKER
            rejected = True
            rejected_to_unknown += 1

        if speaker == UNKNOWN_SPEAKER:
            unknown_words += 1
        speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1

        enriched = word.copy()
        enriched["speaker"] = speaker
        enriched["speaker_confidence"] = confidence
        enriched["speaker_vote_margin"] = vote["margin"]
        enriched["speaker_durations"] = vote["speaker_durations"]
        assigned.append(enriched)

        if debug_state is not None and len(sample_votes) < debug_limit:
            sample_votes.append(
                {
                    "word": _serialize_word(enriched),
                    "vote_best_ratio": _round_float(vote["best_ratio"]),
                    "vote_margin": _round_float(vote["margin"]),
                    "vote_total_overlap": _round_float(vote["total"]),
                    "rejected_to_unknown": rejected,
                }
            )

    if debug_state is not None:
        debug_state["word_speaker_assignment"] = {
            "word_count": len(words),
            "unknown_words": unknown_words,
            "rejected_to_unknown": rejected_to_unknown,
            "speaker_word_counts": speaker_word_counts,
            "sample_votes": sample_votes,
        }
    return assigned


def segment_words(words, config, split_on_speaker_change=False, debug_state=None):
    if not words:
        if debug_state is not None:
            debug_state["segmentation"] = {
                "segment_count": 0,
                "boundary_event_count": 0,
                "boundary_events": [],
            }
        return []

    segments = []
    current_words = []
    boundary_events = []
    debug_limit = max(1, int(config.debug_max_items))

    def flush_segment():
        if not current_words:
            return
        segment_words_copy = [word.copy() for word in current_words]
        segments.append(
            {
                "start": segment_words_copy[0]["start"],
                "end": segment_words_copy[-1]["end"],
                "text": _format_words(segment_words_copy),
                "speaker": None,
                "words": segment_words_copy,
            }
        )
        current_words.clear()

    for word in words:
        if not current_words:
            current_words.append(word)
            continue

        prev_word = current_words[-1]
        gap = max(0.0, word["start"] - prev_word["end"])
        ends_strong = bool(prev_word.get("text") and prev_word["text"].strip().endswith((".", "!", "?")))
        hard_max = (word["end"] - current_words[0]["start"]) > config.max_segment_duration_sec

        speaker_boundary = False
        if split_on_speaker_change:
            prev_speaker = prev_word.get("speaker")
            curr_speaker = word.get("speaker")
            prev_conf = float(prev_word.get("speaker_confidence", 0.0))
            curr_conf = float(word.get("speaker_confidence", 0.0))
            speaker_boundary = (
                prev_speaker not in {None, UNKNOWN_SPEAKER}
                and curr_speaker not in {None, UNKNOWN_SPEAKER}
                and prev_speaker != curr_speaker
                and prev_conf >= config.min_word_overlap_ratio
                and curr_conf >= config.min_word_overlap_ratio
            )

        if ends_strong or gap > config.max_word_gap_sec or hard_max or speaker_boundary:
            if debug_state is not None and len(boundary_events) < debug_limit:
                reasons = []
                if ends_strong:
                    reasons.append("punctuation")
                if gap > config.max_word_gap_sec:
                    reasons.append("silence_gap")
                if hard_max:
                    reasons.append("hard_max_duration")
                if speaker_boundary:
                    reasons.append("speaker_change")
                boundary_events.append(
                    {
                        "reasons": reasons,
                        "gap_sec": _round_float(gap),
                        "segment_start": _round_float(current_words[0]["start"]),
                        "segment_end": _round_float(prev_word["end"]),
                        "prev_word": _serialize_word(prev_word),
                        "next_word": _serialize_word(word),
                    }
                )
            flush_segment()
        current_words.append(word)

    flush_segment()
    if debug_state is not None:
        debug_state["segmentation"] = {
            "segment_count": len(segments),
            "boundary_event_count": len(boundary_events),
            "boundary_events": boundary_events,
        }
    return segments


def assign_sentence_speakers_v2(sentences, config, debug_state=None):
    decision_events = []
    debug_limit = max(1, int(config.debug_max_items))
    unknown_sentences = 0

    for sentence in sentences:
        words = sentence.get("words", [])
        speaker_scores = {}
        for word in words:
            speaker = word.get("speaker", UNKNOWN_SPEAKER)
            if speaker in {None, UNKNOWN_SPEAKER}:
                continue
            duration = max(0.01, word["end"] - word["start"])
            confidence = max(0.01, float(word.get("speaker_confidence", 0.0)))
            speaker_scores[speaker] = speaker_scores.get(speaker, 0.0) + duration * confidence

        if not speaker_scores:
            sentence["speaker"] = UNKNOWN_SPEAKER
            sentence["speaker_confidence"] = 0.0
            sentence["speaker_scores"] = {}
            unknown_sentences += 1
            if debug_state is not None and len(decision_events) < debug_limit:
                decision_events.append(
                    {
                        "segment": _serialize_sentence(sentence, include_words=False),
                        "decision": "unknown_no_scores",
                    }
                )
            continue

        ranked = sorted(speaker_scores.items(), key=lambda item: item[1], reverse=True)
        best_speaker, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total_score = sum(speaker_scores.values())
        best_ratio = best_score / total_score if total_score > 0 else 0.0
        margin = (best_score - second_score) / total_score if total_score > 0 else 0.0
        decision = "accept"

        if best_ratio < config.min_word_overlap_ratio or margin < config.min_speaker_margin:
            sentence["speaker"] = UNKNOWN_SPEAKER
            unknown_sentences += 1
            decision = "unknown_low_confidence"
        else:
            sentence["speaker"] = best_speaker

        sentence["speaker_confidence"] = best_ratio
        sentence["speaker_scores"] = speaker_scores

        if debug_state is not None and len(decision_events) < debug_limit:
            decision_events.append(
                {
                    "segment": _serialize_sentence(sentence, include_words=False),
                    "decision": decision,
                    "best_speaker": best_speaker,
                    "best_ratio": _round_float(best_ratio),
                    "margin": _round_float(margin),
                    "speaker_scores": {
                        str(speaker): _round_float(score)
                        for speaker, score in speaker_scores.items()
                    },
                }
            )

    if debug_state is not None:
        debug_state["sentence_speaker_assignment"] = {
            "sentence_count": len(sentences),
            "unknown_sentences": unknown_sentences,
            "decision_events": decision_events,
        }
    return sentences


def merge_consecutive_sentences(sentences, config, debug_state=None):
    if not sentences:
        if debug_state is not None:
            debug_state["merge"] = {
                "input_count": 0,
                "output_count": 0,
                "merge_count": 0,
                "events": [],
            }
        return []

    ordered = sorted(sentences, key=lambda item: item["start"])
    merged = [ordered[0].copy()]
    merge_count = 0
    merge_events = []
    debug_limit = max(1, int(config.debug_max_items))

    for sentence in ordered[1:]:
        current = merged[-1]
        gap = sentence["start"] - current["end"]
        same_speaker = sentence.get("speaker") == current.get("speaker")
        can_merge = same_speaker
        merge_reasons = []
        if not same_speaker:
            merge_reasons.append("speaker_mismatch")
        if can_merge and gap > config.max_block_merge_gap_sec:
            can_merge = False
            merge_reasons.append("gap_too_large")
        if can_merge:
            current_conf = float(current.get("speaker_confidence", 0.0))
            next_conf = float(sentence.get("speaker_confidence", 0.0))
            if min(current_conf, next_conf) < config.min_word_overlap_ratio:
                can_merge = False
                merge_reasons.append("low_confidence")

        if can_merge:
            merge_count += 1
            current["end"] = max(current["end"], sentence["end"])
            current["text"] = _format_words(
                [{"text": current["text"]}, {"text": sentence["text"]}]
            )
            current.setdefault("diarized_segments", []).extend(sentence.get("diarized_segments", []))
            if "words" in current or "words" in sentence:
                current.setdefault("words", []).extend(sentence.get("words", []))
            if "speaker_confidence" in sentence:
                current["speaker_confidence"] = min(
                    float(current.get("speaker_confidence", sentence["speaker_confidence"])),
                    float(sentence.get("speaker_confidence", 0.0)),
                )
            if debug_state is not None and len(merge_events) < debug_limit:
                merge_events.append(
                    {
                        "merged": True,
                        "gap_sec": _round_float(gap),
                        "speaker": current.get("speaker", UNKNOWN_SPEAKER),
                        "left_start": _round_float(current.get("start")),
                        "right_start": _round_float(sentence.get("start")),
                    }
                )
        else:
            merged.append(sentence.copy())
            if debug_state is not None and len(merge_events) < debug_limit:
                merge_events.append(
                    {
                        "merged": False,
                        "gap_sec": _round_float(gap),
                        "left_speaker": current.get("speaker", UNKNOWN_SPEAKER),
                        "right_speaker": sentence.get("speaker", UNKNOWN_SPEAKER),
                        "reasons": merge_reasons or ["unknown"],
                    }
                )

    if debug_state is not None:
        debug_state["merge"] = {
            "input_count": len(sentences),
            "output_count": len(merged),
            "merge_count": merge_count,
            "events": merge_events,
        }

    return merged


def _round_float(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def _serialize_raw_chunk(chunk):
    timestamp = chunk.get("timestamp")
    start = None
    end = None
    if isinstance(timestamp, (tuple, list)) and len(timestamp) == 2:
        start = _round_float(timestamp[0])
        end = _round_float(timestamp[1])
    return {"start": start, "end": end, "text": (chunk.get("text") or "").strip()}


def _serialize_word(word):
    payload = {
        "start": _round_float(word.get("start")),
        "end": _round_float(word.get("end")),
        "text": (word.get("text") or "").strip(),
    }
    if "speaker" in word:
        payload["speaker"] = word.get("speaker")
    if "speaker_confidence" in word:
        payload["speaker_confidence"] = _round_float(word.get("speaker_confidence"))
    if "speaker_vote_margin" in word:
        payload["speaker_vote_margin"] = _round_float(word.get("speaker_vote_margin"))
    speaker_durations = word.get("speaker_durations") or {}
    if speaker_durations:
        payload["speaker_durations"] = {
            str(speaker): _round_float(duration)
            for speaker, duration in speaker_durations.items()
        }
    return payload


def _serialize_sentence(sentence, include_words=False, max_words_per_sentence=0):
    payload = {
        "start": _round_float(sentence.get("start")),
        "end": _round_float(sentence.get("end")),
        "speaker": sentence.get("speaker", UNKNOWN_SPEAKER),
        "speaker_confidence": _round_float(sentence.get("speaker_confidence")),
        "text": (sentence.get("text") or "").strip(),
    }
    speaker_scores = sentence.get("speaker_scores") or {}
    if speaker_scores:
        payload["speaker_scores"] = {
            str(speaker): _round_float(score)
            for speaker, score in speaker_scores.items()
        }
    if include_words and sentence.get("words"):
        words = sentence["words"]
        limit = len(words) if max_words_per_sentence <= 0 else max_words_per_sentence
        payload["words"] = [_serialize_word(word) for word in words[:limit]]
        if len(words) > limit:
            payload["words_truncated_count"] = len(words) - limit
    return payload


def _sample_list(items, max_items):
    if len(items) <= max_items:
        return items
    half = max_items // 2
    return items[:half] + items[-half:]


def _tokenize_for_quality(text):
    normalized = re.sub(r"[^\w\- ]+", " ", (text or "").lower(), flags=re.UNICODE)
    return [token for token in normalized.split() if token]


def _detect_text_anomalies(segments, unique_ratio_threshold=0.45, min_token_count=10):
    anomalies = []
    for idx, segment in enumerate(segments):
        tokens = _tokenize_for_quality(segment.get("text", ""))
        token_count = len(tokens)
        if token_count < min_token_count:
            continue
        unique_ratio = len(set(tokens)) / token_count if token_count else 1.0
        if unique_ratio < unique_ratio_threshold:
            anomalies.append(
                {
                    "segment_index": idx,
                    "start": _round_float(segment.get("start")),
                    "end": _round_float(segment.get("end")),
                    "speaker": segment.get("speaker", UNKNOWN_SPEAKER),
                    "token_count": token_count,
                    "unique_ratio": _round_float(unique_ratio),
                    "text_preview": (segment.get("text") or "")[:240],
                }
            )
    return anomalies


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_debug_artifacts(
    file_name,
    config,
    script,
    raw_chunks,
    turns,
    prepared_words,
    deduped_words,
    speaker_words,
    sentences,
    merged_sentences,
    debug_state,
):
    debug_dir = Path(file_name).parent / "transcripts" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(file_name).stem
    max_items = max(10, int(config.debug_max_items))
    unknown_word_count = sum(1 for word in speaker_words if word.get("speaker") == UNKNOWN_SPEAKER)
    unknown_segment_count = sum(
        1 for segment in merged_sentences if (segment.get("speaker") or UNKNOWN_SPEAKER) == UNKNOWN_SPEAKER
    )
    anomalies = _detect_text_anomalies(merged_sentences)

    speaker_word_counts = {}
    for word in speaker_words:
        speaker = word.get("speaker") or UNKNOWN_SPEAKER
        speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(file_name),
        "pipeline": "v2",
        "config": asdict(config),
        "counts": {
            "raw_chunk_count": len(raw_chunks),
            "prepared_word_count": len(prepared_words),
            "deduplicated_word_count": len(deduped_words),
            "speaker_word_count": len(speaker_words),
            "diarization_turn_count": len(turns),
            "segment_count_pre_merge": len(sentences),
            "segment_count_final": len(merged_sentences),
            "unknown_word_count": unknown_word_count,
            "unknown_segment_count": unknown_segment_count,
            "text_anomaly_count": len(anomalies),
        },
        "asr": {
            "raw_text": script.get("text", ""),
            "raw_chunks_sample": _sample_list(
                [_serialize_raw_chunk(chunk) for chunk in raw_chunks], max_items
            ),
        },
        "diarization": {
            "turns_sample": _sample_list(
                [
                    {
                        "speaker": turn["speaker"],
                        "start": _round_float(turn["start"]),
                        "end": _round_float(turn["end"]),
                    }
                    for turn in turns
                ],
                max_items,
            ),
        },
        "word_pipeline": {
            "speaker_word_counts": speaker_word_counts,
            "prepared_words_sample": _sample_list(
                [_serialize_word(word) for word in prepared_words], max_items
            ),
            "deduplicated_words_sample": _sample_list(
                [_serialize_word(word) for word in deduped_words], max_items
            ),
            "speaker_words_sample": _sample_list(
                [_serialize_word(word) for word in speaker_words], max_items
            ),
        },
        "segment_pipeline": {
            "segments_pre_merge_sample": _sample_list(
                [_serialize_sentence(sentence, include_words=False) for sentence in sentences], max_items
            ),
            "segments_final": [
                _serialize_sentence(sentence, include_words=False)
                for sentence in merged_sentences
            ],
            "text_anomaly_samples": anomalies[: min(len(anomalies), max_items)],
        },
        "decision_debug": debug_state or {},
    }

    debug_json_file = debug_dir / f"{stem}.debug.json"
    with open(debug_json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    raw_asr_file = debug_dir / f"{stem}.asr_raw.txt"
    raw_asr_file.write_text(script.get("text", ""), encoding="utf-8")

    _write_jsonl(
        debug_dir / f"{stem}.raw_chunks.jsonl",
        [_serialize_raw_chunk(chunk) for chunk in raw_chunks],
    )
    _write_jsonl(
        debug_dir / f"{stem}.turns.jsonl",
        [
            {
                "speaker": turn["speaker"],
                "start": _round_float(turn["start"]),
                "end": _round_float(turn["end"]),
            }
            for turn in turns
        ],
    )
    _write_jsonl(
        debug_dir / f"{stem}.words.jsonl",
        [_serialize_word(word) for word in speaker_words],
    )
    _write_jsonl(
        debug_dir / f"{stem}.segments.pre_merge.jsonl",
        [_serialize_sentence(sentence, include_words=True, max_words_per_sentence=40) for sentence in sentences],
    )
    _write_jsonl(
        debug_dir / f"{stem}.segments.final.jsonl",
        [_serialize_sentence(sentence, include_words=True, max_words_per_sentence=40) for sentence in merged_sentences],
    )

    logging.info("Debug artifacts written to %s", debug_dir)


def _write_job_metrics(file_name, segments, config, raw_chunk_count, usable_word_count, turn_count):
    if not segments:
        return

    metrics_dir = Path(file_name).parent / "transcripts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    durations = [max(0.0, seg["end"] - seg["start"]) for seg in segments]
    total_span = max(0.0, segments[-1]["end"] - segments[0]["start"])
    unknown_duration = sum(
        duration
        for seg, duration in zip(segments, durations)
        if (seg.get("speaker") or UNKNOWN_SPEAKER) == UNKNOWN_SPEAKER
    )

    speaker_switches = 0
    for i in range(1, len(segments)):
        prev_speaker = segments[i - 1].get("speaker") or UNKNOWN_SPEAKER
        curr_speaker = segments[i].get("speaker") or UNKNOWN_SPEAKER
        if prev_speaker != curr_speaker:
            speaker_switches += 1
    anomalies = _detect_text_anomalies(segments)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(file_name),
        "pipeline": "v2",
        "profile": config.profile,
        "raw_chunk_count": raw_chunk_count,
        "usable_word_count": usable_word_count,
        "diarization_turn_count": turn_count,
        "segment_count": len(segments),
        "max_segment_duration_sec": max(durations) if durations else 0.0,
        "avg_segment_duration_sec": (sum(durations) / len(durations)) if durations else 0.0,
        "speaker_switch_count": speaker_switches,
        "unknown_speaker_duration_ratio": (unknown_duration / total_span) if total_span > 0 else 0.0,
        "text_anomaly_count": len(anomalies),
        "text_anomaly_samples": anomalies[: min(len(anomalies), 10)],
        "config": asdict(config),
    }

    metrics_file = metrics_dir / f"{Path(file_name).stem}.metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def transcription_factory(whisper_model_id, diarization_model_id, config=None):
    config = normalize_runtime_config(config)
    _apply_runtime_determinism(config)
    _assert_runtime_dependencies()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(
        "Initializing ASR/diarization pipelines on device=%s | pipeline=v2 | profile=%s",
        device,
        config.profile,
    )

    try:
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        whisper_model.config.forced_decoder_ids = None
        whisper_model.to(device)
        whisper_processor = AutoProcessor.from_pretrained(whisper_model_id, token=HF_TOKEN)
    except Exception as e:
        logging.error("Failed to initialize Whisper model: %s", e)
        raise

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
    logging.info(
        "Whisper generate kwargs: language=%s task=%s deterministic=%s",
        config.force_language,
        config.force_task,
        config.deterministic_decoding,
    )

    def transcript(file_name):
        logging.info("=============> started with %s", file_name)
        script = whisper_pipe(file_name, return_timestamps="word", generate_kwargs=generate_kwargs)
        raw_chunks = script.get("chunks", [])
        debug_state = {} if config.debug_trace else None

        diarization_kwargs = {}
        if config.num_speakers is not None:
            diarization_kwargs["num_speakers"] = config.num_speakers

        diarized_result = (
            diarization_pipeline(file_name, **diarization_kwargs)
            if diarization_kwargs
            else diarization_pipeline(file_name)
        )
        turns = _prepare_diarization_turns(diarized_result, config.min_turn_duration_sec)

        raw_chunk_count = len(raw_chunks)
        prepared_words = _prepare_word_chunks(raw_chunks)
        deduped_words = deduplicate_word_chunks(
            prepared_words,
            config.dedup_overlap_ratio,
            debug_state=debug_state,
            debug_limit=config.debug_max_items,
        )
        speaker_words = assign_word_speakers(deduped_words, turns, config, debug_state=debug_state)
        sentences = segment_words(
            speaker_words,
            config,
            split_on_speaker_change=True,
            debug_state=debug_state,
        )
        sentences = assign_sentence_speakers_v2(sentences, config, debug_state=debug_state)
        usable_word_count = len(speaker_words)

        merged_sentences = merge_consecutive_sentences(sentences, config, debug_state=debug_state)

        unknown_word_count = sum(1 for word in speaker_words if word.get("speaker") == UNKNOWN_SPEAKER)
        text_anomalies = _detect_text_anomalies(merged_sentences)
        logging.info(
            "Pipeline stats: raw_chunks=%d prepared_words=%d deduped_words=%d turns=%d sentences=%d final_blocks=%d unknown_words=%d",
            raw_chunk_count,
            len(prepared_words),
            len(deduped_words),
            len(turns),
            len(sentences),
            len(merged_sentences),
            unknown_word_count,
        )
        if text_anomalies:
            logging.warning(
                "Detected %d text-quality anomalies (low unique-token ratio). Example segment: %s",
                len(text_anomalies),
                text_anomalies[0],
            )

        trans_folder = os.path.join(os.path.dirname(file_name), "transcripts")
        os.makedirs(trans_folder, exist_ok=True)
        trans_file = os.path.join(trans_folder, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        text = save_speech_to_file_with_indent(merged_sentences, trans_file)

        if config.debug_trace:
            _write_debug_artifacts(
                file_name=file_name,
                config=config,
                script=script,
                raw_chunks=raw_chunks,
                turns=turns,
                prepared_words=prepared_words,
                deduped_words=deduped_words,
                speaker_words=speaker_words,
                sentences=sentences,
                merged_sentences=merged_sentences,
                debug_state=debug_state,
            )

        if config.emit_metrics:
            _write_job_metrics(
                file_name,
                merged_sentences,
                config,
                raw_chunk_count=raw_chunk_count,
                usable_word_count=usable_word_count,
                turn_count=len(turns),
            )

        logging.info("<============= Done with %s", file_name)
        return text

    return transcript


def _get_or_create_transcriptor(whisper_model_id, diarization_model_id, config):
    key = (whisper_model_id, diarization_model_id, config)
    with _TRANSCRIPTOR_LOCK:
        transcriptor = _TRANSCRIPTOR_CACHE.get(key)
        if transcriptor is None:
            transcriptor = transcription_factory(whisper_model_id, diarization_model_id, config=config)
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


def move_to_done(filename):
    dest_dir = "./audio/done"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_path = os.path.join(dest_dir, os.path.basename(filename))
    os.rename(filename, dest_path)


def run_transcription(file_name, config=None, move_processed=True):
    config = normalize_runtime_config(config)
    transcriptor = _get_or_create_transcriptor(
        whisper_model_id=DEFAULT_WHISPER_MODEL,
        diarization_model_id=DEFAULT_DIARIZATION_MODEL,
        config=config,
    )
    transcription_text = transcribe(file_name, transcriptor)
    if move_processed:
        move_to_done(file_name)
    return transcription_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config_from_env()

    transcriptor = _get_or_create_transcriptor(
        whisper_model_id=DEFAULT_WHISPER_MODEL,
        diarization_model_id=DEFAULT_DIARIZATION_MODEL,
        config=config,
    )

    folder_path = "./audio/"
    audio_extensions = {".mp3", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".wav", ".mp4"}

    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in audio_extensions:
            full_path = file.resolve()
            logging.info("Found audio file: %s", full_path)
            transcribe(str(full_path), transcriptor)
            print(f"<============= Done with {full_path}")
