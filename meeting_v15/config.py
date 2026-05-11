from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float value: {raw!r}") from exc


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value: {raw!r}") from exc


def _parse_optional_int(raw: str | None, default: int | None = None) -> int | None:
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid optional integer value: {raw!r}") from exc


@dataclass(frozen=True)
class MeetingConfig:
    whisper_model_id: str = "openai/whisper-large-v3"
    #diarization_model_id: str = "pyannote/speaker-diarization-3.1"
    diarization_model_id: str = "pyannote/speaker-diarization-community-1"
    vad_model_id: str = "pyannote/voice-activity-detection"

    language: str = "russian"
    task: str = "transcribe"

    output_dir: str = "audio/transcripts"
    hf_token: str | None = None
    hf_token_file: str = "hf.txt"
    sample_rate: int = 16000

    vad_min_speech_s: float = 0.30
    vad_min_silence_s: float = 0.40
    vad_pad_s: float = 0.15
    overlap_min_duration_s: float = 0.0
    overlap_merge_gap_s: float = 0.0

    chunk_merge_gap_s: float = 0.50
    chunk_max_segment_s: float = 30.0
    chunk_stride_s: float = 0.75
    chunk_pad_s: float = 0.15
    speaker_assignment_max_gap_s: float = 1.0

    min_speakers: int = 2
    max_speakers: int = 8
    num_speakers: int | None = None
    micro_turn_merge_gap_s: float = 0.05

    decode_num_beams: int = 5
    decode_temperature: float = 0.0
    decode_max_new_tokens: int = 256
    decode_condition_on_prev_tokens: bool = False
    decode_no_speech_threshold: float = 0.6
    decode_compression_ratio_threshold: float = 2.4
    decode_logprob_threshold: float = -1.0
    whisper_chunk_length_s: float = 30.0
    whisper_stride_length_s: float = 5.0

    overlap_second_pass: bool = True
    emit_debug: bool = True
    move_processed_to: str | None = None

    audio_extensions: tuple[str, ...] = (
        ".aac",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".ogg",
        ".wav",
        ".wma",
    )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.min_speakers <= 0 or self.max_speakers <= 0:
            raise ValueError("min_speakers and max_speakers must be positive.")
        if self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers cannot exceed max_speakers.")
        if self.num_speakers is not None and self.num_speakers <= 0:
            raise ValueError("num_speakers must be positive when provided.")

        for field_name in (
            "vad_min_speech_s",
            "vad_min_silence_s",
            "vad_pad_s",
            "overlap_min_duration_s",
            "overlap_merge_gap_s",
            "chunk_merge_gap_s",
            "chunk_max_segment_s",
            "chunk_stride_s",
            "chunk_pad_s",
            "speaker_assignment_max_gap_s",
            "micro_turn_merge_gap_s",
            "whisper_chunk_length_s",
            "whisper_stride_length_s",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative.")

        if self.chunk_max_segment_s <= 0:
            raise ValueError("chunk_max_segment_s must be greater than zero.")
        if self.decode_num_beams <= 0:
            raise ValueError("decode_num_beams must be positive.")
        if self.decode_max_new_tokens <= 0:
            raise ValueError("decode_max_new_tokens must be positive.")


def from_env(prefix: str = "MV15_") -> MeetingConfig:
    key = lambda name: os.getenv(f"{prefix}{name}")
    return MeetingConfig(
        whisper_model_id=key("WHISPER_MODEL_ID") or MeetingConfig.whisper_model_id,
        diarization_model_id=key("DIARIZATION_MODEL_ID") or MeetingConfig.diarization_model_id,
        vad_model_id=key("VAD_MODEL_ID") or MeetingConfig.vad_model_id,
        language=key("LANGUAGE") or MeetingConfig.language,
        task=key("TASK") or MeetingConfig.task,
        output_dir=key("OUTPUT_DIR") or MeetingConfig.output_dir,
        hf_token=key("HF_TOKEN"),
        hf_token_file=key("HF_TOKEN_FILE") or MeetingConfig.hf_token_file,
        sample_rate=_parse_int(key("SAMPLE_RATE"), MeetingConfig.sample_rate),
        vad_min_speech_s=_parse_float(key("VAD_MIN_SPEECH_S"), MeetingConfig.vad_min_speech_s),
        vad_min_silence_s=_parse_float(key("VAD_MIN_SILENCE_S"), MeetingConfig.vad_min_silence_s),
        vad_pad_s=_parse_float(key("VAD_PAD_S"), MeetingConfig.vad_pad_s),
        overlap_min_duration_s=_parse_float(
            key("OVERLAP_MIN_DURATION_S"), MeetingConfig.overlap_min_duration_s
        ),
        overlap_merge_gap_s=_parse_float(key("OVERLAP_MERGE_GAP_S"), MeetingConfig.overlap_merge_gap_s),
        chunk_merge_gap_s=_parse_float(key("CHUNK_MERGE_GAP_S"), MeetingConfig.chunk_merge_gap_s),
        chunk_max_segment_s=_parse_float(key("CHUNK_MAX_SEGMENT_S"), MeetingConfig.chunk_max_segment_s),
        chunk_stride_s=_parse_float(key("CHUNK_STRIDE_S"), MeetingConfig.chunk_stride_s),
        chunk_pad_s=_parse_float(key("CHUNK_PAD_S"), MeetingConfig.chunk_pad_s),
        speaker_assignment_max_gap_s=_parse_float(
            key("SPEAKER_ASSIGNMENT_MAX_GAP_S"), MeetingConfig.speaker_assignment_max_gap_s
        ),
        min_speakers=_parse_int(key("MIN_SPEAKERS"), MeetingConfig.min_speakers),
        max_speakers=_parse_int(key("MAX_SPEAKERS"), MeetingConfig.max_speakers),
        num_speakers=_parse_optional_int(key("NUM_SPEAKERS")),
        micro_turn_merge_gap_s=_parse_float(
            key("MICRO_TURN_MERGE_GAP_S"), MeetingConfig.micro_turn_merge_gap_s
        ),
        decode_num_beams=_parse_int(key("DECODE_NUM_BEAMS"), MeetingConfig.decode_num_beams),
        decode_temperature=_parse_float(key("DECODE_TEMPERATURE"), MeetingConfig.decode_temperature),
        decode_max_new_tokens=_parse_int(
            key("DECODE_MAX_NEW_TOKENS"), MeetingConfig.decode_max_new_tokens
        ),
        decode_condition_on_prev_tokens=_parse_bool(
            key("DECODE_CONDITION_ON_PREV_TOKENS"), MeetingConfig.decode_condition_on_prev_tokens
        ),
        decode_no_speech_threshold=_parse_float(
            key("DECODE_NO_SPEECH_THRESHOLD"), MeetingConfig.decode_no_speech_threshold
        ),
        decode_compression_ratio_threshold=_parse_float(
            key("DECODE_COMPRESSION_RATIO_THRESHOLD"),
            MeetingConfig.decode_compression_ratio_threshold,
        ),
        decode_logprob_threshold=_parse_float(
            key("DECODE_LOGPROB_THRESHOLD"), MeetingConfig.decode_logprob_threshold
        ),
        whisper_chunk_length_s=_parse_float(
            key("WHISPER_CHUNK_LENGTH_S"), MeetingConfig.whisper_chunk_length_s
        ),
        whisper_stride_length_s=_parse_float(
            key("WHISPER_STRIDE_LENGTH_S"), MeetingConfig.whisper_stride_length_s
        ),
        overlap_second_pass=_parse_bool(
            key("OVERLAP_SECOND_PASS"), MeetingConfig.overlap_second_pass
        ),
        emit_debug=_parse_bool(key("EMIT_DEBUG"), MeetingConfig.emit_debug),
        move_processed_to=key("MOVE_PROCESSED_TO"),
    )
