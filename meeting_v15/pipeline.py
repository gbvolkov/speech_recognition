from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .asr import WhisperAsr
from .auth import load_hf_token
from .audio_io import normalize_audio
from .chunk_builder import build_fallback_chunks_from_vad
from .config import MeetingConfig, from_env
from .diarization import run_diarization
from .exporters import write_debug_artifacts, write_segments_json, write_text_transcript
from .postprocess import build_segments, normalize_text
from .quality import pick_best_candidate, rerun_reasons, temperature_ladder
from .types import ChunkItem, TranscriptResult
from .vad import run_vad


def _run_stage(stage: str, action):
    try:
        return action()
    except Exception as exc:
        raise RuntimeError(f"{stage} failed: {exc}") from exc


def _decode_chunk_with_quality(
    asr: WhisperAsr,
    wav_path: str,
    chunk: ChunkItem,
    work_dir: Path,
    overlap_second_pass: bool,
) -> tuple[dict, dict]:
    
    start_s = chunk["start_s"]
    end_s = chunk["end_s"]
    duration_s = max(0.0, end_s - start_s)

    initial = asr.decode_window(
        wav_path=wav_path,
        start_s=start_s,
        end_s=end_s,
        work_dir=work_dir,
        generation_config=asr.build_generation_config(),
        temperature=asr.config.decode_temperature,
    )

    reasons = rerun_reasons(
        text=initial.get("text", ""),
        duration_s=duration_s,
        has_vad_speech=True,
        overlap_flag=chunk["overlap_flag"],
        overlap_second_pass=overlap_second_pass,
    )

    candidates = [{"text": initial["text"], "words": initial["words"], "meta": {"pass": "initial"}}]
    rerun_attempts = []

    if reasons:
        for temperature in temperature_ladder():
            candidate = asr.decode_window(
                wav_path=wav_path,
                start_s=start_s,
                end_s=end_s,
                work_dir=work_dir,
                generation_config=asr.build_generation_config(
                    num_beams=1,
                    disable_whisper_internal_fallback=True,
                ),
                temperature=temperature,
                word_timestamps=False,
            )
            rerun_attempts.append(
                {
                    "temperature": temperature,
                    "text_preview": (candidate.get("text") or "")[:160],
                    "word_count": len(candidate.get("words", [])),
                }
            )
            candidates.append(
                {
                    "text": candidate.get("text", ""),
                    "words": [],
                    "meta": {"pass": "rerun", "temperature": temperature, "num_beams": 1},
                }
            )

    selected = pick_best_candidate(candidates)
    selected_meta = selected.get("meta", {})
    if selected_meta.get("pass") == "rerun":
        selected = asr.decode_window(
            wav_path=wav_path,
            start_s=start_s,
            end_s=end_s,
            work_dir=work_dir,
            generation_config=asr.build_generation_config(
                num_beams=int(selected_meta.get("num_beams", 1)),
                disable_whisper_internal_fallback=True,
            ),
            temperature=float(selected_meta.get("temperature", asr.config.decode_temperature)),
            word_timestamps=True,
        ) | {"meta": selected_meta}

    record = {
        "start_s": start_s,
        "end_s": end_s,
        "speaker_id": chunk["speaker_id"],
        "overlap_flag": chunk["overlap_flag"],
        "source": chunk["source"],
        "text": selected.get("text", ""),
        "words": selected.get("words", []),
    }
    event = {
        "chunk_id": chunk["chunk_id"],
        "reasons": reasons,
        "rerun_attempts": rerun_attempts,
        "selected_meta": selected_meta,
        "selected_text_preview": (selected.get("text") or "")[:200],
    }
    return record, event


def _intersects_overlap(start_s: float, end_s: float, overlap_windows: list[dict]) -> bool:
    for window in overlap_windows:
        if min(end_s, float(window["end_s"])) > max(start_s, float(window["start_s"])):
            return True
    return False


def _speaker_at(
    midpoint_s: float,
    exclusive_turns: list[dict],
    speaker_turns: list[dict],
    max_gap_s: float,
) -> str:
    for turn in exclusive_turns:
        if float(turn["start_s"]) <= midpoint_s < float(turn["end_s"]):
            return str(turn["speaker_id"])

    nearest_label = "UNKNOWN"
    nearest_distance = None
    for turn in speaker_turns:
        start_s = float(turn["start_s"])
        end_s = float(turn["end_s"])
        if start_s <= midpoint_s < end_s:
            return str(turn["speaker_id"])
        distance = min(abs(midpoint_s - start_s), abs(midpoint_s - end_s))
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_label = str(turn["speaker_id"])

    if nearest_distance is not None and nearest_distance <= max_gap_s:
        return nearest_label
    return "UNKNOWN"


def _build_full_context_records(
    words: list[dict],
    full_text: str,
    duration_s: float,
    exclusive_turns: list[dict],
    speaker_turns: list[dict],
    overlap_windows: list[dict],
    max_gap_s: float,
    speaker_assignment_max_gap_s: float,
) -> list[dict]:
    if not words:
        text = normalize_text(full_text or "")
        if not text:
            return []
        return [
            {
                "start_s": 0.0,
                "end_s": duration_s,
                "speaker_id": "UNKNOWN",
                "overlap_flag": _intersects_overlap(0.0, duration_s, overlap_windows),
                "source": "whisper_full_context",
                "text": text,
                "words": [],
            }
        ]

    ordered_words = sorted(words, key=lambda item: (float(item["start_s"]), float(item["end_s"])))
    records: list[dict] = []
    current: dict | None = None

    for word in ordered_words:
        word_start = float(word["start_s"])
        word_end = float(word["end_s"])
        midpoint = (word_start + word_end) / 2.0
        speaker_id = _speaker_at(
            midpoint,
            exclusive_turns=exclusive_turns,
            speaker_turns=speaker_turns,
            max_gap_s=speaker_assignment_max_gap_s,
        )
        token = (word.get("word") or "").strip()
        if not token:
            continue

        if current is None:
            current = {
                "start_s": word_start,
                "end_s": word_end,
                "speaker_id": speaker_id,
                "overlap_flag": False,
                "source": "whisper_full_context",
                "text": token,
                "words": [word],
            }
            continue

        gap = word_start - float(current["end_s"])
        same_speaker = speaker_id == str(current["speaker_id"])
        if same_speaker and gap <= max_gap_s:
            current["end_s"] = word_end
            current["text"] = f"{current['text']} {token}"
            current["words"].append(word)
        else:
            current["text"] = normalize_text(str(current["text"]))
            current["overlap_flag"] = _intersects_overlap(
                float(current["start_s"]), float(current["end_s"]), overlap_windows
            )
            records.append(current)
            current = {
                "start_s": word_start,
                "end_s": word_end,
                "speaker_id": speaker_id,
                "overlap_flag": False,
                "source": "whisper_full_context",
                "text": token,
                "words": [word],
            }

    if current is not None:
        current["text"] = normalize_text(str(current["text"]))
        current["overlap_flag"] = _intersects_overlap(
            float(current["start_s"]), float(current["end_s"]), overlap_windows
        )
        records.append(current)
    return records


def _output_paths(config: MeetingConfig, source_path: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    return (
        output_dir / f"{stem}.txt",
        output_dir / f"{stem}.segments.json",
        output_dir / "debug_v15",
    )


def _move_processed_file(source_file: Path, destination_dir: str | None) -> str | None:
    if not destination_dir:
        return None
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source_file.name
    shutil.move(str(source_file), str(target))
    return str(target)


def run_meeting_transcription(
    input_path: str,
    config: MeetingConfig | None = None,
) -> TranscriptResult:
    cfg = config or from_env()
    source_file = Path(input_path).resolve()
    if not source_file.exists() or not source_file.is_file():
        raise RuntimeError(f"Input path is not a file: {source_file}")

    text_path, json_path, debug_dir = _output_paths(cfg, source_file)
    token = _run_stage("auth", lambda: load_hf_token(cfg.hf_token, cfg.hf_token_file))
    asr = _run_stage("asr_init", lambda: WhisperAsr(cfg))

    with tempfile.TemporaryDirectory(prefix="meeting_v15_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        normalized = _run_stage(
            "audio_normalization",
            lambda: normalize_audio(source_file, tmp_dir, sample_rate=cfg.sample_rate),
        )
        wav_path = normalized["wav_path"]
        duration_s = float(normalized["duration_s"])

        vad_regions = _run_stage(
            "vad",
            lambda: run_vad(
                wav_path=wav_path,
                model_id=cfg.vad_model_id,
                token=token,
                min_speech_s=cfg.vad_min_speech_s,
                min_silence_s=cfg.vad_min_silence_s,
                pad_s=cfg.vad_pad_s,
                duration_s=duration_s,
            ),
        )

        if not vad_regions:
            result: TranscriptResult = {
                "source_file": str(source_file),
                "duration_s": duration_s,
                "segments": [],
                "metadata": {
                    "reason": "no_vad_speech",
                    "config": asdict(cfg),
                },
                "outputs": {"text_path": str(text_path), "json_path": str(json_path)},
            }
            write_text_transcript([], text_path)
            write_segments_json(result, json_path)
            return result

        diarization_payload = _run_stage(
            "diarization",
            lambda: run_diarization(
                wav_path=wav_path,
                model_id=cfg.diarization_model_id,
                token=token,
                min_speakers=cfg.min_speakers,
                max_speakers=cfg.max_speakers,
                num_speakers=cfg.num_speakers,
                micro_gap_s=cfg.micro_turn_merge_gap_s,
                min_overlap_duration_s=cfg.overlap_min_duration_s,
                overlap_merge_gap_s=cfg.overlap_merge_gap_s,
            ),
        )
        speaker_turns = diarization_payload["speaker_turns"]
        turns = diarization_payload["exclusive_turns"]
        overlap_windows = diarization_payload["overlap_regions"]

        full_context = _run_stage(
            "asr_full_context_decode",
            lambda: asr.decode_full_audio(
                wav_path=wav_path,
                generation_config=asr.build_generation_config(),
                temperature=cfg.decode_temperature,
                word_timestamps=True,
            ),
        )
        decoded_records = _build_full_context_records(
            words=full_context.get("words", []),
            full_text=full_context.get("text", ""),
            duration_s=duration_s,
            exclusive_turns=turns,
            speaker_turns=speaker_turns,
            overlap_windows=overlap_windows,
            max_gap_s=cfg.chunk_merge_gap_s,
            speaker_assignment_max_gap_s=cfg.speaker_assignment_max_gap_s,
        )
        rerun_events = [
            {
                "chunk_id": "full_context",
                "reasons": [],
                "rerun_attempts": [],
                "selected_meta": {"pass": "full_context"},
                "selected_text_preview": (full_context.get("text") or "")[:200],
            }
        ]

        overlap_second_pass_chunks: list[ChunkItem] = []
        if cfg.overlap_second_pass and overlap_windows:
            overlap_second_pass_chunks = build_fallback_chunks_from_vad(
                vad_regions=overlap_windows,
                overlap_windows=overlap_windows,
                audio_duration_s=duration_s,
                max_segment_s=cfg.chunk_max_segment_s,
                stride_s=cfg.chunk_stride_s,
                pad_s=cfg.chunk_pad_s,
                speaker_id="OVERLAP",
                source="overlap_second_pass",
                )
            for overlap_chunk in overlap_second_pass_chunks:
                record, event = _run_stage(
                    "overlap_asr_decode",
                    lambda current_chunk=overlap_chunk: _decode_chunk_with_quality(
                        asr=asr,
                        wav_path=wav_path,
                        chunk=current_chunk,
                        work_dir=tmp_dir,
                        overlap_second_pass=cfg.overlap_second_pass,
                    ),
                )
                decoded_records.append(record)
                rerun_events.append(event)

        segments = build_segments(decoded_records)
        result: TranscriptResult = {
            "source_file": str(source_file),
            "duration_s": duration_s,
            "segments": segments,
            "metadata": {
                "models": {
                    "whisper": cfg.whisper_model_id,
                    "diarization": cfg.diarization_model_id,
                    "vad": cfg.vad_model_id,
                    "overlap": "from_speaker_diarization_get_overlap",
                },
                "counts": {
                    "vad_region_count": len(vad_regions),
                    "speaker_turn_count": len(speaker_turns),
                    "exclusive_turn_count": len(turns),
                    "chunk_count": len(overlap_second_pass_chunks),
                    "base_segment_count": len(decoded_records)
                    - len(overlap_second_pass_chunks),
                    "overlap_window_count": len(overlap_windows),
                    "overlap_second_pass_chunk_count": len(overlap_second_pass_chunks),
                    "segment_count": len(segments),
                },
                "device": asr.device,
                "config": asdict(cfg),
                "rerun_events": rerun_events,
            },
            "outputs": {
                "text_path": str(text_path),
                "json_path": str(json_path),
                "debug_dir": str(debug_dir),
            },
        }

        write_text_transcript(segments, text_path)
        write_segments_json(result, json_path)

        moved_to = _move_processed_file(source_file, cfg.move_processed_to)
        if moved_to:
            result["metadata"]["moved_to"] = moved_to

        if cfg.emit_debug:
            write_debug_artifacts(
                debug_dir=debug_dir,
                stem=source_file.stem,
                payloads={
                    "normalized_audio": normalized,
                    "vad_regions": vad_regions,
                    "speaker_turns": speaker_turns,
                    "exclusive_turns": turns,
                    "overlap_windows": overlap_windows,
                    "full_context_asr": full_context,
                    "overlap_second_pass_chunks": overlap_second_pass_chunks,
                    "rerun_events": rerun_events,
                },
            )

        return result


def run_batch(input_path: str, config: MeetingConfig | None = None) -> list[TranscriptResult]:
    cfg = config or from_env()
    source = Path(input_path).resolve()
    if source.is_file():
        return [run_meeting_transcription(str(source), config=cfg)]
    if not source.is_dir():
        raise RuntimeError(f"Input path does not exist: {source}")

    allowed = {extension.lower() for extension in cfg.audio_extensions}
    files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in allowed)
    return [run_meeting_transcription(str(path), config=cfg) for path in files]
