from __future__ import annotations

import json
from pathlib import Path

from .types import SegmentItem, TranscriptResult


def _format_timestamp(total_seconds: float) -> str:
    seconds = int(max(0, total_seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_text_transcript(segments: list[SegmentItem], output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for segment in segments:
        header = (
            f"**{segment['speaker_id']}** "
            f"[{_format_timestamp(segment['start_s'])} - {_format_timestamp(segment['end_s'])}]:"
        )
        text = segment["text"]
        if segment["overlap_flag"] and text:
            text = f"[OVERLAP] {text}"
        lines.extend([header, text, ""])

    text_payload = "\n".join(lines).rstrip() + "\n"
    target.write_text(text_payload, encoding="utf-8")
    return text_payload


def _sanitize_result(result: TranscriptResult) -> dict:
    sanitized_segments = []
    for segment in result.get("segments", []):
        clean_segment = {
            "start_s": float(segment["start_s"]),
            "end_s": float(segment["end_s"]),
            "speaker_id": str(segment["speaker_id"]),
            "overlap_flag": bool(segment["overlap_flag"]),
            "text": str(segment.get("text", "")),
            "source": str(segment.get("source", "")),
            "words": [],
        }
        for word in segment.get("words", []):
            clean_word = {
                "start_s": float(word["start_s"]),
                "end_s": float(word["end_s"]),
                "word": str(word["word"]),
            }
            if "conf" in word:
                clean_word["conf"] = float(word["conf"])
            clean_segment["words"].append(clean_word)
        sanitized_segments.append(clean_segment)

    return {
        "source_file": str(result.get("source_file", "")),
        "duration_s": float(result.get("duration_s", 0.0)),
        "segments": sanitized_segments,
        "metadata": result.get("metadata", {}),
        "outputs": result.get("outputs", {}),
    }


def write_segments_json(result: TranscriptResult, output_path: str | Path) -> dict:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_result(result)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_debug_artifacts(debug_dir: str | Path, stem: str, payloads: dict[str, object]) -> None:
    directory = Path(debug_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        output_file = directory / f"{stem}.{name}.json"
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

