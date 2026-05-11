from __future__ import annotations

import re

from .types import SegmentItem, WordItem


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized


def _normalize_word_item(word: WordItem) -> WordItem:
    normalized: WordItem = {
        "start_s": round(float(word["start_s"]), 3),
        "end_s": round(float(word["end_s"]), 3),
        "word": normalize_text(word.get("word", "")),
    }
    if "conf" in word:
        normalized["conf"] = round(float(word["conf"]), 4)
    return normalized


def build_segments(decoded_records: list[dict]) -> list[SegmentItem]:
    segments: list[SegmentItem] = []
    for record in decoded_records:
        words = [_normalize_word_item(word) for word in record.get("words", [])]
        text = normalize_text(record.get("text", ""))
        segment: SegmentItem = {
            "start_s": round(float(record["start_s"]), 3),
            "end_s": round(float(record["end_s"]), 3),
            "speaker_id": str(record.get("speaker_id", "UNKNOWN")),
            "overlap_flag": bool(record.get("overlap_flag", False)),
            "text": text,
            "words": words,
            "source": str(record.get("source", "unknown")),
        }
        segments.append(segment)

    segments.sort(key=lambda item: (item["start_s"], item["end_s"], item["source"]))
    return segments

