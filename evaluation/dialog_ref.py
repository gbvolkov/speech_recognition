from __future__ import annotations

import json
import re
from pathlib import Path

SEGMENT_LINE_RE = re.compile(
    r'^\s*\((?P<start>\d+(?:\.\d+)?),\s*(?P<speaker>[01]),\s*"(?P<text>.*)"\),?\s*$'
)


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_test_dialog_script(script_path: str | Path):
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Source script not found: {path}")

    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = SEGMENT_LINE_RE.match(line.rstrip("\n"))
            if not match:
                continue
            segments.append(
                {
                    "start": float(match.group("start")),
                    "speaker": f"SPEAKER_0{match.group('speaker')}",
                    "text": match.group("text").strip(),
                }
            )

    if not segments:
        raise ValueError(f"No DEFAULT_SEGMENTS lines parsed from: {path}")

    for i, segment in enumerate(segments):
        if i + 1 < len(segments):
            segment["end"] = segments[i + 1]["start"]
        else:
            segment["end"] = segment["start"] + 4.0
        segment["id"] = i
        segment["text_normalized"] = normalize_text(segment["text"])

    return segments


def write_reference_jsonl(segments, output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(json.dumps(segment, ensure_ascii=False) + "\n")


def build_reference(source_script: str | Path, output_path: str | Path):
    segments = parse_test_dialog_script(source_script)
    write_reference_jsonl(segments, output_path)
    return segments
