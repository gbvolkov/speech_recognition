from __future__ import annotations

import re


def temperature_ladder() -> tuple[float, ...]:
    return (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\- ]+", " ", (text or "").lower(), flags=re.UNICODE)
    return [token for token in normalized.split() if token]


def has_repetition(text: str, min_tokens: int = 8, max_unique_ratio: float = 0.5) -> bool:
    tokens = _tokenize(text)
    if len(tokens) < min_tokens:
        return False

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < max_unique_ratio:
        return True

    lowered = " ".join(tokens)
    for size in (3, 4, 5):
        pieces = [tokens[i : i + size] for i in range(max(0, len(tokens) - size + 1))]
        if not pieces:
            continue
        joined = [" ".join(piece) for piece in pieces]
        for phrase in set(joined):
            if len(phrase) < 8:
                continue
            if lowered.count(phrase) >= 3:
                return True
    return False


def rerun_reasons(
    text: str,
    duration_s: float,
    has_vad_speech: bool,
    overlap_flag: bool,
    overlap_second_pass: bool,
) -> list[str]:
    reasons: list[str] = []
    if has_repetition(text):
        reasons.append("repetition")
    if duration_s > 2.0 and has_vad_speech and not (text or "").strip():
        reasons.append("empty_with_speech")
    if overlap_flag and overlap_second_pass:
        reasons.append("overlap")
    return reasons


def _candidate_score(text: str) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return -100.0
    score = min(len(stripped), 500) / 100.0
    if has_repetition(stripped):
        score -= 10.0
    return score


def pick_best_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        return {"text": "", "words": [], "meta": {"reason": "empty_candidates"}}

    best = max(candidates, key=lambda item: _candidate_score(item.get("text", "")))
    return best
