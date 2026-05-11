from __future__ import annotations

from .types import ChunkItem, Region, Turn


def _merge_same_speaker_turns(turns: list[Turn], merge_gap_s: float) -> list[Turn]:
    if not turns:
        return []
    ordered = sorted(turns, key=lambda item: (item["start_s"], item["end_s"]))
    merged = [ordered[0].copy()]
    for turn in ordered[1:]:
        prev = merged[-1]
        gap = turn["start_s"] - prev["end_s"]
        if turn["speaker_id"] == prev["speaker_id"] and gap <= merge_gap_s:
            prev["end_s"] = max(prev["end_s"], turn["end_s"])
        else:
            merged.append(turn.copy())
    return merged


def _intersections(start_s: float, end_s: float, regions: list[Region]) -> list[Region]:
    output: list[Region] = []
    for region in regions:
        left = max(start_s, region["start_s"])
        right = min(end_s, region["end_s"])
        if right > left:
            output.append({"start_s": left, "end_s": right})
    return output


def _apply_padding(start_s: float, end_s: float, pad_s: float, duration_s: float) -> tuple[float, float]:
    return max(0.0, start_s - pad_s), min(duration_s, end_s + pad_s)


def _split_region(start_s: float, end_s: float, max_segment_s: float, stride_s: float) -> list[Region]:
    if end_s - start_s <= max_segment_s:
        return [{"start_s": start_s, "end_s": end_s}]

    chunks: list[Region] = []
    current_start = start_s
    while current_start < end_s:
        current_end = min(current_start + max_segment_s, end_s)
        chunks.append({"start_s": current_start, "end_s": current_end})
        if current_end >= end_s:
            break

        next_start = max(start_s, current_end - stride_s)
        if next_start <= current_start:
            next_start = current_start + max_segment_s
        current_start = next_start

    return chunks


def _intersects_overlap(start_s: float, end_s: float, overlap_windows: list[Region]) -> bool:
    for window in overlap_windows:
        if min(end_s, window["end_s"]) > max(start_s, window["start_s"]):
            return True
    return False


def build_fallback_chunks_from_vad(
    vad_regions: list[Region],
    overlap_windows: list[Region],
    audio_duration_s: float,
    max_segment_s: float,
    stride_s: float,
    pad_s: float,
    speaker_id: str = "UNKNOWN",
    source: str = "vad_fallback",
) -> list[ChunkItem]:
    chunks: list[ChunkItem] = []
    chunk_counter = 0
    for region in vad_regions:
        padded_start, padded_end = _apply_padding(
            region["start_s"], region["end_s"], pad_s=pad_s, duration_s=audio_duration_s
        )
        for split in _split_region(padded_start, padded_end, max_segment_s=max_segment_s, stride_s=stride_s):
            chunk_counter += 1
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:06d}",
                    "speaker_id": speaker_id,
                    "start_s": split["start_s"],
                    "end_s": split["end_s"],
                    "overlap_flag": _intersects_overlap(
                        split["start_s"], split["end_s"], overlap_windows=overlap_windows
                    ),
                    "source": source,
                }
            )
    return chunks


def build_chunks(
    turns: list[Turn],
    vad_regions: list[Region],
    overlap_windows: list[Region],
    audio_duration_s: float,
    merge_gap_s: float,
    max_segment_s: float,
    stride_s: float,
    pad_s: float,
) -> list[ChunkItem]:
    if not turns:
        return build_fallback_chunks_from_vad(
            vad_regions=vad_regions,
            overlap_windows=overlap_windows,
            audio_duration_s=audio_duration_s,
            max_segment_s=max_segment_s,
            stride_s=stride_s,
            pad_s=pad_s,
            speaker_id="UNKNOWN",
            source="vad_fallback",
        )

    merged_turns = _merge_same_speaker_turns(turns, merge_gap_s=merge_gap_s)
    chunks: list[ChunkItem] = []
    chunk_counter = 0

    for turn in merged_turns:
        intersections = _intersections(turn["start_s"], turn["end_s"], vad_regions)
        for intersection in intersections:
            padded_start, padded_end = _apply_padding(
                intersection["start_s"],
                intersection["end_s"],
                pad_s=pad_s,
                duration_s=audio_duration_s,
            )
            for split in _split_region(
                padded_start, padded_end, max_segment_s=max_segment_s, stride_s=stride_s
            ):
                chunk_counter += 1
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_counter:06d}",
                        "speaker_id": turn["speaker_id"],
                        "start_s": split["start_s"],
                        "end_s": split["end_s"],
                        "overlap_flag": _intersects_overlap(
                            split["start_s"], split["end_s"], overlap_windows=overlap_windows
                        ),
                        "source": "diarized_turn",
                    }
                )

    if chunks:
        return chunks

    return build_fallback_chunks_from_vad(
        vad_regions=vad_regions,
        overlap_windows=overlap_windows,
        audio_duration_s=audio_duration_s,
        max_segment_s=max_segment_s,
        stride_s=stride_s,
        pad_s=pad_s,
        speaker_id="UNKNOWN",
        source="vad_fallback",
    )

