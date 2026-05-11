from __future__ import annotations

from typing import Iterable

from .types import Region


def _iter_segments(timeline) -> Iterable[tuple[float, float]]:
    iterator = getattr(timeline, "itersegments", None)
    if callable(iterator):
        for segment in iterator():
            yield float(segment.start), float(segment.end)
        return

    try:
        for segment in timeline:
            yield float(segment.start), float(segment.end)
        return
    except TypeError:
        pass

    raise RuntimeError("Unsupported overlap timeline format: cannot iterate segments.")


def _merge_regions(regions: list[Region], max_gap_s: float) -> list[Region]:
    if not regions:
        return []
    merged = [regions[0].copy()]
    for region in regions[1:]:
        prev = merged[-1]
        gap = region["start_s"] - prev["end_s"]
        if gap <= max_gap_s:
            prev["end_s"] = max(prev["end_s"], region["end_s"])
        else:
            merged.append(region.copy())
    return merged


def compute_overlap_regions(
    speaker_diarization_annotation,
    min_overlap_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> list[Region]:
    get_overlap = getattr(speaker_diarization_annotation, "get_overlap", None)
    if not callable(get_overlap):
        raise RuntimeError(
            "Diarization annotation does not expose get_overlap(). "
            "Use a pyannote 4.x speaker diarization pipeline output."
        )

    overlap_timeline = get_overlap()
    regions: list[Region] = []
    for start_s, end_s in _iter_segments(overlap_timeline):
        if end_s <= start_s:
            continue
        if (end_s - start_s) < float(min_overlap_duration_s):
            continue
        regions.append({"start_s": start_s, "end_s": end_s})

    regions.sort(key=lambda item: (item["start_s"], item["end_s"]))
    return _merge_regions(regions, max_gap_s=float(merge_gap_s))
