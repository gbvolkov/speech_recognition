from __future__ import annotations

from typing import NotRequired, TypedDict


class WordItem(TypedDict):
    start_s: float
    end_s: float
    word: str
    conf: NotRequired[float]


class SegmentItem(TypedDict):
    start_s: float
    end_s: float
    speaker_id: str
    overlap_flag: bool
    text: str
    words: list[WordItem]
    source: str


class TranscriptResult(TypedDict):
    source_file: str
    duration_s: float
    segments: list[SegmentItem]
    metadata: dict
    outputs: dict


class Region(TypedDict):
    start_s: float
    end_s: float


class Turn(TypedDict):
    speaker_id: str
    start_s: float
    end_s: float


class ChunkItem(TypedDict):
    chunk_id: str
    speaker_id: str
    start_s: float
    end_s: float
    overlap_flag: bool
    source: str


class DiarizationArtifacts(TypedDict):
    speaker_turns: list[Turn]
    exclusive_turns: list[Turn]
    overlap_regions: list[Region]
