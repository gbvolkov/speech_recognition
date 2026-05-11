from __future__ import annotations

from .overlap import compute_overlap_regions
from .types import DiarizationArtifacts, Turn


def _split_hf_revision(model_id: str) -> tuple[str, str | None]:
    if "@" not in model_id:
        return model_id, None
    checkpoint, revision = model_id.rsplit("@", 1)
    checkpoint = checkpoint.strip()
    revision = revision.strip()
    if not checkpoint or not revision:
        return model_id, None
    return checkpoint, revision


def _extract_turns(annotation) -> list[Turn]:
    iterator = getattr(annotation, "itertracks", None)
    if not callable(iterator):
        raise RuntimeError(
            "Unsupported diarization output format: expected itertracks(yield_label=True)."
        )

    turns: list[Turn] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        start_s = float(segment.start)
        end_s = float(segment.end)
        if end_s <= start_s:
            continue
        turns.append({"speaker_id": str(label), "start_s": start_s, "end_s": end_s})
    turns.sort(key=lambda item: (item["start_s"], item["end_s"]))
    return turns


def _merge_micro_turns(turns: list[Turn], micro_gap_s: float) -> list[Turn]:
    if not turns:
        return []
    merged = [turns[0].copy()]
    for turn in turns[1:]:
        prev = merged[-1]
        gap = turn["start_s"] - prev["end_s"]
        if turn["speaker_id"] == prev["speaker_id"] and gap <= micro_gap_s:
            prev["end_s"] = max(prev["end_s"], turn["end_s"])
        else:
            merged.append(turn.copy())
    return merged


def run_diarization(
    wav_path: str,
    model_id: str,
    token: str,
    min_speakers: int,
    max_speakers: int,
    num_speakers: int | None,
    micro_gap_s: float,
    min_overlap_duration_s: float,
    overlap_merge_gap_s: float,
) -> DiarizationArtifacts:
    from pyannote.audio import Pipeline

    checkpoint, revision = _split_hf_revision(model_id)
    load_kwargs = {"token": token}
    if revision:
        load_kwargs["revision"] = revision
    pipeline = Pipeline.from_pretrained(checkpoint, **load_kwargs)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        kwargs["min_speakers"] = int(min_speakers)
        kwargs["max_speakers"] = int(max_speakers)

    diarization_result = pipeline(wav_path, **kwargs) if kwargs else pipeline(wav_path)
    speaker_diarization = getattr(diarization_result, "speaker_diarization", diarization_result)
    exclusive_diarization = getattr(diarization_result, "exclusive_speaker_diarization", None)
    if exclusive_diarization is None:
        raise RuntimeError(
            "Diarization output does not expose exclusive_speaker_diarization. "
            "Use a pyannote 4.x speaker diarization pipeline that provides it "
            "(e.g., pyannote/speaker-diarization-community-1)."
        )

    speaker_turns = _merge_micro_turns(
        _extract_turns(speaker_diarization), micro_gap_s=micro_gap_s
    )
    exclusive_turns = _merge_micro_turns(
        _extract_turns(exclusive_diarization), micro_gap_s=micro_gap_s
    )
    overlap_regions = compute_overlap_regions(
        speaker_diarization_annotation=speaker_diarization,
        min_overlap_duration_s=min_overlap_duration_s,
        merge_gap_s=overlap_merge_gap_s,
    )
    return {
        "speaker_turns": speaker_turns,
        "exclusive_turns": exclusive_turns,
        "overlap_regions": overlap_regions,
    }
