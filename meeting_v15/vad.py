from __future__ import annotations

from typing import Iterable

import yaml
from huggingface_hub import hf_hub_download

from .types import Region


def _split_hf_revision(model_id: str) -> tuple[str, str | None]:
    # Support "repo@revision" shorthand by converting it to an explicit revision arg.
    if "@" not in model_id:
        return model_id, None
    checkpoint, revision = model_id.rsplit("@", 1)
    checkpoint = checkpoint.strip()
    revision = revision.strip()
    if not checkpoint or not revision:
        return model_id, None
    return checkpoint, revision


def _normalize_model_ref(model_ref):
    if not isinstance(model_ref, str):
        return model_ref
    if "@" not in model_ref:
        return model_ref
    checkpoint, revision = model_ref.rsplit("@", 1)
    checkpoint = checkpoint.strip()
    revision = revision.strip()
    if not checkpoint or not revision:
        return model_ref
    return {"checkpoint": checkpoint, "revision": revision}


def _load_vad_pipeline_definition(
    model_id: str,
    revision: str | None,
    token: str,
) -> tuple[object, dict]:
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.yaml",
        revision=revision,
        token=token,
    )
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    pipeline_params = (payload.get("pipeline") or {}).get("params") or {}
    segmentation_ref = pipeline_params.get("segmentation")
    if segmentation_ref is None:
        raise RuntimeError(
            f"VAD config at {model_id} does not define pipeline.params.segmentation."
        )

    instantiate_params = payload.get("params") or {}
    if not isinstance(instantiate_params, dict):
        raise RuntimeError(f"VAD config params must be a dictionary for model {model_id}.")

    return _normalize_model_ref(segmentation_ref), dict(instantiate_params)


def _iter_segments(annotation) -> Iterable[tuple[float, float]]:
    if hasattr(annotation, "get_timeline"):
        timeline = annotation.get_timeline()
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

    iterator = getattr(annotation, "itersegments", None)
    if callable(iterator):
        for segment in iterator():
            yield float(segment.start), float(segment.end)
        return

    # Some pyannote objects expose segment+track tuples via itertracks.
    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        for segment, *_ in itertracks(yield_label=True):
            yield float(segment.start), float(segment.end)
        return

    # Last-resort iterable support for timeline-like containers.
    try:
        for segment in annotation:
            if hasattr(segment, "start") and hasattr(segment, "end"):
                yield float(segment.start), float(segment.end)
            elif isinstance(segment, tuple) and segment and hasattr(segment[0], "start"):
                yield float(segment[0].start), float(segment[0].end)
        return
    except TypeError:
        pass

    raise RuntimeError("Unsupported VAD output format: cannot iterate segments.")


def _merge_regions(regions: list[Region], min_silence_s: float) -> list[Region]:
    if not regions:
        return []
    merged = [regions[0].copy()]
    for region in regions[1:]:
        prev = merged[-1]
        gap = region["start_s"] - prev["end_s"]
        if gap <= min_silence_s:
            prev["end_s"] = max(prev["end_s"], region["end_s"])
        else:
            merged.append(region.copy())
    return merged


def _pad_and_clamp_regions(regions: list[Region], pad_s: float, duration_s: float) -> list[Region]:
    padded = []
    for region in regions:
        start_s = max(0.0, region["start_s"] - pad_s)
        end_s = min(duration_s, region["end_s"] + pad_s)
        if end_s <= start_s:
            continue
        padded.append({"start_s": start_s, "end_s": end_s})
    return _merge_regions(padded, min_silence_s=0.0)


def run_vad(
    wav_path: str,
    model_id: str,
    token: str,
    min_speech_s: float,
    min_silence_s: float,
    pad_s: float,
    duration_s: float,
) -> list[Region]:
    from pyannote.audio.pipelines import VoiceActivityDetection

    checkpoint, revision = _split_hf_revision(model_id)
    segmentation_ref, instantiate_params = _load_vad_pipeline_definition(
        model_id=checkpoint,
        revision=revision,
        token=token,
    )

    pipeline = VoiceActivityDetection(segmentation=segmentation_ref, token=token)
    instantiate_params["min_duration_on"] = float(min_speech_s)
    instantiate_params["min_duration_off"] = float(min_silence_s)
    pipeline.instantiate(instantiate_params)
    result = pipeline(wav_path)

    regions: list[Region] = []
    for start_s, end_s in _iter_segments(result):
        if end_s - start_s < min_speech_s:
            continue
        regions.append({"start_s": start_s, "end_s": end_s})

    regions.sort(key=lambda item: (item["start_s"], item["end_s"]))
    regions = _merge_regions(regions, min_silence_s=min_silence_s)
    regions = _pad_and_clamp_regions(regions, pad_s=pad_s, duration_s=duration_s)
    return regions
