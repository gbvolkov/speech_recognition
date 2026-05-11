from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evaluation.dialog_ref import build_reference
from evaluation.run_benchmark import compute_metrics, parse_reference, parse_transcript_text


def score_metrics(metrics):
    speaker_acc = metrics.get("speaker_sequence_accuracy") or 0.0
    start_error = metrics.get("start_alignment_median_abs_error_sec")
    start_score = 0.0 if start_error is None else 1.0 / (1.0 + start_error)
    wer = metrics.get("wer_optional")
    text_score = 0.0 if wer is None else 1.0 - min(max(wer, 0.0), 1.0)
    return (0.5 * speaker_acc) + (0.3 * text_score) + (0.2 * start_score)


def run_single_config(audio_path, reference_rows, base_config, chunk_length_s, stride_length_s):
    from transcribe import TranscriptionConfig, run_transcription

    config = TranscriptionConfig(
        profile="default",
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        deterministic_decoding=base_config["deterministic_decoding"],
        num_speakers=base_config["num_speakers"],
    )

    tmp_root = Path(".tmp_bench")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_root / f"dialog-tune-{uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        tmp_audio = tmp_path / audio_path.name
        shutil.copy2(audio_path, tmp_audio)
        run_transcription(str(tmp_audio), config=config, move_processed=False)
        transcript_path = tmp_path / "transcripts" / f"{tmp_audio.stem}.txt"
        transcript_text = transcript_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    predicted_rows = parse_transcript_text(transcript_text)
    metrics = compute_metrics(reference_rows, predicted_rows)
    return {"chunk_length_s": chunk_length_s, "stride_length_s": stride_length_s, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="Grid-search decode chunk/stride for dialog benchmark.")
    parser.add_argument("--audio", default="audio/dialog_ru_timed.wav")
    parser.add_argument("--ref", default="evaluation/gold/dialog_ru_timed.ref.jsonl")
    parser.add_argument("--source-script", default="data/test_dialog.py")
    parser.add_argument("--deterministic-decoding", action="store_true")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--chunk-grid", default="20,30")
    parser.add_argument("--stride-grid", default="5,8")
    parser.add_argument(
        "--report-out",
        default="evaluation/reports/dialog_ru_timed.tuning.json",
        help="Output JSON report with ranked candidates.",
    )
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        build_reference(args.source_script, ref_path)
    reference_rows = parse_reference(ref_path)

    chunk_values = [int(value.strip()) for value in args.chunk_grid.split(",") if value.strip()]
    stride_values = [int(value.strip()) for value in args.stride_grid.split(",") if value.strip()]
    audio_path = Path(args.audio)

    base_config = {
        "deterministic_decoding": args.deterministic_decoding,
        "num_speakers": args.num_speakers,
    }

    candidates = []
    for chunk in chunk_values:
        for stride in stride_values:
            result = run_single_config(
                audio_path=audio_path,
                reference_rows=reference_rows,
                base_config=base_config,
                chunk_length_s=chunk,
                stride_length_s=stride,
            )
            result["score"] = score_metrics(result["metrics"])
            candidates.append(result)
            print(
                f"chunk={chunk} stride={stride} score={result['score']:.4f} "
                f"speaker_seq={result['metrics'].get('speaker_sequence_accuracy')}"
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else None

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio": str(audio_path),
        "reference": str(ref_path),
        "base_config": {
            "pipeline": "v2",
            "deterministic_decoding": args.deterministic_decoding,
            "num_speakers": args.num_speakers,
        },
        "candidates": candidates,
        "best": best,
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tuning report written to: {report_path}")
    if best:
        print(
            "Best config:",
            f"chunk={best['chunk_length_s']}, stride={best['stride_length_s']}, score={best['score']:.4f}",
        )


if __name__ == "__main__":
    main()
