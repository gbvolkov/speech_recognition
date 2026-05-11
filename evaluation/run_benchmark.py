from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evaluation.dialog_ref import build_reference, normalize_text

HEADER_RE = re.compile(
    r"^\*\*(?P<speaker>[^*]+)\*\* \[(?P<start>\d{2}:\d{2}:\d{2}) - (?P<end>\d{2}:\d{2}:\d{2})\]:\s*$"
)

DEFAULT_ANCHORS = [
    "привет",
    "дожд",
    "капает",
    "зонтик",
    "радуг",
    "первые капли",
    "двадцать процентов",
    "северо-западный",
    "обувь",
    "фото",
]


def parse_hms(value: str) -> int:
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_reference(ref_path: str | Path):
    rows = []
    with open(ref_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["text_normalized"] = row.get("text_normalized") or normalize_text(row.get("text", ""))
            rows.append(row)
    return rows


def parse_transcript_text(text: str):
    segments = []
    current = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        header = HEADER_RE.match(line.strip())
        if header:
            if current is not None:
                current["text"] = " ".join(current.pop("text_lines")).strip()
                current["text_normalized"] = normalize_text(current["text"])
                segments.append(current)
            current = {
                "speaker": header.group("speaker").strip(),
                "start": parse_hms(header.group("start")),
                "end": parse_hms(header.group("end")),
                "text_lines": [],
            }
            continue

        if current is not None and line.strip():
            current["text_lines"].append(line.strip())

    if current is not None:
        current["text"] = " ".join(current.pop("text_lines")).strip()
        current["text_normalized"] = normalize_text(current["text"])
        segments.append(current)

    return segments


def read_transcript(transcript_path: str | Path):
    with open(transcript_path, encoding="utf-8") as f:
        return parse_transcript_text(f.read())


def levenshtein_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, a_item in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, b_item in enumerate(b, start=1):
            old = dp[j]
            if a_item == b_item:
                dp[j] = prev
            else:
                dp[j] = min(prev, dp[j - 1], dp[j]) + 1
            prev = old
    return dp[-1]


def compute_wer(ref_text: str, pred_text: str):
    ref_tokens = [token for token in normalize_text(ref_text).split(" ") if token]
    pred_tokens = [token for token in normalize_text(pred_text).split(" ") if token]
    if not ref_tokens:
        return None
    return levenshtein_distance(ref_tokens, pred_tokens) / len(ref_tokens)


def lcs_length(a, b):
    if not a or not b:
        return 0
    rows, cols = len(a) + 1, len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def monotonic_start_errors(ref_starts, pred_starts):
    if not ref_starts or not pred_starts:
        return []
    errors = []
    j = 0
    for ref_start in ref_starts:
        if j >= len(pred_starts):
            break
        best_j = min(range(j, len(pred_starts)), key=lambda idx: abs(pred_starts[idx] - ref_start))
        errors.append(abs(pred_starts[best_j] - ref_start))
        j = best_j + 1
    return errors


def compute_metrics(reference_rows, predicted_rows):
    ref_starts = [float(row["start"]) for row in reference_rows]
    pred_starts = [float(row["start"]) for row in predicted_rows]
    start_errors = monotonic_start_errors(ref_starts, pred_starts)

    ref_speakers = [row["speaker"] for row in reference_rows]
    pred_speakers = [row["speaker"] for row in predicted_rows]
    speaker_sequence_accuracy = (
        lcs_length(ref_speakers, pred_speakers) / len(ref_speakers) if ref_speakers else None
    )

    pred_text_all = " ".join(row.get("text", "") for row in predicted_rows)
    pred_text_norm = normalize_text(pred_text_all)
    anchor_hits = sum(1 for anchor in DEFAULT_ANCHORS if normalize_text(anchor) in pred_text_norm)

    false_overlap_before_45 = False
    for i in range(1, len(predicted_rows)):
        if predicted_rows[i]["start"] < predicted_rows[i - 1]["end"] and predicted_rows[i]["start"] < 45:
            false_overlap_before_45 = True
            break

    max_segment_duration = 0.0
    for row in predicted_rows:
        max_segment_duration = max(max_segment_duration, float(row["end"] - row["start"]))

    metrics = {
        "reference_segment_count": len(reference_rows),
        "predicted_segment_count": len(predicted_rows),
        "utterance_count_delta": len(predicted_rows) - len(reference_rows),
        "max_segment_duration_sec": max_segment_duration,
        "start_alignment_median_abs_error_sec": statistics.median(start_errors) if start_errors else None,
        "start_alignment_mean_abs_error_sec": (sum(start_errors) / len(start_errors)) if start_errors else None,
        "speaker_sequence_accuracy": speaker_sequence_accuracy,
        "anchor_hits": anchor_hits,
        "anchor_total": len(DEFAULT_ANCHORS),
        "anchor_recall": anchor_hits / len(DEFAULT_ANCHORS) if DEFAULT_ANCHORS else None,
        "false_overlap_before_45_sec": false_overlap_before_45,
        "segments_starting_after_43_sec": sum(1 for row in predicted_rows if float(row["start"]) >= 43.0),
        "wer_optional": compute_wer(
            " ".join(row.get("text", "") for row in reference_rows),
            pred_text_all,
        ),
    }
    return metrics


def evaluate_gate(metrics, phase, baseline_metrics=None):
    checks = []
    gate_pass = True

    def add_check(name, ok, detail):
        nonlocal gate_pass
        gate_pass = gate_pass and ok
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if phase == "phase1":
        add_check(
            "segment_count>=16",
            metrics["predicted_segment_count"] >= 16,
            f"predicted_segment_count={metrics['predicted_segment_count']}",
        )
        add_check(
            "max_segment_duration<=12s",
            metrics["max_segment_duration_sec"] <= 12.0,
            f"max_segment_duration_sec={metrics['max_segment_duration_sec']}",
        )
        add_check(
            "no_false_overlap_before_45s",
            not metrics["false_overlap_before_45_sec"],
            f"false_overlap_before_45_sec={metrics['false_overlap_before_45_sec']}",
        )
        if baseline_metrics and baseline_metrics.get("start_alignment_median_abs_error_sec") is not None:
            baseline = baseline_metrics["start_alignment_median_abs_error_sec"]
            current = metrics.get("start_alignment_median_abs_error_sec")
            add_check(
                "start_error_improves_30pct",
                current is not None and current <= baseline * 0.7,
                f"current={current}, baseline={baseline}",
            )
    elif phase == "phase2":
        if baseline_metrics and baseline_metrics.get("speaker_sequence_accuracy") is not None:
            baseline = baseline_metrics["speaker_sequence_accuracy"]
            current = metrics.get("speaker_sequence_accuracy")
            add_check(
                "speaker_sequence_improves_20pct",
                current is not None and current >= baseline * 1.2,
                f"current={current}, baseline={baseline}",
            )
        add_check(
            "long_tail_split_after_43s",
            metrics["segments_starting_after_43_sec"] >= 3,
            f"segments_starting_after_43_sec={metrics['segments_starting_after_43_sec']}",
        )
        add_check(
            "max_segment_duration<=12s",
            metrics["max_segment_duration_sec"] <= 12.0,
            f"max_segment_duration_sec={metrics['max_segment_duration_sec']}",
        )
    elif phase == "phase3":
        add_check(
            "anchor_hits>=8",
            metrics["anchor_hits"] >= 8,
            f"anchor_hits={metrics['anchor_hits']}",
        )
        if baseline_metrics and baseline_metrics.get("speaker_sequence_accuracy") is not None:
            baseline = baseline_metrics["speaker_sequence_accuracy"]
            current = metrics.get("speaker_sequence_accuracy")
            add_check(
                "no_speaker_regression",
                current is not None and current >= baseline,
                f"current={current}, baseline={baseline}",
            )

    return gate_pass, checks


def maybe_build_reference(ref_path: Path, source_script: Path):
    if ref_path.exists():
        return
    build_reference(source_script, ref_path)


def run_transcription_for_audio(audio_path: Path, profile: str):
    from transcribe import TranscriptionConfig, run_transcription

    config = TranscriptionConfig(profile=profile)
    tmp_root = Path(".tmp_bench")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_root / f"dialog-bench-{uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        tmp_audio = tmp_path / audio_path.name
        shutil.copy2(audio_path, tmp_audio)
        run_transcription(str(tmp_audio), config=config, move_processed=False)
        generated_transcript = tmp_path / "transcripts" / f"{tmp_audio.stem}.txt"
        with open(generated_transcript, encoding="utf-8") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark against dialog_ru_timed transcript output.")
    parser.add_argument("--ref", default="evaluation/gold/dialog_ru_timed.ref.jsonl")
    parser.add_argument("--source-script", default="data/test_dialog.py")
    parser.add_argument("--transcript", default="audio/transcripts/dialog_ru_timed.txt")
    parser.add_argument("--audio", default="audio/dialog_ru_timed.wav")
    parser.add_argument("--run-transcription", action="store_true")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--gate", choices=["phase1", "phase2", "phase3"], default=None)
    parser.add_argument("--enforce-gate", action="store_true")
    parser.add_argument(
        "--report-out",
        default="evaluation/reports/dialog_ru_timed.latest.json",
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--generated-transcript-out",
        default=None,
        help="Optional path to persist generated transcript when --run-transcription is used.",
    )
    args = parser.parse_args()

    ref_path = Path(args.ref)
    source_script_path = Path(args.source_script)
    maybe_build_reference(ref_path, source_script_path)
    reference_rows = parse_reference(ref_path)

    if args.run_transcription:
        transcript_text = run_transcription_for_audio(
            Path(args.audio),
            profile=args.profile,
        )
        if args.generated_transcript_out:
            output_path = Path(args.generated_transcript_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(transcript_text, encoding="utf-8")
    else:
        transcript_text = Path(args.transcript).read_text(encoding="utf-8")

    predicted_rows = parse_transcript_text(transcript_text)
    metrics = compute_metrics(reference_rows, predicted_rows)

    baseline_metrics = None
    if args.baseline_report:
        baseline_data = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
        baseline_metrics = baseline_data.get("metrics", baseline_data)

    gate_pass = None
    gate_checks = []
    if args.gate:
        gate_pass, gate_checks = evaluate_gate(metrics, args.gate, baseline_metrics=baseline_metrics)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_path": str(ref_path),
        "source_script_path": str(source_script_path),
        "transcript_source": "generated" if args.run_transcription else str(args.transcript),
        "config": {
            "run_transcription": args.run_transcription,
            "pipeline": "v2",
            "profile": args.profile,
        },
        "metrics": metrics,
        "gate": {"phase": args.gate, "pass": gate_pass, "checks": gate_checks} if args.gate else None,
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report written to: {report_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.gate and args.enforce_gate and not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
