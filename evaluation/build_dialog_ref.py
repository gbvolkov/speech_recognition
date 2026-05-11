from __future__ import annotations

import argparse

from evaluation.dialog_ref import build_reference


def main():
    parser = argparse.ArgumentParser(description="Build JSONL reference from data/test_dialog.py")
    parser.add_argument(
        "--source",
        default="data/test_dialog.py",
        help="Path to test dialog generator script with DEFAULT_SEGMENTS.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/gold/dialog_ru_timed.ref.jsonl",
        help="Output reference JSONL path.",
    )
    args = parser.parse_args()

    segments = build_reference(args.source, args.output)
    print(f"Wrote {len(segments)} reference segments to {args.output}")


if __name__ == "__main__":
    main()
