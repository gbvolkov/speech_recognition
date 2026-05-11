from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    # Allow `python meeting_v15/__main__.py` execution by adding repo root to sys.path.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from meeting_v15.config import from_env
    from meeting_v15.pipeline import run_batch
else:
    from .config import from_env
    from .pipeline import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone meeting transcription pipeline (Whisper large-v3 + Pyannote)."
    )
    parser.add_argument("--input", required=False, default="audio/dialog_ru_timed.wav", help="Input audio file or directory.")
    parser.add_argument("--output-dir", default="audio/transcripts", help="Output directory for transcript artifacts.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token override.")
    parser.add_argument("--hf-token-file", default=None, help="Path to token file (default: hf.txt).")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument(
        "--disable-overlap-second-pass",
        action="store_true",
        help="Disable overlap-specific second-pass ASR.",
    )
    parser.add_argument(
        "--emit-debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit debug artifacts under output_dir/debug_v15.",
    )
    parser.add_argument(
        "--move-processed-to",
        default=None,
        help="Optional directory where processed input files are moved.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = from_env()
    overrides = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.hf_token is not None:
        overrides["hf_token"] = args.hf_token
    if args.hf_token_file is not None:
        overrides["hf_token_file"] = args.hf_token_file
    if args.min_speakers is not None:
        overrides["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        overrides["max_speakers"] = args.max_speakers
    if args.num_speakers is not None:
        overrides["num_speakers"] = args.num_speakers
    if args.disable_overlap_second_pass:
        overrides["overlap_second_pass"] = False
    if args.emit_debug is not None:
        overrides["emit_debug"] = bool(args.emit_debug)
    if args.move_processed_to is not None:
        overrides["move_processed_to"] = args.move_processed_to

    if overrides:
        config = replace(config, **overrides)

    results = run_batch(args.input, config=config)
    for result in results:
        source = result.get("source_file", "")
        outputs = result.get("outputs", {})
        print(f"[meeting_v15] completed: {source}")
        print(f"  text: {outputs.get('text_path')}")
        print(f"  json: {outputs.get('json_path')}")
    print(f"[meeting_v15] processed {len(results)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
