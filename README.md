# Flask Transcription Service

This project is a simple Flask application that receives an audio file, transcribes it, displays the transcript in an editor for speaker renaming, and allows you to download the transcript as a Markdown or Word document.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://your-repo-url.git
   cd flask_transcription_service
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Provide Hugging Face token**

   Create `hf.txt` in the repo root and put your token in it.

4. **Run web app**

   ```bash
   python app_sr.py
   ```

## Runtime Pipeline Flags

`transcribe.py` supports non-breaking internal runtime flags via environment variables:

- `SR_PROFILE=default|quality|fast`
- `SR_CHUNK_LENGTH_S=<int>`
- `SR_STRIDE_LENGTH_S=<int>`
- `SR_NUM_SPEAKERS=<int>`
- `SR_FORCE_LANGUAGE=<name>` (canonical language name, e.g. `russian`)
- `SR_FORCE_TASK=transcribe|translate` (default `transcribe`)
- `SR_STRICT_DETERMINISM=true|false` (enforce deterministic torch/CUDA behavior)
- `SR_RANDOM_SEED=<int>` (default `1337`)
- `SR_EMIT_METRICS=true|false`
- `SR_DEBUG_TRACE=true|false` (write stage-by-stage debug artifacts under `audio/transcripts/debug/`)
- `SR_DEBUG_MAX_ITEMS=<int>` (limit samples in debug JSON)

Only the single `v2` pipeline exists.
Deprecated mode env vars are hard-disabled:
- `SR_SEGMENTATION_MODE`
- `SR_SPEAKER_MODE`

## Benchmark: `dialog_ru_timed.wav`

Build reference from `data/test_dialog.py`:

```bash
python -m evaluation.build_dialog_ref \
  --source data/test_dialog.py \
  --output evaluation/gold/dialog_ru_timed.ref.jsonl
```

Run benchmark against an existing transcript:

```bash
python -m evaluation.run_benchmark \
  --ref evaluation/gold/dialog_ru_timed.ref.jsonl \
  --transcript audio/transcripts/dialog_ru_timed.txt \
  --report-out evaluation/reports/dialog_ru_timed.baseline.json
```

Run benchmark by generating a fresh transcript from audio:

```bash
python -m evaluation.run_benchmark \
  --run-transcription \
  --audio audio/dialog_ru_timed.wav \
  --profile quality \
  --ref evaluation/gold/dialog_ru_timed.ref.jsonl \
  --report-out evaluation/reports/dialog_ru_timed.v2.json
```

Run decode tuning grid (chunk/stride search):

```bash
python -m evaluation.tune_profiles \
  --audio audio/dialog_ru_timed.wav \
  --chunk-grid 20,30 \
  --stride-grid 5,8 \
  --report-out evaluation/reports/dialog_ru_timed.tuning.json
```
