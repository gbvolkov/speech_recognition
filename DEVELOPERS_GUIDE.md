# Developer Guide

This document describes the current implementation of the `speech_recognition` project as it exists in this repository.
It covers architecture, runtime requirements, transcription and segmentation algorithms, models, web flow, and current risks.

## 1. Project Purpose

The project provides:
- Speech transcription from uploaded audio/video files.
- Speaker diarization and speaker-labeled transcript generation.
- A browser editor for transcript cleanup and speaker rename.
- Export to Markdown or DOCX.

There are two execution modes:
- Web mode via Flask (`app_sr.py`) with queued background transcription.
- Batch/CLI mode via `main.py` (scan `./audio` and process files).

## 2. Codebase Map

Core runtime files:
- `transcribe.py`: model loading, ASR + diarization pipeline, segmentation, output writing.
- `app_sr.py`: Flask app, upload endpoint, background queue worker, status polling, editor and download endpoints.
- `main.py`: batch processor for local files in `./audio`.
- `tools.py`: utility to cut WAV segments by start/duration.
- `prompts.py`: Russian prompt templates for downstream LLM tasks (currently not imported by runtime flow).
- `evaluation/run_benchmark.py`: benchmark harness and phase-gate checker for `dialog_ru_timed`.
- `evaluation/build_dialog_ref.py`: reference generator from `data/test_dialog.py`.
- `evaluation/tune_profiles.py`: small chunk/stride grid-search utility for decode tuning.

UI files:
- `templates/index.html`: upload form.
- `templates/waiting.html`: progress page with queue status polling.
- `templates/editor.html`: Quill-based transcript editor and speaker rename sidebar.
- `static/js/waiting.js`: polling loop for `/check_status/<job_id>`.
- `static/js/editor.js`: editor initialization and sidebar behavior.
- `static/js/rename.js`: speaker rename and speaker-list refresh logic.
- `static/css/style.css`, `static/css/editor.css`: base and editor styling.

Supporting/config files:
- `pyproject.toml`, `requirements.in`, `requirements.txt`: dependencies.
- `hf.txt`: Hugging Face access token (required at import time in `transcribe.py`).
- `get_models.cmd`: helper to clone model repos from Hugging Face.
- `start.sh`, `kill.sh`, `check.sh`: Linux shell scripts for running/checking the Flask app.

Legacy/experimental artifacts:
- `models/` contains large model files and an example diarization script not used by runtime.
- `data/` contains CSV snapshots from prior experiments.

## 3. Runtime Requirements

Python and libraries:
- Python `>=3.12` (from `pyproject.toml`).
- Core packages: `flask`, `transformers`, `torch`, `torchaudio`, `torchcodec`, `pyannote-audio`, `pysbd`, `html2docx`, `html2text`, `python-docx`.

External dependencies:
- FFmpeg executable available on `PATH`.
- On Windows, FFmpeg must be a shared build containing DLLs (`av*.dll`/`sw*.dll`) because runtime checks enforce this.

Secrets:
- `hf.txt` must contain a valid Hugging Face token.
- `transcribe.py` reads `hf.txt` at module import time:
  - Missing file or invalid token will fail pipeline initialization.

Hardware:
- GPU is optional.
- If CUDA is available, both Whisper and diarization pipeline are moved to GPU.

## 4. Filesystem and Data Flow

Input/output directories used by code:
- `audio/`: upload destination in web mode and scan source in batch mode.
- `audio/done/`: processed original files are moved here.
- `audio/transcripts/`: generated transcript text files.

Other directories:
- `data/`: offline CSV analysis artifacts (not consumed by runtime).
- `models/`: local model artifacts and exploratory files.

## 5. End-to-End Pipeline (Current Implementation)

### 5.1 Initialization and Dependency Checks

In `transcribe.py`:
1. Read Hugging Face token from `hf.txt`.
2. Configure FFmpeg runtime path/DLL discovery on Windows (`_configure_ffmpeg_runtime`).
3. Enforce dependencies (`_assert_runtime_dependencies`):
   - FFmpeg present.
   - FFmpeg shared DLLs present on Windows.
   - `torchcodec` import succeeds.

If checks fail, transcription is aborted with explicit runtime errors.

### 5.2 Model Construction

`transcription_factory(whisper_model_id, diarization_model_id)` builds:
- Whisper ASR model (`openai/whisper-large-v3` by default):
  - `AutoModelForSpeechSeq2Seq.from_pretrained(...)`
  - `AutoProcessor.from_pretrained(...)`
  - Transformers `pipeline("automatic-speech-recognition", ...)`
- Diarization model (`pyannote/speaker-diarization-community-1` by default):
  - `pyannote.audio.Pipeline.from_pretrained(...)`

ASR pipeline parameters:
- `chunk_length_s=30`
- `stride_length_s=5`
- `return_timestamps='word'`
- `generate_kwargs={"language": "ru"}`

### 5.3 Audio Intake and Conversion

`transcribe(audio_name, transcriptor)`:
- If input is not `.wav`, convert with FFmpeg to temporary WAV (`convert_audio_to_wav`).
- Run the assembled transcription function on WAV.
- Remove temporary WAV after success.
- Run GPU memory cleanup (`gc.collect()` + `torch.cuda.empty_cache()` when CUDA exists).

### 5.4 ASR + Diarization Processing

Inside the closure returned by `transcription_factory`:
1. Run ASR on file to get `script['chunks']` with word timestamps.
2. Run diarization pipeline to get speaker turns.
3. Sort and deduplicate ASR chunks.
4. Build transcription chunk list with `start`, `end`, `text`, `speaker=None`.
5. Apply text segmentation stages (section 6).
6. Assign a speaker label to each sentence by overlap-duration scoring with diarization turns.
7. Merge consecutive sentences with the same speaker.
8. Save transcript text to `audio/transcripts/<source_basename>.txt`.

### 5.5 Output Format

Final output lines are saved as:
- `**SPEAKER_LABEL** [HH:MM:SS - HH:MM:SS]:`
- followed by wrapped text body (`textwrap.fill`, width 128).

The same generated text is returned to the Flask app for the editor flow.

## 6. Segmentation and Speaker Assignment Algorithms

This section describes the exact current logic in `transcribe.py`.

### 6.1 Deduplication (`deduplicate`)

Input: sorted ASR chunk list with fields `text` and `timestamp`.

Logic:
- Keep a running `current_text`.
- If next chunk text differs from `current_text`, append chunk.
- If text is identical and overlaps previous deduped timestamp window, extend previous window.
- Otherwise skip as duplicate.

Effect:
- Removes repeated neighboring tokens/chunks that appear in Whisper output.
- Preserves one time window for repeated text where overlap exists.

### 6.2 Merge into Coarse Sentences (`merge_chunks_into_sentences`)

Input: deduplicated chunks with start/end/text.

Logic:
- Sort by start time.
- Start a current segment.
- Append following chunk text if current segment does not end with strong punctuation (`.`, `!`, `?`).
- If strong punctuation is present, finalize segment and start a new one.

Important behavior:
- Time gap is not considered in merging (only punctuation).

### 6.3 Sentence Boundary Split with PySBD (`split_segments_with_pysbd`)

Input: coarse merged segments.

Logic:
- Apply `pysbd.Segmenter(language="ru", clean=False)` to each segment text.
- If multiple sentences are detected:
  - Segment duration is split uniformly across those sentences.
  - Sentence `i` gets:
    - `start = seg.start + duration * i / N`
    - `end = seg.start + duration * (i+1) / N`

Effect:
- Produces sentence-sized units suitable for diarization alignment.
- Timing is approximate (uniform distribution, not token-aligned).

### 6.4 Speaker Assignment by Overlap Duration

For each sentence:
1. Collect all diarization turns that overlap sentence time window.
2. For each overlapping turn:
   - Compute intersection duration with sentence.
3. Sum intersection duration per speaker.
4. Assign sentence speaker = speaker with max total overlap.
5. If no overlap, assign `"Unknown"`.

This is a majority-duration vote on overlap.

### 6.5 Merge Consecutive Same-Speaker Sentences

After speaker assignment:
- Sort sentences by start.
- Merge adjacent sentences if `speaker` labels match.
- Update block `end` and concatenate text with a space.

Result:
- Final output becomes speaker blocks rather than isolated sentences.

## 7. Flask App and Queueing Pipeline

`app_sr.py` implements a single-process in-memory job queue.

### 7.1 Upload and Queue

`POST /transcribe`:
- Validates extension against `ALLOWED_EXTENSIONS`.
- Saves file to `audio/<uuid>_<original_filename>`.
- Creates `job_id`.
- Stores `transcription_jobs[job_id] = None` (pending marker).
- Appends job ID to `pending_jobs` and enqueues `(job_id, file_path)` in `transcription_queue`.
- Returns `waiting.html`.

### 7.2 Worker Thread

`transcription_worker` (daemon thread):
- Pops job from queue.
- Marks as current (`current_job`) and removes from pending list.
- Runs `run_transcription(file_path)` from `transcribe.py`.
- Stores transcript or error string into `transcription_jobs[job_id]`.
- Clears `current_job`, marks queue task done.

### 7.3 Status Polling

`GET /check_status/<job_id>` returns JSON:
- `{"status":"completed"}` when transcript exists.
- `{"status":"processing","position":1}` when current job matches.
- `{"status":"pending","position":<n>,"queue_length":<m>}` when queued.
- `{"status":"unknown"}` otherwise.

`waiting.js` polls every 2 seconds and redirects to `/editor?job_id=<id>` when complete.

### 7.4 Editor and Export

`GET /editor`:
- Loads stored transcript text.
- Converts speaker headers into HTML (`<b>` + `<i>` formatting).
- Extracts speaker tags (`**...**`) for rename form.
- Renders `editor.html`.

`POST /download`:
- If `format=word`: converts editor HTML via `html2docx` and returns DOCX.
- Else: converts HTML to Markdown via `html2text` and returns `.md`.

## 8. Model Inventory

Runtime models actually used:
- ASR:
  - Model ID: `openai/whisper-large-v3`
  - Library: Hugging Face Transformers
  - Task: word-timestamp speech-to-text
- Diarization:
  - Model ID: `pyannote/speaker-diarization-community-1`
  - Library: pyannote-audio
  - Task: speaker turn segmentation

Non-runtime artifacts:
- `models/large-v3.pt`, `models/medium.pt`, and `models/diarization.py` are not referenced by runtime code paths.

## 9. Operational Notes

- `run_transcription` now reuses a cached transcriptor per `(model IDs + config)` in-process.
- Processed original files are moved to `audio/done`.
- Transcript files are stored in `audio/transcripts`.
- `main.py` provides batch mode but has a separate move-to-done flow from Flask mode.
- `.process` PID management is only used when launching `app_sr.py` directly.

## 10. Review Findings (Current Risks and Gaps)

High severity:
- In-memory queue/state only:
  - `transcription_jobs`, `pending_jobs`, and queue are process-local globals.
  - Impact: no durability, no cross-process coordination, state loss on restart.
- No automated tests in repository:
  - Impact: regressions are likely during refactors (especially segmentation logic and Flask flows).

Medium severity:
- Approximate sentence timing:
  - PySBD split distributes time uniformly, not by token timing.
  - Impact: less accurate timestamps and overlap scoring near boundaries.
- Merge logic ignores silence/time gaps:
  - Sentence merge is punctuation-driven only.
  - Impact: may join distant utterances into one segment.
- Fixed ASR language:
  - `generate_kwargs={"language": "ru"}` is hardcoded.
  - Impact: poor quality for non-Russian media.
- Upload validation is extension-based only:
  - No MIME sniffing/content verification.
  - Impact: malformed inputs may reach converter/model pipeline.
- Temporary WAV cleanup is not in `finally`:
  - Conversion temp file may remain if downstream fails.
- Job dictionaries grow unbounded:
  - No retention policy/expiration for completed jobs.

Low severity:
- Speaker rename UI relies on literal `**name**` markers in editor text.
- Shell helper scripts are Linux-centric, while code includes explicit Windows handling.

## 11. Recommended Next Steps

1. Initialize models once per worker process and reuse pipelines across jobs.
2. Add an automated test suite:
   - Unit tests for segmentation and overlap speaker assignment.
   - Integration test for Flask upload -> status -> editor flow.
3. Improve segmentation quality:
   - Include silence-gap thresholds in merge.
   - Use token-level timing for sentence boundaries where possible.
4. Harden upload and job lifecycle:
   - Validate media content type.
   - Add cleanup policy for `transcription_jobs` and pending data.
5. Externalize queue/state if multi-worker deployment is needed (for example Redis + task worker).
