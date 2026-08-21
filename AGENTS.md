# Marketing Engineers: Transcription Automator

This project automates the extraction, processing, and transcription of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python, `ffmpeg`, and Google Cloud Speech-to-Text V2 (Chirp 3).

## 1. Identity, Environment & Tone
- Act as a senior Data Engineer and Cloud Architect for Marketing Engineers.
- Prioritize robust infrastructure, proactive observability, and minimal human intervention.
- Do not implement "quick fixes" or sloppy code. Write production-ready, highly defensive Python.
- **Virtual Environment Requirement:** Agents MUST activate the project's virtual environment (`source venv/bin/activate`) or invoke the venv binary directly (`venv/bin/python`, `venv/bin/pytest`) before running tests, executing Python modules, or debugging scripts. Never attempt to run system-level Python tools directly without the virtual environment.

## 2. Project Phasing & Scope

### [CURRENT SCOPE] Version 0.2: End-to-End Transcription Pipeline
The script handles the full pipeline: audio extraction, GCS upload, Chirp 3 transcription via BatchRecognize, and transcript output.
- Uses `google-cloud-speech` (V2) and `google-cloud-storage` libraries.
- Authentication via Application Default Credentials (`gcloud auth application-default login`).
- The `--prepare-only` flag provides backwards-compatible v0.1 behavior (FLAC prep only, no GCP calls).

### [FUTURE SCOPE] Version 0.3+
- **Cross-Chunk Speaker Reconciliation via Gemini 3.6 Flash Global Speaker Map**:
  - Diarized audio chunks produce per-chunk speaker labels (`1`, `2`, `3`) that reset across chunks.
  - In v0.3, pass GCS audio chunk URIs directly to `gemini-3.6-flash` via GCP Vertex AI / Google GenAI SDK to analyze cross-chunk vocal profiles.
  - Generate a structured `GlobalSpeakerMap` JSON (e.g., `{"chunk_1": {"1": "Speaker A"}, "chunk_2": {"1": "Speaker B"}}`).
  - Apply mapping to `WordInfo.speaker_label` protobuf fields prior to stitching, restoring 100% speaker label consistency across long recordings without local PyTorch/heavy binary dependencies.
- **Automatic LLM Summarization Pipeline**: Summarize stitched transcripts natively via Gemini.
- **Denoiser Toggle**: FFmpeg noise reduction for phone recordings.
- **Model Adaptation**: Domain-specific vocabulary hints.
- **Batch Processing**: Process all files in `input/` directory concurrently.

## 3. Core Architecture & Rules

### Phase A: Input Handling & Audio Extraction (FFmpeg)
- **Input Flexibility:** The script must accept various input media formats (e.g., `.mp4`, `.m4a`, `.wav`, `.mov`). Do not hardcode validation strictly to `.mp4`.
- **Rule:** Never prepare raw compressed containers directly for GCP. The system must normalize all audio inputs.
- **Action:** Use `ffmpeg` via Python's `subprocess` module to extract audio, drop any video streams if present, and transcode losslessly to FLAC. 
- **Why:** FLAC headers are auto-detected by the GCP API, bypassing the need to manually declare sample rates and channel counts.

### Phase B: Performance Model, Sizing Strategies & Language Logic

#### Linear Processing Performance Model
Empirical measurement demonstrates that Chirp 3 BatchRecognize processing time scales linearly with audio duration:
```
per_chunk_time ≈ CHUNK_OVERHEAD_SECONDS + (chunk_duration / CHUNK_REALTIME_FACTOR)
               ≈ 15s + (chunk_duration / 25)
```
- Short chunks (<10 min) waste wall-clock time because the ~15s fixed overhead dominates.
- Historical 5-minute Dutch chunking was an over-fit to a single transient regional incident on 2026-07-22.
- The 20-minute BatchRecognize limit only applies when explicit word-level time offsets are enabled (`enable_word_time_offsets=True`). Diarization alone without word offsets operates seamlessly up to the API ceiling (tested and verified up to 45 minutes / 2700s).

#### Route 1: Dutch (`nl-NL`)
- **Constraint:** Speaker Diarization is explicitly unsupported for Dutch on Chirp 3.
- **Strategy:** **Adaptive Parallel Wave (`N = TRANSCRIPTION_CONCURRENCY`)**. Because boundaries carry no diarization fragmentation cost in Dutch, chunk sizing optimizes purely for parallel wall clock.
- **Action:** 
  - Subdivides audio into `N = min(TRANSCRIPTION_CONCURRENCY, floor(duration / CHUNK_MIN_LENGTH_SECONDS))` chunks (floored at `CHUNK_MIN_LENGTH_SECONDS=600`, capped at `CHUNK_MAX_LENGTH_SECONDS_DUTCH=2700`).
  - Nominal cut points snap to natural pauses via FFmpeg `silencedetect` within a `±CHUNK_SILENCE_SEARCH_WINDOW_SECONDS` (15s) window.
  - Short files (≤ 45 min ceiling when `N=1`) naturally process as a single monolithic FLAC.
  - Transcribes without diarization in a single parallel wave; stitches with chunk boundary headers.
  - Formats plain text into clean, line-bounded sentences (max 120 chars per line, target 3 sentences per line).

#### Route 2: English (`en-US` / `en-GB`)
- **Constraint:** Diarization is supported, but speaker IDs are chunk-local.
- **Strategy:** **Ceiling-Driven Boundary Minimization**. Maximizes chunk duration up to the ceiling (`CHUNK_MAX_LENGTH_SECONDS_ENGLISH=2700`, 45 min) to minimize speaker fragmentation across chunk cuts.
- **Action:** 
  - Subdivides audio based strictly on `n_ceiling = ceil(duration / (max_chunk_length - window))`.
  - Nominal boundaries snap to the nearest silence midpoint within `±15s`.
  - Short files (≤ 45 min) process monolithically with zero chunk boundaries.
  - Prompts user for speaker count (or auto-detects).
  - Transcribes with diarization; outputs speaker-labeled transcript.

### Phase C: GCP Integration, Resumable Uploads & Bounded Concurrency
- **GCS Upload & Resumable Streaming:** Temporary upload for transcription, cleaned up after completion.
  - Bucket: `gs://your-transcriber-bucket` (configured with 7-day retention, uniform access, and public access prevention).
  - Resumable streaming uploads use 2 MB chunk sizes (`blob.chunk_size = 2 * 1024 * 1024`) with `ProgressFileReader` to deliver fine-grained in-place upload feedback every 5 seconds without blocking socket reads.
- **Regional Endpoint:** Use `{region}-speech.googleapis.com` (default: `eu`). Note: This region must match the Chirp 3 API supported regions (e.g., `eu`), which is independent of the GCS bucket region.
- **Authentication:** Must run `gcloud auth application-default login` (with `GCP_PROJECT_ID` configured programmatically via `.env`). The Cloud Speech-to-Text API must also be enabled.
- **Bounded Concurrency & Rate Protection:** Multi-chunk requests are submitted in bounded waves (controlled by `TRANSCRIPTION_CONCURRENCY`, default: 5) with 1-second submission spacing. Operation polling uses adaptive pre-delays and intervals scaled to expected chunk duration, incorporating automated backoff/retry on transient `429 RESOURCE_EXHAUSTED` errors.
- **Duration-Aware Timeout:** Maximum allowable transcription time scales dynamically: `max(TRANSCRIPTION_MAX_TIMEOUT_SECONDS, ceil(total_audio_seconds / 3600) * 1800)`.
- **BatchRecognize:** Single-file requests with InlineOutputConfig for each chunk.
- **Cleanup:** Delete GCS objects in `finally` blocks; log paths on failure for manual inspection.

### Phase D: GCS Re-transcription (`--from-gcs`)
- **Direct Cloud Processing:** Allows re-transcribing or processing audio chunks already uploaded to GCS without re-uploading or local re-encoding.
- **CLI Options:** `python -m src.main --from-gcs <prefix_or_gs_uri> --lang <lang>` or `python -m src.transcribe_gcs <gcs_target> --lang <lang>`.
- **Workflow:** Discovers all valid audio files in the target GCS prefix, sorts them alphanumerically into sequential chunk order, transcribes via the bounded concurrency pipeline, and stitches outputs into `transcripts/`.

## 4. Coding Standards & Observability
- **Typing:** Use strict Python type hints. Use `list[str]` (not `typing.List`).
- **Logging & UX Architecture:** Use standard Python `logging`. Do not use `print()` statements.
  - **Terminal (`stdout`):** `TerminalProgressHandler` renders live in-place progress lines (`\r\x1b[2K`) for transient upload ticks and full ASCII status matrices during transcription.
  - **Log Files (`app.log`, `gcp_api.log`):** `CleanFileFormatter` strips terminal escape codes and multi-line ASCII art, preserving clean, formal, single-line entries for automated ingestion and grep parsing.
- **Configuration:** No magic numbers. All configurable values via `.env` file (`python-dotenv`). Migration guard enforces renamed configuration keys.
- **Error Handling:** Wrap GCP and subprocess errors in `ProcessingError` with actionable messages.
- **Cleanup:** Ensure temporary/intermediate files are cleaned up, leaving only the original input and final transcript.
