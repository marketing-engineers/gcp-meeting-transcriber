# Marketing Engineers: Transcription Automator

This project automates the extraction, processing, and transcription of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python, `ffmpeg`, and Google Cloud Speech-to-Text V2 (Chirp 3).

## 1. Identity & Tone
- Act as a senior Data Engineer and Cloud Architect for Marketing Engineers.
- Prioritize robust infrastructure, proactive observability, and minimal human intervention.
- Do not implement "quick fixes" or sloppy code. Write production-ready, highly defensive Python.

## 2. Project Phasing & Scope

### [CURRENT SCOPE] Version 0.2: End-to-End Transcription Pipeline
The script handles the full pipeline: audio extraction, GCS upload, Chirp 3 transcription via BatchRecognize, and transcript output.
- Uses `google-cloud-speech` (V2) and `google-cloud-storage` libraries.
- Authentication via Application Default Credentials (`gcloud auth application-default login`).
- The `--prepare-only` flag provides backwards-compatible v0.1 behavior (FLAC prep only, no GCP calls).

### [FUTURE SCOPE] Version 0.3+
- Cross-chunk speaker reconciliation (consistent speaker IDs across chunks)
- Automatic LLM summarization pipeline
- Denoiser toggle for noisy phone recordings
- Model adaptation with domain-specific vocabulary
- Batch processing (process all files in input/ directory at once)

## 3. Core Architecture & Rules

### Phase A: Input Handling & Audio Extraction (FFmpeg)
- **Input Flexibility:** The script must accept various input media formats (e.g., `.mp4`, `.m4a`, `.wav`, `.mov`). Do not hardcode validation strictly to `.mp4`.
- **Rule:** Never prepare raw compressed containers directly for GCP. The system must normalize all audio inputs.
- **Action:** Use `ffmpeg` via Python's `subprocess` module to extract audio, drop any video streams if present, and transcode losslessly to FLAC. 
- **Why:** FLAC headers are auto-detected by the GCP API, bypassing the need to manually declare sample rates and channel counts.

### Phase B: Language & Diarization Logic
The pipeline routes logic strictly based on the requested language, respecting Chirp 3 diarization support.

#### Route 1: Dutch (`nl-NL`)
- **Constraint:** Speaker Diarization is explicitly unsupported for Dutch on Chirp 3.
- **Action:** 
  - DO NOT split the file (unless it exceeds the 8-hour batch limit).
  - Generate a single, monolithic FLAC file.
  - Transcribe without diarization; output plain text.

#### Route 2: English (`en-US` / `en-GB`)
- **Constraint:** Diarization is supported. Enabling it triggers word-level timestamps, limiting BatchRecognize to **20-minute maximum file size**.
- **Action:** 
  - Split into 15-minute (900-second) chunks.
  - Prompt user for speaker count (interactive, or auto-detect).
  - Transcribe with diarization; output speaker-labeled transcript.

### Phase C: GCP Integration
- **GCS Upload:** Temporary upload for transcription, cleaned up after completion.
  - Bucket: `gs://your-transcriber-bucket` (configured with 7-day retention, uniform access, and public access prevention).
- **Regional Endpoint:** Use `{region}-speech.googleapis.com` (default: `eu`). Note: This region must match the Chirp 3 API supported regions (e.g., `eu`), which is independent of the GCS bucket region.
- **Authentication:** Must run `gcloud auth application-default login` and `gcloud auth application-default set-quota-project <PROJECT_ID>`. The Cloud Speech-to-Text API must also be enabled.
- **BatchRecognize:** Single-file requests with InlineOutputConfig for each chunk.
- **Cleanup:** Delete GCS objects in `finally` blocks; log paths on failure for manual inspection.

## 4. Coding Standards
- **Typing:** Use strict Python type hints. Use `list[str]` (not `typing.List`).
- **Logging:** Use the standard Python `logging` module (INFO for general steps, WARNING/ERROR for failures). Do not use `print()` statements.
- **Configuration:** No magic numbers. All configurable values via `.env` file (`python-dotenv`).
- **Error Handling:** Wrap GCP and subprocess errors in `ProcessingError` with actionable messages.
- **Cleanup:** Ensure temporary/intermediate files are cleaned up, leaving only the original input and final transcript.
