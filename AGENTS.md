# Marketing Engineers: Transcription Automator

This project automates the extraction, processing, and transcription of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python, `ffmpeg`, and Google Cloud AI models: Google Cloud Speech-to-Text V2 (Chirp 3) and Gemini 3.5 Transcribe (`gemini-3.5-transcribe`).

## 1. Identity, Environment & Tone
- Act as a senior Data Engineer and Cloud Architect for Marketing Engineers.
- Prioritize robust infrastructure, proactive observability, and minimal human intervention.
- Do not implement "quick fixes" or sloppy code. Write production-ready, highly defensive Python.
- **Virtual Environment Requirement:** Agents MUST activate the project's virtual environment (`source venv/bin/activate`) or invoke the venv binary directly (`venv/bin/python`, `venv/bin/pytest`) before running tests, executing Python modules, or debugging scripts. Never attempt to run system-level Python tools directly without the virtual environment.
- **Harness & Agent Skill Registration:**
  - This repository provides an `agentskills.io` skill in `skills/gcp-audio-transcribe/SKILL.md`.
  - When cloning or setting up this repository on an agent host (such as Hermes), link the skill into the harness directory:
    `mkdir -p ~/.hermes/skills && ln -sf "$(pwd)/skills/gcp-audio-transcribe" ~/.hermes/skills/`
  - When asked how to install or set up this repository in an autonomous environment, ensure the skill is linked and guide the human operator to `docs/hermes-deployment.md` for GCP IAM provisioning.


## 2. Project Phasing & Scope

### [CURRENT SCOPE] Version 0.2: End-to-End Dual-Model Transcription Pipeline
The pipeline supports both `gemini-3.5-transcribe` and `chirp_3` with intelligent duration-aware and language-aware routing:
- **Default Preferred Model:** `gemini-3.5-transcribe` for Dutch & English recordings <= 28 min (monolithic with diarization), multilingual code-switching, and smart dictation.
- **Chirp 3 Model:** Used for English recordings between 28–45 min (monolithic with diarization without chunk cuts), or fast adaptive parallel waves on long recordings when diarization is disabled.
- **Hybrid Authentication:**
  - `api_key`: Zero-GCP setup using `GEMINI_API_KEY` and the Gemini Files API.
  - `vertex` / `adc` / `service_account_key`: Enterprise GCP deployment via Vertex AI / GCS / Speech-to-Text V2.
- **Non-Interactive by Default (Agent Friendly):** The CLI runs non-interactively without blocking on `input()`. When a trade-off decision is needed, it cleanly exits with exit code `2` (`EXIT_DECISION_REQUIRED`) and outputs a structured JSON block describing the options and exact CLI flags.
- **Interactive Mode:** Human operators running in a terminal pass `-i` / `--interactive` for guided prompts.
- **Backwards Compatibility:** `--prepare-only` provides FLAC extraction only without GCP/Gemini API calls.

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

### Phase B: Performance Model, Sizing Strategies & Model Routing

#### Safe Duration Ceilings
- `GEMINI_DIARIZATION_MAX_SECONDS = 1680` (28 min; safe ceiling below 30m API limit)
- `CHIRP_MAX_SECONDS = 2700` (45 min; verified ceiling for Speech V2 BatchRecognize without word offsets)
- `GEMINI_PLAIN_MAX_SECONDS = 3480` (58 min; safe ceiling below 60m plain audio API limit)

#### Decision Tree & Language Routing
1. **Short Recordings (duration <= 1680s / 28 min) — All Languages:**
   - Auto-resolves to `gemini-3.5-transcribe` with diarization enabled (monolithic, 0 chunk cuts).
   - If only Chirp 3 credentials configured, falls back to Chirp 3 (English diarized, Dutch plain).
2. **Medium Recordings (1680s < duration <= 2700s / 28–45 min):**
   - **English:** Auto-switches to `chirp_3` with diarization enabled to transcribe monolithically without chunk cuts (if GCS configured).
   - **Dutch:** Requires user decision (Gemini chunked diarization vs Gemini plain monolithic up to 58m vs Chirp 3 parallel waves).
3. **Long Recordings (duration > 2700s / > 45 min):**
   - Diarization without chunking is impossible on any model. Requires decision: keep diarization (with chunk boundary reset notice) or disable diarization (smart/verbatim plain text or fast parallel waves).

#### Non-Interactive Agent Workflow (Exit Code 2)
When the pipeline encounters a duration requiring a routing trade-off and `--diarize` / `--no-diarize` was not specified:
- Emits structured JSON to `stderr`:
  ```json
  {
    "status": "decision_required",
    "duration_seconds": 3600.0,
    "duration_formatted": "60m00s",
    "reason": "...",
    "options": [
      {
        "option": 1,
        "title": "Keep Diarization via Gemini 3.5 Transcribe",
        "flags": "--model gemini-3.5-transcribe --diarize",
        "command": "python -m src.main input/rec.mp4 --lang nl-NL --model gemini-3.5-transcribe --diarize"
      }
    ]
  }
  ```
- Exits immediately with code `2`. Autonomous agents should catch exit code `2`, present the options to the user, and re-invoke `venv/bin/python -m src.main` with the chosen flags.

### Phase C: GCP & Gemini Integration, Uploads & Bounded Concurrency
- **AI Studio (`api_key`):** Audio files are uploaded to Gemini Files API (`client.files.upload`), transcribed, and cleaned up in `finally` blocks (`client.files.delete`).
- **Vertex AI (`vertex`):** Audio files are uploaded to GCS and referenced by `gs://` URI via `types.Part.from_uri`.
- **Chirp 3 (`chirp_3`):** Audio files are uploaded to GCS, submitted via `BatchRecognize`, and polled with adaptive intervals.
- **Bounded Concurrency & Rate Protection:** Multi-chunk requests are submitted in bounded waves (`TRANSCRIPTION_CONCURRENCY`, default 5) with ThreadPoolExecutor. Exponential backoff and retry handle transient `429 RESOURCE_EXHAUSTED` errors.
- **Cleanup:** Temporary GCS objects and local chunk files are deleted in `finally` blocks.

### Phase D: GCS Re-transcription (`--from-gcs`)
- **Direct Cloud Processing:** Allows re-transcribing audio files already in GCS:
  `python -m src.main --from-gcs <prefix_or_gs_uri> --lang <lang>` or `python -m src.transcribe_gcs <gcs_target> --lang <lang>`.
- Supports `--model`, `--diarize`/`--no-diarize`, and `--mode`.

## 4. Coding Standards & Observability
- **Typing:** Use strict Python type hints. Use `list[str]` (not `typing.List`).
- **Logging & UX Architecture:** Use standard Python `logging`. Do not use `print()` statements.
  - **Terminal (`stdout`):** `TerminalProgressHandler` renders live in-place progress lines (`\r\x1b[2K`) and ASCII status matrices during transcription.
  - **Log Files (`app.log`, `gcp_api.log`):** `CleanFileFormatter` strips terminal escape codes, preserving clean single-line entries.
- **Configuration:** No magic numbers. All configurable values via `.env` file (`python-dotenv`). Migration guard enforces renamed configuration keys.
- **Error Handling:** Wrap errors in `ProcessingError` with actionable remediation guidance.
- **Error Handling:** Wrap GCP and subprocess errors in `ProcessingError` with actionable messages.
- **Cleanup:** Ensure temporary/intermediate files are cleaned up, leaving only the original input and final transcript.
