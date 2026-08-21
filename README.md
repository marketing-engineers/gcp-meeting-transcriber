# Marketing Engineers: Transcription Automator v0.2

This project automates the extraction, processing, and transcription of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python, `ffmpeg`, and Google Cloud Speech-to-Text V2.

## Background
There is a huge benefit to having a transcript of your meetings, but we don't like sharing recordings with third parties. Since we have been working with Google Cloud for years, we wanted to build a solution that is secure, private, and cost-effective.

## Prerequisites

- Python 3.10+
- `ffmpeg` installed on your system and accessible via the command line (e.g., `sudo apt install ffmpeg` or `brew install ffmpeg`).
- `gcloud` CLI installed and configured.

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth application-default login
   ```
   *Note: Ensure the **Cloud Speech-to-Text API** is enabled in your GCP project. Quota project attribution is managed programmatically via the `.env` configuration file.*

4. **Configure environment variables:**
   Copy the example file and adjust as needed:
   ```bash
   cp .env.example .env
   ```
   *Note: Ensure you update `GCP_PROJECT_ID` with your actual GCP project ID. The default bucket `your-transcriber-bucket` is pre-configured for temporary audio uploads with a 7-day retention policy and public access prevention.*

    Key environment variables:
    - `TRANSCRIPTION_BATCH_SIZE` (Default: `5`): Maximum number of audio chunks submitted and monitored concurrently per batch to prevent GCP rate-limit (`429 RESOURCE_EXHAUSTED`) issues.
    - `GCS_UPLOAD_TIMEOUT_SECONDS` (Default: `600`): Socket/upload timeout in seconds for GCS audio uploads. If you have large files and slow upload speeds, increase this to prevent timeout errors.
    - `TRANSCRIPTION_MAX_TIMEOUT_SECONDS` (Default: `1800`): Overall maximum time cap in seconds (30 minutes) across the entire transcription job.
    - `TRANSCRIPTION_MAX_CHARS_PER_LINE` (Default: `120`): Target maximum line length for transcript post-processing.
    - `TRANSCRIPTION_MAX_SENTENCES_PER_LINE` (Default: `3`): Target maximum sentences per line.
    - `TRANSCRIPTION_CUSTOM_PROMPT` (Optional): Custom prompt instruction passed to Chirp 3 models via `CustomPromptConfig`.

## Usage

1. Place your input media file in a directory (e.g., `input/meeting.mp4`).

2. Run the script via the CLI, specifying the input file and the language route:
   ```bash
   python -m src.main input/meeting.mp4 --lang en-US
   ```

3. **Interactive Prompt:** If processing an English recording (which supports diarization), the script will prompt you for the number of speakers:
   ```text
   How many speakers were in this recording? (press Enter to auto-detect):
   ```
   You can enter a number (e.g., `2`) or press Enter to let the Chirp 3 model auto-detect the speaker count.

4. The pipeline will automatically upload the audio to the GCS bucket (`gs://your-transcriber-bucket`), run the transcription via BatchRecognize, download the results, and clean up the GCS bucket.

5. **Output:** The final transcripts will be saved in the `transcripts/` directory as both a human-readable text file and a raw JSON API response.
   Transcripts are automatically post-processed at sentence boundaries (`.`, `?`, `!`) to guarantee clean, line-bounded plain text (capped by default at 120 characters per line) for optimal compatibility with CLI tools, git diffs, and LLM contexts.

### Bounded Batch Execution & Rate Protection
When chunking long recordings, requests are submitted in bounded batches (controlled by `TRANSCRIPTION_BATCH_SIZE`, default: 5) with a 1-second submission spacing. Operation polling incorporates automated backoff and retry on transient `429 RESOURCE_EXHAUSTED` errors to prevent regional queue congestion and API throttling.

### Transcribing Existing GCS Files (`--from-gcs`)
If audio chunks are already uploaded to Google Cloud Storage (e.g. following a recovery or re-run scenario), you can process them directly without local re-encoding or re-uploading:

```bash
# Using the main entrypoint
python -m src.main --from-gcs sample-meeting-2026-08-20_20260820_152628 --lang nl-NL

# Or with full gs:// URI
python -m src.main --from-gcs gs://your-transcriber-bucket/sample-meeting-2026-08-20_20260820_152628 --lang nl-NL

# Or using the dedicated module
python -m src.transcribe_gcs sample-meeting-2026-08-20_20260820_152628 --lang nl-NL
```
The GCS re-transcription workflow automatically discovers and sorts all audio chunks under the prefix, executes the bounded batch pipeline, stitches the transcripts, and saves output to `transcripts/`.

### Partial Run (v0.1 Behavior)
If you only want to extract and chunk the FLAC files without calling the GCP API, use the `--prepare-only` flag:
```bash
python -m src.main input/meeting.mp4 --lang en-US --prepare-only
```
You can then manually upload the generated FLACs from the `output/` directory.

### Language Routes

The behavior of the script changes based on the language requested, to accommodate Google Cloud's Chirp 3 model constraints.

#### Dutch (`--lang nl-NL`)
- **Behavior:** Transcribes without diarization (as Chirp 3 does not support diarization for Dutch). Extracts audio losslessly into FLAC. If duration exceeds `CHUNK_LENGTH_SECONDS_DUTCH` (default: 5 minutes / 300s), audio is automatically split into chunks to prevent GCP BatchRecognize timeouts on the EU regional queue.

#### English (`--lang en-US` or `--lang en-GB`)
- **Behavior:** Extracts audio losslessly and splits the file into 15-minute chunks to stay under the 20-minute BatchRecognize limit for diarized audio. Automatically stitches the transcripts together after processing and inserts explicit chunk boundary headers with notices warning that speaker numbers do not correspond across chunk boundaries.

## Project Structure

```
.
├── .env.example          # Template for environment variables
├── .gitignore
├── pyproject.toml        # Project metadata and configuration
├── requirements.txt      # Python dependencies
├── input/                # (Optional) Drop raw media files here
├── output/               # Intermediate FLAC files appear here
├── transcripts/          # Final generated transcripts
├── tests/                # Unit tests
└── src/
    ├── audio.py           # FFmpeg subprocess logic
    ├── config.py          # Environment configuration
    ├── gcs.py             # Google Cloud Storage operations
    ├── logger.py          # Centralized logging setup
    ├── main.py            # CLI entry point and orchestration
    ├── transcribe.py      # Speech-to-Text V2 BatchRecognize logic
    ├── transcribe_gcs.py  # Direct GCS audio transcription & recovery
    └── transcript_output.py # Transcript formatting and stitching
```

## Roadmap & Future Scope (v0.3+)

- **Cross-Chunk Speaker Reconciliation (`gemini-3.6-flash`)**:
  - Pass GCS audio chunk URIs directly to `gemini-3.6-flash` via Vertex AI / Google GenAI SDK to analyze cross-chunk vocal profiles.
  - Automatically generate a `GlobalSpeakerMap` JSON to align speaker labels (`Speaker 1`, `Speaker 2`) across long, multi-chunk recordings without local PyTorch dependencies.
- **Automatic LLM Summarization Pipeline**: Summarize stitched transcripts natively via Gemini.
- **Denoiser Toggle**: FFmpeg noise reduction for phone recordings.
- **Model Adaptation**: Domain-specific vocabulary hints.

---

## Troubleshooting & Performance

### Dutch (`nl-NL`) Queue Stalls & GDPR EU Residency Compliance
The `GCP_REGION` is set to `eu` to enforce strict European data residency under GDPR compliance guidelines. However, Google Cloud's European multi-region endpoint (`eu-speech.googleapis.com`) occasionally places heavy 15-minute chunks on extended queue holds, leading to job stalls and timeouts.

**GDPR-Compliant Resolution:**
To bypass European queue congestion while maintaining 100% EU data residency, the pipeline splits Dutch recordings into shorter **5-minute chunks** (`CHUNK_LENGTH_SECONDS_DUTCH=300`). 
* Smaller 5-minute chunks require significantly less processing footprint on the GCP backend.
* Google's European queue slots these smaller chunks with high priority, completing them rapidly in parallel while keeping your audio data entirely inside the European Union.
* To tune this behavior, adjust `CHUNK_LENGTH_SECONDS_DUTCH` in your `.env` file. Do not set `GCP_REGION=us` if you are bound by EU privacy laws.
