# Marketing Engineers: Transcription Automator v0.3

This project automates the extraction, processing, and transcription of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python, `ffmpeg`, and Google Cloud AI models: Google Cloud Speech-to-Text V2 (Chirp 3) and Gemini 3.5 Transcribe (`gemini-3.5-transcribe`).

## Background
There is a huge benefit to having a transcript of your meetings, but we don't like sharing recordings with third parties. Since we have been working with Google Cloud for years, we wanted to build a solution that is secure, private, and cost-effective.

## Quickstart Options

### Option A: Zero-GCP Setup (Gemini API Key)
Get started in 2 minutes with no GCP project or cloud bucket needed:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Set it in `.env`:
   ```bash
   cp .env.example .env
   # Add your key to .env:
   # GEMINI_API_KEY=AIzaSy...
   ```
3. Run transcription:
   ```bash
   python -m src.main input/meeting.mp4 --lang nl-NL
   ```

### Option B: Enterprise GCP / GCS Setup (Vertex AI & Chirp 3)
For high-volume enterprise workloads, Speech-to-Text V2, or Vertex AI:
1. Authenticate with GCP:
   ```bash
   gcloud auth application-default login
   ```
2. Provision a GCS bucket and configure `GCP_PROJECT_ID` and `GCS_BUCKET_NAME` in `.env`.
3. Verify authentication:
   ```bash
   python -m src.main --check-auth
   ```

## Data Handling & Privacy

- **Your Cloud, Your Data:** Audio is processed using your own Google AI Studio key or your private Google Cloud Storage (GCS) bucket.
- **Automated Cleanup:** Audio chunks uploaded to Gemini Files API or GCS are automatically deleted in a `finally` execution block upon completion or failure.
- **Bucket Hardening:** We recommend configuring your GCS bucket with Uniform Bucket-Level Access, Public Access Prevention, and a 7-day lifecycle deletion policy.
- **Local Isolation:** Raw media (`input/`), intermediate audio (`output/`), generated transcripts (`transcripts/`), and local logs (`logs/`) are strictly gitignored with built-in pre-commit guardrails to prevent accidental commits of confidential data.

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` installed on your system and accessible via the command line (e.g., `sudo apt install ffmpeg` or `brew install ffmpeg`).
- `gcloud` CLI installed and configured (if using Vertex AI or Chirp 3).

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

3. **Install local pre-commit guardrails:**
   ```bash
   .githooks/install.sh
   ```

4. **Authentication Modes:**
   - **Gemini API Key (AI Studio, Zero-GCP):**
     Set `GEMINI_API_KEY=your_key` in `.env`. Files are uploaded via the Gemini Files API and purged immediately after transcription.
   - **Local Workstation (GCP ADC):**
     Authenticate via Application Default Credentials:
     ```bash
     gcloud auth application-default login
     ```
     With `GCP_AUTH_MODE=adc`, the application uses your user ADC.
   - **Automated / Agent Deployments (Service Account Key):**
     For running on a VPS or autonomous agent platform (e.g., Hermes), configure `GCP_AUTH_MODE=service_account_key` with a dedicated, least-privilege service account key file.
     See the [Hermes Deployment Guide](docs/hermes-deployment.md) for full provisioning commands.
   - **Validate Credentials & Permissions:**
     Verify your configuration at any time without processing audio:
     ```bash
     python -m src.main --check-auth
     ```
   *Note on GCP APIs:*
   - If using Chirp 3: ensure `speech.googleapis.com` is enabled.
   - If using Vertex AI (`GEMINI_BACKEND=vertex` or `auto`): ensure `aiplatform.googleapis.com` is enabled and your principal has `aiplatform.endpoints.predict` (`roles/aiplatform.user`).
   - If you see `403 PERMISSION_DENIED (SERVICE_DISABLED)` for `aiplatform.googleapis.com`, run `gcloud services enable aiplatform.googleapis.com`, or switch to `GEMINI_BACKEND=api_key` with a `GEMINI_API_KEY` from Google AI Studio.

5. **Configure environment variables:**
   Copy the example file and adjust as needed:
   ```bash
   cp .env.example .env
   ```

   Key environment variables:
   - `TRANSCRIPTION_MODEL` (Default: `auto`): Model routing policy (`auto`, `gemini-3.5-transcribe`, or `chirp_3`).
   - `GEMINI_API_KEY`: API key for Gemini AI Studio.
   - `GEMINI_BACKEND` (Default: `auto`): Backend selection (`auto`, `api_key`, or `vertex`).
   - `TRANSCRIPTION_CONCURRENCY` (Default: `5`): Maximum number of audio chunks submitted concurrently per wave.
   - `CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED` (Default: `1680` / 28 min): Safe diarization ceiling for Gemini.
   - `CHUNK_MAX_LENGTH_SECONDS_DUTCH` / `CHUNK_MAX_LENGTH_SECONDS_ENGLISH` (Default: `2700` / 45 min): Safe ceiling for Chirp 3.
   - `CHUNK_MAX_LENGTH_SECONDS_GEMINI_PLAIN` (Default: `3480` / 58 min): Safe plain text ceiling for Gemini.

## Usage

1. Place your input media file in a directory (e.g., `input/meeting.mp4`).

2. Run the script via the CLI:
   ```bash
   python -m src.main input/meeting.mp4 --lang nl-NL
   ```

3. **Interactive Mode (`-i` / `--interactive`):**
   When running in a terminal, pass `-i` for interactive prompts on language, model, or trade-offs:
   ```bash
   python -m src.main input/meeting.mp4 -i
   ```

4. **CLI Flags:**
   - `--model {auto,gemini-3.5-transcribe,chirp_3}`: Select or force a specific model.
   - `--diarize` / `--no-diarize`: Force enable or disable speaker diarization.
   - `--mode {verbatim,smart}`: Transcription formatting mode for Gemini plain audio.
   - `--lang {nl-nl,en-us,en-gb,auto}`: Language code. `auto` is supported by Gemini.
   - `--speakers <N>`: Optional speaker count hint.
   - `--prepare-only`: Extract and prepare FLAC chunks without making API calls.
   - `--from-gcs <prefix>`: Transcribe audio files already residing in GCS.

5. **Agent-Friendly Non-Interactive Mode (Exit Code 2):**
   In non-interactive environments (CI, autonomous agents), if a recording exceeds single-pass limits without explicit `--diarize`/`--no-diarize` flags, the automator exits with exit code `2` (`EXIT_DECISION_REQUIRED`) and emits a structured JSON block describing the options and exact CLI commands.

### Process Flow & Architecture

```mermaid
flowchart TD
    A["Input Media File<br/>(.mp4, .m4a, .wav, .mov, etc.)"] --> B["Phase 1: Audio Extraction<br/>FFmpeg lossless extraction to FLAC"]
    B --> C["Measure Audio Duration &<br/>Detect Conversational Pauses"]
    
    C --> D{"Duration?"}
    
    %% Short Path
    D -->|"<= 28 min (1680s)"| E["Short Path (All Languages)<br/>Model: gemini-3.5-transcribe<br/>Diarization: Enabled<br/>Monolithic (0 chunk cuts)"]
    
    %% Medium Path
    D -->|"28 to 45 min (1680s-2700s)"| F{"Language & Backend?"}
    F -->|"English + GCS"| G["Auto-switch to chirp_3<br/>Diarization: Enabled<br/>Monolithic (0 chunk cuts)"]
    F -->|"Dutch or API Key Only"| H{"Decision Required<br/>(Exit Code 2 or -i Prompt)"}
    H -->|"Option 1"| I["gemini-3.5-transcribe --diarize<br/>(Split into <= 28m chunks)"]
    H -->|"Option 2"| J["gemini-3.5-transcribe --no-diarize<br/>(Monolithic up to 58m)"]
    H -->|"Option 3"| K["chirp_3 --no-diarize<br/>(Parallel wave chunks)"]
    
    %% Long Path
    D -->|"> 45 min (> 2700s)"| L{"Decision Required<br/>(Exit Code 2 or -i Prompt)"}
    L -->|"Diarize"| M["Chunked Diarization<br/>gemini-3.5-transcribe (<=28m)<br/>or chirp_3 (<=45m)"]
    L -->|"No Diarize"| N["Continuous Plain Text<br/>gemini-3.5-transcribe smart (<=58m)<br/>or chirp_3 fast parallel wave"]
    
    E --> O["Phase 2: Transcription Engine<br/>AI Studio (Files API) or GCP GCS / Speech V2"]
    G --> O
    I --> O
    J --> O
    K --> O
    M --> O
    N --> O
    
    O --> P["Phase 3: Post-Processing & Stitching<br/>• Diarized: Speaker 1, Speaker 2 formatting<br/>• Plain: Sentence-bounded wrapping (120 chars)<br/>• Smart: Preserves markdown lists & bullets"]
    
    P --> Q["Final Artifacts Saved in transcripts/<br/>• .txt: Clean Transcript<br/>• .json: Full API Metadata"]
```

#### Functional Step Breakdown

1. **Audio Extraction & Normalization (FFmpeg):**
   - Extracts audio from compressed video or voice recording containers (`.mp4`, `.m4a`, `.mov`, `.wav`) and transcodes losslessly to FLAC.
   - FLAC provides standardized, uncompressed headers that allow Google Cloud to auto-detect sample rates and channel layouts without manual audio declaration.

2. **Intelligent Language Routing & Chunk Sizing:**
   - **Dutch (`nl-NL`) — Parallel Speed Optimization:** Diarization is not supported on Chirp 3 for Dutch, meaning chunk boundaries carry zero speaker-label penalty. Long files are split into up to 5 balanced chunks (`N = TRANSCRIPTION_CONCURRENCY`, minimum 10 min per chunk) so the entire recording transcribes concurrently in a single parallel wave in under ~2–3 minutes.
   - **English (`en-US` / `en-GB`) — Boundary Minimization:** Speaker diarization is supported, but speaker IDs reset across chunk cuts. The pipeline maximizes chunk duration up to the 45-minute API ceiling (`CHUNK_MAX_LENGTH_SECONDS_ENGLISH=2700`) to keep long meetings in as few chunks as possible (e.g. a 90-minute meeting becomes just 2 chunks).
   - **Silence Detection & Snapping:** Instead of cutting audio mid-sentence or mid-word, FFmpeg `silencedetect` locates natural conversational pauses (≥ 0.5s) within a `±15s` window around nominal cut points, ensuring conversational context remains unbroken.
   - **Monolithic Route for Short Recordings:** Any recording ≤ 45 minutes bypasses chunking entirely and processes as a single monolithic FLAC.

3. **Secure Cloud Streaming with Real-Time Feedback:**
   - Audio is uploaded to your private GCS bucket configured with 7-day retention and public access prevention.
   - Uploads stream in 2 MB resumable chunks with live in-place terminal updates (like `wget`/`curl`) and clean, timestamped logs in `logs/app.log`.

4. **Speech-to-Text V2 Batch Transcription:**
   - Submits single-file BatchRecognize requests to Google Cloud Speech-to-Text V2 using Chirp 3.
   - Uses adaptive polling intervals and automated exponential backoff on transient rate limits (`429 RESOURCE_EXHAUSTED`).

5. **Post-Processing & Formatting:**
   - English transcripts format dialogues by speaker (e.g., `Speaker 1: ...`, `Speaker 2: ...`).
   - All transcripts are formatted across sentence boundaries (`.`, `?`, `!`) with a target line length of 120 characters, avoiding thousands of characters on a single line and optimizing for LLMs, diffs, and reading.

---

### Real-Time Upload Feedback & Dual Logging Architecture
- **GCS Upload Streaming:** Large audio chunks are streamed to GCS in 2 MB resumable chunks. When running in an interactive terminal, upload progress updates in place on a single line (like `wget`/`curl`) every 5 seconds, displaying transferred megabytes, percentage, and real-time upload speed (MB/s).
- **Dual Logging (`TerminalProgressHandler` + `CleanFileFormatter`):** The terminal receives dynamic in-place updates and multi-line ASCII progress matrices during transcription. Concurrently, `logs/app.log` and `logs/gcp_api.log` receive clean, formal, single-line entries without terminal escape sequences or multi-line art blocks.

### Bounded Concurrency & Adaptive Polling
Multi-chunk requests are submitted in bounded waves (controlled by `TRANSCRIPTION_CONCURRENCY`, default: 5) with 1-second submission spacing. Operation polling uses adaptive pre-delays and intervals scaled to expected chunk duration, with automated retry on transient `429 RESOURCE_EXHAUSTED` errors to prevent regional queue congestion and API throttling.

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
The GCS re-transcription workflow automatically discovers and sorts all audio chunks under the prefix, executes the bounded concurrency pipeline, stitches the transcripts, and saves output to `transcripts/`.

### Partial Run (v0.1 Behavior)
If you only want to extract and chunk the FLAC files without calling the GCP API, use the `--prepare-only` flag:
```bash
python -m src.main input/meeting.mp4 --lang en-US --prepare-only
```
You can then manually upload the generated FLACs from the `output/` directory.

### Language Routes

The behavior of the script changes based on the language requested, to accommodate Google Cloud's Chirp 3 model constraints.

#### Dutch (`--lang nl-NL`)
- **Behavior:** Transcribes without diarization (as Chirp 3 does not support diarization for Dutch).
- **Adaptive Parallel Wave:** Sizing optimizes purely for parallel wall-clock completion in one wave (`N = TRANSCRIPTION_CONCURRENCY`), floored at `CHUNK_MIN_LENGTH_SECONDS` (10 min) and capped at `CHUNK_MAX_LENGTH_SECONDS_DUTCH` (45 min).
- **Silence Snapping:** Boundaries snap to natural pauses via FFmpeg `silencedetect` within a `±15s` window.
- **Monolithic Path:** Short recordings (≤ 45 min) naturally process as a single monolithic FLAC with zero chunk cuts.

#### English (`--lang en-US` or `--lang en-GB`)
- **Behavior:** Transcribes with speaker diarization.
- **Ceiling-Driven Boundary Minimization:** Maximizes chunk duration up to the ceiling (`CHUNK_MAX_LENGTH_SECONDS_ENGLISH=2700`, 45 min) to minimize speaker ID fragmentation across chunk cuts.
- **Silence Snapping:** Boundaries snap to natural pauses within `±15s`.
- **Monolithic Path:** Short recordings (≤ 45 min) process monolithically without boundaries.

## Project Structure

```
.
├── .env.example          # Template for environment variables
├── .githooks/            # Pre-commit security and confidentiality guard
├── .gitignore
├── CONTRIBUTING.md       # Contribution guidelines and workflow
├── LICENSE               # MIT License
├── pyproject.toml        # Project metadata and configuration
├── README.md
├── requirements.txt      # Python dependencies
├── SECURITY.md           # Security disclosure policy
├── input/                # (Optional) Drop raw media files here
├── output/               # Intermediate FLAC files appear here
├── transcripts/          # Final generated transcripts
├── logs/                 # Rotating logs (app.log and gcp_api.log)
├── tests/                # Unit and integration test suite
│   ├── test_audio.py
│   ├── test_config.py
│   ├── test_gcs.py
│   ├── test_logger.py
│   ├── test_main.py
│   ├── test_transcribe.py
│   ├── test_transcribe_gcs.py
│   ├── test_transcribe_gemini.py
│   └── test_transcript_output.py
└── src/
    ├── audio.py           # FFmpeg subprocess logic & silence-aware chunking
    ├── config.py          # Environment configuration & migration guard
    ├── gcs.py             # Google Cloud Storage operations & progress reader
    ├── logger.py          # Dual logging architecture (TTY in-place & clean file logs)
    ├── main.py            # CLI entry point and orchestration
    ├── transcribe.py      # Speech-to-Text V2 BatchRecognize logic
    ├── transcribe_gcs.py  # Direct GCS audio transcription & recovery
    ├── transcribe_gemini.py # Gemini 3.5 Transcribe integration
    └── transcript_output.py # Transcript formatting and stitching
```

## Roadmap & Future Scope (v0.4+)

- **Cross-Chunk Speaker Reconciliation (`gemini-3.6-flash`)**:
  - Pass GCS audio chunk URIs directly to `gemini-3.6-flash` via Vertex AI / Google GenAI SDK to analyze cross-chunk vocal profiles.
  - Automatically generate a `GlobalSpeakerMap` JSON to align speaker labels (`Speaker 1`, `Speaker 2`) across long, multi-chunk recordings without local PyTorch dependencies.
- **Automatic LLM Summarization Pipeline**: Summarize stitched transcripts natively via Gemini.
- **Denoiser Toggle**: FFmpeg noise reduction for phone recordings.
- **Model Adaptation**: Domain-specific vocabulary hints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
