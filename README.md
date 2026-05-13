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
   gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>
   ```
   *Note: Ensure the **Cloud Speech-to-Text API** is enabled in your GCP project.*

4. **Configure environment variables:**
   Copy the example file and adjust as needed:
   ```bash
   cp .env.example .env
   ```
   *Note: Ensure you update `GCP_PROJECT_ID` with your actual GCP project ID. The default bucket `your-transcriber-bucket` is pre-configured for temporary audio uploads with a 7-day retention policy and public access prevention.*

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

### Partial Run (v0.1 Behavior)
If you only want to extract and chunk the FLAC files without calling the GCP API, use the `--prepare-only` flag:
```bash
python -m src.main input/meeting.mp4 --lang en-US --prepare-only
```
You can then manually upload the generated FLACs from the `output/` directory.

### Language Routes

The behavior of the script changes based on the language requested, to accommodate Google Cloud's Chirp 3 model constraints.

#### Dutch (`--lang nl-NL`)
- **Behavior:** Extracts audio losslessly into a single, monolithic `.flac` file. Transcribes without diarization (as Chirp 3 does not support diarization for Dutch).

#### English (`--lang en-US` or `--lang en-GB`)
- **Behavior:** Extracts audio losslessly and splits the file into 15-minute chunks to stay under the 20-minute BatchRecognize limit for diarized audio. Automatically stitches the transcripts together after processing.

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
    └── transcript_output.py # Transcript formatting and stitching
```
