# Marketing Engineers: Transcription Automator v0.1

This project automates the extraction, processing, and transcription preparation of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python and `ffmpeg`.

## Prerequisites

- Python 3.8+
- `ffmpeg` installed on your system and accessible via the command line (e.g., `sudo apt install ffmpeg` or `brew install ffmpeg`).

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

3. **Configure environment variables:**
   Copy the example file and adjust as needed:
   ```bash
   cp .env.example .env
   ```
   Available variables:
   | Variable | Default | Description |
   |---|---|---|
   | `CHUNK_LENGTH_SECONDS` | `900` | Duration (seconds) per audio chunk for the English route |
   | `SUPPORTED_EXTENSIONS` | `.mp4,.m4a,.wav,.mov,.mkv` | Allowed input file extensions |
   | `DUTCH_LANGUAGE_CODES` | `nl-NL` | Language codes routed as Dutch (monolithic) |
   | `ENGLISH_LANGUAGE_CODES` | `en-US,en-GB` | Language codes routed as English (chunked) |
   | `OUTPUT_DIR` | `output` | Directory for generated FLAC files |

## Usage

1. Place your input media file in the `input/` directory (e.g., `input/meeting.mp4`).

2. Run the script via the package module, specifying the input file and the language route:
   ```bash
   python -m src.main input/meeting.mp4 --lang en-US
   ```

3. Collect the prepared `.flac` files from the `output/` directory and upload them manually to the GCP Console.

### Language Routes

The behavior of the script changes strictly based on the language requested, to accommodate Google Cloud's Chirp model constraints.

#### Dutch (`--lang nl-NL`)
- **Behavior:** Extracts audio losslessly into a single, monolithic `.flac` file.
- **Next Steps:** Upload the resulting file from the `output/` directory directly to the GCP console. **Do NOT enable Speaker Diarization** when running the transcription on GCP.

#### English (`--lang en-US` or `--lang en-GB`)
- **Behavior:** Extracts audio losslessly and splits the file into 15-minute chunks (e.g., `meeting_chunk_000.flac`, `meeting_chunk_001.flac`).
- **Next Steps:** Upload all generated chunks from the `output/` directory to GCP. **ENABLE Speaker Diarization** (2 speakers). You will manually stitch the transcripts together.

## Project Structure

```
.
├── .env.example          # Template for environment variables
├── .gitignore
├── requirements.txt      # Python dependencies
├── input/                # Drop raw media files here
├── output/               # Generated FLAC files appear here
└── src/
    ├── __init__.py
    ├── audio.py           # FFmpeg subprocess logic (extraction, chunking)
    ├── config.py          # Loads and validates .env configuration
    ├── logger.py          # Centralized logging setup
    └── main.py            # CLI entry point and orchestration
```
