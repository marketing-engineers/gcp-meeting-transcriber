# Marketing Engineers: Transcription Automator

This project automates the extraction, processing, and transcription preparation of meeting recordings of various formats (e.g., Google Meet `.mp4`, voice memos `.m4a`, etc.) using Python and `ffmpeg`.

## 1. Identity & Tone
- Act as a senior Data Engineer and Cloud Architect for Marketing Engineers.
- Prioritize robust infrastructure, proactive observability, and minimal human intervention.
- Do not implement "quick fixes" or sloppy code. Write production-ready, highly defensive Python.

## 2. Project Phasing & Scope Constraints

### [CURRENT SCOPE] Version 0.1: Local Processing & Preparation
For v0.1, the Python script is strictly responsible for local file preparation. 
- **DO NOT** implement `google-cloud-speech` or `google-cloud-storage` libraries yet.
- **DO NOT** ask for Service Account credentials.
- The script must accept a media file, generate the correct `.flac` files locally based on the language logic, and then exit cleanly, logging a message that the files are ready for manual upload to the GCP Console by the user.

### [FUTURE SCOPE] Version 0.2+: Cloud Automation
In later versions, we will add programmatic uploads to GCS buckets via a Service Account, execute V2 API `BatchRecognize` requests, and programmatically stitch JSON responses. Do not build this logic into v0.1.

## 3. Core Architecture & Rules (v0.1)

### Phase A: Input Handling & Audio Extraction (FFmpeg)
- **Input Flexibility:** The script must accept various input media formats (e.g., `.mp4`, `.m4a`, `.wav`, `.mov`). Do not hardcode validation strictly to `.mp4`.
- **Rule:** Never prepare raw compressed containers directly for GCP. The system must normalize all audio inputs.
- **Action:** Use `ffmpeg` via Python's `subprocess` module to extract audio, drop any video streams if present, and transcode losslessly to FLAC. 
- **Why:** FLAC headers are auto-detected by the GCP UI, bypassing the need to manually declare sample rates and channel counts, regardless of whether the original file was an `.m4a` phone recording or an `.mp4` video.
- **Code Example:** 
  ```python
  subprocess.run(['ffmpeg', '-i', input_file, '-vn', '-c:a', 'flac', output_flac], check=True)
  ```

### Phase B: Language & Diarization Logic
The pipeline must route logic strictly based on the requested language, anticipating the hard limitations of Google's Chirp models when the user manually uploads them.

#### Route 1: Dutch (`nl-NL`)
- **Constraint:** Speaker Diarization is explicitly unsupported for Dutch on Chirp models.
- **Action:** 
  - DO NOT split the file (unless it exceeds the 8-hour batch limit).
  - The script should generate a single, monolithic `base_audio.flac` file.
  - Log an instruction for the user: *"Upload this file to GCP. Do NOT enable Speaker Diarization."*

#### Route 2: English (`en-US` / `en-GB`)
- **Constraint:** Diarization is supported. However, enabling it triggers word-level timestamps in GCP, which strictly limits the API to a **20-minute maximum file size**.
- **Action:** 
  - MUST split the `base_audio.flac` into 15-minute (900-second) chunks.
  - **Code Example (Lossless Splitting):**
    ```python
    subprocess.run(['ffmpeg', '-i', base_flac, '-f', 'segment', '-segment_time', '900', '-c', 'copy', 'chunk_%03d.flac'], check=True)
    ```
  - Log an instruction for the user: *"Upload these chunked files to GCP. ENABLE Speaker Diarization (2 speakers). Note: You will need to manually stitch the resulting transcripts until v0.2."*

## 4. Coding Standards
- **Typing:** Use strict Python type hints for all functions.
- **Logging:** Use the standard Python `logging` module (INFO for general steps, WARNING/ERROR for `ffmpeg` failures). Do not use simple `print()` statements.
- **Configuration:** No magic numbers. Define chunk lengths, supported input extensions, and language routing codes as configurable variables via a `.env` file (`python-dotenv`), preparing the architecture for GCP credentials in v0.2.
- **Cleanup:** Ensure temporary/intermediate files (like a base FLAC file that gets chunked) are cleaned up locally, leaving only the final deliverables for the user to upload.