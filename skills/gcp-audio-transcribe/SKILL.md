---
name: gcp-audio-transcribe
description: Automates audio extraction, chunking, and speech-to-text transcription for meeting recordings and voice memos using Google Cloud Speech-to-Text V2 (Chirp 3).
compatibility: Linux or macOS with Python 3.10+, FFmpeg, and Google Cloud credentials configured.
---

# GCP Audio Transcribe Skill

This skill provides autonomous transcription of audio and video recordings (such as Google Meet `.mp4`, voice memos `.m4a`, `.wav`, `.mov`, `.mkv`) using Google Cloud Speech-to-Text V2 (Chirp 3) and Google Cloud Storage.

## Prerequisites

1. **Python Virtual Environment:** The project repository contains a dedicated virtual environment at `venv/`. Always invoke commands using the virtual environment binary (`venv/bin/python`).
2. **FFmpeg & FFprobe:** Installed and accessible on `PATH`.
3. **Environment & GCP Configuration:** The project `.env` file must be provisioned by the system operator prior to execution.

## Directory Layout

- `input/`: Source media files for transcription (e.g. `input/meeting.mp4`).
- `output/`: Temporary working directory for FLAC extraction and chunking (managed automatically).
- `transcripts/`: Final transcript outputs saved as `<basename>_<lang>.txt` (human-readable, line-bounded text) and `<basename>_<lang>_raw.json` (GCP API metadata).

## Core Commands

All commands must be executed from the repository root folder.

### 1. Validate Authentication & Permissions
Before initiating a new transcription job, verify GCP credentials and bucket access:
```bash
venv/bin/python -m src.main --check-auth
```
- If the command exits with code `0`, authentication and bucket permissions are verified.
- If the command exits with code `1`, stop and report the error output to the operator. Do not attempt to fix or modify credentials.

### 2. Transcribe a Local Media File
Execute end-to-end extraction, upload, Chirp 3 transcription, and cleanup:
```bash
# Dutch (nl-NL) - Optimized for parallel wave processing (no diarization)
venv/bin/python -m src.main input/recording.mp4 --lang nl-NL

# English (en-US or en-GB) - Supports speaker diarization
venv/bin/python -m src.main input/meeting.mp4 --lang en-US
```

### 3. Transcribe Files Already in GCS (`--from-gcs`)
If audio files were already uploaded to a GCS bucket prefix:
```bash
venv/bin/python -m src.main --from-gcs gs://my-bucket/meeting_20261009_120000 --lang en-US
```

### 4. Audio Preparation Only (`--prepare-only`)
Extract audio and split into FLAC chunks without uploading to GCP or making cloud API calls:
```bash
venv/bin/python -m src.main input/recording.mp4 --lang nl-NL --prepare-only
```

## Output Delivery & Post-Processing Guidelines

### 1. Deliver as File Attachment (Required)
- **Send File Attachment:** Once transcription finishes, the primary deliverable is the generated transcript file: `transcripts/<basename>_<lang>_transcript.txt`.
- When attachments/file uploads are supported by the messaging platform or agent environment, the agent **MUST send this `.txt` file as an attachment** immediately.
- **Do NOT Paste Full Transcripts in Chat:** The agent must never dump the raw or full transcript text directly into chat messages. Dumping large transcripts causes message truncation, token bloat, formatting issues, and accidental hallucination. Only provide a concise status message confirming completion (e.g., output file location, language processed).

### 2. No Autonomous Post-Processing (Strict Guardrail)
- **Zero Hallucination / No Name Inference:** GCP Speech-to-Text V2 (Chirp 3) only produces generic numerical speaker tags (`Speaker 1:`, `Speaker 2:`) for English, and plain text without speaker tags for Dutch. GCP does not have access to attendee names.
- The agent **MUST NEVER** autonomously infer speaker identities or replace `Speaker 1`/`Speaker 2` with real person names based on conversation context or greetings.
- The agent **MUST NEVER** alter, rewrite, or summarize the transcript text prior to delivering the file attachment. The raw transcript must always be delivered untouched.

### 3. User-Directed Post-Processing Flow
- **Transcribe First, Ask Second:** The agent must follow a strict two-phase workflow:
  1. Complete transcription and immediately deliver the untouched transcript file as an attachment.
  2. Ask the user if they would like any post-processing performed (for example: identifying speakers, resolving entities against a glossary, extracting action items, or generating a meeting summary).
- Do not execute any post-processing or text manipulation unless the user explicitly requests it in a follow-up prompt.

## Critical Security Guardrails & Operational Constraints

1. **Service Account Provisioning & Least-Privilege Scope:**
   - The agent MUST NEVER attempt to create or grant GCP permissions autonomously.
   - If the operator asks how to set up or scope the service account, direct them to `docs/hermes-deployment.md`.
   - The required minimum permissions are:
     - Project level: `speech.recognizers.recognize`, `speech.operations.get`, `serviceusage.services.use`.
     - Dedicated bucket level only: `storage.objects.create`, `storage.objects.get`, `storage.objects.list`, `storage.objects.delete`.
     - No broad admin roles (`speech.admin`, `storage.admin`, or project-wide storage roles).

2. **Never Inspect Secrets or Keys:**
   - NEVER read, print, display, copy, or transmit the service account key file (`*.json`) or the project `.env` file.
   - NEVER inspect files in `~/.config/gcp-audio-transcribe/`.

3. **Never Execute IAM or Authentication CLI Commands:**
   - NEVER execute `gcloud auth *`, `gcloud iam *`, `gcloud projects add-iam-policy-binding`, or any IAM modification commands.
   - NEVER attempt to create, rotate, or modify Google Cloud service accounts, keys, or roles.

4. **Never Modify Authentication Mode:**
   - NEVER modify `GCP_AUTH_MODE` or attempt to switch authentication back to user ADC.

5. **Handling Failures:**
   - If `--check-auth` fails or an API call returns `PermissionDenied`, immediately stop execution and report the error message back to the human operator.
   - Do not attempt workarounds or credential modifications.

