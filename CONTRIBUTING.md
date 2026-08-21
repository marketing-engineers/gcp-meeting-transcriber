# Contributing to GCP Meeting Transcriber

Thank you for your interest in contributing to `gcp-meeting-transcriber`!

## Mandatory Setup & Guardrails

This project processes confidential audio and transcription artifacts locally. To prevent accidental commits of sensitive data, all contributors must activate the repository's git hooks before making changes.

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/marketing-engineers/gcp-meeting-transcriber.git
cd gcp-meeting-transcriber

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Mandatory: Install local pre-commit guard
.githooks/install.sh
```

The hook automatically checks staged files against disallowed directories (`input/`, `output/`, `transcripts/`, `logs/`), sensitive extensions, large blobs (> 5 MB), and known secret patterns.

## Development Workflow

### Testing & Code Quality
Before submitting code, ensure all tests pass:

```bash
source venv/bin/activate
pytest
```

### Confidentiality and Test Fixtures
- **Never commit real client media or transcripts:** All files under `input/`, `output/`, `transcripts/`, and `logs/` are gitignored and blocked by git hooks.
- **Use synthetic test fixtures:** If you add or modify tests, use generic synthetic names (e.g. `sample-meeting`, `example-bucket`) and generated mock audio/responses. Never hardcode real corporate identifiers or bucket names in test files.

## Pull Request Guidelines

1. Ensure strict type hints (`list[str]`, etc.) and Python 3.10+ compatibility.
2. Keep logging and progress output aligned with the project's terminal/file logger architecture (`TerminalProgressHandler` vs `CleanFileFormatter`).
3. Ensure no `.env`, credentials, or temporary files are included.
4. Run `pytest` locally to confirm 100% pass rate.
