import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def _get_csv_list(key: str, default: str) -> list[str]:
    """Parses a comma-separated environment variable into a list of stripped, lowercase strings."""
    raw = os.environ.get(key, default)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _get_int(key: str, default: int) -> int:
    """Retrieves an integer environment variable, falling back to a default on parse failure."""
    raw = os.environ.get(key, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.error("%s must be an integer. Falling back to %d.", key, default)
        return default


def _get_str(key: str, default: str) -> str:
    """Retrieves a string environment variable, falling back to a default."""
    return os.environ.get(key, default).strip()


def _get_optional_str(key: str) -> str | None:
    """Retrieves an optional string environment variable. Returns None if unset or empty."""
    val = os.environ.get(key, "").strip()
    return val if val else None


# ---------------------------------------------------------------------------
# Audio Processing Configuration
# ---------------------------------------------------------------------------

CHUNK_LENGTH_SECONDS_ENGLISH: int = _get_int("CHUNK_LENGTH_SECONDS_ENGLISH", 900)
CHUNK_LENGTH_SECONDS_DUTCH: int = _get_int("CHUNK_LENGTH_SECONDS_DUTCH", 300)
SUPPORTED_EXTENSIONS: list[str] = _get_csv_list("SUPPORTED_EXTENSIONS", ".mp4,.m4a,.wav,.mov,.mkv")

# Language Routing (Group-first naming, plural for lists)
LANGUAGE_CODES_DUTCH: list[str] = _get_csv_list("LANGUAGE_CODES_DUTCH", "nl-nl")
LANGUAGE_CODES_ENGLISH: list[str] = _get_csv_list("LANGUAGE_CODES_ENGLISH", "en-us,en-gb")
LANGUAGE_CODES_ALL: list[str] = LANGUAGE_CODES_DUTCH + LANGUAGE_CODES_ENGLISH

# Directory Configuration
OUTPUT_DIR: str = _get_str("OUTPUT_DIR", "output")
TRANSCRIPT_OUTPUT_DIR: str = _get_str("TRANSCRIPT_OUTPUT_DIR", "transcripts")

# Transcript Formatting Configuration
TRANSCRIPTION_MAX_CHARS_PER_LINE: int = _get_int("TRANSCRIPTION_MAX_CHARS_PER_LINE", 120)
TRANSCRIPTION_MAX_SENTENCES_PER_LINE: int = _get_int("TRANSCRIPTION_MAX_SENTENCES_PER_LINE", 3)

# ---------------------------------------------------------------------------
# GCP Configuration (required for transcription, optional for --prepare-only)
# ---------------------------------------------------------------------------

GCP_PROJECT_ID: str | None = _get_optional_str("GCP_PROJECT_ID")
GCS_BUCKET_NAME: str | None = _get_optional_str("GCS_BUCKET_NAME")
GCP_REGION: str = _get_str("GCP_REGION", "eu")
TRANSCRIPTION_MODEL: str = _get_str("TRANSCRIPTION_MODEL", "chirp_3")
TRANSCRIPTION_BATCH_SIZE: int = _get_int("TRANSCRIPTION_BATCH_SIZE", 5)
TRANSCRIPTION_TIMEOUT_SECONDS: int = _get_int("TRANSCRIPTION_TIMEOUT_SECONDS", 1800)
TRANSCRIPTION_MAX_TIMEOUT_SECONDS: int = _get_int("TRANSCRIPTION_MAX_TIMEOUT_SECONDS", 1800)
GCS_UPLOAD_TIMEOUT_SECONDS: int = _get_int("GCS_UPLOAD_TIMEOUT_SECONDS", 600)
TRANSCRIPTION_CUSTOM_PROMPT: str | None = _get_optional_str("TRANSCRIPTION_CUSTOM_PROMPT")


def validate_extension(file_path: str) -> bool:
    """Checks if a given file has a supported extension."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in SUPPORTED_EXTENSIONS


def validate_gcp_config() -> None:
    """
    Validates that required GCP configuration is present.
    Raises ValueError with actionable message if missing.
    """
    missing: list[str] = []
    if not GCP_PROJECT_ID:
        missing.append("GCP_PROJECT_ID")
    if not GCS_BUCKET_NAME:
        missing.append("GCS_BUCKET_NAME")

    if missing:
        raise ValueError(
            f"Missing required GCP configuration: {', '.join(missing)}. "
            f"Set these in your .env file. See .env.example for reference."
        )
