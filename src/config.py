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

CHUNK_LENGTH_SECONDS: int = _get_int("CHUNK_LENGTH_SECONDS", 900)
SUPPORTED_EXTENSIONS: list[str] = _get_csv_list("SUPPORTED_EXTENSIONS", ".mp4,.m4a,.wav,.mov,.mkv")
DUTCH_LANGUAGE_CODES: list[str] = _get_csv_list("DUTCH_LANGUAGE_CODES", "nl-nl")
ENGLISH_LANGUAGE_CODES: list[str] = _get_csv_list("ENGLISH_LANGUAGE_CODES", "en-us,en-gb")
ALL_LANGUAGE_CODES: list[str] = DUTCH_LANGUAGE_CODES + ENGLISH_LANGUAGE_CODES
OUTPUT_DIR: str = _get_str("OUTPUT_DIR", "output")

# ---------------------------------------------------------------------------
# GCP Configuration (required for transcription, optional for --prepare-only)
# ---------------------------------------------------------------------------

GCP_PROJECT_ID: str | None = _get_optional_str("GCP_PROJECT_ID")
GCS_BUCKET_NAME: str | None = _get_optional_str("GCS_BUCKET_NAME")
GCP_REGION: str = _get_str("GCP_REGION", "eu")
TRANSCRIPTION_MODEL: str = _get_str("TRANSCRIPTION_MODEL", "chirp_3")
TRANSCRIPT_OUTPUT_DIR: str = _get_str("TRANSCRIPT_OUTPUT_DIR", "transcripts")
TRANSCRIPTION_TIMEOUT: int = _get_int("TRANSCRIPTION_TIMEOUT", 600)


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
