import logging
import os
from typing import List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def _get_csv_list(key: str, default: str) -> List[str]:
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


# ---------------------------------------------------------------------------
# Exposed configuration constants — imported by other modules
# ---------------------------------------------------------------------------

CHUNK_LENGTH_SECONDS: int = _get_int("CHUNK_LENGTH_SECONDS", 900)
SUPPORTED_EXTENSIONS: List[str] = _get_csv_list("SUPPORTED_EXTENSIONS", ".mp4,.m4a,.wav,.mov,.mkv")
DUTCH_LANGUAGE_CODES: List[str] = _get_csv_list("DUTCH_LANGUAGE_CODES", "nl-nl")
ENGLISH_LANGUAGE_CODES: List[str] = _get_csv_list("ENGLISH_LANGUAGE_CODES", "en-us,en-gb")
ALL_LANGUAGE_CODES: List[str] = DUTCH_LANGUAGE_CODES + ENGLISH_LANGUAGE_CODES
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output")


def validate_extension(file_path: str) -> bool:
    """Checks if a given file has a supported extension."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in SUPPORTED_EXTENSIONS
