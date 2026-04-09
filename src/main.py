import argparse
import logging
import os
import sys

from src.config import (
    ALL_LANGUAGE_CODES,
    CHUNK_LENGTH_SECONDS,
    DUTCH_LANGUAGE_CODES,
    ENGLISH_LANGUAGE_CODES,
    OUTPUT_DIR,
    validate_extension,
)
from src.audio import extract_audio_to_flac, split_flac
from src.logger import setup_logging

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Raised when audio processing fails for a known, handled reason."""


def process_file(input_path: str, lang: str) -> None:
    """
    Orchestrates the audio processing for a single file based on language constraints.

    Args:
        input_path: Path to the source media file.
        lang: Language code determining the processing route.

    Raises:
        ProcessingError: On validation failures or processing errors.
    """
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")

    if not validate_extension(input_path):
        raise ProcessingError(f"Unsupported file extension for: {input_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    lang_lower = lang.lower()

    if lang_lower in DUTCH_LANGUAGE_CODES:
        _process_dutch(input_path, base_name, lang)
    elif lang_lower in ENGLISH_LANGUAGE_CODES:
        _process_english(input_path, base_name, lang)
    else:
        raise ProcessingError(
            f"Unsupported language route: {lang}. "
            f"Configured languages: {', '.join(ALL_LANGUAGE_CODES)}"
        )


def _process_dutch(input_path: str, base_name: str, lang: str) -> None:
    """Route 1: Dutch — monolithic FLAC, no diarization."""
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")
    logger.info("Processing route for Dutch (%s): Monolithic file without diarization.", lang)

    extract_audio_to_flac(input_path, final_output)

    logger.info("-------------------------------------------------------------------")
    logger.info("READY FOR GCP: %s", final_output)
    logger.info("Upload this file to GCP. Do NOT enable Speaker Diarization.")
    logger.info("-------------------------------------------------------------------")


def _process_english(input_path: str, base_name: str, lang: str) -> None:
    """Route 2: English — chunked FLACs, diarization enabled."""
    temp_flac = os.path.join(OUTPUT_DIR, f".{base_name}_temp.flac")
    logger.info("Processing route for English (%s): Chunked files with diarization.", lang)

    try:
        # 1. Extract to temporary monolithic FLAC
        extract_audio_to_flac(input_path, temp_flac)

        # 2. Split into chunks — pass the original base_name so chunks are named cleanly
        split_flac(temp_flac, OUTPUT_DIR, base_name, CHUNK_LENGTH_SECONDS)

        logger.info("-------------------------------------------------------------------")
        logger.info("READY FOR GCP: Files generated in %s/ matching %s_chunk_*.flac", OUTPUT_DIR, base_name)
        logger.info("Upload these chunked files to GCP. ENABLE Speaker Diarization (2 speakers).")
        logger.info("Note: You will need to manually stitch the resulting transcripts until v0.2.")
        logger.info("-------------------------------------------------------------------")

    finally:
        # 3. Always clean up temp file
        if os.path.exists(temp_flac):
            os.remove(temp_flac)
            logger.info("Cleaned up temporary file: %s", temp_flac)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Marketing Engineers: Transcription Automator v0.1",
    )
    parser.add_argument(
        "input_file",
        help="Path to the input media file (e.g., input/meeting.mp4)",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=[code.lower() for code in ALL_LANGUAGE_CODES],
        type=str.lower,
        help="Language of the recording (determines processing route)",
    )

    args = parser.parse_args()

    logger.info("Starting Marketing Engineers Transcription Automator v0.1")

    try:
        process_file(args.input_file, args.lang)
    except ProcessingError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)

    logger.info("Finished execution cleanly.")


if __name__ == "__main__":
    main()
