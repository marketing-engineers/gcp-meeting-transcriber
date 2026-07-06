import argparse
import logging
import os
import sys
from datetime import datetime

from src.config import (
    ALL_LANGUAGE_CODES,
    CHUNK_LENGTH_SECONDS,
    DUTCH_LANGUAGE_CODES,
    ENGLISH_LANGUAGE_CODES,
    GCP_PROJECT_ID,
    GCP_REGION,
    GCS_BUCKET_NAME,
    OUTPUT_DIR,
    TRANSCRIPT_OUTPUT_DIR,
    TRANSCRIPTION_MODEL,
    TRANSCRIPTION_TIMEOUT,
    GCS_UPLOAD_TIMEOUT,
    validate_extension,
    validate_gcp_config,
)
from src.audio import ProcessingError, extract_audio_to_flac, split_flac, get_audio_duration
from src.gcs import delete_gcs_prefix, upload_files_to_gcs, upload_to_gcs
from src.transcribe import transcribe_files, transcribe_single_file
from src.transcript_output import (
    format_plain_transcript,
    save_transcript,
    stitch_diarized_transcripts,
    stitch_plain_transcripts,
)
from src.logger import setup_logging

logger = logging.getLogger(__name__)


def _prompt_speaker_count() -> int | None:
    """
    Interactively prompt the user for the number of speakers in the recording.

    Only called for diarization-supported languages (English).
    Skipped when stdin is not a TTY (CI/pipe contexts default to auto-detect).

    Returns:
        int if the user provides a valid positive integer, None for auto-detect.
    """
    if not sys.stdin.isatty():
        logger.info("Non-interactive mode detected. Using auto-detect for speaker count.")
        return None

    while True:
        try:
            raw = input("How many speakers were in this recording? (press Enter to auto-detect): ")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D / Ctrl+C gracefully
            print()  # newline after ^C
            logger.info("Using auto-detect for speaker count.")
            return None

        raw = raw.strip()

        if not raw:
            logger.info("Speaker count: auto-detect (empty input).")
            return None

        try:
            count = int(raw)
            if count < 1:
                print("Please enter a positive integer (1 or greater), or press Enter to auto-detect.")
                continue
            logger.info("Speaker count: %d (user-provided).", count)
            return count
        except ValueError:
            print("Invalid input. Please enter a number or press Enter to auto-detect.")


def process_file(input_path: str, lang: str, prepare_only: bool = False) -> None:
    """
    Orchestrates the full audio processing and transcription pipeline for a single file.

    Pipeline steps:
        1. Validate input
        2. Extract audio to FLAC
        3. Split into chunks if needed (English only)
        4. [If not prepare_only] Upload to GCS
        5. [If not prepare_only] Transcribe via Speech-to-Text V2
        6. [If not prepare_only] Process and save transcript
        7. Clean up temporary files

    Args:
        input_path: Path to the source media file.
        lang: Language code determining the processing route.
        prepare_only: If True, stop after FLAC preparation (v0.1 behavior).

    Raises:
        ProcessingError: On validation failures or processing errors.
    """
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")

    if not validate_extension(input_path):
        raise ProcessingError(f"Unsupported file extension for: {input_path}")

    # Validate GCP config early (before doing any work) unless prepare-only
    if not prepare_only:
        validate_gcp_config()
        assert GCP_PROJECT_ID is not None
        assert GCS_BUCKET_NAME is not None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    lang_lower = lang.lower()

    if lang_lower in DUTCH_LANGUAGE_CODES:
        _process_dutch(input_path, base_name, lang, prepare_only)
    elif lang_lower in ENGLISH_LANGUAGE_CODES:
        _process_english(input_path, base_name, lang, prepare_only)
    else:
        raise ProcessingError(
            f"Unsupported language route: {lang}. "
            f"Configured languages: {', '.join(ALL_LANGUAGE_CODES)}"
        )


def _process_dutch(input_path: str, base_name: str, lang: str, prepare_only: bool) -> None:
    """
    Route 1: Dutch — monolithic FLAC unless > 60 minutes, no diarization.

    Full pipeline: extract -> upload -> transcribe (no diarization) -> save transcript.
    """
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")
    logger.info("Processing route for Dutch (%s).", lang)

    extract_audio_to_flac(input_path, final_output)

    duration = get_audio_duration(final_output)
    is_chunked = duration > 3600.0

    if is_chunked:
        logger.info("Audio duration (%.2fs) exceeds 60 minutes limit. Switching to chunked mode.", duration)
        chunk_paths = split_flac(final_output, OUTPUT_DIR, base_name, CHUNK_LENGTH_SECONDS)

        if prepare_only:
            logger.info("-------------------------------------------------------------------")
            logger.info(
                "READY FOR GCP: Files generated in %s/ matching %s_chunk_*.flac",
                OUTPUT_DIR, base_name,
            )
            logger.info("Upload these chunked files to GCP. Do NOT enable Speaker Diarization.")
            logger.info(
                "Note: You will need to manually stitch the resulting transcripts "
                "until the full pipeline is used (without --prepare-only)."
            )
            logger.info("-------------------------------------------------------------------")
            if os.path.exists(final_output):
                os.remove(final_output)
            return
    else:
        logger.info("Audio duration (%.2fs) is within 60 minutes limit. Proceeding with monolithic file.", duration)
        if prepare_only:
            logger.info("-------------------------------------------------------------------")
            logger.info("READY FOR GCP: %s", final_output)
            logger.info("Upload this file to GCP. Do NOT enable Speaker Diarization.")
            logger.info("-------------------------------------------------------------------")
            return

    # --- Full pipeline: upload, transcribe, save, cleanup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_prefix = f"{base_name}_{timestamp}"

    try:
        if is_chunked:
            # Upload chunks to GCS
            gcs_uris = upload_files_to_gcs(chunk_paths, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT)

            # Transcribe all chunks
            logger.info(
                "Starting transcription (model: %s, lang: %s, diarization: disabled, chunked)...",
                TRANSCRIPTION_MODEL, lang,
            )
            responses = transcribe_files(
                gcs_uris=gcs_uris,
                lang=lang,
                enable_diarization=False,
                speaker_count=None,
                project_id=GCP_PROJECT_ID,
                region=GCP_REGION,
                model=TRANSCRIPTION_MODEL,
                timeout=TRANSCRIPTION_TIMEOUT,
            )

            # Stitch and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_plain_transcripts(responses)
            txt_path, json_path = save_transcript(
                transcript_text=transcript_text,
                raw_responses=responses,
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
            )
        else:
            # Upload monolithic FLAC to GCS
            gcs_uri = upload_to_gcs(final_output, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT)

            # Transcribe single monolithic file
            logger.info(
                "Starting transcription (model: %s, lang: %s, diarization: disabled, monolithic)...",
                TRANSCRIPTION_MODEL, lang,
            )
            response = transcribe_single_file(
                gcs_uri=gcs_uri,
                lang=lang,
                enable_diarization=False,
                speaker_count=None,
                project_id=GCP_PROJECT_ID,
                region=GCP_REGION,
                model=TRANSCRIPTION_MODEL,
                timeout=TRANSCRIPTION_TIMEOUT,
            )

            # Process and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_plain_transcripts([response])
            txt_path, json_path = save_transcript(
                transcript_text=transcript_text,
                raw_responses=[response],
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
            )

    finally:
        # Cleanup GCS objects
        logger.info("Cleaning up temporary GCS files...")
        delete_gcs_prefix(GCS_BUCKET_NAME, gcs_prefix)

        # Cleanup local chunk files if any
        if is_chunked:
            for chunk_path in chunk_paths:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            logger.info("Removed %d local chunk file(s).", len(chunk_paths))

        # Cleanup local monolithic FLAC
        if os.path.exists(final_output):
            os.remove(final_output)
            logger.info("Removed local FLAC: %s", final_output)


def _process_english(input_path: str, base_name: str, lang: str, prepare_only: bool) -> None:
    """
    Route 2: English — chunked FLACs, diarization enabled.

    Full pipeline: extract -> chunk -> prompt speakers -> upload -> transcribe -> stitch -> save.
    """
    temp_flac = os.path.join(OUTPUT_DIR, f".{base_name}_temp.flac")
    logger.info("Processing route for English (%s): Chunked files with diarization.", lang)

    try:
        # 1. Extract to temporary monolithic FLAC
        extract_audio_to_flac(input_path, temp_flac)

        # 2. Split into chunks
        chunk_paths = split_flac(temp_flac, OUTPUT_DIR, base_name, CHUNK_LENGTH_SECONDS)
        logger.info("Splitting into 15-minute chunks (%d chunks created).", len(chunk_paths))

        if prepare_only:
            logger.info("-------------------------------------------------------------------")
            logger.info(
                "READY FOR GCP: Files generated in %s/ matching %s_chunk_*.flac",
                OUTPUT_DIR, base_name,
            )
            logger.info("Upload these chunked files to GCP. ENABLE Speaker Diarization.")
            logger.info(
                "Note: You will need to manually stitch the resulting transcripts "
                "until the full pipeline is used (without --prepare-only)."
            )
            logger.info("-------------------------------------------------------------------")
            return

        # 3. Interactive speaker count prompt (after audio prep, before upload)
        speaker_count = _prompt_speaker_count()

        # --- Full pipeline: upload, transcribe, stitch, save, cleanup ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_prefix = f"{base_name}_{timestamp}"

        try:
            # 4. Upload chunks to GCS
            gcs_uris = upload_files_to_gcs(chunk_paths, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT)

            # 5. Transcribe all chunks
            speaker_str = str(speaker_count) if speaker_count else "auto-detect"
            logger.info(
                "Starting transcription (model: %s, lang: %s, diarization: enabled, speakers: %s)...",
                TRANSCRIPTION_MODEL, lang, speaker_str,
            )
            responses = transcribe_files(
                gcs_uris=gcs_uris,
                lang=lang,
                enable_diarization=True,
                speaker_count=speaker_count,
                project_id=GCP_PROJECT_ID,
                region=GCP_REGION,
                model=TRANSCRIPTION_MODEL,
                timeout=TRANSCRIPTION_TIMEOUT,
            )

            # 6. Stitch and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_diarized_transcripts(responses)
            txt_path, json_path = save_transcript(
                transcript_text=transcript_text,
                raw_responses=responses,
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
            )

        finally:
            # Cleanup: GCS objects
            logger.info("Cleaning up temporary files...")
            delete_gcs_prefix(GCS_BUCKET_NAME, gcs_prefix)
            # Cleanup: local chunk files
            for chunk_path in chunk_paths:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            logger.info("Removed %d local chunk file(s).", len(chunk_paths))

    finally:
        # Always clean up temp monolithic FLAC
        if os.path.exists(temp_flac):
            os.remove(temp_flac)
            logger.info("Cleaned up temporary file: %s", temp_flac)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Marketing Engineers: Transcription Automator v0.2",
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
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        default=False,
        help="Stop after generating FLAC files (v0.1 behavior). "
             "Files can then be manually uploaded to GCP Console.",
    )

    args = parser.parse_args()

    logger.info("Starting Marketing Engineers Transcription Automator v0.2")

    try:
        process_file(args.input_file, args.lang, prepare_only=args.prepare_only)
    except ProcessingError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
