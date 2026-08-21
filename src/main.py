import argparse
import logging
import os
import sys
from datetime import datetime

from src.config import (
    CHUNK_MAX_LENGTH_SECONDS_DUTCH,
    CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
    CHUNK_MIN_LENGTH_SECONDS,
    CHUNK_SILENCE_MIN_DURATION_SECONDS,
    CHUNK_SILENCE_NOISE_THRESHOLD_DB,
    CHUNK_SILENCE_SEARCH_WINDOW_SECONDS,
    GCP_PROJECT_ID,
    GCP_REGION,
    GCS_BUCKET_NAME,
    GCS_UPLOAD_TIMEOUT_SECONDS,
    LANGUAGE_CODES_ALL,
    LANGUAGE_CODES_DUTCH,
    LANGUAGE_CODES_ENGLISH,
    OUTPUT_DIR,
    TRANSCRIPT_OUTPUT_DIR,
    TRANSCRIPTION_CONCURRENCY,
    TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
    TRANSCRIPTION_MODEL,
    validate_extension,
    validate_gcp_config,
)
from src.audio import (
    ProcessingError,
    compute_chunk_boundaries,
    detect_silences,
    extract_audio_to_flac,
    get_audio_duration,
    split_flac,
)
from src.gcs import delete_gcs_prefix, upload_files_to_gcs, upload_to_gcs
from src.transcribe import transcribe_files, transcribe_single_file
from src.transcript_output import (
    save_transcript,
    stitch_diarized_transcripts,
    stitch_plain_transcripts,
)
from src.transcribe_gcs import transcribe_from_gcs
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
        3. Detect silences and compute adaptive chunk boundaries
        4. Split into chunks if N > 1
        5. [If not prepare_only] Upload to GCS
        6. [If not prepare_only] Transcribe via Speech-to-Text V2
        7. [If not prepare_only] Process and save transcript
        8. Clean up temporary files

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

    if lang_lower in LANGUAGE_CODES_DUTCH:
        _process_dutch(input_path, base_name, lang, prepare_only)
    elif lang_lower in LANGUAGE_CODES_ENGLISH:
        _process_english(input_path, base_name, lang, prepare_only)
    else:
        raise ProcessingError(
            f"Unsupported language route: {lang}. "
            f"Configured languages: {', '.join(LANGUAGE_CODES_ALL)}"
        )


def _process_dutch(input_path: str, base_name: str, lang: str, prepare_only: bool) -> None:
    """
    Route 1: Dutch — adaptive one-wave parallel chunking, no diarization.
    """
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")
    logger.info("Processing route for Dutch (%s): Adaptive parallel wave.", lang)

    extract_audio_to_flac(input_path, final_output)
    duration = get_audio_duration(final_output)

    silences = detect_silences(
        final_output,
        noise_threshold=CHUNK_SILENCE_NOISE_THRESHOLD_DB,
        min_duration=CHUNK_SILENCE_MIN_DURATION_SECONDS,
    )
    boundaries = compute_chunk_boundaries(
        duration=duration,
        max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_DUTCH,
        silences=silences,
        window=CHUNK_SILENCE_SEARCH_WINDOW_SECONDS,
        min_chunk_length=CHUNK_MIN_LENGTH_SECONDS,
        target_parallelism=TRANSCRIPTION_CONCURRENCY,
    )

    is_chunked = len(boundaries) > 0
    chunk_paths: list[str] = []

    if is_chunked:
        logger.info(
            "Audio duration (%.2fs) divided into %d chunks with adaptive pause snapping.",
            duration, len(boundaries) + 1,
        )
        chunk_paths = split_flac(final_output, OUTPUT_DIR, base_name, boundaries)

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
        logger.info(
            "Audio duration (%.2fs) within monolithic limit (%ds). Proceeding with monolithic file.",
            duration, CHUNK_MAX_LENGTH_SECONDS_DUTCH,
        )
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
            gcs_uris = upload_files_to_gcs(chunk_paths, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS)

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
                max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
                concurrency=TRANSCRIPTION_CONCURRENCY,
                total_audio_seconds=duration,
                chunk_duration=duration / len(chunk_paths),
            )

            # Stitch and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_plain_transcripts(responses)
            save_transcript(
                transcript_text=transcript_text,
                raw_responses=responses,
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
            )
        else:
            # Upload monolithic FLAC to GCS
            gcs_uri = upload_to_gcs(final_output, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS)

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
                max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
                total_audio_seconds=duration,
                chunk_duration=duration,
            )

            # Process and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_plain_transcripts([response])
            save_transcript(
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
    Route 2: English — ceiling-driven chunking to minimise boundaries, diarization enabled.
    """
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")
    logger.info("Processing route for English (%s): Ceiling-driven chunking with diarization.", lang)

    extract_audio_to_flac(input_path, final_output)
    duration = get_audio_duration(final_output)

    silences = detect_silences(
        final_output,
        noise_threshold=CHUNK_SILENCE_NOISE_THRESHOLD_DB,
        min_duration=CHUNK_SILENCE_MIN_DURATION_SECONDS,
    )
    boundaries = compute_chunk_boundaries(
        duration=duration,
        max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
        silences=silences,
        window=CHUNK_SILENCE_SEARCH_WINDOW_SECONDS,
        min_chunk_length=CHUNK_MIN_LENGTH_SECONDS,
        target_parallelism=None,
    )

    is_chunked = len(boundaries) > 0
    chunk_paths: list[str] = []

    if is_chunked:
        logger.info(
            "Audio duration (%.2fs) divided into %d chunks (ceiling: %ds).",
            duration, len(boundaries) + 1, CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
        )
        chunk_paths = split_flac(final_output, OUTPUT_DIR, base_name, boundaries)

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
            if os.path.exists(final_output):
                os.remove(final_output)
            return
    else:
        logger.info(
            "Audio duration (%.2fs) within monolithic limit (%ds). Proceeding with monolithic file.",
            duration, CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
        )
        if prepare_only:
            logger.info("-------------------------------------------------------------------")
            logger.info("READY FOR GCP: %s", final_output)
            logger.info("Upload this file to GCP. ENABLE Speaker Diarization.")
            logger.info("-------------------------------------------------------------------")
            return

    # Interactive speaker count prompt (after audio prep, before upload)
    speaker_count = _prompt_speaker_count()

    # --- Full pipeline: upload, transcribe, stitch, save, cleanup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_prefix = f"{base_name}_{timestamp}"

    try:
        if is_chunked:
            # Upload chunks to GCS
            gcs_uris = upload_files_to_gcs(chunk_paths, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS)

            # Transcribe all chunks
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
                max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
                concurrency=TRANSCRIPTION_CONCURRENCY,
                total_audio_seconds=duration,
                chunk_duration=duration / len(chunk_paths),
            )

            # Stitch and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_diarized_transcripts(responses)
            save_transcript(
                transcript_text=transcript_text,
                raw_responses=responses,
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
            )
        else:
            # Upload monolithic FLAC to GCS
            gcs_uri = upload_to_gcs(final_output, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS)

            # Transcribe single monolithic file
            speaker_str = str(speaker_count) if speaker_count else "auto-detect"
            logger.info(
                "Starting transcription (model: %s, lang: %s, diarization: enabled, speakers: %s, monolithic)...",
                TRANSCRIPTION_MODEL, lang, speaker_str,
            )
            response = transcribe_single_file(
                gcs_uri=gcs_uri,
                lang=lang,
                enable_diarization=True,
                speaker_count=speaker_count,
                project_id=GCP_PROJECT_ID,
                region=GCP_REGION,
                model=TRANSCRIPTION_MODEL,
                max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
                total_audio_seconds=duration,
                chunk_duration=duration,
            )

            # Process and save transcript
            logger.info("Transcription complete. Processing results...")
            transcript_text = stitch_diarized_transcripts([response])
            save_transcript(
                transcript_text=transcript_text,
                raw_responses=[response],
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
            )

    finally:
        # Cleanup: GCS objects
        logger.info("Cleaning up temporary files...")
        delete_gcs_prefix(GCS_BUCKET_NAME, gcs_prefix)

        # Cleanup: local chunk files
        if is_chunked:
            for chunk_path in chunk_paths:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            logger.info("Removed %d local chunk file(s).", len(chunk_paths))

        # Cleanup local monolithic FLAC
        if os.path.exists(final_output):
            os.remove(final_output)
            logger.info("Removed local FLAC: %s", final_output)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Marketing Engineers: Transcription Automator v0.2",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to the input media file (e.g., input/meeting.mp4)",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=[code.lower() for code in LANGUAGE_CODES_ALL],
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
    parser.add_argument(
        "--from-gcs",
        metavar="GCS_TARGET",
        default=None,
        help="Transcribe audio files already uploaded to GCS (prefix or gs://bucket/prefix).",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Optional speaker count hint for English diarization when using --from-gcs.",
    )

    args = parser.parse_args()

    logger.info("Starting Marketing Engineers Transcription Automator v0.2")

    if not args.input_file and not args.from_gcs:
        parser.error("Either input_file or --from-gcs must be specified.")

    if args.input_file and args.from_gcs:
        parser.error("Cannot specify both input_file and --from-gcs.")

    try:
        if args.from_gcs:
            speaker_count = args.speakers
            if speaker_count is None and args.lang in LANGUAGE_CODES_ENGLISH:
                speaker_count = _prompt_speaker_count()
            txt_path, json_path = transcribe_from_gcs(
                target=args.from_gcs,
                lang=args.lang,
                speaker_count=speaker_count,
            )
            logger.info("Transcript saved to: %s", txt_path)
            logger.info("Raw API response saved to: %s", json_path)
        else:
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
