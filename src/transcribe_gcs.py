"""Module and CLI for transcribing audio files already uploaded to Google Cloud Storage."""

import argparse
import logging
import os
import re
import sys

from src.audio import ProcessingError
from src.config import (
    GCP_PROJECT_ID,
    GCP_REGION,
    GCS_BUCKET_NAME,
    LANGUAGE_CODES_ALL,
    LANGUAGE_CODES_DUTCH,
    LANGUAGE_CODES_ENGLISH,
    TRANSCRIPT_OUTPUT_DIR,
    TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
    TRANSCRIPTION_MODEL,
    validate_gcp_config,
)
from src.gcs import list_gcs_uris_under_prefix
from src.logger import setup_logging
from src.transcribe import transcribe_files
from src.transcript_output import (
    save_transcript,
    stitch_diarized_transcripts,
    stitch_plain_transcripts,
)

logger = logging.getLogger(__name__)


def parse_gcs_target(target: str, default_bucket: str | None = None) -> tuple[str, str]:
    """
    Parse a GCS target string into (bucket_name, prefix).

    Supports formats:
        - gs://my-bucket/path/to/prefix
        - my-bucket/path/to/prefix (if default_bucket is not provided or target starts with known bucket)
        - path/to/prefix (uses default_bucket)

    Args:
        target: Input target string.
        default_bucket: Fallback bucket name if target is just a prefix.

    Returns:
        Tuple of (bucket_name, clean_prefix).

    Raises:
        ProcessingError: If bucket name cannot be resolved.
    """
    clean = target.strip()
    if clean.startswith("gs://"):
        without_scheme = clean[5:]
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix.strip("/")

    if default_bucket:
        if clean.startswith(f"{default_bucket}/"):
            return default_bucket, clean[len(default_bucket) + 1:].strip("/")
        if clean == default_bucket:
            return default_bucket, ""
        return default_bucket, clean.strip("/")

    # No default bucket provided, try splitting on first slash
    parts = clean.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1].strip("/")

    raise ProcessingError(
        f"Cannot determine GCS bucket and prefix from '{target}'. "
        f"Use format: gs://<bucket-name>/<prefix-path>"
    )


def derive_base_name(prefix: str, gcs_uris: list[str]) -> str:
    """
    Derive a clean base filename for transcript outputs.

    Strips timestamp suffixes (e.g., _20260820_152628) and chunk identifiers (_chunk_000).

    Args:
        prefix: GCS prefix folder name.
        gcs_uris: List of GCS object URIs found.

    Returns:
        Clean base name string.
    """
    candidate = prefix.strip("/").split("/")[-1] if prefix else ""
    
    # Strip timestamp suffix like _20260820_152628
    candidate = re.sub(r"_\d{8}_\d{6}$", "", candidate)

    if not candidate and gcs_uris:
        first_file = gcs_uris[0].split("/")[-1]
        name_no_ext = os.path.splitext(first_file)[0]
        candidate = re.sub(r"_chunk_\d+$", "", name_no_ext)
        candidate = re.sub(r"_\d{8}_\d{6}$", "", candidate)

    return candidate if candidate else "gcs_transcription"


def transcribe_from_gcs(
    target: str,
    lang: str,
    speaker_count: int | None = None,
) -> tuple[str, str]:
    """
    Transcribe audio files already residing in a GCS prefix.

    Pipeline steps:
        1. Validate GCP configuration.
        2. Resolve bucket name and prefix.
        3. List and sort all audio files under prefix.
        4. Route by language (diarized for English, non-diarized for Dutch).
        5. Submit batch transcription requests with bounded concurrency and 429 retry.
        6. Stitch results and save plain text transcript and raw JSON.

    Args:
        target: GCS URI (gs://bucket/prefix) or prefix under default bucket.
        lang: Language code (e.g. 'nl-NL', 'en-US').
        speaker_count: Optional speaker count hint for English diarization.

    Returns:
        Tuple of (transcript_file_path, json_file_path).

    Raises:
        ProcessingError: On invalid input, API errors, or configuration issues.
    """
    validate_gcp_config()
    assert GCP_PROJECT_ID is not None
    assert GCS_BUCKET_NAME is not None

    bucket_name, prefix = parse_gcs_target(target, default_bucket=GCS_BUCKET_NAME)
    gcs_uris = list_gcs_uris_under_prefix(bucket_name, prefix)

    lang_lower = lang.lower()
    base_name = derive_base_name(prefix, gcs_uris)

    logger.info(
        "Initiating GCS transcription for %d file(s) under gs://%s/%s/ (base name: %s, lang: %s)...",
        len(gcs_uris), bucket_name, prefix, base_name, lang,
    )

    if lang_lower in LANGUAGE_CODES_DUTCH:
        responses = transcribe_files(
            gcs_uris=gcs_uris,
            lang=lang,
            enable_diarization=False,
            speaker_count=None,
            project_id=GCP_PROJECT_ID,
            region=GCP_REGION,
            model=TRANSCRIPTION_MODEL,
            max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
        )
        logger.info("Transcription complete. Processing Dutch results...")
        transcript_text = stitch_plain_transcripts(responses)
    elif lang_lower in LANGUAGE_CODES_ENGLISH:
        responses = transcribe_files(
            gcs_uris=gcs_uris,
            lang=lang,
            enable_diarization=True,
            speaker_count=speaker_count,
            project_id=GCP_PROJECT_ID,
            region=GCP_REGION,
            model=TRANSCRIPTION_MODEL,
            max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
        )
        logger.info("Transcription complete. Processing English results...")
        transcript_text = stitch_diarized_transcripts(responses)
    else:
        raise ProcessingError(
            f"Unsupported language route: {lang}. Configured languages: {', '.join(LANGUAGE_CODES_ALL)}"
        )

    txt_path, json_path = save_transcript(
        transcript_text=transcript_text,
        raw_responses=responses,
        output_dir=TRANSCRIPT_OUTPUT_DIR,
        base_name=base_name,
        lang=lang,
    )

    return txt_path, json_path


def main() -> None:
    """CLI entrypoint for standalone GCS re-transcription."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Transcribe audio files already uploaded to Google Cloud Storage.",
    )
    parser.add_argument(
        "gcs_target",
        help="GCS URI (e.g., gs://your-transcriber-bucket/my-recording_20260820_152628) or prefix name.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=[code.lower() for code in LANGUAGE_CODES_ALL],
        type=str.lower,
        help="Language of the recording (determines processing route).",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Optional speaker count hint for English diarization.",
    )

    args = parser.parse_args()

    logger.info("Starting GCS Transcription for: %s", args.gcs_target)

    try:
        txt_path, json_path = transcribe_from_gcs(
            target=args.gcs_target,
            lang=args.lang,
            speaker_count=args.speakers,
        )
        logger.info("Transcription completed successfully.")
        logger.info("Transcript: %s", txt_path)
        logger.info("Raw JSON: %s", json_path)
    except ProcessingError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
