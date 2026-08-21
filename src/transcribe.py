"""Google Cloud Speech-to-Text V2 BatchRecognize integration using Chirp 3."""

import logging
import time
from typing import Any

from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as gcp_exceptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from src.audio import ProcessingError
from src.config import TRANSCRIPTION_BATCH_SIZE, TRANSCRIPTION_MAX_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# BatchRecognize accepts max 15 files per request
_MAX_FILES_PER_BATCH = 15


def _build_recognition_config(
    lang: str,
    model: str,
    enable_diarization: bool,
    speaker_count: int | None,
) -> cloud_speech.RecognitionConfig:
    """
    Build the RecognitionConfig for a BatchRecognize request.

    Args:
        lang: Language code (e.g., "en-US", "nl-NL").
        model: Model name (e.g., "chirp_3").
        enable_diarization: Whether to enable speaker diarization.
        speaker_count: Number of speakers hint. None = auto-detect.

    Returns:
        Configured RecognitionConfig.
    """
    features_kwargs: dict[str, Any] = {
        "enable_automatic_punctuation": True,
    }

    if enable_diarization:
        if speaker_count is not None:
            diarization_config = cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=speaker_count,
                max_speaker_count=speaker_count,
            )
        else:
            # Empty config = Chirp 3 auto-detects speaker count
            diarization_config = cloud_speech.SpeakerDiarizationConfig()

        features_kwargs["diarization_config"] = diarization_config

    from src.config import TRANSCRIPTION_CUSTOM_PROMPT
    if TRANSCRIPTION_CUSTOM_PROMPT:
        if "chirp" in model.lower():
            features_kwargs["custom_prompt_config"] = cloud_speech.CustomPromptConfig(
                custom_prompt=TRANSCRIPTION_CUSTOM_PROMPT
            )
        else:
            logger.warning(
                "TRANSCRIPTION_CUSTOM_PROMPT is only supported for Chirp models (current model: %s). "
                "Omitting custom prompt.",
                model,
            )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[lang],
        model=model,
        features=cloud_speech.RecognitionFeatures(**features_kwargs),
    )
    return config


def _get_speech_client(region: str) -> SpeechClient:
    """Create a Speech-to-Text V2 client with the regional endpoint."""
    from src.config import GCP_PROJECT_ID
    api_endpoint = f"{region}-speech.googleapis.com"
    client_options = ClientOptions(
        api_endpoint=api_endpoint,
        quota_project_id=GCP_PROJECT_ID,
    )
    try:
        return SpeechClient(client_options=client_options)
    except Exception as e:
        raise ProcessingError(
            f"Failed to create Speech-to-Text client (endpoint: {api_endpoint}).\n"
            f"Ensure you have authenticated: gcloud auth application-default login\n"
            f"Original error: {e}"
        ) from e


def format_ascii_matrix(
    total_chunks: int,
    chunk_statuses: dict[int, str],
    current_batch: int,
    total_batches: int,
    elapsed_seconds: int,
    columns: int = 5,
) -> str:
    """
    Renders an ASCII status matrix representing all transcription chunks and their progress.

    Args:
        total_chunks: Total number of chunks in the transcription job.
        chunk_statuses: Dictionary mapping chunk index (1-based) to status string
                        ('pending', 'in_progress', 'completed', 'error').
        current_batch: Current batch index (1-based).
        total_batches: Total number of batches.
        elapsed_seconds: Total elapsed seconds since transcription started.
        columns: Number of chunk cells per row in the matrix display.

    Returns:
        Formatted multi-line ASCII box string.
    """
    completed_count = sum(1 for s in chunk_statuses.values() if s == "completed")
    pct = int((completed_count / total_chunks) * 100) if total_chunks > 0 else 100

    status_symbols = {
        "completed": "[●]",
        "in_progress": "[◌]",
        "pending": "[·]",
        "error": "[!]",
    }

    header = (
        f"[TRANSCRIPTION MATRIX] Batch {current_batch}/{total_batches} | "
        f"Completed: {completed_count}/{total_chunks} ({pct}%) | "
        f"Elapsed: {elapsed_seconds}s"
    )

    rows: list[str] = []
    current_row: list[str] = []
    for i in range(1, total_chunks + 1):
        status = chunk_statuses.get(i, "pending")
        sym = status_symbols.get(status, "[·]")
        cell = f"{i:02d} {sym}"
        current_row.append(cell)
        if len(current_row) == columns:
            rows.append("   ".join(current_row))
            current_row = []
    if current_row:
        rows.append("   ".join(current_row))

    max_row_len = max(len(r) for r in rows) if rows else 30
    box_width = max(max_row_len + 4, len(header) + 4)

    top = "┌" + "─" * (box_width - 2) + "┐"
    bottom = "└" + "─" * (box_width - 2) + "┘"
    body = "\n".join("│  " + r.ljust(box_width - 6) + "  │" for r in rows)
    legend = "Legend: [●] Done   [◌] Processing   [·] Queued"

    return f"{header}\n{top}\n{body}\n{bottom}\n{legend}"


def transcribe_single_file(
    gcs_uri: str,
    lang: str,
    enable_diarization: bool,
    speaker_count: int | None,
    project_id: str,
    region: str,
    model: str,
    timeout: int | None = None,
    max_timeout: int = TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
) -> cloud_speech.BatchRecognizeResponse:
    """
    Submit a single-file BatchRecognize request and wait for results using transcribe_files.

    Args:
        gcs_uri: GCS URI of the audio file (gs://bucket/path).
        lang: Language code for recognition.
        enable_diarization: Whether to enable speaker diarization.
        speaker_count: Speaker count hint (None = auto-detect).
        project_id: GCP project ID.
        region: GCP region (e.g., "eu", "us").
        model: Recognition model name.
        timeout: Legacy parameter for backward compatibility (maps to max_timeout if set).
        max_timeout: Maximum allowable time cap in seconds.

    Returns:
        BatchRecognizeResponse with inline results.

    Raises:
        ProcessingError: On API errors or timeouts.
    """
    effective_max_timeout = timeout if timeout is not None else max_timeout
    responses = transcribe_files(
        gcs_uris=[gcs_uri],
        lang=lang,
        enable_diarization=enable_diarization,
        speaker_count=speaker_count,
        project_id=project_id,
        region=region,
        model=model,
        max_timeout=effective_max_timeout,
    )
    return responses[0]


def transcribe_files(
    gcs_uris: list[str],
    lang: str,
    enable_diarization: bool,
    speaker_count: int | None,
    project_id: str,
    region: str,
    model: str,
    timeout: int | None = None,
    max_timeout: int = TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
    poll_interval: int = 15,
    batch_size: int = TRANSCRIPTION_BATCH_SIZE,
) -> list[cloud_speech.BatchRecognizeResponse]:
    """
    Transcribe multiple files by submitting BatchRecognize requests in bounded batches
    and monitoring them with retry on transient 429 ResourceExhausted errors.

    Args:
        gcs_uris: List of GCS URIs to transcribe.
        lang: Language code.
        enable_diarization: Whether to enable speaker diarization.
        speaker_count: Speaker count hint (None = auto-detect).
        project_id: GCP project ID.
        region: GCP region.
        model: Recognition model name.
        timeout: Legacy parameter for backward compatibility (maps to max_timeout if set).
        max_timeout: Maximum overall time cap in seconds across all batches.
        poll_interval: Seconds between progress monitoring checks.
        batch_size: Maximum number of concurrent transcription requests per batch.

    Returns:
        List of BatchRecognizeResponse objects (one per file, in original order).

    Raises:
        ProcessingError: On API errors or timeout cap exceeded.
    """
    if not gcs_uris:
        return []

    effective_max_timeout = timeout if timeout is not None else max_timeout
    effective_batch_size = max(1, batch_size)

    client = _get_speech_client(region)
    config = _build_recognition_config(lang, model, enable_diarization, speaker_count)
    recognizer = f"projects/{project_id}/locations/{region}/recognizers/_"

    diarization_str = "enabled" if enable_diarization else "disabled"
    speaker_str = str(speaker_count) if speaker_count else "auto-detect"

    total_files = len(gcs_uris)
    num_batches = (total_files + effective_batch_size - 1) // effective_batch_size
    logger.info(
        "Submitting %d transcription request(s) across %d batch(es) (batch size: %d)...",
        total_files, num_batches, effective_batch_size,
    )
    logger.info(
        "Config: model=%s, lang=%s, diarization=%s, speakers=%s",
        model, lang, diarization_str, speaker_str,
    )

    all_responses: list[cloud_speech.BatchRecognizeResponse] = []
    chunk_statuses: dict[int, str] = {i: "pending" for i in range(1, total_files + 1)}
    start_time = time.time()

    for batch_num, batch_start in enumerate(range(0, total_files, effective_batch_size), start=1):
        batch_uris = gcs_uris[batch_start:batch_start + effective_batch_size]
        logger.info(
            "--- Starting Batch %d/%d (files %d-%d of %d) ---",
            batch_num, num_batches, batch_start + 1, batch_start + len(batch_uris), total_files,
        )

        pending_items: list[dict[str, Any]] = []

        try:
            for offset, uri in enumerate(batch_uris):
                global_index = batch_start + offset + 1
                chunk_statuses[global_index] = "in_progress"
                request = cloud_speech.BatchRecognizeRequest(
                    recognizer=recognizer,
                    config=config,
                    files=[cloud_speech.BatchRecognizeFileMetadata(uri=uri)],
                    recognition_output_config=cloud_speech.RecognitionOutputConfig(
                        inline_response_config=cloud_speech.InlineOutputConfig(),
                    ),
                )
                logger.info("Submitting request %d/%d: %s", global_index, total_files, uri)
                operation = client.batch_recognize(request=request)
                pending_items.append({
                    "chunk_index": global_index,
                    "gcs_uri": uri,
                    "operation": operation,
                    "completed_response": None,
                })
                # Small pause between submissions to prevent burst rate-limit rejection
                if offset < len(batch_uris) - 1 and poll_interval > 0:
                    time.sleep(1)
        except gcp_exceptions.InvalidArgument as e:
            raise ProcessingError(
                f"Invalid transcription request. Check language code and model.\nOriginal error: {e}"
            ) from e
        except gcp_exceptions.PermissionDenied as e:
            raise ProcessingError(
                f"Permission denied for Speech-to-Text API.\nOriginal error: {e}"
            ) from e
        except gcp_exceptions.NotFound as e:
            raise ProcessingError(
                f"Speech-to-Text API resource not found.\nOriginal error: {e}"
            ) from e
        except Exception as e:
            raise ProcessingError(
                f"Failed to submit transcription requests.\nOriginal error: {e}"
            ) from e

        logger.info(
            "Monitoring %d operation(s) in Batch %d (overall timeout cap: %ds)...",
            len(pending_items), batch_num, effective_max_timeout,
        )

        # Initial wait before first polling check to allow backend workers to initialize
        if poll_interval > 0:
            time.sleep(min(15, poll_interval))

        while True:
            total_elapsed = time.time() - start_time
            if total_elapsed > effective_max_timeout:
                raise ProcessingError(
                    f"Transcription exceeded maximum allowable time of {effective_max_timeout}s."
                )

            all_done = True

            for item in pending_items:
                if item["completed_response"] is not None:
                    continue

                op = item["operation"]

                # Pacing between individual status checks to prevent burst query rates
                if poll_interval > 0:
                    time.sleep(0.2)

                try:
                    if op.done():
                        item["completed_response"] = op.result()
                        chunk_statuses[item["chunk_index"]] = "completed"
                        continue
                except gcp_exceptions.ResourceExhausted:
                    logger.debug(
                        "Chunk %d: transient 429 ResourceExhausted on status poll, will retry in %ds.",
                        item["chunk_index"], poll_interval,
                    )
                except Exception as e:
                    chunk_statuses[item["chunk_index"]] = "error"
                    raise ProcessingError(
                        f"Transcription failed for Chunk {item['chunk_index']} ({item['gcs_uri']}).\n"
                        f"Original error: {e}"
                    ) from e

                all_done = False

            batch_completed = sum(1 for it in pending_items if it["completed_response"] is not None)
            total_completed = sum(1 for s in chunk_statuses.values() if s == "completed")
            pct = int((total_completed / total_files) * 100) if total_files > 0 else 100
            matrix_str = format_ascii_matrix(
                total_chunks=total_files,
                chunk_statuses=chunk_statuses,
                current_batch=batch_num,
                total_batches=num_batches,
                elapsed_seconds=int(total_elapsed),
            )

            logger.info(
                "Batch %d/%d: %d/%d chunks completed (Total progress: %d/%d - %d%%)\n%s",
                batch_num, num_batches, batch_completed, len(batch_uris), total_completed, total_files, pct, matrix_str,
            )

            if all_done:
                break

            time.sleep(poll_interval)

        for item in pending_items:
            all_responses.append(item["completed_response"])

        if batch_num < num_batches and poll_interval > 0:
            time.sleep(2)

    total_time = int(time.time() - start_time)
    final_matrix = format_ascii_matrix(
        total_chunks=total_files,
        chunk_statuses=chunk_statuses,
        current_batch=num_batches,
        total_batches=num_batches,
        elapsed_seconds=total_time,
    )
    logger.info("All %d file(s) transcribed successfully in %ds.\n%s", total_files, total_time, final_matrix)
    return all_responses
