"""Google Cloud Speech-to-Text V2 BatchRecognize integration using Chirp 3."""

import logging
import time
from typing import Any

from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as gcp_exceptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from src.audio import ProcessingError
from src.config import TRANSCRIPTION_MAX_TIMEOUT_SECONDS

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
    poll_interval: int = 20,
) -> list[cloud_speech.BatchRecognizeResponse]:
    """
    Transcribe multiple files concurrently by submitting individual BatchRecognize
    requests up front and monitoring them in parallel with informational progress polling.

    Args:
        gcs_uris: List of GCS URIs to transcribe.
        lang: Language code.
        enable_diarization: Whether to enable speaker diarization.
        speaker_count: Speaker count hint (None = auto-detect).
        project_id: GCP project ID.
        region: GCP region.
        model: Recognition model name.
        timeout: Legacy parameter for backward compatibility (maps to max_timeout if set).
        max_timeout: Maximum overall time cap in seconds.
        poll_interval: Seconds between progress monitoring checks.

    Returns:
        List of BatchRecognizeResponse objects (one per file, in original order).

    Raises:
        ProcessingError: On API errors or timeout cap exceeded.
    """
    if not gcs_uris:
        return []

    effective_max_timeout = timeout if timeout is not None else max_timeout

    if len(gcs_uris) > _MAX_FILES_PER_BATCH:
        logger.warning(
            "Processing %d files exceeds the %d-file batch limit. "
            "Submitting concurrently as individual requests.",
            len(gcs_uris), _MAX_FILES_PER_BATCH,
        )

    client = _get_speech_client(region)
    config = _build_recognition_config(lang, model, enable_diarization, speaker_count)
    recognizer = f"projects/{project_id}/locations/{region}/recognizers/_"

    diarization_str = "enabled" if enable_diarization else "disabled"
    speaker_str = str(speaker_count) if speaker_count else "auto-detect"

    logger.info("Submitting %d transcription request(s) concurrently...", len(gcs_uris))
    logger.info(
        "Config: model=%s, lang=%s, diarization=%s, speakers=%s",
        model, lang, diarization_str, speaker_str,
    )

    pending_items: list[dict[str, Any]] = []

    try:
        for i, uri in enumerate(gcs_uris, start=1):
            request = cloud_speech.BatchRecognizeRequest(
                recognizer=recognizer,
                config=config,
                files=[cloud_speech.BatchRecognizeFileMetadata(uri=uri)],
                recognition_output_config=cloud_speech.RecognitionOutputConfig(
                    inline_response_config=cloud_speech.InlineOutputConfig(),
                ),
            )
            logger.info("Submitting request %d/%d: %s", i, len(gcs_uris), uri)
            operation = client.batch_recognize(request=request)
            pending_items.append({
                "chunk_index": i,
                "gcs_uri": uri,
                "operation": operation,
                "last_progress": -1,
                "completed_response": None,
            })
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

    start_time = time.time()
    logger.info(
        "Monitoring %d parallel operation(s) (overall timeout cap: %ds)...",
        len(pending_items), effective_max_timeout,
    )

    while True:
        now = time.time()
        total_elapsed = now - start_time

        if total_elapsed > effective_max_timeout:
            raise ProcessingError(
                f"Transcription batch exceeded maximum allowable time of {effective_max_timeout}s."
            )

        all_done = True
        progress_reports: list[str] = []

        for item in pending_items:
            if item["completed_response"] is not None:
                progress_reports.append(f"Chunk {item['chunk_index']}: Completed")
                continue

            op = item["operation"]

            try:
                if op.done():
                    item["completed_response"] = op.result()
                    progress_reports.append(f"Chunk {item['chunk_index']}: Completed")
                    continue
            except Exception as e:
                raise ProcessingError(
                    f"Transcription failed for Chunk {item['chunk_index']} ({item['gcs_uri']}).\n"
                    f"Original error: {e}"
                ) from e

            all_done = False
            progress_reports.append(f"Chunk {item['chunk_index']}: In Progress")

        if all_done:
            break

        logger.info("Progress (%ds elapsed): %s", int(total_elapsed), " | ".join(progress_reports))
        time.sleep(poll_interval)

    logger.info("All %d file(s) transcribed successfully in %ds.", len(gcs_uris), int(time.time() - start_time))
    return [item["completed_response"] for item in pending_items]
