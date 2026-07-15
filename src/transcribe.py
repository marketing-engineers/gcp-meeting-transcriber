"""Google Cloud Speech-to-Text V2 BatchRecognize integration using Chirp 3."""

import logging
from typing import Any

from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as gcp_exceptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from src.audio import ProcessingError

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
    timeout: int,
) -> cloud_speech.BatchRecognizeResponse:
    """
    Submit a single-file BatchRecognize request and wait for results.

    Uses InlineOutputConfig so results are returned directly in the response
    (only works for single-file requests).

    Args:
        gcs_uri: GCS URI of the audio file (gs://bucket/path).
        lang: Language code for recognition.
        enable_diarization: Whether to enable speaker diarization.
        speaker_count: Speaker count hint (None = auto-detect).
        project_id: GCP project ID.
        region: GCP region (e.g., "eu", "us").
        model: Recognition model name.
        timeout: Timeout in seconds for the operation.

    Returns:
        BatchRecognizeResponse with inline results.

    Raises:
        ProcessingError: On API errors.
    """
    client = _get_speech_client(region)
    config = _build_recognition_config(lang, model, enable_diarization, speaker_count)

    recognizer = f"projects/{project_id}/locations/{region}/recognizers/_"

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer,
        config=config,
        files=[
            cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri),
        ],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    logger.info("Submitting transcription request for: %s", gcs_uri)
    diarization_str = "enabled" if enable_diarization else "disabled"
    speaker_str = str(speaker_count) if speaker_count else "auto-detect"
    logger.info(
        "Config: model=%s, lang=%s, diarization=%s, speakers=%s",
        model, lang, diarization_str, speaker_str,
    )

    try:
        operation = client.batch_recognize(request=request)
        logger.info("Waiting for transcription to complete (timeout: %ds)...", timeout)
        response = operation.result(timeout=timeout)
        return response
    except gcp_exceptions.InvalidArgument as e:
        raise ProcessingError(
            f"Invalid transcription request. Check language code and model.\n"
            f"Original error: {e}"
        ) from e
    except gcp_exceptions.PermissionDenied as e:
        raise ProcessingError(
            f"Permission denied for Speech-to-Text API. "
            f"Ensure your account has 'Cloud Speech Administrator' role "
            f"and the Speech-to-Text API is enabled.\n"
            f"Original error: {e}"
        ) from e
    except gcp_exceptions.NotFound as e:
        raise ProcessingError(
            f"Speech-to-Text API resource not found. "
            f"Ensure the API is enabled for project '{project_id}' in region '{region}'.\n"
            f"Original error: {e}"
        ) from e
    except TimeoutError as e:
        raise ProcessingError(
            f"Transcription timed out after {timeout}s. "
            f"Try increasing TRANSCRIPTION_TIMEOUT in .env or check GCP console for status.\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise ProcessingError(
            f"Transcription failed unexpectedly.\nOriginal error: {e}"
        ) from e


def transcribe_files(
    gcs_uris: list[str],
    lang: str,
    enable_diarization: bool,
    speaker_count: int | None,
    project_id: str,
    region: str,
    model: str,
    timeout: int,
) -> list[cloud_speech.BatchRecognizeResponse]:
    """
    Transcribe multiple files by submitting individual BatchRecognize requests.

    Each file is submitted as a separate single-file request with InlineOutputConfig
    to get results directly in the response (avoids needing GcsOutputConfig for multi-file).

    Args:
        gcs_uris: List of GCS URIs to transcribe.
        lang: Language code.
        enable_diarization: Whether to enable speaker diarization.
        speaker_count: Speaker count hint (None = auto-detect).
        project_id: GCP project ID.
        region: GCP region.
        model: Recognition model name.
        timeout: Timeout per file in seconds.

    Returns:
        List of BatchRecognizeResponse objects (one per file, in order).

    Raises:
        ProcessingError: On API errors.
    """
    if len(gcs_uris) > _MAX_FILES_PER_BATCH:
        logger.warning(
            "Processing %d files exceeds the %d-file batch limit. "
            "Submitting as individual requests.",
            len(gcs_uris), _MAX_FILES_PER_BATCH,
        )

    responses: list[cloud_speech.BatchRecognizeResponse] = []
    for i, uri in enumerate(gcs_uris, start=1):
        logger.info("Transcribing file %d/%d: %s", i, len(gcs_uris), uri)
        response = transcribe_single_file(
            gcs_uri=uri,
            lang=lang,
            enable_diarization=enable_diarization,
            speaker_count=speaker_count,
            project_id=project_id,
            region=region,
            model=model,
            timeout=timeout,
        )
        responses.append(response)

    logger.info("All %d file(s) transcribed successfully.", len(gcs_uris))
    return responses
