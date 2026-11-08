"""Gemini 3.5 Transcribe integration via Google GenAI SDK (AI Studio and Vertex AI)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import random
import time
from typing import Any

from google import genai
from google.genai import errors, types

from src.audio import ProcessingError
from src.auth import auth_hint, get_credentials, get_gemini_backend_mode
from src.config import (
    GCP_PROJECT_ID,
    GCP_REGION,
    GEMINI_API_KEY,
    TRANSCRIPTION_CONCURRENCY,
    TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
    resolve_gemini_language_codes,
)
from src.transcribe import format_ascii_matrix

logger = logging.getLogger(__name__)


def _is_rate_limit_error(e: Exception) -> bool:
    """Checks if an exception represents a 429 Resource Exhausted / Rate Limit error."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code == 429:
        return True
    msg = str(e).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE_LIMIT" in msg


def _get_gemini_client(backend: str | None = None) -> genai.Client:
    """
    Creates and returns a google-genai Client configured for the specified backend.

    Args:
        backend: 'api_key', 'vertex', or None (auto-resolved).

    Returns:
        genai.Client instance.

    Raises:
        ProcessingError: If client initialization fails.
    """
    effective_backend = backend or get_gemini_backend_mode()

    try:
        if effective_backend == "api_key":
            if not GEMINI_API_KEY:
                raise ProcessingError(
                    "GEMINI_API_KEY is not configured in .env for Gemini API Studio mode."
                )
            return genai.Client(api_key=GEMINI_API_KEY)
        elif effective_backend == "vertex":
            credentials = get_credentials()
            return genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_REGION,
                credentials=credentials,
            )
        else:
            raise ProcessingError(f"Unsupported Gemini backend mode: '{effective_backend}'")
    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(
            f"Failed to create Gemini client (backend: {effective_backend}). {auth_hint()}\n"
            f"Original error: {e}"
        ) from e


def transcribe_gemini_file(
    audio_path_or_uri: str,
    lang: str,
    diarize: bool,
    mode: str = "verbatim",
    backend: str | None = None,
    client: genai.Client | None = None,
    max_retries: int = 5,
) -> Any:
    """
    Transcribes a single audio file or GCS URI using gemini-3.5-transcribe.

    Args:
        audio_path_or_uri: Local FLAC file path or GCS URI (gs://...).
        lang: Language code (e.g., 'nl-NL', 'en-US', or 'auto').
        diarize: Whether to enable speaker diarization.
        mode: Transcription mode ('verbatim' or 'smart'). Diarization enforces verbatim.
        backend: Backend to use ('api_key' or 'vertex'). None = auto-detect.
        client: Pre-configured genai.Client instance, or None to create one.
        max_retries: Number of exponential backoff retries on rate limit (429) errors.

    Returns:
        Gemini response object (Interaction or GenerateContentResponse).

    Raises:
        ProcessingError: On unrecoverable API errors or missing files.
    """
    effective_backend = backend or get_gemini_backend_mode()
    if client is None:
        client = _get_gemini_client(effective_backend)

    is_gcs = audio_path_or_uri.startswith("gs://")

    for attempt in range(max_retries + 1):
        try:
            if effective_backend == "api_key":
                if is_gcs:
                    raise ProcessingError(
                        f"Cannot transcribe GCS URI directly with Gemini AI Studio (api_key mode): {audio_path_or_uri}. "
                        "GCS URIs require Vertex AI backend mode or a local audio file."
                    )
                if not os.path.exists(audio_path_or_uri):
                    raise ProcessingError(f"Audio file not found for Gemini transcription: {audio_path_or_uri}")

                uploaded_file = None
                try:
                    logger.info("Uploading audio chunk to Gemini Files API: %s", audio_path_or_uri)
                    uploaded_file = client.files.upload(file=audio_path_or_uri)

                    # Build generation_config for interactions.create
                    transcription_config: dict[str, Any] = {}
                    gemini_lang_codes = resolve_gemini_language_codes(lang)
                    if gemini_lang_codes:
                        transcription_config["language_codes"] = gemini_lang_codes

                    if diarize:
                        transcription_config["mode"] = {
                            "type": "verbatim",
                            "diarization_mode": "speaker",
                        }
                    elif mode == "smart":
                        transcription_config["mode"] = "smart"
                    else:
                        transcription_config["mode"] = {"type": "verbatim"}

                    try:
                        response = client.interactions.create(
                            model="gemini-3.5-transcribe",
                            input=[uploaded_file],
                            generation_config={"transcription_config": transcription_config},
                        )
                        return response
                    except Exception as interaction_err:
                        if _is_rate_limit_error(interaction_err):
                            raise interaction_err
                        logger.debug(
                            "interactions.create failed (%s); trying models.generate_content",
                            interaction_err,
                        )

                        audio_cfg = types.AudioTranscriptionConfig(
                            diarization=diarize,
                            language_codes=gemini_lang_codes,
                            mode="SMART" if (not diarize and mode == "smart") else "VERBATIM",
                        )
                        response = client.models.generate_content(
                            model="gemini-3.5-transcribe",
                            contents=[uploaded_file],
                            config=types.GenerateContentConfig(audio_transcription_config=audio_cfg),
                        )
                        return response
                finally:
                    if uploaded_file is not None:
                        try:
                            client.files.delete(name=uploaded_file.name)
                            logger.debug("Deleted temporary Gemini file: %s", uploaded_file.name)
                        except Exception as del_err:
                            logger.warning(
                                "Failed to delete temporary Gemini file %s: %s",
                                uploaded_file.name,
                                del_err,
                            )

            else:  # effective_backend == "vertex"
                if not is_gcs:
                    raise ProcessingError(
                        f"Vertex AI transcription expects a GCS URI (gs://...), got: {audio_path_or_uri}"
                    )

                part = types.Part.from_uri(file_uri=audio_path_or_uri, mime_type="audio/flac")
                gemini_lang_codes = resolve_gemini_language_codes(lang)
                audio_cfg = types.AudioTranscriptionConfig(
                    diarization=diarize,
                    language_codes=gemini_lang_codes,
                    mode="SMART" if (not diarize and mode == "smart") else "VERBATIM",
                )
                response = client.models.generate_content(
                    model="gemini-3.5-transcribe",
                    contents=[part],
                    config=types.GenerateContentConfig(audio_transcription_config=audio_cfg),
                )
                return response

        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "Gemini API rate limit (429) encountered on attempt %d/%d for %s. Backing off for %.1fs...",
                    attempt + 1, max_retries, audio_path_or_uri, delay,
                )
                time.sleep(delay)
                continue
            if isinstance(e, ProcessingError):
                raise
            err_str = str(e)
            if "aiplatform.googleapis.com" in err_str or ("403" in err_str and "PERMISSION_DENIED" in err_str):
                raise ProcessingError(
                    f"Gemini Vertex AI transcription failed: {e}\n\n"
                    "Remediation Guidance:\n"
                    f"1. Enable the Vertex AI API on project '{GCP_PROJECT_ID}':\n"
                    f"   gcloud services enable aiplatform.googleapis.com --project={GCP_PROJECT_ID}\n"
                    "2. Ensure your service account has 'aiplatform.endpoints.predict' (or role 'roles/aiplatform.user').\n"
                    "3. Alternatively, switch to Google AI Studio by setting GEMINI_API_KEY in .env and GEMINI_BACKEND=api_key."
                ) from e
            raise ProcessingError(
                f"Gemini transcription failed for {audio_path_or_uri}.\nOriginal error: {e}"
            ) from e

    raise ProcessingError(f"Gemini transcription exceeded {max_retries} retries for {audio_path_or_uri}")


def transcribe_gemini_files(
    audio_targets: list[str],
    lang: str,
    diarize: bool,
    mode: str = "verbatim",
    backend: str | None = None,
    concurrency: int = TRANSCRIPTION_CONCURRENCY,
    total_audio_seconds: float | None = None,
    chunk_duration: float | None = None,
    max_timeout: int = TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
) -> list[Any]:
    """
    Transcribes multiple audio files or GCS URIs using gemini-3.5-transcribe in bounded concurrency waves.

    Args:
        audio_targets: List of local audio file paths or GCS URIs in chunk order.
        lang: Language code ('nl-NL', 'en-US', 'auto').
        diarize: Whether to enable speaker diarization.
        mode: Transcription mode ('verbatim' or 'smart').
        backend: 'api_key' or 'vertex' (None = auto).
        concurrency: Max concurrent requests per wave.
        total_audio_seconds: Total audio duration in seconds.
        chunk_duration: Representative duration per chunk.
        max_timeout: Timeout limit in seconds.

    Returns:
        List of Gemini response objects corresponding to each audio target in order.

    Raises:
        ProcessingError: If any chunk fails to transcribe.
    """
    if not audio_targets:
        return []

    effective_backend = backend or get_gemini_backend_mode()
    client = _get_gemini_client(effective_backend)
    total_files = len(audio_targets)
    effective_concurrency = max(1, concurrency)

    num_batches = (total_files + effective_concurrency - 1) // effective_concurrency
    logger.info(
        "Submitting %d Gemini transcription request(s) across %d wave(s) (concurrency: %d, backend: %s)...",
        total_files, num_batches, effective_concurrency, effective_backend,
    )

    chunk_statuses: dict[int, str] = {i: "pending" for i in range(1, total_files + 1)}
    all_responses: list[Any] = [None] * total_files
    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_num = batch_idx + 1
        batch_start_idx = batch_idx * effective_concurrency
        batch_end_idx = min(batch_start_idx + effective_concurrency, total_files)
        batch_targets = audio_targets[batch_start_idx:batch_end_idx]

        for i in range(batch_start_idx, batch_end_idx):
            chunk_statuses[i + 1] = "in_progress"

        matrix_str = format_ascii_matrix(
            total_chunks=total_files,
            chunk_statuses=chunk_statuses,
            current_batch=batch_num,
            total_batches=num_batches,
            elapsed_seconds=int(time.time() - start_time),
        )
        logger.info(
            "Wave %d/%d: Starting %d chunks (concurrency: %d)...\n%s",
            batch_num, num_batches, len(batch_targets), effective_concurrency, matrix_str,
        )

        def _transcribe_worker(index_and_target: tuple[int, str]) -> tuple[int, Any]:
            idx, target = index_and_target
            res = transcribe_gemini_file(
                audio_path_or_uri=target,
                lang=lang,
                diarize=diarize,
                mode=mode,
                backend=effective_backend,
                client=client,
            )
            return idx, res

        wave_items = list(enumerate(batch_targets, start=batch_start_idx))
        with ThreadPoolExecutor(max_workers=len(batch_targets)) as executor:
            future_to_idx = {
                executor.submit(_transcribe_worker, item): item[0]
                for item in wave_items
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk_num = idx + 1
                try:
                    res_idx, response_obj = future.result()
                    all_responses[res_idx] = response_obj
                    chunk_statuses[chunk_num] = "completed"
                except Exception as e:
                    chunk_statuses[chunk_num] = "error"
                    matrix_str = format_ascii_matrix(
                        total_chunks=total_files,
                        chunk_statuses=chunk_statuses,
                        current_batch=batch_num,
                        total_batches=num_batches,
                        elapsed_seconds=int(time.time() - start_time),
                    )
                    logger.error("Error transcribing chunk %d:\n%s", chunk_num, matrix_str)
                    raise ProcessingError(
                        f"Transcription failed for Chunk {chunk_num} ({audio_targets[idx]}):\n{e}"
                    ) from e

        completed_in_batch = sum(1 for i in range(batch_start_idx, batch_end_idx) if chunk_statuses[i + 1] == "completed")
        total_completed = sum(1 for s in chunk_statuses.values() if s == "completed")
        pct = int((total_completed / total_files) * 100)
        matrix_str = format_ascii_matrix(
            total_chunks=total_files,
            chunk_statuses=chunk_statuses,
            current_batch=batch_num,
            total_batches=num_batches,
            elapsed_seconds=int(time.time() - start_time),
        )
        logger.info(
            "Wave %d/%d: %d/%d chunks completed (Total progress: %d/%d - %d%%)\n%s",
            batch_num, num_batches, completed_in_batch, len(batch_targets),
            total_completed, total_files, pct, matrix_str,
        )

    total_time = int(time.time() - start_time)
    final_matrix = format_ascii_matrix(
        total_chunks=total_files,
        chunk_statuses=chunk_statuses,
        current_batch=num_batches,
        total_batches=num_batches,
        elapsed_seconds=total_time,
    )
    logger.info("All %d file(s) transcribed successfully via Gemini in %ds.\n%s", total_files, total_time, final_matrix)
    return all_responses
