"""Authentication management for Google Cloud APIs (ADC and Service Account Key modes)."""

import json
import logging
import os
import stat
from typing import Any

import google.auth.credentials
from google.oauth2 import service_account

from src.audio import ProcessingError

logger = logging.getLogger(__name__)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_credentials: google.auth.credentials.Credentials | None = None
_credentials_loaded: bool = False


def reset_credentials_cache() -> None:
    """Resets the module-level credentials cache (primarily for tests)."""
    global _credentials, _credentials_loaded
    _credentials = None
    _credentials_loaded = False


def auth_hint() -> str:
    """Returns mode-specific remediation text for authentication failures."""
    from src.config import GCP_AUTH_MODE

    if GCP_AUTH_MODE == "service_account_key":
        return (
            "Service account key mode is active. Ensure your key file exists, is owned by the current user, "
            "has permissions 0600 ('chmod 600 <key_file>'), matches GCP_EXPECTED_SERVICE_ACCOUNT and string GCP_PROJECT_ID, "
            "and has the required IAM roles bound on the project and bucket."
        )
    return "Ensure you have authenticated:\n  Run: gcloud auth application-default login"


def get_credentials() -> google.auth.credentials.Credentials | None:
    """
    Retrieves Google Cloud credentials based on GCP_AUTH_MODE.

    Modes:
        - 'adc': Returns None, allowing GCP client libraries to use implicit Application Default Credentials.
        - 'service_account_key': Validates file existence, permissions (0600), ownership, and JSON metadata,
          returning explicit service account credentials scoped to cloud-platform.

    Returns:
        google.auth.credentials.Credentials if in service_account_key mode, None if in adc mode.

    Raises:
        ProcessingError: If service account key file validation or loading fails.
    """
    global _credentials, _credentials_loaded
    if _credentials_loaded:
        return _credentials

    from src.config import (
        GCP_AUTH_MODE,
        GCP_EXPECTED_SERVICE_ACCOUNT,
        GCP_PROJECT_ID,
        GCP_SERVICE_ACCOUNT_KEY_FILE,
    )

    if GCP_AUTH_MODE == "adc":
        logger.info("Using Application Default Credentials (mode: adc)")
        _credentials = None
        _credentials_loaded = True
        return None

    if GCP_AUTH_MODE != "service_account_key":
        raise ProcessingError(
            f"Unsupported GCP_AUTH_MODE '{GCP_AUTH_MODE}'. Allowed modes: 'adc', 'service_account_key'."
        )

    if not GCP_SERVICE_ACCOUNT_KEY_FILE:
        raise ProcessingError(
            "GCP_SERVICE_ACCOUNT_KEY_FILE is not configured in .env for service_account_key mode."
        )

    if not GCP_EXPECTED_SERVICE_ACCOUNT:
        raise ProcessingError(
            "GCP_EXPECTED_SERVICE_ACCOUNT is not configured in .env for service_account_key mode."
        )

    key_path = os.path.abspath(GCP_SERVICE_ACCOUNT_KEY_FILE)

    if not os.path.exists(key_path):
        raise ProcessingError(f"Service account key file not found: {key_path}")

    try:
        file_stat = os.stat(key_path)
    except OSError as e:
        raise ProcessingError(f"Failed to access service account key file '{key_path}': {e}") from e

    if not stat.S_ISREG(file_stat.st_mode):
        raise ProcessingError(f"Service account key path is not a regular file: {key_path}")

    current_uid = os.getuid()
    if file_stat.st_uid != current_uid:
        raise ProcessingError(
            f"Service account key file '{key_path}' is owned by UID {file_stat.st_uid}, "
            f"but current process is running as UID {current_uid}. "
            "Key file must be owned by the running user."
        )

    insecure_bits = file_stat.st_mode & 0o077
    if insecure_bits != 0:
        mode_oct = oct(file_stat.st_mode & 0o777)
        raise ProcessingError(
            f"Service account key file '{key_path}' has insecure permissions ({mode_oct}). "
            f"Permissions must not allow group or other access. "
            f"Run: chmod 600 \"{key_path}\""
        )

    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS is set in environment but will be ignored "
            "because GCP_AUTH_MODE=service_account_key is active."
        )

    try:
        with open(key_path, "r", encoding="utf-8") as f:
            info: Any = json.load(f)
    except json.JSONDecodeError as e:
        raise ProcessingError(f"Service account key file '{key_path}' is not valid JSON: {e}") from e
    except OSError as e:
        raise ProcessingError(f"Failed to read service account key file '{key_path}': {e}") from e

    if not isinstance(info, dict):
        raise ProcessingError(
            f"Service account key file '{key_path}' is malformed: expected JSON object at root."
        )

    key_type = info.get("type")
    if key_type != "service_account":
        raise ProcessingError(
            f"Service account key file '{key_path}' has credential type '{key_type}', expected 'service_account'. "
            "Authorized user tokens and external accounts are not permitted."
        )

    actual_email = info.get("client_email")
    if actual_email != GCP_EXPECTED_SERVICE_ACCOUNT:
        raise ProcessingError(
            f"Service account email in key file ('{actual_email}') does not match "
            f"configured GCP_EXPECTED_SERVICE_ACCOUNT ('{GCP_EXPECTED_SERVICE_ACCOUNT}')."
        )

    actual_project = info.get("project_id")
    if actual_project != GCP_PROJECT_ID:
        raise ProcessingError(
            f"Project ID in key file ('{actual_project}') does not match configured GCP_PROJECT_ID ('{GCP_PROJECT_ID}'). "
            "Note: GCP_PROJECT_ID in .env must be the string project ID (e.g., 'my-project-id'), "
            "not a numeric project number."
        )

    try:
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=[CLOUD_PLATFORM_SCOPE],
        )
    except Exception as e:
        raise ProcessingError(
            f"Failed to load service account credentials from '{key_path}': {e}"
        ) from e

    logger.info(
        "Loaded service account credentials for %s (mode: service_account_key)",
        actual_email,
    )

    _credentials = creds
    _credentials_loaded = True
    return _credentials


def get_gemini_backend_mode() -> str:
    """
    Resolves the active backend mode for Gemini 3.5 Transcribe.
    Returns:
        'vertex'  if GEMINI_BACKEND == 'vertex' or
                  (GEMINI_BACKEND == 'auto' and GCP_PROJECT_ID is set).
        'api_key' if GEMINI_BACKEND == 'api_key' or
                  (GEMINI_BACKEND == 'auto' and not GCP_PROJECT_ID and GEMINI_API_KEY is set).
        Defaults to 'api_key'.
    """
    from src.config import GCP_PROJECT_ID, GEMINI_API_KEY, GEMINI_BACKEND

    if GEMINI_BACKEND == "api_key":
        return "api_key"
    if GEMINI_BACKEND == "vertex":
        return "vertex"
    # GEMINI_BACKEND == "auto": GCP credentials take precedence (unified GCP setup for Chirp + Gemini)
    if GCP_PROJECT_ID:
        return "vertex"
    if GEMINI_API_KEY:
        return "api_key"
    return "api_key"


def check_auth() -> bool:
    """
    Validates credentials, configuration, and permissions for whichever transcription
    backends are configured (GEMINI_API_KEY, GCP Speech V2 + GCS, and/or Vertex AI).

    Returns:
        True if all configured backends pass, False otherwise.
    """
    from src.config import (
        GCP_AUTH_MODE,
        GCP_EXPECTED_SERVICE_ACCOUNT,
        GCP_PROJECT_ID,
        GCP_REGION,
        GCS_BUCKET_NAME,
        GEMINI_API_KEY,
        GEMINI_BACKEND,
        validate_backend_config,
        validate_gcp_config,
    )
    from src.gcs import _get_client
    from src.transcribe import _get_speech_client

    has_gemini_key = bool(GEMINI_API_KEY)
    has_gcp = bool(GCP_PROJECT_ID or GCS_BUCKET_NAME)

    if not has_gemini_key and not has_gcp:
        try:
            validate_backend_config()
        except Exception as e:
            logger.error("Configuration validation failed: %s", e)
            return False

    all_passed = True

    if has_gemini_key:
        logger.info("=== Gemini API (AI Studio) Authentication Check ===")
        try:
            from google import genai

            ai_client = genai.Client(api_key=GEMINI_API_KEY)
            try:
                _ = list(ai_client.models.list(config={"page_size": 1}))
                logger.info("  Gemini API (AI Studio) key validated and active.")
                logger.info("=== Gemini API Authentication Check PASSED ===")
            except Exception as ping_err:
                logger.error("Failed to validate Gemini API key: %s", ping_err)
                all_passed = False
        except Exception as e:
            logger.error("Failed to initialize Gemini API client: %s", e)
            all_passed = False

    if has_gcp or not has_gemini_key:
        try:
            validate_gcp_config()
        except Exception as e:
            logger.error("GCP configuration validation failed: %s", e)
            return False

        principal = (
            GCP_EXPECTED_SERVICE_ACCOUNT
            if GCP_AUTH_MODE == "service_account_key"
            else "user ADC"
        )

        logger.info("=== GCP Authentication & Permissions Check ===")
        logger.info("  Auth Mode: %s", GCP_AUTH_MODE)
        logger.info("  Principal: %s", principal)
        logger.info("  Project:   %s", GCP_PROJECT_ID)
        logger.info("  Bucket:    %s", GCS_BUCKET_NAME)
        logger.info("  Region:    %s", GCP_REGION)

        try:
            gcs_client = _get_client()
            logger.info("GCS client initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize GCS client: %s", e)
            return False

        try:
            speech_client = _get_speech_client(GCP_REGION)
            logger.info(
                "Speech-to-Text client initialized successfully for region '%s' (projects/%s/locations/%s/recognizers/_).",
                GCP_REGION,
                GCP_PROJECT_ID,
                GCP_REGION,
            )
            # Active check to verify speech.googleapis.com is enabled
            try:
                _ = list(
                    speech_client.list_recognizers(
                        parent=f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}",
                        page_size=1,
                    )
                )
                logger.info("Speech-to-Text V2 API (speech.googleapis.com) verified active.")
            except Exception as api_err:
                err_str = str(api_err)
                if "SERVICE_DISABLED" in err_str or ("403" in err_str and "has not been used" in err_str):
                    logger.error(
                        "Speech-to-Text V2 API (speech.googleapis.com) is disabled on project '%s'.\n"
                        "Enable it with: gcloud services enable speech.googleapis.com --project=%s",
                        GCP_PROJECT_ID,
                        GCP_PROJECT_ID,
                    )
                    return False
                elif "PERMISSION_DENIED" in err_str or ("403" in err_str and "Permission" in err_str):
                    logger.warning(
                        "Speech-to-Text V2 API permission notice on project '%s': %s\n"
                        "Ensure your service account/user has role 'roles/speech.client' or 'roles/speech.admin'.",
                        GCP_PROJECT_ID,
                        api_err,
                    )
                else:
                    logger.info("Speech-to-Text API connectivity confirmed (notice: %s).", api_err)
        except Exception as e:
            logger.error("Failed to initialize Speech-to-Text client: %s", e)
            return False

        required_permissions = [
            "storage.objects.create",
            "storage.objects.get",
            "storage.objects.list",
            "storage.objects.delete",
        ]

        try:
            bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            held_permissions = bucket.test_iam_permissions(required_permissions)
            missing_permissions = set(required_permissions) - set(held_permissions)
            if missing_permissions:
                logger.error(
                    "Missing required GCS object permissions on bucket '%s': %s",
                    GCS_BUCKET_NAME,
                    ", ".join(sorted(missing_permissions)),
                )
                return False

            logger.info(
                "GCS bucket '%s' IAM permissions verified: all %d required permissions held.",
                GCS_BUCKET_NAME,
                len(required_permissions),
            )

            list(bucket.list_blobs(max_results=1))
            logger.info("GCS bucket '%s' list_blobs check succeeded.", GCS_BUCKET_NAME)
        except Exception as e:
            logger.error("Failed verifying GCS bucket '%s': %s", GCS_BUCKET_NAME, e)
            return False

        if GEMINI_BACKEND in ("auto", "vertex"):
            try:
                from google import genai

                vertex_client = genai.Client(
                    vertexai=True,
                    project=GCP_PROJECT_ID,
                    location=GCP_REGION,
                    credentials=get_credentials(),
                )
                logger.info("Gemini Vertex AI client initialized successfully.")

                # Active API check to verify aiplatform.googleapis.com is enabled
                try:
                    _ = list(vertex_client.models.list(config={"page_size": 1}))
                    logger.info("Gemini Vertex AI API (aiplatform.googleapis.com) verified active.")
                except Exception as vertex_err:
                    err_str = str(vertex_err)
                    if "SERVICE_DISABLED" in err_str or ("403" in err_str and "has not been used" in err_str):
                        msg = (
                            f"Vertex AI API (aiplatform.googleapis.com) is disabled on project '{GCP_PROJECT_ID}'.\n"
                            f"Remediation:\n"
                            f"  1. Enable it with: gcloud services enable aiplatform.googleapis.com --project={GCP_PROJECT_ID}\n"
                            f"  2. Or if using Google AI Studio, configure GEMINI_API_KEY in .env and set GEMINI_BACKEND=api_key."
                        )
                        if GEMINI_BACKEND == "vertex":
                            logger.error("%s", msg)
                            return False
                        elif not has_gemini_key:
                            logger.error("%s", msg)
                            all_passed = False
                        else:
                            logger.warning(
                                "Vertex AI check note (fallback available via GEMINI_API_KEY): %s",
                                vertex_err,
                            )
                    elif "PERMISSION_DENIED" in err_str or ("403" in err_str and "Permission" in err_str):
                        msg = (
                            f"Vertex AI permission check failed for project '{GCP_PROJECT_ID}': {vertex_err}\n"
                            f"Ensure service account has 'aiplatform.endpoints.predict' (or role 'roles/aiplatform.user')."
                        )
                        if GEMINI_BACKEND == "vertex" or not has_gemini_key:
                            logger.error("%s", msg)
                            all_passed = False
                        else:
                            logger.warning("%s", msg)
                    else:
                        logger.warning("Vertex AI API check note: %s", vertex_err)
            except Exception as e:
                logger.warning("Vertex AI client initialization check notice: %s", e)

        logger.info(
            "Note: The first transcription job will validate specific model execution permissions "
            "('speech.recognizers.recognize' for Chirp, 'aiplatform.endpoints.predict' for Vertex AI)."
        )
        logger.info("=== GCP Authentication Check PASSED ===")

    return all_passed

