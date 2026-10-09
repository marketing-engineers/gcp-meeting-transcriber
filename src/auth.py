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


def check_auth() -> bool:
    """
    Validates GCP credentials, configuration, and permissions for the transcription pipeline.

    Checks:
        1. Validates GCP configuration variables.
        2. Initializes GCS client and Speech-to-Text client.
        3. Tests GCS bucket permissions via testIamPermissions:
           - storage.objects.create
           - storage.objects.get
           - storage.objects.list
           - storage.objects.delete
        4. Verifies GCS list_blobs(max_results=1).
        5. Reports Speech client initialization (Speech API has no testIamPermissions).

    Returns:
        True if all checks pass, False otherwise.
    """
    from src.config import (
        GCP_AUTH_MODE,
        GCP_EXPECTED_SERVICE_ACCOUNT,
        GCP_PROJECT_ID,
        GCP_REGION,
        GCS_BUCKET_NAME,
        validate_gcp_config,
    )
    from src.gcs import _get_client
    from src.transcribe import _get_speech_client

    try:
        validate_gcp_config()
    except Exception as e:
        logger.error("Configuration validation failed: %s", e)
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

    logger.info(
        "Note: Speech-to-Text V2 does not support testIamPermissions. "
        "The first transcription job will validate Speech API permissions "
        "('speech.recognizers.recognize', 'speech.operations.get', 'serviceusage.services.use')."
    )
    logger.info("=== GCP Authentication Check PASSED ===")
    return True

