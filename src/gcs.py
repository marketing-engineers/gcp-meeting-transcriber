"""Google Cloud Storage operations for audio file upload and cleanup."""

import logging
import os

from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions

from src.audio import ProcessingError

logger = logging.getLogger(__name__)

_client: storage.Client | None = None


def _get_client() -> storage.Client:
    """Create or return cached GCS client with error handling for authentication issues."""
    global _client
    if _client is not None:
        return _client
    try:
        _client = storage.Client()
        return _client
    except Exception as e:
        raise ProcessingError(
            f"Failed to create GCS client. Ensure you have authenticated:\n"
            f"  Run: gcloud auth application-default login\n"
            f"Original error: {e}"
        ) from e


def upload_to_gcs(local_path: str, bucket_name: str, gcs_prefix: str, timeout: int = 600) -> str:
    """
    Upload a local file to GCS.

    Args:
        local_path: Path to the local file.
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix (folder) in the bucket.
        timeout: Timeout in seconds for the upload operation.

    Returns:
        The gs:// URI of the uploaded object.

    Raises:
        ProcessingError: On upload failure.
    """
    client = _get_client()
    filename = os.path.basename(local_path)
    blob_name = f"{gcs_prefix}/{filename}"

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path, timeout=timeout)
    except gcp_exceptions.NotFound as e:
        raise ProcessingError(
            f"GCS bucket '{bucket_name}' not found. "
            f"Ensure the bucket exists and you have access."
        ) from e
    except gcp_exceptions.Forbidden as e:
        raise ProcessingError(
            f"Permission denied uploading to gs://{bucket_name}/{blob_name}. "
            f"Ensure your account has the 'Storage Object Admin' role.\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise ProcessingError(
            f"Failed to upload {local_path} to GCS.\nOriginal error: {e}"
        ) from e

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("Uploaded: %s -> %s", local_path, gcs_uri)
    return gcs_uri


def upload_files_to_gcs(
    local_paths: list[str], bucket_name: str, gcs_prefix: str, timeout: int = 600
) -> list[str]:
    """
    Upload multiple local files to GCS.

    Args:
        local_paths: List of local file paths to upload.
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix (folder) in the bucket.
        timeout: Timeout in seconds for each upload operation.

    Returns:
        List of gs:// URIs for uploaded objects.

    Raises:
        ProcessingError: On upload failure.
    """
    logger.info(
        "Uploading %d file(s) to gs://%s/%s/...",
        len(local_paths), bucket_name, gcs_prefix,
    )
    uris: list[str] = []
    for path in local_paths:
        uri = upload_to_gcs(path, bucket_name, gcs_prefix, timeout=timeout)
        uris.append(uri)
    return uris


def delete_gcs_prefix(bucket_name: str, gcs_prefix: str) -> None:
    """
    Delete all objects under a GCS prefix (cleanup after transcription).

    Args:
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix (folder) to delete all objects under.
    """
    client = _get_client()
    try:
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        if not blobs:
            logger.info("No GCS objects found under gs://%s/%s/", bucket_name, gcs_prefix)
            return

        for blob in blobs:
            blob.delete()
        logger.info(
            "Cleaned up %d GCS object(s) under gs://%s/%s/",
            len(blobs), bucket_name, gcs_prefix,
        )
    except Exception as e:
        # Cleanup failure should not crash the pipeline — log and continue
        if isinstance(e, gcp_exceptions.Forbidden) and "retention policy" in str(e).lower():
            logger.info(
                "Skipping cleanup for gs://%s/%s/: Objects are preserved by bucket retention policy.",
                bucket_name, gcs_prefix
            )
        else:
            logger.warning(
                "Failed to clean up GCS prefix gs://%s/%s/: %s",
                bucket_name, gcs_prefix, e,
            )
