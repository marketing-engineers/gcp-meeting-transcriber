"""Google Cloud Storage operations for audio file upload and cleanup."""

import logging
import os
import time
from typing import Any

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
    from src.config import GCP_PROJECT_ID
    try:
        _client = storage.Client(project=GCP_PROJECT_ID)
        return _client
    except Exception as e:
        raise ProcessingError(
            f"Failed to create GCS client. Ensure you have authenticated:\n"
            f"  Run: gcloud auth application-default login\n"
            f"Original error: {e}"
        ) from e


class ProgressFileReader:
    """File-like wrapper that reports upload progress periodically."""

    def __init__(
        self,
        file_path: str,
        callback: Any = None,
        min_interval_seconds: float = 5.0,
    ) -> None:
        self.file_path = file_path
        self._file = open(file_path, "rb")
        self.total_bytes = os.path.getsize(file_path)
        self.bytes_read = 0
        self.callback = callback
        self.min_interval_seconds = min_interval_seconds
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.last_report_bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self._file.read(size)
        if data:
            self.bytes_read += len(data)
            now = time.time()
            elapsed_since_last = now - self.last_report_time
            if self.callback and elapsed_since_last >= self.min_interval_seconds:
                pct = (self.bytes_read / self.total_bytes) * 100 if self.total_bytes > 0 else 100.0
                speed_mb = (
                    ((self.bytes_read - self.last_report_bytes) / elapsed_since_last) / (1024 * 1024)
                    if elapsed_since_last > 0
                    else 0.0
                )
                self.callback(self.bytes_read, self.total_bytes, pct, speed_mb)
                self.last_report_time = now
                self.last_report_bytes = self.bytes_read
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        res = self._file.seek(offset, whence)
        self.bytes_read = self._file.tell()
        return res

    def tell(self) -> int:
        return self._file.tell()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "ProgressFileReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def upload_to_gcs(
    local_path: str,
    bucket_name: str,
    gcs_prefix: str,
    timeout: int = 600,
    file_index: int | None = None,
    total_files: int | None = None,
    progress_interval_seconds: float = 5.0,
) -> str:
    """
    Upload a local file to GCS with intermittent progress logging.

    Args:
        local_path: Path to the local file.
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix (folder) in the bucket.
        timeout: Timeout in seconds for the upload operation.
        file_index: Optional 1-based index when uploading as part of a batch.
        total_files: Optional total count of files in the batch.
        progress_interval_seconds: Progress reporting throttle interval (default 5.0s).

    Returns:
        The gs:// URI of the uploaded object.

    Raises:
        ProcessingError: On upload failure.
    """
    client = _get_client()
    filename = os.path.basename(local_path)
    blob_name = f"{gcs_prefix}/{filename}"
    file_size_bytes = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    total_mb = file_size_bytes / (1024 * 1024)

    prefix_str = f"[{file_index}/{total_files}] " if file_index is not None and total_files is not None else ""

    logger.info(
        "Uploading %s%s (%.1f MB) -> gs://%s/%s...",
        prefix_str, filename, total_mb, bucket_name, blob_name,
    )

    t0 = time.time()

    def _progress_callback(bytes_read: int, total_bytes: int, pct: float, speed_mb: float) -> None:
        mb_read = bytes_read / (1024 * 1024)
        logger.info(
            "Upload progress %s%s: %.1f/%.1f MB (%d%%) @ %.1f MB/s",
            prefix_str, filename, mb_read, total_mb, int(pct), speed_mb,
            extra={"in_place": True},
        )

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Set chunk size to 2MB (multiple of 256KB) so resumable upload streams in chunks,
        # allowing fine-grained progress callbacks during upload.
        blob.chunk_size = 2 * 1024 * 1024
        with ProgressFileReader(
            local_path,
            callback=_progress_callback,
            min_interval_seconds=progress_interval_seconds,
        ) as reader:
            blob.upload_from_file(reader, size=reader.total_bytes, timeout=timeout)
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

    upload_elapsed = time.time() - t0
    avg_speed_mb = total_mb / upload_elapsed if upload_elapsed > 0 else 0.0
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info(
        "Uploaded %s%s: %.1f MB in %.1fs (avg %.1f MB/s) -> %s",
        prefix_str, filename, total_mb, upload_elapsed, avg_speed_mb, gcs_uri,
    )
    return gcs_uri


def upload_files_to_gcs(
    local_paths: list[str], bucket_name: str, gcs_prefix: str, timeout: int = 600
) -> list[str]:
    """
    Upload multiple local files to GCS sequentially with progress updates.

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
    total_bytes = sum(os.path.getsize(p) for p in local_paths if os.path.exists(p))
    total_mb = total_bytes / (1024 * 1024)
    total_count = len(local_paths)

    logger.info(
        "Starting upload of %d file(s) (total: %.1f MB) to gs://%s/%s/...",
        total_count, total_mb, bucket_name, gcs_prefix,
    )
    uris: list[str] = []
    for i, path in enumerate(local_paths, start=1):
        uri = upload_to_gcs(
            path, bucket_name, gcs_prefix,
            timeout=timeout,
            file_index=i,
            total_files=total_count,
        )
        uris.append(uri)
    return uris


def list_gcs_uris_under_prefix(bucket_name: str, gcs_prefix: str) -> list[str]:
    """
    List all audio object URIs under a given GCS prefix, sorted alphanumerically.

    Args:
        bucket_name: GCS bucket name.
        gcs_prefix: Prefix (folder) to search in bucket.

    Returns:
        List of gs:// URIs for matching audio objects, sorted in natural order.

    Raises:
        ProcessingError: If no audio blobs are found or on GCS failure.
    """
    client = _get_client()
    clean_prefix = gcs_prefix.strip("/")
    try:
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=clean_prefix))
    except Exception as e:
        raise ProcessingError(
            f"Failed to list GCS objects in bucket '{bucket_name}' under prefix '{clean_prefix}'.\n"
            f"Original error: {e}"
        ) from e

    from src.config import SUPPORTED_EXTENSIONS
    valid_exts = tuple(SUPPORTED_EXTENSIONS) + (".flac",)

    audio_blobs = [
        b for b in blobs
        if any(b.name.lower().endswith(ext) for ext in valid_exts)
    ]

    if not audio_blobs:
        raise ProcessingError(
            f"No audio files found in gs://{bucket_name}/{clean_prefix}/. "
            f"Expected extensions: {', '.join(valid_exts)}"
        )

    # Sort blobs alphanumerically so chunks are ordered: chunk_000, chunk_001, etc.
    audio_blobs.sort(key=lambda b: b.name)
    uris = [f"gs://{bucket_name}/{b.name}" for b in audio_blobs]
    logger.info("Found %d audio file(s) under gs://%s/%s/", len(uris), bucket_name, clean_prefix)
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
