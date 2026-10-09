"""Tests for src/gcs.py — upload progress tracking, bucket listing, and error handling."""

import os
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as gcp_exceptions

from src.audio import ProcessingError
from src.gcs import (
    ProgressFileReader,
    delete_gcs_prefix,
    list_gcs_uris_under_prefix,
    upload_files_to_gcs,
    upload_to_gcs,
    _get_client,
)


class TestGcsClientCreation:
    """Tests for _get_client credential pass-through and caching."""

    @patch("google.cloud.storage.Client")
    @patch("src.auth.get_credentials")
    def test_get_client_passes_credentials(self, mock_get_creds, mock_storage_client):
        """_get_client passes resolved credentials to storage.Client."""
        mock_creds = MagicMock()
        mock_get_creds.return_value = mock_creds

        client = _get_client()

        mock_get_creds.assert_called_once()
        mock_storage_client.assert_called_once_with(project="test-project", credentials=mock_creds)
        assert client == mock_storage_client.return_value


class TestProgressFileReader:
    """Tests for ProgressFileReader."""

    def test_progress_file_reader_reads_and_calls_callback(self, tmp_path):
        """ProgressFileReader streams data and triggers progress callback."""
        test_file = tmp_path / "test.bin"
        content = b"x" * 1000
        test_file.write_bytes(content)

        callback_calls = []

        def callback(bytes_read, total_bytes, pct, speed_mb):
            callback_calls.append((bytes_read, total_bytes, pct))

        with ProgressFileReader(str(test_file), callback=callback, min_interval_seconds=0.0) as reader:
            data1 = reader.read(400)
            data2 = reader.read(600)

        assert data1 + data2 == content
        assert len(callback_calls) >= 1
        last_call = callback_calls[-1]
        assert last_call[0] == 1000
        assert last_call[1] == 1000
        assert last_call[2] == 100.0


class TestUploadToGcs:
    """Tests for upload_to_gcs and upload_files_to_gcs."""

    @patch("src.gcs._get_client")
    def test_upload_to_gcs_success(self, mock_get_client, tmp_path):
        """upload_to_gcs uploads file and returns gs:// URI."""
        test_file = tmp_path / "audio.flac"
        test_file.write_bytes(b"flacdata" * 100)

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_get_client.return_value = mock_client

        uri = upload_to_gcs(str(test_file), "my-bucket", "my-prefix")

        assert uri == "gs://my-bucket/my-prefix/audio.flac"
        mock_blob.upload_from_file.assert_called_once()

    @patch("src.gcs._get_client")
    def test_upload_to_gcs_bucket_not_found(self, mock_get_client, tmp_path):
        """upload_to_gcs raises ProcessingError on bucket NotFound."""
        test_file = tmp_path / "audio.flac"
        test_file.write_bytes(b"data")

        mock_client = MagicMock()
        mock_client.bucket.side_effect = gcp_exceptions.NotFound("Bucket not found")
        mock_get_client.return_value = mock_client

        with pytest.raises(ProcessingError, match="GCS bucket 'my-bucket' not found"):
            upload_to_gcs(str(test_file), "my-bucket", "my-prefix")

    @patch("src.gcs.upload_to_gcs")
    def test_upload_files_to_gcs(self, mock_upload, tmp_path):
        """upload_files_to_gcs uploads each file sequentially and passes indexing."""
        f1 = tmp_path / "c1.flac"
        f2 = tmp_path / "c2.flac"
        f1.write_bytes(b"1")
        f2.write_bytes(b"2")

        mock_upload.side_effect = ["gs://b/p/c1.flac", "gs://b/p/c2.flac"]

        uris = upload_files_to_gcs([str(f1), str(f2)], "b", "p")

        assert uris == ["gs://b/p/c1.flac", "gs://b/p/c2.flac"]
        assert mock_upload.call_count == 2
        call1 = mock_upload.call_args_list[0]
        assert call1.kwargs["file_index"] == 1
        assert call1.kwargs["total_files"] == 2
        call2 = mock_upload.call_args_list[1]
        assert call2.kwargs["file_index"] == 2
        assert call2.kwargs["total_files"] == 2


class TestListAndCleanupGcs:
    """Tests for listing and deleting GCS prefixes."""

    @patch("src.gcs._get_client")
    def test_list_gcs_uris_under_prefix(self, mock_get_client):
        """list_gcs_uris_under_prefix sorts and returns audio blobs."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        blob1 = MagicMock(name="prefix/chunk_001.flac")
        blob1.name = "prefix/chunk_001.flac"
        blob0 = MagicMock(name="prefix/chunk_000.flac")
        blob0.name = "prefix/chunk_000.flac"
        mock_bucket.list_blobs.return_value = [blob1, blob0]
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        uris = list_gcs_uris_under_prefix("my-bucket", "prefix")

        assert uris == [
            "gs://my-bucket/prefix/chunk_000.flac",
            "gs://my-bucket/prefix/chunk_001.flac",
        ]

    @patch("src.gcs._get_client")
    def test_delete_gcs_prefix(self, mock_get_client):
        """delete_gcs_prefix deletes all objects in prefix."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        blob = MagicMock()
        mock_bucket.list_blobs.return_value = [blob]
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        delete_gcs_prefix("my-bucket", "prefix")
        blob.delete.assert_called_once()
