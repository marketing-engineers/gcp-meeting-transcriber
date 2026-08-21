"""Tests for src/transcribe_gcs.py — GCS re-transcription pipeline and CLI."""

from unittest.mock import MagicMock, patch
import pytest

from src.audio import ProcessingError
from src.transcribe_gcs import (
    derive_base_name,
    parse_gcs_target,
    transcribe_from_gcs,
)


class TestParseGcsTarget:
    """Tests for parse_gcs_target utility."""

    def test_parse_full_gs_uri(self):
        bucket, prefix = parse_gcs_target("gs://my-bucket/path/to/folder")
        assert bucket == "my-bucket"
        assert prefix == "path/to/folder"

    def test_parse_full_gs_uri_trailing_slash(self):
        bucket, prefix = parse_gcs_target("gs://my-bucket/path/to/folder/")
        assert bucket == "my-bucket"
        assert prefix == "path/to/folder"

    def test_parse_full_gs_uri_root(self):
        bucket, prefix = parse_gcs_target("gs://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""

    def test_parse_with_default_bucket(self):
        bucket, prefix = parse_gcs_target("sample-meeting_prefix", default_bucket="example-bucket")
        assert bucket == "example-bucket"
        assert prefix == "sample-meeting_prefix"

    def test_parse_with_default_bucket_prefixed(self):
        bucket, prefix = parse_gcs_target("example-bucket/sample-meeting_prefix", default_bucket="example-bucket")
        assert bucket == "example-bucket"
        assert prefix == "sample-meeting_prefix"

    def test_parse_two_part_without_default_bucket(self):
        bucket, prefix = parse_gcs_target("my-bucket/subfolder/prefix")
        assert bucket == "my-bucket"
        assert prefix == "subfolder/prefix"

    def test_parse_invalid_fails(self):
        with pytest.raises(ProcessingError, match="Cannot determine GCS bucket"):
            parse_gcs_target("invalid_single_token", default_bucket=None)


class TestDeriveBaseName:
    """Tests for derive_base_name."""

    def test_derive_from_timestamped_prefix(self):
        prefix = "sample-meeting-2026-08-20_20260820_152628"
        base = derive_base_name(prefix, [])
        assert base == "sample-meeting-2026-08-20"

    def test_derive_from_plain_prefix(self):
        prefix = "interview-recording"
        base = derive_base_name(prefix, [])
        assert base == "interview-recording"

    def test_derive_fallback_to_first_uri(self):
        uris = ["gs://bucket/prefix/interview_chunk_000.flac"]
        base = derive_base_name("", uris)
        assert base == "interview"


class TestTranscribeFromGcs:
    """Tests for transcribe_from_gcs workflow."""

    @patch("src.transcribe_gcs.save_transcript")
    @patch("src.transcribe_gcs.stitch_plain_transcripts")
    @patch("src.transcribe_gcs.transcribe_files")
    @patch("src.transcribe_gcs.list_gcs_uris_under_prefix")
    @patch("src.transcribe_gcs.validate_gcp_config")
    def test_transcribe_from_gcs_dutch(
        self, mock_val, mock_list, mock_tf, mock_stitch, mock_save
    ):
        mock_list.return_value = ["gs://b/p/chunk_000.flac", "gs://b/p/chunk_001.flac"]
        mock_tf.return_value = [MagicMock(), MagicMock()]
        mock_stitch.return_value = "Stitched text"
        mock_save.return_value = ("transcripts/t.txt", "transcripts/t.json")

        txt, raw_json = transcribe_from_gcs(
            target="gs://b/meeting_20260820_152628",
            lang="nl-NL",
        )

        assert txt == "transcripts/t.txt"
        assert raw_json == "transcripts/t.json"
        mock_tf.assert_called_once()
        # Diarization should be False for Dutch
        assert mock_tf.call_args.kwargs["enable_diarization"] is False

    @patch("src.transcribe_gcs.save_transcript")
    @patch("src.transcribe_gcs.stitch_diarized_transcripts")
    @patch("src.transcribe_gcs.transcribe_files")
    @patch("src.transcribe_gcs.list_gcs_uris_under_prefix")
    @patch("src.transcribe_gcs.validate_gcp_config")
    def test_transcribe_from_gcs_english(
        self, mock_val, mock_list, mock_tf, mock_stitch, mock_save
    ):
        mock_list.return_value = ["gs://b/p/chunk_000.flac"]
        mock_tf.return_value = [MagicMock()]
        mock_stitch.return_value = "Speaker 1: Hi"
        mock_save.return_value = ("transcripts/t.txt", "transcripts/t.json")

        txt, raw_json = transcribe_from_gcs(
            target="gs://b/meeting_20260820_152628",
            lang="en-US",
            speaker_count=3,
        )

        assert txt == "transcripts/t.txt"
        assert raw_json == "transcripts/t.json"
        # Diarization should be True for English with 3 speakers
        assert mock_tf.call_args.kwargs["enable_diarization"] is True
        assert mock_tf.call_args.kwargs["speaker_count"] == 3
