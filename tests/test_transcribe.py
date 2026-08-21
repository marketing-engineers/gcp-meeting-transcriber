"""Tests for src/transcribe.py — parallel chunk transcription and progress polling."""

from unittest.mock import MagicMock, patch

import pytest

from src.audio import ProcessingError
from src.transcribe import transcribe_files, transcribe_single_file


def _make_mock_op(done: bool = True, progress: int = 100, result_val: MagicMock | None = None) -> MagicMock:
    """Helper to construct a mock GCP LRO operation."""
    op = MagicMock()
    op.done.return_value = done
    if result_val is None:
        result_val = MagicMock()
    op.result.return_value = result_val

    meta = MagicMock()
    item_meta = MagicMock()
    item_meta.progress_percent = progress
    meta.transcription_metadata = {"gs://bucket/chunk.flac": item_meta}
    op.metadata = meta

    return op


class TestTranscribeFilesParallel:
    """Tests for transcribe_files parallel execution."""

    def test_transcribe_empty_list_returns_empty(self):
        """Empty list of URIs returns empty list immediately."""
        res = transcribe_files(
            gcs_uris=[],
            lang="en-us",
            enable_diarization=True,
            speaker_count=2,
            project_id="p",
            region="eu",
            model="chirp_3",
        )
        assert res == []

    @patch("src.transcribe._get_speech_client")
    @patch("time.sleep")
    def test_transcribe_files_parallel_success(self, mock_sleep, mock_get_client):
        """Multiple chunks are submitted in parallel and complete successfully."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        op1 = _make_mock_op(done=False)
        op2 = _make_mock_op(done=False)

        mock_client.batch_recognize.side_effect = [op1, op2]

        uris = ["gs://b/chunk1.flac", "gs://b/chunk2.flac"]

        # Mock op.done.side_effect to return False on first loop, then True
        op1.done.side_effect = [False, True]
        op2.done.side_effect = [False, True]

        results = transcribe_files(
            gcs_uris=uris,
            lang="nl-nl",
            enable_diarization=False,
            speaker_count=None,
            project_id="test-proj",
            region="eu",
            model="chirp_3",
            poll_interval=0,
        )

        assert len(results) == 2
        assert mock_client.batch_recognize.call_count == 2

    @patch("src.transcribe._get_speech_client")
    @patch("time.sleep")
    def test_transcribe_files_batches_and_retries_429(self, mock_sleep, mock_get_client):
        """Chunks are split into batches, and transient 429 errors during polling are retried."""
        from google.api_core import exceptions as gcp_exceptions

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        op1 = _make_mock_op(done=False)
        op2 = _make_mock_op(done=False)
        op3 = _make_mock_op(done=False)

        mock_client.batch_recognize.side_effect = [op1, op2, op3]

        uris = ["gs://b/chunk1.flac", "gs://b/chunk2.flac", "gs://b/chunk3.flac"]

        # op1 in batch 1 throws ResourceExhausted on first check, then succeeds
        op1.done.side_effect = [gcp_exceptions.ResourceExhausted("Rate exceeded"), True]
        op2.done.side_effect = [True]
        # op3 in batch 2 completes immediately
        op3.done.side_effect = [True]

        results = transcribe_files(
            gcs_uris=uris,
            lang="nl-nl",
            enable_diarization=False,
            speaker_count=None,
            project_id="test-proj",
            region="eu",
            model="chirp_3",
            batch_size=2,  # batch 1: [chunk1, chunk2], batch 2: [chunk3]
            poll_interval=0,
        )

        assert len(results) == 3
        assert mock_client.batch_recognize.call_count == 3

    @patch("src.transcribe._get_speech_client")
    @patch("time.sleep")
    def test_transcribe_files_max_timeout_raises_error(self, mock_sleep, mock_get_client):
        """Batch exceeding max_timeout raises ProcessingError."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        op = _make_mock_op(done=False, progress=10)
        mock_client.batch_recognize.return_value = op

        with pytest.raises(ProcessingError, match="exceeded maximum allowable time"):
            transcribe_files(
                gcs_uris=["gs://b/chunk1.flac"],
                lang="nl-nl",
                enable_diarization=False,
                speaker_count=None,
                project_id="test-proj",
                region="eu",
                model="chirp_3",
                max_timeout=0.001,  # Force fast max timeout
                poll_interval=0,
            )


class TestTranscribeSingleFile:
    """Tests for transcribe_single_file wrapper."""

    @patch("src.transcribe.transcribe_files")
    def test_transcribe_single_file_delegates(self, mock_tf):
        """transcribe_single_file calls transcribe_files with single URI."""
        mock_resp = MagicMock()
        mock_tf.return_value = [mock_resp]

        res = transcribe_single_file(
            gcs_uri="gs://b/file.flac",
            lang="en-us",
            enable_diarization=True,
            speaker_count=2,
            project_id="proj",
            region="eu",
            model="chirp_3",
            max_timeout=1800,
        )

        assert res == mock_resp
        mock_tf.assert_called_once()


class TestBuildRecognitionConfig:
    """Tests for _build_recognition_config including CustomPromptConfig rules."""

    @patch("src.config.TRANSCRIPTION_CUSTOM_PROMPT", "Test custom prompt instruction")
    def test_chirp_model_uses_custom_prompt(self):
        """Chirp model includes CustomPromptConfig when TRANSCRIPTION_CUSTOM_PROMPT is set."""
        from src.transcribe import _build_recognition_config
        config = _build_recognition_config(
            lang="nl-NL",
            model="chirp_3",
            enable_diarization=False,
            speaker_count=None,
        )
        assert config.features.custom_prompt_config is not None
        assert config.features.custom_prompt_config.custom_prompt == "Test custom prompt instruction"

    @patch("src.config.TRANSCRIPTION_CUSTOM_PROMPT", "Test custom prompt instruction")
    def test_non_chirp_model_omits_custom_prompt_and_logs_warning(self, caplog):
        """Non-Chirp model omits CustomPromptConfig and logs a warning when prompt is set."""
        from src.transcribe import _build_recognition_config
        import logging

        with caplog.at_level(logging.WARNING):
            config = _build_recognition_config(
                lang="nl-NL",
                model="long",
                enable_diarization=False,
                speaker_count=None,
            )
        assert not config.features.custom_prompt_config.custom_prompt
        assert "only supported for Chirp models" in caplog.text


class TestFormatAsciiMatrix:
    """Tests for format_ascii_matrix rendering."""

    def test_format_ascii_matrix_rendering(self):
        """ASCII matrix produces expected box structure and counts."""
        from src.transcribe import format_ascii_matrix

        statuses = {
            1: "completed",
            2: "completed",
            3: "in_progress",
            4: "pending",
            5: "pending",
        }
        output = format_ascii_matrix(
            total_chunks=5,
            chunk_statuses=statuses,
            current_batch=1,
            total_batches=1,
            elapsed_seconds=30,
            columns=5,
        )

        assert "[TRANSCRIPTION MATRIX] Batch 1/1 | Completed: 2/5 (40%) | Elapsed: 30s" in output
        assert "01 [●]" in output
        assert "02 [●]" in output
        assert "03 [◌]" in output
        assert "04 [·]" in output
        assert "05 [·]" in output
        assert "Legend:" in output


