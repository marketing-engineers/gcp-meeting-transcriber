"""Tests for src/main.py — CLI argument parsing, route selection, interactive prompt."""

import sys
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest

from src.main import _prompt_speaker_count, process_file
from src.audio import ProcessingError


class TestPromptSpeakerCount:
    """Tests for the interactive speaker count prompt."""

    def test_auto_detect_on_empty_input(self):
        """Empty input (Enter) returns None for auto-detect."""
        with patch("builtins.input", return_value=""):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = _prompt_speaker_count()
        assert result is None

    def test_valid_integer_input(self):
        """Valid positive integer returns that number."""
        with patch("builtins.input", return_value="3"):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = _prompt_speaker_count()
        assert result == 3

    def test_non_interactive_returns_none(self):
        """Non-TTY stdin returns None (auto-detect) without prompting."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = _prompt_speaker_count()
        assert result is None

    def test_invalid_then_valid_input(self):
        """Invalid input triggers re-prompt, then valid input is accepted."""
        inputs = iter(["abc", "2"])
        with patch("builtins.input", side_effect=inputs):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = _prompt_speaker_count()
        assert result == 2

    def test_negative_then_valid_input(self):
        """Negative number triggers re-prompt."""
        inputs = iter(["-1", "0", "2"])
        with patch("builtins.input", side_effect=inputs):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = _prompt_speaker_count()
        assert result == 2

    def test_eof_returns_none(self):
        """EOFError (Ctrl+D) returns None."""
        with patch("builtins.input", side_effect=EOFError):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = _prompt_speaker_count()
        assert result is None

    def test_keyboard_interrupt_returns_none(self):
        """KeyboardInterrupt (Ctrl+C) returns None."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = _prompt_speaker_count()
        assert result is None


class TestProcessFileValidation:
    """Tests for process_file input validation."""

    def test_missing_file_raises_error(self):
        """Non-existent file raises ProcessingError."""
        with pytest.raises(ProcessingError, match="Input file not found"):
            process_file("/nonexistent/path.mp4", "en-us", prepare_only=True)

    def test_unsupported_extension_raises_error(self, tmp_path):
        """Unsupported extension raises ProcessingError."""
        bad_file = tmp_path / "file.pdf"
        bad_file.touch()
        with pytest.raises(ProcessingError, match="Unsupported file extension"):
            process_file(str(bad_file), "en-us", prepare_only=True)

    def test_unsupported_language_raises_error(self, tmp_path):
        """Unsupported language code raises ProcessingError."""
        good_file = tmp_path / "file.mp4"
        good_file.touch()
        with pytest.raises(ProcessingError, match="Unsupported language route"):
            process_file(str(good_file), "fr-fr", prepare_only=True)


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing via main()."""

    def test_prepare_only_flag_accepted(self):
        """--prepare-only flag is parsed correctly."""
        import argparse
        from src.main import main

        test_args = ["prog", "input.mp4", "--lang", "en-us", "--prepare-only"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.process_file") as mock_pf:
                with patch("src.main.setup_logging"):
                    main()
                    mock_pf.assert_called_once_with("input.mp4", "en-us", prepare_only=True)

    def test_default_no_prepare_only(self):
        """Without --prepare-only, prepare_only=False is passed."""
        test_args = ["prog", "input.mp4", "--lang", "nl-nl"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.process_file") as mock_pf:
                with patch("src.main.setup_logging"):
                    from src.main import main
                    main()
                    mock_pf.assert_called_once_with("input.mp4", "nl-nl", prepare_only=False)


class TestProcessFileDutchRoute:
    """Tests for process_file routing through Dutch route (monolithic vs chunked)."""

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_gcp_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration")
    @patch("src.main.upload_to_gcs")
    @patch("src.main.transcribe_single_file")
    @patch("src.main.stitch_plain_transcripts")
    @patch("src.main.save_transcript")
    @patch("src.main.delete_gcs_prefix")
    @patch("src.main.os.remove")
    def test_dutch_monolithic_route(
        self, mock_remove, mock_delete_gcs, mock_save_tr, mock_stitch,
        mock_transcribe, mock_upload, mock_duration, mock_extract,
        mock_validate_gcp, mock_validate_ext, mock_exists
    ):
        """Dutch file <= CHUNK_LENGTH_SECONDS_DUTCH goes through monolithic route."""
        mock_duration.return_value = 200.0  # 3.3 minutes (within 5-min default threshold)
        mock_upload.return_value = "gs://bucket/prefix/file.flac"
        mock_transcribe.return_value = MagicMock()
        mock_stitch.return_value = "transcribed text"
        mock_save_tr.return_value = ("txt_path", "json_path")

        process_file("input.mp4", "nl-nl", prepare_only=False)

        mock_extract.assert_called_once()
        mock_duration.assert_called_once()
        mock_upload.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_stitch.assert_called_once()
        mock_save_tr.assert_called_once()
        mock_delete_gcs.assert_called_once()
        mock_remove.assert_called_once()

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_gcp_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration")
    @patch("src.main.split_flac")
    @patch("src.main.upload_files_to_gcs")
    @patch("src.main.transcribe_files")
    @patch("src.main.stitch_plain_transcripts")
    @patch("src.main.save_transcript")
    @patch("src.main.delete_gcs_prefix")
    @patch("src.main.os.remove")
    def test_dutch_chunked_route(
        self, mock_remove, mock_delete_gcs, mock_save_tr, mock_stitch,
        mock_transcribe, mock_upload_files, mock_split, mock_duration, mock_extract,
        mock_validate_gcp, mock_validate_ext, mock_exists
    ):
        """Dutch file > CHUNK_LENGTH_SECONDS_DUTCH goes through chunked route."""
        mock_duration.return_value = 1800.0  # 30 minutes (exceeds 5-min default threshold)
        mock_split.return_value = ["chunk1.flac", "chunk2.flac"]
        mock_upload_files.return_value = ["gs://b/p/c1.flac", "gs://b/p/c2.flac"]
        mock_transcribe.return_value = [MagicMock(), MagicMock()]
        mock_stitch.return_value = "stitched text"
        mock_save_tr.return_value = ("txt_path", "json_path")

        process_file("input.mp4", "nl-nl", prepare_only=False)

        mock_extract.assert_called_once()
        mock_duration.assert_called_once()
        mock_split.assert_called_once()
        mock_upload_files.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_stitch.assert_called_once()
        mock_save_tr.assert_called_once()
        mock_delete_gcs.assert_called_once()
        # Should clean up chunks + original file
        assert mock_remove.call_count == 3

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration")
    @patch("src.main.os.remove")
    def test_dutch_monolithic_prepare_only(
        self, mock_remove, mock_duration, mock_extract, mock_validate_ext, mock_exists
    ):
        """Dutch file <= CHUNK_LENGTH_SECONDS_DUTCH in prepare_only returns early without upload or cleanup."""
        mock_duration.return_value = 200.0

        process_file("input.mp4", "nl-nl", prepare_only=True)

        mock_extract.assert_called_once()
        mock_duration.assert_called_once()
        mock_remove.assert_not_called()

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration")
    @patch("src.main.split_flac")
    @patch("src.main.os.remove")
    def test_dutch_chunked_prepare_only(
        self, mock_remove, mock_split, mock_duration, mock_extract, mock_validate_ext, mock_exists
    ):
        """Dutch file > CHUNK_LENGTH_SECONDS_DUTCH in prepare_only chunks and cleans up only monolithic flac."""
        mock_duration.return_value = 1800.0
        mock_split.return_value = ["chunk1.flac", "chunk2.flac"]

        process_file("input.mp4", "nl-nl", prepare_only=True)

        mock_extract.assert_called_once()
        mock_duration.assert_called_once()
        mock_split.assert_called_once()
        mock_remove.assert_called_once()  # Cleans up monolithic file
