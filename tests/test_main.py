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
