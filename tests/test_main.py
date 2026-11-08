"""Tests for src/main.py — CLI argument parsing, route selection, decision trees, and interactive prompt."""

import json
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from src.audio import ProcessingError
from src.main import (
    EXIT_DECISION_REQUIRED,
    _prompt_speaker_count,
    main,
    process_file,
    resolve_transcription_plan,
)


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
        test_args = ["prog", "input.mp4", "--lang", "en-us", "--prepare-only"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.process_file") as mock_pf:
                with patch("src.main.setup_logging"):
                    main()
                    assert mock_pf.call_args.kwargs.get("prepare_only") is True
                    assert mock_pf.call_args.kwargs.get("input_path") == "input.mp4"
                    assert mock_pf.call_args.kwargs.get("lang") == "en-us"

    def test_default_no_prepare_only(self):
        """Without --prepare-only, prepare_only=False is passed."""
        test_args = ["prog", "input.mp4", "--lang", "nl-nl"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.process_file") as mock_pf:
                with patch("src.main.setup_logging"):
                    main()
                    assert mock_pf.call_args.kwargs.get("prepare_only") is False

    def test_from_gcs_option_accepted(self):
        """--from-gcs option triggers transcribe_from_gcs."""
        test_args = ["prog", "--from-gcs", "sample-meeting_prefix", "--lang", "nl-nl"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.transcribe_from_gcs") as mock_tfgcs:
                with patch("src.main.setup_logging"):
                    mock_tfgcs.return_value = ("txt_path", "json_path")
                    main()
                    assert mock_tfgcs.call_args.kwargs.get("target") == "sample-meeting_prefix"
                    assert mock_tfgcs.call_args.kwargs.get("lang") == "nl-nl"

    def test_model_and_diarize_flags_accepted(self):
        """--model, --diarize, and --mode flags are parsed and passed to process_file."""
        test_args = [
            "prog", "input.mp4", "--lang", "nl-nl",
            "--model", "gemini-3.5-transcribe", "--diarize", "--mode", "smart"
        ]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.process_file") as mock_pf:
                with patch("src.main.setup_logging"):
                    main()
                    assert mock_pf.call_args.kwargs.get("model") == "gemini-3.5-transcribe"
                    assert mock_pf.call_args.kwargs.get("diarize") is True
                    assert mock_pf.call_args.kwargs.get("mode") == "smart"

    def test_no_diarize_flag_accepted(self):
        """--no-diarize sets diarize=False."""
        test_args = ["prog", "input.mp4", "--lang", "en-us", "--no-diarize"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.process_file") as mock_pf:
                with patch("src.main.setup_logging"):
                    main()
                    assert mock_pf.call_args.kwargs.get("diarize") is False

    def test_missing_both_input_file_and_from_gcs_errors(self):
        """Providing neither input_file nor --from-gcs raises SystemExit."""
        test_args = ["prog", "--lang", "nl-nl"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.setup_logging"):
                with pytest.raises(SystemExit):
                    main()

    def test_providing_both_input_file_and_from_gcs_errors(self):
        """Providing both input_file and --from-gcs raises SystemExit."""
        test_args = ["prog", "input.mp4", "--from-gcs", "prefix", "--lang", "nl-nl"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.setup_logging"):
                with pytest.raises(SystemExit):
                    main()

    def test_check_auth_flag_success(self):
        """--check-auth flag executes check_auth and exits with 0 on success."""
        test_args = ["prog", "--check-auth"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.setup_logging"), \
                 patch("src.main.check_auth", return_value=True) as mock_check:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
                mock_check.assert_called_once()

    def test_check_auth_flag_failure(self):
        """--check-auth flag executes check_auth and exits with 1 on failure."""
        test_args = ["prog", "--check-auth"]
        with patch.object(sys, "argv", test_args):
            with patch("src.main.setup_logging"), \
                 patch("src.main.check_auth", return_value=False) as mock_check:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1
                mock_check.assert_called_once()


class TestProcessFileRouting:
    """Tests for routing logic, decision tree, and execution."""

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_backend_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration", return_value=500.0)
    @patch("src.main.detect_silences", return_value=[])
    @patch("src.main.compute_chunk_boundaries", return_value=[])
    @patch("src.main.get_gemini_backend_mode", return_value="api_key")
    @patch("src.main.transcribe_gemini_files", return_value=[MagicMock()])
    @patch("src.main.stitch_diarized_transcripts", return_value="Speaker 1: Hallo")
    @patch("src.main.save_transcript", return_value=("t.txt", "t.json"))
    @patch("src.main.os.remove")
    def test_short_recording_auto_selects_gemini_with_diarization(
        self, mock_remove, mock_save, mock_stitch, mock_tgf, mock_backend,
        mock_boundaries, mock_silences, mock_duration, mock_extract, mock_val, mock_ext, mock_exists
    ):
        """Short recording (500s <= 28m) auto-selects gemini-3.5-transcribe with diarization for Dutch."""
        with patch("src.main._has_gemini_backend", return_value=True):
            process_file("input.mp4", "nl-nl", prepare_only=False)

        mock_tgf.assert_called_once()
        assert mock_tgf.call_args.kwargs["diarize"] is True
        mock_stitch.assert_called_once()
        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["duration"] == 500.0
        assert save_kwargs["model"] == "gemini-3.5-transcribe"
        assert save_kwargs["diarize"] is True

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_backend_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration", return_value=2000.0)
    @patch("src.main.detect_silences", return_value=[])
    @patch("src.main.compute_chunk_boundaries", return_value=[])
    @patch("src.main.upload_to_gcs", return_value="gs://b/p/file.flac")
    @patch("src.main.transcribe_single_file", return_value=MagicMock())
    @patch("src.main.stitch_diarized_transcripts", return_value="Speaker 1: Hi")
    @patch("src.main.save_transcript", return_value=("t.txt", "t.json"))
    @patch("src.main.delete_gcs_prefix")
    @patch("src.main.os.remove")
    def test_english_medium_recording_auto_switches_to_chirp_monolithic(
        self, mock_remove, mock_delete_gcs, mock_save, mock_stitch, mock_tsf,
        mock_upload, mock_boundaries, mock_silences, mock_duration, mock_extract, mock_val, mock_ext, mock_exists
    ):
        """English recording between 28m and 45m auto-switches to chirp_3 with diarization for monolithic pass."""
        with patch("src.main._has_chirp_backend", return_value=True):
            process_file("input.mp4", "en-us", prepare_only=False)

        mock_tsf.assert_called_once()
        assert mock_tsf.call_args.kwargs["enable_diarization"] is True
        mock_stitch.assert_called_once()

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.os.remove")
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_backend_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration", return_value=4500.0)
    def test_long_recording_decision_required_non_interactive(
        self, mock_duration, mock_extract, mock_val, mock_ext, mock_remove, mock_exists
    ):
        """Long recording (> 45m) without --diarize/--no-diarize exits with code 2 in non-interactive mode and cleans up FLAC."""
        with patch("src.main._has_gemini_backend", return_value=True), \
             patch("src.main._has_chirp_backend", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                process_file("input.mp4", "nl-nl", prepare_only=False, interactive=False)
            assert exc_info.value.code == EXIT_DECISION_REQUIRED
            mock_remove.assert_called_once()

    def test_long_recording_auto_lang_with_diarize_routes_to_gemini(self):
        """Long recording with --lang auto and --diarize routes to gemini-3.5-transcribe, not chirp_3."""
        with patch("src.main._has_gemini_backend", return_value=True), \
             patch("src.main._has_chirp_backend", return_value=True):
            plan = resolve_transcription_plan(
                duration=3000.0,
                lang="auto",
                requested_model="auto",
                requested_diarize=True,
                mode="verbatim",
                speaker_count=None,
                interactive=False,
            )
            assert plan.model == "gemini-3.5-transcribe"
            assert plan.diarize is True

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_backend_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration", return_value=4500.0)
    @patch("src.main.detect_silences", return_value=[])
    @patch("src.main.compute_chunk_boundaries", return_value=[2600.0])
    @patch("src.main.split_flac", return_value=["chunk1.flac", "chunk2.flac"])
    @patch("src.main.upload_files_to_gcs", return_value=["gs://b/p/c1.flac", "gs://b/p/c2.flac"])
    @patch("src.main.transcribe_files", return_value=[MagicMock(), MagicMock()])
    @patch("src.main.stitch_plain_transcripts", return_value="plain text")
    @patch("src.main.save_transcript", return_value=("t.txt", "t.json"))
    @patch("src.main.delete_gcs_prefix")
    @patch("src.main.os.remove")
    def test_long_recording_with_explicit_no_diarize(
        self, mock_remove, mock_delete_gcs, mock_save, mock_stitch, mock_tf,
        mock_upload, mock_split, mock_boundaries, mock_silences, mock_duration,
        mock_extract, mock_val, mock_ext, mock_exists
    ):
        """Long recording with --no-diarize does not require a decision and proceeds."""
        with patch("src.main._has_chirp_backend", return_value=True):
            process_file("input.mp4", "nl-nl", diarize=False, prepare_only=False)

        mock_tf.assert_called_once()
        assert mock_tf.call_args.kwargs["enable_diarization"] is False
        mock_stitch.assert_called_once()

    @patch("src.main.os.path.exists", return_value=True)
    @patch("src.main.validate_extension", return_value=True)
    @patch("src.main.validate_backend_config")
    @patch("src.main.extract_audio_to_flac")
    @patch("src.main.get_audio_duration", return_value=3000.0)
    @patch("src.main.detect_silences", return_value=[])
    @patch("src.main.compute_chunk_boundaries", return_value=[])
    @patch("src.main.get_gemini_backend_mode", return_value="api_key")
    @patch("src.main.transcribe_gemini_files", return_value=[MagicMock()])
    @patch("src.main.stitch_diarized_transcripts", return_value="text")
    @patch("src.main.save_transcript", return_value=("t.txt", "t.json"))
    @patch("src.main.os.remove")
    def test_interactive_decision_prompt_selection(
        self, mock_remove, mock_save, mock_stitch, mock_tgf, mock_backend,
        mock_boundaries, mock_silences, mock_duration, mock_extract, mock_val, mock_ext, mock_exists
    ):
        """In interactive mode, user selection (1) resolves and executes the selected plan."""
        with patch("src.main._has_gemini_backend", return_value=True), \
             patch("builtins.input", return_value="1"):
            process_file("input.mp4", "nl-nl", interactive=True)

        mock_tgf.assert_called_once()
        assert mock_tgf.call_args.kwargs["diarize"] is True
