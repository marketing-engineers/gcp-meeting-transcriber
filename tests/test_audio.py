"""Tests for src/audio.py — FFmpeg command construction and error handling (mocked subprocess)."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from src.audio import (
    ProcessingError,
    _check_ffmpeg_available,
    extract_audio_to_flac,
    split_flac,
)


class TestFfmpegAvailabilityCheck:
    """Tests for _check_ffmpeg_available."""

    def test_ffmpeg_found(self):
        """No error raised when ffmpeg is on PATH."""
        with patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
            _check_ffmpeg_available()  # Should not raise

    def test_ffmpeg_not_found(self):
        """ProcessingError raised when ffmpeg is not on PATH."""
        with patch("src.audio.shutil.which", return_value=None):
            with pytest.raises(ProcessingError, match="ffmpeg not found"):
                _check_ffmpeg_available()


class TestExtractAudioToFlac:
    """Tests for extract_audio_to_flac."""

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_successful_extraction(self, mock_run, mock_which):
        """Successful extraction calls ffmpeg with correct args."""
        mock_run.return_value = MagicMock(returncode=0)

        extract_audio_to_flac("input.mp4", "output.flac")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-vn" in args
        assert "-c:a" in args
        assert "flac" in args
        assert "input.mp4" in args
        assert "output.flac" in args

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_ffmpeg_failure_raises_processing_error(self, mock_run, mock_which):
        """FFmpeg failure is wrapped in ProcessingError with stderr."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg", "-i", "input.mp4"],
            stderr="Error: invalid input file",
        )

        with pytest.raises(ProcessingError, match="FFmpeg failed during audio extraction"):
            extract_audio_to_flac("input.mp4", "output.flac")


class TestSplitFlac:
    """Tests for split_flac."""

    @patch("src.audio.os.listdir")
    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_successful_split_returns_paths(self, mock_run, mock_which, mock_listdir):
        """Successful split returns sorted list of chunk file paths."""
        mock_run.return_value = MagicMock(returncode=0)
        mock_listdir.return_value = [
            "meeting_chunk_002.flac",
            "meeting_chunk_000.flac",
            "meeting_chunk_001.flac",
            "other_file.txt",
        ]

        result = split_flac("input.flac", "/output", "meeting", 900)

        assert result == [
            "/output/meeting_chunk_000.flac",
            "/output/meeting_chunk_001.flac",
            "/output/meeting_chunk_002.flac",
        ]

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_ffmpeg_failure_raises_processing_error(self, mock_run, mock_which):
        """FFmpeg chunking failure is wrapped in ProcessingError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="Error: segment failed",
        )

        with pytest.raises(ProcessingError, match="FFmpeg failed during FLAC chunking"):
            split_flac("input.flac", "/output", "meeting", 900)
