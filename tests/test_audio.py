"""Tests for src/audio.py — FFmpeg command construction and error handling (mocked subprocess)."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from src.audio import (
    ProcessingError,
    _check_ffmpeg_available,
    extract_audio_to_flac,
    split_flac,
    get_audio_duration,
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


class TestGetAudioDuration:
    """Tests for get_audio_duration."""

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffprobe")
    @patch("src.audio.subprocess.run")
    def test_successful_get_duration(self, mock_run, mock_which):
        """Successful duration retrieval calls ffprobe and parses the float value."""
        mock_run.return_value = MagicMock(returncode=0, stdout=" 3600.817656 \n")

        result = get_audio_duration("input.mp4")

        assert result == 3600.817656
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffprobe"
        assert "format=duration" in args
        assert "input.mp4" in args

    @patch("src.audio.shutil.which", return_value=None)
    def test_ffprobe_not_found(self, mock_which):
        """ProcessingError is raised if ffprobe is not on PATH."""
        with pytest.raises(ProcessingError, match="ffprobe not found"):
            get_audio_duration("input.mp4")

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffprobe")
    @patch("src.audio.subprocess.run")
    def test_empty_output_raises_processing_error(self, mock_run, mock_which):
        """ProcessingError is raised if ffprobe outputs nothing."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        with pytest.raises(ProcessingError, match="ffprobe returned empty output"):
            get_audio_duration("input.mp4")

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffprobe")
    @patch("src.audio.subprocess.run")
    def test_ffprobe_failure_raises_processing_error(self, mock_run, mock_which):
        """If ffprobe exits with non-zero code, wraps in ProcessingError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffprobe"],
            stderr="Error probing file",
        )

        with pytest.raises(ProcessingError, match="ffprobe failed to retrieve audio duration"):
            get_audio_duration("input.mp4")

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffprobe")
    @patch("src.audio.subprocess.run")
    def test_invalid_float_raises_processing_error(self, mock_run, mock_which):
        """If ffprobe outputs an invalid float string, wraps in ProcessingError."""
        mock_run.return_value = MagicMock(returncode=0, stdout="not-a-float")

        with pytest.raises(ProcessingError, match="Failed to parse audio duration"):
            get_audio_duration("input.mp4")
