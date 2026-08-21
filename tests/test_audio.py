"""Tests for src/audio.py — FFmpeg command construction, silence detection, chunking, and error handling."""

import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from src.audio import (
    ProcessingError,
    _check_ffmpeg_available,
    compute_chunk_boundaries,
    detect_silences,
    extract_audio_to_flac,
    get_audio_duration,
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


class TestDetectSilences:
    """Tests for detect_silences."""

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_detect_silences_parses_stderr(self, mock_run, mock_which):
        """Parses representative silencedetect stderr correctly into start/end tuples."""
        stderr_sample = (
            "[silencedetect @ 0x123] silence_start: 10.5\n"
            "[silencedetect @ 0x123] silence_end: 12.0 | silence_duration: 1.5\n"
            "[silencedetect @ 0x123] silence_start: 100.25\n"
            "[silencedetect @ 0x123] silence_end: 101.75 | silence_duration: 1.5\n"
        )
        mock_run.return_value = MagicMock(returncode=0, stderr=stderr_sample)

        silences = detect_silences("test.flac", noise_threshold=-30, min_duration=0.5)

        assert silences == [(10.5, 12.0), (100.25, 101.75)]
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "-af" in args
        assert "silencedetect=noise=-30dB:d=0.5" in args

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_detect_silences_unparseable_returns_empty_list(self, mock_run, mock_which):
        """Unparseable or unexpected output degrades gracefully to empty list."""
        mock_run.return_value = MagicMock(returncode=0, stderr="Invalid output without silences")

        silences = detect_silences("test.flac")
        assert silences == []

    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_detect_silences_subprocess_failure_returns_empty_list(self, mock_run, mock_which):
        """Subprocess exception degrades gracefully to empty list without raising."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Error")

        silences = detect_silences("test.flac")
        assert silences == []


class TestComputeChunkBoundaries:
    """Tests for compute_chunk_boundaries."""

    def test_worked_examples_table_row_1(self):
        """Row 1: 600s (10 min), Dutch ceiling 2700s, concurrency 5 -> monolithic ([])."""
        b = compute_chunk_boundaries(
            duration=600.0,
            max_chunk_length=2700.0,
            silences=[],
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=5,
        )
        assert b == []

    def test_worked_examples_table_row_2(self):
        """Row 2: 1800s (30 min), Dutch ceiling 2700s, concurrency 5 -> 3 chunks (floor-limited)."""
        b = compute_chunk_boundaries(
            duration=1800.0,
            max_chunk_length=2700.0,
            silences=[],
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=5,
        )
        assert len(b) == 2  # 3 chunks -> 2 boundaries
        assert b == [600.0, 1200.0]

    def test_worked_examples_table_row_3(self):
        """Row 3: 7448s (2h04m), Dutch ceiling 2700s, concurrency 5 -> 5 chunks (one wave)."""
        b = compute_chunk_boundaries(
            duration=7448.0,
            max_chunk_length=2700.0,
            silences=[],
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=5,
        )
        assert len(b) == 4  # 5 chunks -> 4 boundaries
        assert b == [1489.6, 2979.2, 4468.8, 5958.4]

    def test_worked_examples_table_row_4(self):
        """Row 4: 21600s (6h), Dutch ceiling 2700s, concurrency 5 -> 9 chunks (ceiling governs)."""
        b = compute_chunk_boundaries(
            duration=21600.0,
            max_chunk_length=2700.0,
            silences=[],
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=5,
        )
        assert len(b) == 8  # 9 chunks -> 8 boundaries
        assert b == [2400.0, 4800.0, 7200.0, 9600.0, 12000.0, 14400.0, 16800.0, 19200.0]

    def test_worked_examples_table_row_5(self):
        """Row 5: 10800s English, ceiling 1080s, parallelism None -> 11 chunks (ceiling governs)."""
        b = compute_chunk_boundaries(
            duration=10800.0,
            max_chunk_length=1080.0,
            silences=[],
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=None,
        )
        assert len(b) == 10  # 11 chunks -> 10 boundaries
        for b1, b2 in zip([0.0] + b, b + [10800.0]):
            assert b2 - b1 <= 1080.0

    def test_english_case_ignores_parallelism(self):
        """English case with target_parallelism=None uses ceiling alone."""
        b = compute_chunk_boundaries(
            duration=7448.0,
            max_chunk_length=2700.0,
            silences=[],
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=None,
        )
        assert len(b) == 2  # 3 chunks (ceil(7448 / (2700-15)) = 3)

    def test_headroom_guarantees_no_ceiling_breach_with_snapping(self):
        """Headroom subtraction ensures forward snapping never breaches ceiling."""
        # Silence at nominal boundary + 10s
        silences = [(982 + 10 - 0.5, 982 + 10 + 0.5)]
        b = compute_chunk_boundaries(
            duration=10800.0,
            max_chunk_length=1080.0,
            silences=silences,
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=None,
        )
        for b1, b2 in zip([0.0] + b, b + [10800.0]):
            assert b2 - b1 <= 1080.0

    def test_snaps_to_nearest_silence_in_window(self):
        """Snaps to silence within +/- window and falls back to nominal if outside."""
        # Nominal is 1489.6. Silences at 1495.0 (midpoint) and 2000.0 (outside window)
        silences = [(1494.0, 1496.0), (1999.0, 2001.0)]
        b = compute_chunk_boundaries(
            duration=7448.0,
            max_chunk_length=2700.0,
            silences=silences,
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=5,
        )
        assert b[0] == 1495.0  # Snapped to 1495.0 midpoint
        assert b[1] == 2979.2  # Fallback to nominal since 2000 is outside window around 2979.2

    def test_strictly_increasing_and_respects_floor(self):
        """Boundaries are strictly increasing and chunks respect min_chunk_length."""
        silences = [(100.0, 101.0), (700.0, 701.0), (1300.0, 1301.0)]
        b = compute_chunk_boundaries(
            duration=1800.0,
            max_chunk_length=2700.0,
            silences=silences,
            window=15.0,
            min_chunk_length=600.0,
            target_parallelism=5,
        )
        assert len(b) == 2
        assert all(b1 < b2 for b1, b2 in zip([0.0] + b, b + [1800.0]))


class TestSplitFlac:
    """Tests for split_flac."""

    @patch("src.audio.os.makedirs")
    @patch("src.audio.os.remove")
    @patch("src.audio.os.path.exists", return_value=True)
    @patch("src.audio.os.listdir")
    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_successful_split_returns_paths_and_deletes_stale_files(
        self, mock_run, mock_which, mock_listdir, mock_exists, mock_remove, mock_makedirs
    ):
        """Successful split deletes stale chunk files and invokes ffmpeg with -segment_times."""
        mock_run.return_value = MagicMock(returncode=0)
        # First listdir for stale cleanup, second listdir for generated chunks
        mock_listdir.side_effect = [
            ["meeting_chunk_000.flac", "meeting_chunk_001.flac", "other.txt"],
            ["meeting_chunk_000.flac", "meeting_chunk_001.flac"],
        ]

        result = split_flac("input.flac", "/output", "meeting", [600.0, 1200.0])

        assert mock_remove.call_count == 2
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "-segment_times" in args
        assert "600.0,1200.0" in args
        assert "-c" in args
        assert "copy" in args

        assert result == [
            "/output/meeting_chunk_000.flac",
            "/output/meeting_chunk_001.flac",
        ]

    def test_split_flac_empty_segment_times_returns_empty(self):
        """split_flac with empty segment_times returns empty list."""
        with patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
            res = split_flac("input.flac", "/output", "meeting", [])
            assert res == []

    @patch("src.audio.os.makedirs")
    @patch("src.audio.os.listdir", return_value=[])
    @patch("src.audio.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("src.audio.subprocess.run")
    def test_ffmpeg_failure_raises_processing_error(self, mock_run, mock_which, mock_listdir, mock_makedirs):
        """FFmpeg chunking failure is wrapped in ProcessingError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="Error: segment failed",
        )

        with pytest.raises(ProcessingError, match="FFmpeg failed during FLAC chunking"):
            split_flac("input.flac", "/output", "meeting", [600.0])


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
