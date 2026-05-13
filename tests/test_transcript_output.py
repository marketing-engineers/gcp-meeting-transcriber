"""Tests for src/transcript_output.py — transcript formatting logic (no API calls needed)."""

from unittest.mock import MagicMock

import pytest

from src.transcript_output import (
    format_diarized_transcript,
    format_plain_transcript,
)


def _make_word(word: str, speaker_label: str) -> MagicMock:
    """Create a mock word_info object."""
    mock = MagicMock()
    mock.word = word
    mock.speaker_label = speaker_label
    return mock


def _make_result(transcript: str = "", words: list | None = None) -> MagicMock:
    """Create a mock SpeechRecognitionResult."""
    alt = MagicMock()
    alt.transcript = transcript
    alt.words = words or []
    result = MagicMock()
    result.alternatives = [alt]
    return result


class TestFormatDiarizedTranscript:
    """Tests for format_diarized_transcript."""

    def test_empty_results(self):
        """Empty input returns empty string."""
        assert format_diarized_transcript([]) == ""

    def test_single_speaker(self):
        """Single speaker produces correct output."""
        words = [
            _make_word("Hello", "1"),
            _make_word("world", "1"),
        ]
        result = _make_result(words=words)
        output = format_diarized_transcript([result])
        assert output == "Speaker 1: Hello world"

    def test_speaker_change(self):
        """Speaker change creates new line."""
        words = [
            _make_word("Good", "1"),
            _make_word("morning", "1"),
            _make_word("Thanks", "2"),
            _make_word("everyone", "2"),
        ]
        result = _make_result(words=words)
        output = format_diarized_transcript([result])
        assert "Speaker 1: Good morning" in output
        assert "Speaker 2: Thanks everyone" in output

    def test_multiple_speaker_changes(self):
        """Multiple speaker changes produce correct turns."""
        words = [
            _make_word("Hi", "1"),
            _make_word("Hello", "2"),
            _make_word("Bye", "1"),
        ]
        result = _make_result(words=words)
        output = format_diarized_transcript([result])
        lines = output.split('\n')
        assert len(lines) == 3
        assert lines[0] == "Speaker 1: Hi"
        assert lines[1] == "Speaker 2: Hello"
        assert lines[2] == "Speaker 1: Bye"

    def test_fallback_to_transcript_when_no_words(self):
        """Falls back to transcript text when no word-level info."""
        result = _make_result(transcript="Hello world", words=[])
        output = format_diarized_transcript([result])
        assert output == "Hello world"


class TestFormatPlainTranscript:
    """Tests for format_plain_transcript."""

    def test_empty_results(self):
        """Empty input returns empty string."""
        assert format_plain_transcript([]) == ""

    def test_single_segment(self):
        """Single segment returns its transcript."""
        result = _make_result(transcript="Hello world")
        output = format_plain_transcript([result])
        assert output == "Hello world"

    def test_multiple_segments_joined(self):
        """Multiple segments are joined with space."""
        r1 = _make_result(transcript="Hello")
        r2 = _make_result(transcript="world")
        output = format_plain_transcript([r1, r2])
        assert output == "Hello world"

    def test_empty_transcript_skipped(self):
        """Empty transcript segments are skipped."""
        r1 = _make_result(transcript="Hello")
        r2 = _make_result(transcript="   ")
        r3 = _make_result(transcript="world")
        output = format_plain_transcript([r1, r2, r3])
        assert output == "Hello world"
