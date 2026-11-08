"""Tests for src/transcript_output.py — transcript formatting logic (no API calls needed)."""

import json
import os
from unittest.mock import MagicMock

import pytest

from src.transcript_output import (
    format_diarized_transcript,
    format_duration_human,
    format_plain_transcript,
    format_smart_transcript,
    format_timestamp,
    format_transcript_lines,
    generate_frontmatter,
    parse_duration_seconds,
    save_transcript,
    stitch_diarized_transcripts,
    stitch_plain_transcripts,
)


class TestFormatTranscriptLines:
    """Tests for format_transcript_lines helper."""

    def test_empty_string(self):
        assert format_transcript_lines("") == ""
        assert format_transcript_lines("   ") == ""

    def test_sentence_splitting_and_line_wrapping(self):
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        formatted = format_transcript_lines(text, max_chars_per_line=50, max_sentences_per_line=2)
        lines = formatted.split("\n")
        assert len(lines) >= 2
        for line in lines:
            assert len(line) <= 50

    def test_max_sentences_per_line_cap(self):
        text = "One. Two. Three. Four. Five."
        formatted = format_transcript_lines(text, max_chars_per_line=200, max_sentences_per_line=2)
        lines = formatted.split("\n")
        assert len(lines) == 3
        assert lines[0] == "One. Two."
        assert lines[1] == "Three. Four."
        assert lines[2] == "Five."

    def test_long_single_sentence_word_wrapping(self):
        long_sentence = "Word " * 30  # 150 chars sentence without punctuation
        formatted = format_transcript_lines(long_sentence, max_chars_per_line=50)
        lines = formatted.split("\n")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 50


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

    def test_speaker_turn_line_wrapping(self):
        """Long speaker turns wrap onto new lines without re-stamping Speaker label."""
        words = [
            _make_word("This", "1"),
            _make_word("is", "1"),
            _make_word("sentence", "1"),
            _make_word("one.", "1"),
            _make_word("This", "1"),
            _make_word("is", "1"),
            _make_word("sentence", "1"),
            _make_word("two.", "1"),
        ]
        result = _make_result(words=words)
        output = format_diarized_transcript([result], max_chars_per_line=25)
        lines = output.split("\n")
        assert lines[0] == "Speaker 1: This is sentence one."
        assert lines[1] == "This is sentence two."
        assert not lines[1].startswith("Speaker 1:")

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


def _make_batch_response(results: list) -> MagicMock:
    """Create a mock BatchRecognizeResponse object containing transcript results."""
    file_result = MagicMock()
    file_result.inline_result.transcript.results = results
    resp = MagicMock()
    resp.results = {"gs://bucket/file.flac": file_result}
    return resp


class TestStitchTranscripts:
    """Tests for stitch_diarized_transcripts and stitch_plain_transcripts."""

    def test_stitch_diarized_transcripts_includes_boundary_and_notice(self):
        """Multiple diarized chunks include Markdown Part headers and speaker warning notices."""
        r1 = _make_result(words=[_make_word("Hello", "1")])
        r2 = _make_result(words=[_make_word("World", "1")])

        resp1 = _make_batch_response([r1])
        resp2 = _make_batch_response([r2])

        stitched = stitch_diarized_transcripts([resp1, resp2])

        assert "Speaker 1: Hello" in stitched
        assert "Speaker 1: World" in stitched
        assert "## Part A (Chunk 1 of 2)" in stitched
        assert "## Part B (Chunk 2 of 2)" in stitched
        assert "Speaker Attribution Scope" in stitched
        assert "Speaker Attribution Notice" in stitched
        assert "DO NOT correspond to speaker numbers in previous chunks" in stitched
        assert "Speaker labels from Part A might not align with Part B" in stitched

    def test_stitch_plain_transcripts_single_chunk_has_no_boundary(self):
        """Single plain transcript chunk has no boundary headers."""
        r1 = _make_result(transcript="Hello world")
        resp1 = _make_batch_response([r1])

        stitched = stitch_plain_transcripts([resp1])
        assert stitched == "Hello world"
        assert "Part A" not in stitched
        assert "Part B" not in stitched

    def test_stitch_plain_transcripts_multiple_chunks_includes_boundary(self):
        """Multiple plain transcript chunks include Markdown chunk headers."""
        r1 = _make_result(transcript="Chunk one content.")
        r2 = _make_result(transcript="Chunk two content.")

        resp1 = _make_batch_response([r1])
        resp2 = _make_batch_response([r2])

        stitched = stitch_plain_transcripts([resp1, resp2])
        assert "Chunk one content." in stitched
        assert "## Part A (Chunk 1 of 2)" in stitched
        assert "## Part B (Chunk 2 of 2)" in stitched
        assert "Chunk two content." in stitched


class TestFrontmatterAndFormatting:
    """Tests for frontmatter generation and time formatting helpers."""

    def test_format_duration_human(self):
        assert format_duration_human(None) == "unknown"
        assert format_duration_human(-5) == "unknown"
        assert format_duration_human(45.2) == "0m 45s"
        assert format_duration_human(300.0) == "5m 00s"
        assert format_duration_human(2712.5) == "45m 12s"
        assert format_duration_human(3665.0) == "1h 01m 05s"

    def test_format_timestamp(self):
        assert format_timestamp(0) == "[00:00]"
        assert format_timestamp(300) == "[05:00]"
        assert format_timestamp(3600) == "[01:00:00]"
        assert format_timestamp(300, include_hours=True) == "[00:05:00]"
        assert format_timestamp(3905, include_hours=True) == "[01:05:05]"

    def test_parse_duration_seconds(self):
        assert parse_duration_seconds(None) is None
        assert parse_duration_seconds(42) == 42.0
        assert parse_duration_seconds(42.5) == 42.5
        assert parse_duration_seconds("15.5s") == 15.5
        assert parse_duration_seconds("05:30") == 330.0
        assert parse_duration_seconds("01:05:30") == 3930.0
        assert parse_duration_seconds({"seconds": 10, "nanos": 500000000}) == 10.5

    def test_generate_frontmatter_full(self):
        fm = generate_frontmatter(
            source_file="input/quarterly_review.mp4",
            duration=2712.5,
            repository="marketing-engineers-gcp-audio-transcribe",
            model="gemini-3.5-transcribe",
            mode="verbatim",
            lang="nl-NL",
            languages=["nl-NL", "en-US"],
            diarize=True,
            speakers=3,
            generated_at="2026-11-08T16:00:00Z",
        )
        assert fm.startswith("---\n")
        assert fm.endswith("---")
        assert 'generated_at: "2026-11-08T16:00:00Z"' in fm
        assert 'source_file: "quarterly_review.mp4"' in fm
        assert 'duration: "45m 12s"' in fm
        assert "duration_seconds: 2712.5" in fm
        assert 'repository: "marketing-engineers-gcp-audio-transcribe"' in fm
        assert 'model: "gemini-3.5-transcribe"' in fm
        assert 'mode: "verbatim"' in fm
        assert 'language: "nl-NL"' in fm
        assert '- "nl-NL"' in fm
        assert '- "en-US"' in fm
        assert "diarization: true" in fm
        assert "speakers: 3" in fm

    def test_generate_frontmatter_escapes_quotes(self):
        """Filenames with quotes or special characters are safely escaped for YAML."""
        fm = generate_frontmatter(
            source_file='Meeting "Q3 Strategy" & Review.mp4',
            lang="en-US",
            generated_at="2026-11-08T12:00:00Z",
        )
        assert r'source_file: "Meeting \"Q3 Strategy\" & Review.mp4"' in fm

    def test_generate_frontmatter_defaults(self):
        fm = generate_frontmatter(
            lang="en-US",
            generated_at="2026-11-08T12:00:00Z",
        )
        assert 'generated_at: "2026-11-08T12:00:00Z"' in fm
        assert 'repository: "marketing-engineers-gcp-audio-transcribe"' in fm
        assert 'language: "en-US"' in fm


class TestMilestoneTimestamps:
    """Tests for milestone indicator insertion."""

    def test_diarized_milestones_every_5_minutes(self):
        """A recording longer than 5 minutes receives a milestone indicator."""
        words1 = [_make_word(f"word{i}", "1") for i in range(50)]
        words2 = [_make_word(f"word{i}", "2") for i in range(50)]
        r = _make_result(words=words1 + words2)

        # Total duration 600 seconds (10 min), milestone interval 300 (5 min)
        formatted = format_diarized_transcript(
            [r],
            duration=600.0,
            milestone_interval=300.0,
        )
        assert "_**[05:00]**_" in formatted

    def test_plain_milestones_every_5_minutes(self):
        """Plain transcript longer than 5 minutes receives milestone indicator."""
        text = ". ".join([f"Sentence number {i} discussing progress" for i in range(20)]) + "."
        r = _make_result(transcript=text)

        formatted = format_plain_transcript(
            [r],
            duration=600.0,
            milestone_interval=300.0,
        )
        assert "_**[05:00]**_" in formatted

    def test_plain_milestones_short_chunk_crossing_cumulative_milestone(self):
        """Short remainder chunk crossing a milestone on cumulative timeline receives milestone."""
        text = "Short wrap up segment of the discussion."
        r = _make_result(transcript=text)

        # Chunk is only 100s (< 300s), but starts at 250s, so crosses 300s milestone
        formatted = format_plain_transcript(
            [r],
            time_offset=250.0,
            duration=100.0,
            milestone_interval=300.0,
        )
        assert "_**[05:00]**_" in formatted

    def test_milestone_on_exact_boundary(self):
        """A chunk starting on an exact milestone multiple (e.g. 300s) does not skip it."""
        words = [_make_word(f"w{i}", "1") for i in range(20)]
        r = _make_result(words=words)

        formatted = format_diarized_transcript(
            [r],
            time_offset=300.0,
            duration=300.0,
            milestone_interval=300.0,
        )
        assert "_**[05:00]**_" in formatted

    def test_smart_milestones_every_5_minutes(self):
        """Smart transcript with bullet points receives milestone indicator."""
        lines = [f"* Action item {i}: follow up with team" for i in range(20)]
        text = "\n".join(lines)

        formatted = format_smart_transcript(
            text,
            duration=600.0,
            milestone_interval=300.0,
        )
        assert "_**[05:00]**_" in formatted

    def test_smart_transcript_preserves_paragraph_blank_lines(self):
        """Smart transcript preserves blank lines separating paragraphs."""
        input_text = "# Header\n\nFirst paragraph content.\n\nSecond paragraph content.\n\n* Bullet 1\n* Bullet 2"
        formatted = format_smart_transcript(input_text, duration=100.0, milestone_interval=0.0)
        assert "\n\nFirst paragraph content.\n\n" in formatted
        assert "\n\nSecond paragraph content.\n\n" in formatted

    def test_custom_milestone_interval(self):
        """Configurable milestone interval (e.g. 60 seconds)."""
        words = [_make_word(f"w{i}", "1") for i in range(20)]
        r = _make_result(words=words)

        formatted = format_diarized_transcript(
            [r],
            duration=150.0,
            milestone_interval=60.0,
        )
        assert "_**[01:00]**_" in formatted
        assert "_**[02:00]**_" in formatted

    def test_milestone_disabled_when_zero(self):
        """Milestone indicators are suppressed when milestone_interval is 0."""
        words = [_make_word(f"w{i}", "1") for i in range(20)]
        r = _make_result(words=words)

        formatted = format_diarized_transcript(
            [r],
            duration=600.0,
            milestone_interval=0.0,
        )
        assert "_**[" not in formatted

    def test_multi_chunk_cumulative_milestones(self):
        """Multi-chunk stitching properly tracks cumulative timeline across chunks."""
        # Chunk 1: 1680s (28m), Chunk 2: 1020s (17m). Total 2700s (45m).
        r1 = _make_result(words=[_make_word(f"c1_{i}", "1") for i in range(40)])
        r2 = _make_result(words=[_make_word(f"c2_{i}", "1") for i in range(40)])
        resp1 = _make_batch_response([r1])
        resp2 = _make_batch_response([r2])

        stitched = stitch_diarized_transcripts(
            [resp1, resp2],
            chunk_durations=[1680.0, 1020.0],
            total_duration=2700.0,
            milestone_interval=300.0,
        )
        # Part A has 05:00, 10:00, 15:00, 20:00, 25:00
        assert "_**[05:00]**_" in stitched
        assert "_**[25:00]**_" in stitched
        # Part B begins at 28:00, so first milestone in Part B is 30:00
        assert "_**[30:00]**_" in stitched
        assert "_**[35:00]**_" in stitched


class TestSaveTranscriptMarkdown:
    """Tests for saving .md files with YAML frontmatter."""

    def test_save_transcript_creates_markdown_and_raw_json(self, tmp_path):
        out_dir = str(tmp_path / "transcripts")
        raw_responses = [{"output_text": "Speaker 1: Hi", "confidence": 0.98}]

        md_path, json_path = save_transcript(
            transcript_text="Speaker 1: Hi",
            raw_responses=raw_responses,
            output_dir=out_dir,
            base_name="meeting_2026",
            lang="nl-NL",
            duration=1200.0,
            model="gemini-3.5-transcribe",
            diarize=True,
            speaker_count=2,
            mode="verbatim",
            languages=["nl-NL", "en-US"],
            source_file="input/meeting_2026.mp4",
        )

        assert os.path.exists(md_path)
        assert os.path.exists(json_path)
        assert md_path.endswith("_transcript.md")
        assert json_path.endswith("_raw.json")

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        assert md_content.startswith("---")
        assert 'source_file: "meeting_2026.mp4"' in md_content
        assert 'duration: "20m 00s"' in md_content
        assert "duration_seconds: 1200.0" in md_content
        assert 'repository: "marketing-engineers-gcp-audio-transcribe"' in md_content
        assert 'model: "gemini-3.5-transcribe"' in md_content
        assert 'mode: "verbatim"' in md_content
        assert 'language: "nl-NL"' in md_content
        assert "diarization: true" in md_content
        assert "speakers: 2" in md_content
        assert "# Transcript: meeting_2026.mp4" in md_content
        assert "Speaker 1: Hi" in md_content
