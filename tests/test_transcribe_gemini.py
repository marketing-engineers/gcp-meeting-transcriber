"""Unit tests for src/transcribe_gemini.py and Gemini output formatting."""

import json
import os
from unittest.mock import MagicMock, call, patch
import pytest

from src.audio import ProcessingError
from src.transcribe_gemini import (
    _is_rate_limit_error,
    transcribe_gemini_file,
    transcribe_gemini_files,
)
from src.transcript_output import (
    format_gemini_diarized_transcript,
    format_gemini_plain_transcript,
    format_smart_transcript,
    normalize_speaker_label,
    save_transcript,
    stitch_diarized_transcripts,
    stitch_plain_transcripts,
)


class TestNormalizeSpeakerLabel:
    """Tests for speaker label normalization."""

    def test_spk_tags(self):
        assert normalize_speaker_label("spk_1") == "Speaker 1"
        assert normalize_speaker_label("spk_2") == "Speaker 2"
        assert normalize_speaker_label("spk_01") == "Speaker 1"
        assert normalize_speaker_label("spk_0") == "Speaker 1"
        assert normalize_speaker_label("spk:0") == "Speaker 1"
        assert normalize_speaker_label("spk:1") == "Speaker 1"
        assert normalize_speaker_label("spk:2") == "Speaker 2"

    def test_speaker_tags(self):
        assert normalize_speaker_label("Speaker 1") == "Speaker 1"
        assert normalize_speaker_label("Speaker 3") == "Speaker 3"
        assert normalize_speaker_label("speaker 2") == "Speaker 2"

    def test_numeric_tags(self):
        assert normalize_speaker_label("1") == "Speaker 1"
        assert normalize_speaker_label(2) == "Speaker 2"

    def test_empty_or_none(self):
        assert normalize_speaker_label(None) == "Speaker 1"
        assert normalize_speaker_label("") == "Speaker 1"


class TestFormatSmartTranscript:
    """Tests for smart formatting preservation and line wrapping."""

    def test_preserves_bullets_and_numbered_lists(self):
        raw = (
            "* Review quarterly revenue\n"
            "* Confirm launch milestones\n"
            "1. First action item\n"
            "2. Second action item\n\n"
            "The team agreed on all objectives and will reconvene next week."
        )
        formatted = format_smart_transcript(raw, max_chars_per_line=120)
        assert "* Review quarterly revenue" in formatted
        assert "* Confirm launch milestones" in formatted
        assert "1. First action item" in formatted
        assert "2. Second action item" in formatted
        assert "The team agreed on all objectives" in formatted

    def test_wraps_long_paragraphs(self):
        long_paragraph = "This is a sentence that goes on. " * 10
        formatted = format_smart_transcript(long_paragraph, max_chars_per_line=80)
        lines = formatted.split("\n")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 80


class TestFormatGeminiDiarizedTranscript:
    """Tests for format_gemini_diarized_transcript."""

    def test_format_from_candidates_parts(self):
        part1 = MagicMock()
        part1.text = "Hello everyone."
        part1.audio_transcription.speaker_label = "spk_1"

        part2 = MagicMock()
        part2.text = "Hi there. Thanks for joining."
        part2.audio_transcription.speaker_label = "spk_2"

        response = MagicMock()
        candidate = MagicMock()
        candidate.content.parts = [part1, part2]
        response.candidates = [candidate]

        result = format_gemini_diarized_transcript(response)
        assert "Speaker 1: Hello everyone." in result
        assert "Speaker 2: Hi there. Thanks for joining." in result

    def test_format_from_interaction_output_text(self):
        response = MagicMock(spec=["output_text"])
        response.output_text = (
            "spk_1: Good morning team.\n"
            "spk_2: Good morning. Let's start the agenda."
        )

        result = format_gemini_diarized_transcript(response)
        assert "Speaker 1: Good morning team." in result
        assert "Speaker 2: Good morning. Let's start the agenda." in result

    def test_format_from_dict_structure(self):
        response = {
            "output_text": "Speaker 1: We are on track.\nSpeaker 2: Perfect."
        }
        result = format_gemini_diarized_transcript(response)
        assert "Speaker 1: We are on track." in result
        assert "Speaker 2: Perfect." in result


class TestFormatGeminiPlainTranscript:
    """Tests for format_gemini_plain_transcript."""

    def test_plain_verbatim(self):
        response = MagicMock(spec=["output_text"])
        response.output_text = "This is a continuous meeting transcript without speaker tags."
        result = format_gemini_plain_transcript(response, mode="verbatim")
        assert "This is a continuous meeting transcript" in result

    def test_plain_smart(self):
        response = {
            "output_text": "* Point A\n* Point B\n\nDiscussion summary goes here."
        }
        result = format_gemini_plain_transcript(response, mode="smart")
        assert "* Point A" in result
        assert "* Point B" in result
        assert "Discussion summary goes here." in result


class TestTranscribeGeminiFile:
    """Tests for transcribe_gemini_file."""

    def test_is_rate_limit_error(self):
        e1 = Exception("429 Resource exhausted")
        assert _is_rate_limit_error(e1) is True

        e2 = Exception("RESOURCE_EXHAUSTED: quota exceeded")
        assert _is_rate_limit_error(e2) is True

        mock_e3 = MagicMock()
        mock_e3.code = 429
        assert _is_rate_limit_error(mock_e3) is True

        e4 = Exception("Internal 500 error")
        assert _is_rate_limit_error(e4) is False

    @patch("os.path.exists", return_value=True)
    def test_api_key_upload_and_cleanup_in_finally(self, mock_exists):
        """Verifies Files API upload and that client.files.delete is called in finally."""
        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/test123"
        mock_client.files.upload.return_value = mock_uploaded

        mock_resp = MagicMock()
        mock_resp.output_text = "Transcribed text"
        mock_client.interactions.create.return_value = mock_resp

        res = transcribe_gemini_file(
            audio_path_or_uri="/tmp/test.flac",
            lang="nl-NL",
            diarize=True,
            mode="verbatim",
            backend="api_key",
            client=mock_client,
        )

        assert res is mock_resp
        mock_client.files.upload.assert_called_once_with(file="/tmp/test.flac")
        mock_client.files.delete.assert_called_once_with(name="files/test123")

    @patch("os.path.exists", return_value=True)
    def test_api_key_cleanup_on_interaction_failure(self, mock_exists):
        """Verifies client.files.delete is called even when interactions.create fails."""
        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/err_cleanup"
        mock_client.files.upload.return_value = mock_uploaded

        mock_client.interactions.create.side_effect = Exception("API error")
        mock_client.models.generate_content.side_effect = Exception("Fallback error")

        with pytest.raises(ProcessingError, match="Gemini transcription failed"):
            transcribe_gemini_file(
                audio_path_or_uri="/tmp/test.flac",
                lang="en-US",
                diarize=False,
                backend="api_key",
                client=mock_client,
            )

        mock_client.files.delete.assert_called_once_with(name="files/err_cleanup")

    def test_vertex_ai_transcription_from_gcs_uri(self):
        """Verifies Vertex AI path uses types.Part.from_uri and models.generate_content."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp

        res = transcribe_gemini_file(
            audio_path_or_uri="gs://test-bucket/prefix/audio.flac",
            lang="en-US",
            diarize=True,
            mode="verbatim",
            backend="vertex",
            client=mock_client,
        )

        assert res is mock_resp
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-transcribe"
        assert len(call_kwargs["contents"]) == 1
        assert call_kwargs["config"].audio_transcription_config.diarization is True

    def test_vertex_ai_service_disabled_remediation_guidance(self):
        """Verifies 403 / aiplatform.googleapis.com errors return actionable remediation guidance."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "403 PERMISSION_DENIED: Agent Platform API (aiplatform.googleapis.com) has not been used..."
        )

        with pytest.raises(ProcessingError, match="Remediation Guidance") as exc_info:
            transcribe_gemini_file(
                audio_path_or_uri="gs://test-bucket/prefix/audio.flac",
                lang="nl-NL",
                diarize=True,
                backend="vertex",
                client=mock_client,
            )

        err_msg = str(exc_info.value)
        assert "gcloud services enable aiplatform.googleapis.com" in err_msg
        assert "aiplatform.endpoints.predict" in err_msg
        assert "GEMINI_API_KEY" in err_msg


class TestTranscribeGeminiFilesConcurrency:
    """Tests for multi-chunk execution in transcribe_gemini_files."""

    @patch("src.transcribe_gemini.transcribe_gemini_file")
    @patch("src.transcribe_gemini._get_gemini_client")
    def test_concurrency_wave_execution(self, mock_get_client, mock_transcribe_file):
        mock_get_client.return_value = MagicMock()
        mock_transcribe_file.side_effect = lambda audio_path_or_uri, **kw: f"response_{audio_path_or_uri}"

        targets = ["chunk1.flac", "chunk2.flac", "chunk3.flac"]
        responses = transcribe_gemini_files(
            audio_targets=targets,
            lang="nl-NL",
            diarize=True,
            mode="verbatim",
            backend="api_key",
            concurrency=2,
        )

        assert len(responses) == 3
        assert responses == ["response_chunk1.flac", "response_chunk2.flac", "response_chunk3.flac"]
        assert mock_transcribe_file.call_count == 3


class TestSaveTranscriptHybridSerialization:
    """Tests save_transcript with Gemini dict/Pydantic responses."""

    def test_save_transcript_with_gemini_dict(self, tmp_path):
        out_dir = str(tmp_path / "transcripts")
        raw_responses = [{"output_text": "Speaker 1: Testing", "custom_id": 42}]

        txt_p, json_p = save_transcript(
            transcript_text="Speaker 1: Testing",
            raw_responses=raw_responses,
            output_dir=out_dir,
            base_name="meeting",
            lang="en-US",
        )

        assert os.path.exists(txt_p)
        assert os.path.exists(json_p)
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["custom_id"] == 42
