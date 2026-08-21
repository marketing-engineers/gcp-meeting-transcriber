"""Transcript formatting, chunk stitching, and file output."""

import json
import logging
import os
import re
from typing import Any

from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict

from src.config import (
    TRANSCRIPTION_MAX_CHARS_PER_LINE,
    TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
)

logger = logging.getLogger(__name__)


def format_transcript_lines(
    text: str,
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
) -> str:
    """
    Splits a continuous transcript string into short, readable lines
    based on sentence boundaries (. ? !) while capping line length.

    Args:
        text: Input raw transcript text.
        max_chars_per_line: Target maximum character length per line.
        max_sentences_per_line: Target maximum sentences per line.

    Returns:
        Line-bounded, sentence-wrapped transcript text joined with newlines.
    """
    if not text or not text.strip():
        return ""

    # Split into raw sentence strings using sentence boundary punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Expand any individual sentences that exceed max_chars_per_line via word wrapping
    sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars_per_line:
            sentences.append(s)
        else:
            # Word-wrap long sentences
            words = s.split()
            curr_words: list[str] = []
            curr_len = 0
            for w in words:
                if curr_words and curr_len + len(w) + 1 > max_chars_per_line:
                    sentences.append(" ".join(curr_words))
                    curr_words = [w]
                    curr_len = len(w)
                else:
                    curr_words.append(w)
                    curr_len += len(w) + (1 if len(curr_words) > 1 else 0)
            if curr_words:
                sentences.append(" ".join(curr_words))

    lines: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        needed_len = sentence_len if not current_chunk else current_length + 1 + sentence_len

        if (needed_len > max_chars_per_line) or (len(current_chunk) >= max_sentences_per_line):
            if current_chunk:
                lines.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length = needed_len

    if current_chunk:
        lines.append(" ".join(current_chunk))

    return "\n".join(lines)


def format_diarized_transcript(
    results: list[cloud_speech.SpeechRecognitionResult],
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
) -> str:
    """
    Process word-level speaker labels into speaker turns.
    Groups consecutive words by the same speaker into paragraphs, wrapping
    long speaker turns onto multiple lines at sentence boundaries.

    Output format:
        Speaker 1: Good morning everyone, let's start with the quarterly review.
        Speaker 2: Thanks. So looking at the numbers from last quarter...

    Args:
        results: List of SpeechRecognitionResult from the API response.
        max_chars_per_line: Target maximum character length per line.
        max_sentences_per_line: Target maximum sentences per line.

    Returns:
        Formatted transcript string with speaker labels.
    """
    if not results:
        return ""

    lines: list[str] = []
    current_speaker: int | None = None
    current_words: list[str] = []

    def _flush_turn(speaker: int | None, words: list[str]) -> None:
        if not words:
            return
        turn_text = ' '.join(words)
        formatted_turn = format_transcript_lines(
            turn_text,
            max_chars_per_line=max_chars_per_line,
            max_sentences_per_line=max_sentences_per_line,
        )
        if not formatted_turn:
            return
        turn_lines = formatted_turn.split('\n')
        if speaker is not None:
            prefix = f"Speaker {speaker}: "
            lines.append(f"{prefix}{turn_lines[0]}")
            lines.extend(turn_lines[1:])
        else:
            lines.extend(turn_lines)

    for result in results:
        if not result.alternatives:
            continue

        alternative = result.alternatives[0]

        # If no word-level info, fall back to transcript text
        if not alternative.words:
            transcript = alternative.transcript.strip()
            if transcript:
                formatted_text = format_transcript_lines(
                    transcript,
                    max_chars_per_line=max_chars_per_line,
                    max_sentences_per_line=max_sentences_per_line,
                )
                if formatted_text:
                    lines.extend(formatted_text.split('\n'))
            continue

        for word_info in alternative.words:
            speaker_tag = word_info.speaker_label
            try:
                speaker_num = int(speaker_tag) if speaker_tag else None
            except (ValueError, TypeError):
                speaker_num = None

            word = word_info.word

            if speaker_num != current_speaker:
                _flush_turn(current_speaker, current_words)
                current_words = [word]
                current_speaker = speaker_num
            else:
                current_words.append(word)

    # Flush remaining words
    _flush_turn(current_speaker, current_words)

    return '\n'.join(lines)


def format_plain_transcript(
    results: list[cloud_speech.SpeechRecognitionResult],
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
) -> str:
    """
    Concatenate transcript segments into plain text formatted at sentence boundaries.
    No speaker labels available (used for Dutch).

    Args:
        results: List of SpeechRecognitionResult from the API response.
        max_chars_per_line: Target maximum character length per line.
        max_sentences_per_line: Target maximum sentences per line.

    Returns:
        Plain transcript text split cleanly into readable lines.
    """
    if not results:
        return ""

    segments: list[str] = []
    for result in results:
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript.strip()
        if transcript:
            segments.append(transcript)

    raw_text = ' '.join(segments)
    return format_transcript_lines(
        raw_text,
        max_chars_per_line=max_chars_per_line,
        max_sentences_per_line=max_sentences_per_line,
    )


def _extract_results_from_response(
    response: cloud_speech.BatchRecognizeResponse,
) -> list[cloud_speech.SpeechRecognitionResult]:
    """Extract SpeechRecognitionResult list from a BatchRecognizeResponse."""
    all_results: list[cloud_speech.SpeechRecognitionResult] = []

    # BatchRecognizeResponse.results is a map of URI -> BatchRecognizeFileResult
    for _uri, file_result in response.results.items():
        if file_result.inline_result and file_result.inline_result.transcript:
            for result in file_result.inline_result.transcript.results:
                all_results.append(result)

    return all_results


def stitch_diarized_transcripts(
    responses: list[cloud_speech.BatchRecognizeResponse],
) -> str:
    """
    Stitch together diarized transcripts from multiple chunked files.

    Inserts chunk boundary markers and explicit speaker non-alignment warning notices
    between segments. Speaker numbering resets per chunk — cross-chunk speaker
    reconciliation is deferred to v0.3.

    Args:
        responses: Ordered list of BatchRecognizeResponse (one per chunk).

    Returns:
        Full stitched transcript with chunk boundary markers and speaker notices.
    """
    parts: list[str] = []
    total_chunks = len(responses)

    for i, response in enumerate(responses):
        results = _extract_results_from_response(response)
        chunk_text = format_diarized_transcript(results)

        if chunk_text:
            if i > 0:
                boundary_marker = (
                    f"\n\n--- Chunk Boundary (Chunk {i + 1} of {total_chunks}) ---\n"
                    f"[NOTICE: Speaker numbers in Chunk {i + 1} are local to this chunk "
                    f"and DO NOT correspond to speaker numbers in previous chunks.]\n\n"
                )
                parts.append(boundary_marker)
            parts.append(chunk_text)

    return ''.join(parts)


def stitch_plain_transcripts(
    responses: list[cloud_speech.BatchRecognizeResponse],
) -> str:
    """
    Stitch together plain (non-diarized) transcripts.

    Inserts chunk boundary markers if multiple chunks are present.

    Args:
        responses: Ordered list of BatchRecognizeResponse.

    Returns:
        Full stitched transcript text.
    """
    parts: list[str] = []
    total_chunks = len(responses)

    for i, response in enumerate(responses):
        results = _extract_results_from_response(response)
        chunk_text = format_plain_transcript(results)
        if chunk_text:
            if i > 0 and total_chunks > 1:
                parts.append(f"\n\n--- Chunk Boundary (Chunk {i + 1} of {total_chunks}) ---\n\n")
            parts.append(chunk_text)

    return ''.join(parts) if total_chunks > 1 else ' '.join(parts)


def _responses_to_dict(
    responses: list[cloud_speech.BatchRecognizeResponse],
) -> list[dict[str, Any]]:
    """Convert protobuf responses to serializable dicts."""
    return [MessageToDict(r._pb) for r in responses]


def save_transcript(
    transcript_text: str,
    raw_responses: list[cloud_speech.BatchRecognizeResponse],
    output_dir: str,
    base_name: str,
    lang: str,
) -> tuple[str, str]:
    """
    Save the formatted transcript and raw API response to disk.

    Args:
        transcript_text: Formatted transcript string.
        raw_responses: Raw BatchRecognizeResponse objects for JSON dump.
        output_dir: Directory to save output files.
        base_name: Base filename (without extension).
        lang: Language code used.

    Returns:
        Tuple of (transcript_path, json_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save human/LLM-readable transcript
    txt_path = os.path.join(output_dir, f"{base_name}_{lang}_transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    logger.info("Transcript saved to: %s", txt_path)

    # Save raw API response as JSON
    json_path = os.path.join(output_dir, f"{base_name}_{lang}_raw.json")
    raw_data = _responses_to_dict(raw_responses)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    logger.info("Raw API response saved to: %s", json_path)

    return txt_path, json_path
