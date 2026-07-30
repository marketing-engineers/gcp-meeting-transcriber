"""Transcript formatting, chunk stitching, and file output."""

import json
import logging
import os
from typing import Any

from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)


def format_diarized_transcript(results: list[cloud_speech.SpeechRecognitionResult]) -> str:
    """
    Process word-level speaker labels into speaker turns.
    Groups consecutive words by the same speaker into paragraphs.

    Output format:
        Speaker 1: Good morning everyone, let's start with the quarterly review.
        Speaker 2: Thanks. So looking at the numbers from last quarter...

    Args:
        results: List of SpeechRecognitionResult from the API response.

    Returns:
        Formatted transcript string with speaker labels.
    """
    if not results:
        return ""

    lines: list[str] = []
    current_speaker: int | None = None
    current_words: list[str] = []

    for result in results:
        if not result.alternatives:
            continue

        alternative = result.alternatives[0]

        # If no word-level info, fall back to transcript text
        if not alternative.words:
            transcript = alternative.transcript.strip()
            if transcript:
                lines.append(transcript)
            continue

        for word_info in alternative.words:
            speaker_tag = word_info.speaker_label
            # speaker_label is a string like "1", "2", etc.
            try:
                speaker_num = int(speaker_tag) if speaker_tag else None
            except (ValueError, TypeError):
                speaker_num = None

            word = word_info.word

            if speaker_num != current_speaker:
                # Flush current speaker's words
                if current_words and current_speaker is not None:
                    lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
                elif current_words:
                    lines.append(' '.join(current_words))
                current_words = [word]
                current_speaker = speaker_num
            else:
                current_words.append(word)

    # Flush remaining words
    if current_words and current_speaker is not None:
        lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
    elif current_words:
        lines.append(' '.join(current_words))

    return '\n'.join(lines)


def format_plain_transcript(results: list[cloud_speech.SpeechRecognitionResult]) -> str:
    """
    Concatenate transcript segments into plain text.
    No speaker labels available (used for Dutch).

    Args:
        results: List of SpeechRecognitionResult from the API response.

    Returns:
        Plain transcript text.
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

    return ' '.join(segments)


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
