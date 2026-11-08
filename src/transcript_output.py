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


def _responses_to_dict(
    responses: list[Any],
) -> list[dict[str, Any]]:
    """Convert responses (protobuf, Pydantic, dict, or Interaction) to serializable dicts."""
    converted: list[dict[str, Any]] = []
    for r in responses:
        if hasattr(r, "_pb"):
            converted.append(MessageToDict(r._pb))
        elif hasattr(r, "model_dump"):
            converted.append(r.model_dump(mode="json"))
        elif hasattr(r, "to_dict") and callable(r.to_dict):
            converted.append(r.to_dict())
        elif isinstance(r, dict):
            converted.append(r)
        elif hasattr(r, "__dict__"):
            try:
                d = {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
                converted.append(json.loads(json.dumps(d, default=str)))
            except Exception:
                converted.append({"output_text": getattr(r, "output_text", str(r))})
        else:
            converted.append({"output_text": str(r)})
    return converted


def normalize_speaker_label(label: Any) -> str:
    """
    Normalizes a speaker label to a standard 'Speaker X' format.
    Handles 'spk_1', 'spk_01', 'spk_0', 'spk:0', 'spk:1', '1', 1, 'Speaker 1', etc.
    0-based speaker tags (e.g. 'spk:0', 'spk_0') use colon or explicit 0 to convert to 1-based (num + 1).
    1-based tags (e.g. 'spk_1', 'spk_2', 'Speaker 1', '1') preserve their 1-based indices (1, 2, ...).
    """
    if label is None:
        return "Speaker 1"

    label_str = str(label).strip()
    if not label_str:
        return "Speaker 1"

    # Colon delimiter 'spk:X' is specifically used by Vertex AI 0-indexed tags (spk:0, spk:1, etc.)
    m_spk_colon = re.match(r"^spk:(\d+)$", label_str, re.I)
    if m_spk_colon:
        num = int(m_spk_colon.group(1))
        return f"Speaker {num + 1}"

    m_spk = re.match(r"^spk_?(\d+)$", label_str, re.I)
    if m_spk:
        num = int(m_spk.group(1))
        # 0-based tag like spk_0 offsets to 1
        speaker_idx = num + 1 if num == 0 else num
        return f"Speaker {speaker_idx}"

    m_speaker = re.match(r"^Speaker\s*(\d+)$", label_str, re.I)
    if m_speaker:
        num = int(m_speaker.group(1))
        speaker_idx = num + 1 if num == 0 else num
        return f"Speaker {speaker_idx}"

    if label_str.isdigit():
        num = int(label_str)
        speaker_idx = num + 1 if num == 0 else num
        return f"Speaker {speaker_idx}"

    return label_str


def format_smart_transcript(
    text: str,
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
) -> str:
    """
    Formats smart transcription text while preserving structured markdown
    (bullets, numbered lists, headings) and applying line-length wrapping to paragraphs.
    """
    if not text or not text.strip():
        return ""

    raw_blocks = text.strip().split("\n")
    formatted_lines: list[str] = []

    for line in raw_blocks:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue

        list_match = re.match(r"^([*+-]|\d+\.)\s+(.*)$", stripped)
        if list_match:
            prefix = list_match.group(1) + " "
            item_text = list_match.group(2)
            if len(stripped) <= max_chars_per_line:
                formatted_lines.append(stripped)
            else:
                wrapped = format_transcript_lines(item_text, max_chars_per_line=max(30, max_chars_per_line - len(prefix)))
                wrapped_lines = wrapped.split("\n")
                formatted_lines.append(f"{prefix}{wrapped_lines[0]}")
                indent = " " * len(prefix)
                for wl in wrapped_lines[1:]:
                    formatted_lines.append(f"{indent}{wl}")
        elif stripped.startswith("#"):
            formatted_lines.append(stripped)
        else:
            if len(stripped) <= max_chars_per_line:
                formatted_lines.append(stripped)
            else:
                formatted_lines.append(format_transcript_lines(stripped, max_chars_per_line=max_chars_per_line))

    result_lines: list[str] = []
    consecutive_empty = 0
    for line in formatted_lines:
        if not line:
            consecutive_empty += 1
            if consecutive_empty <= 1:
                result_lines.append(line)
        else:
            consecutive_empty = 0
            result_lines.append(line)

    return "\n".join(result_lines)


def format_gemini_diarized_transcript(
    response: Any,
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
) -> str:
    """
    Processes a Gemini transcription response with diarization into standard speaker turns:
        Speaker 1: ...
        Speaker 2: ...
    """
    if not response:
        return ""

    turns: list[tuple[str, list[str]]] = []

    candidates = getattr(response, "candidates", None)
    if candidates is None and isinstance(response, dict):
        candidates = response.get("candidates")

    if candidates and len(candidates) > 0:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content is None and isinstance(candidate, dict):
            content = candidate.get("content")

        parts = getattr(content, "parts", None)
        if parts is None and isinstance(content, dict):
            parts = content.get("parts")

        if parts:
            current_speaker: str | None = None
            current_texts: list[str] = []

            for part in parts:
                audio_tx = getattr(part, "audio_transcription", None)
                if audio_tx is None and isinstance(part, dict):
                    audio_tx = part.get("audio_transcription")

                speaker_raw = None
                if audio_tx:
                    speaker_raw = (
                        getattr(audio_tx, "speaker_label", None)
                        or getattr(audio_tx, "speakerLabel", None)
                    )
                    if speaker_raw is None and isinstance(audio_tx, dict):
                        speaker_raw = audio_tx.get("speaker_label") or audio_tx.get("speakerLabel")

                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if not text and audio_tx:
                    text = getattr(audio_tx, "text", None)
                    if text is None and isinstance(audio_tx, dict):
                        text = audio_tx.get("text")

                text_str = str(text).strip() if text else ""
                if not text_str:
                    continue

                spk = normalize_speaker_label(speaker_raw) if speaker_raw else (current_speaker or "Speaker 1")

                if spk != current_speaker:
                    if current_speaker is not None and current_texts:
                        turns.append((current_speaker, current_texts))
                    current_speaker = spk
                    current_texts = [text_str]
                else:
                    current_texts.append(text_str)

            if current_speaker is not None and current_texts:
                turns.append((current_speaker, current_texts))

    if not turns:
        output_text = getattr(response, "output_text", None)
        if output_text is None:
            if isinstance(response, dict):
                output_text = response.get("output_text")
            elif isinstance(response, str):
                output_text = response

        if output_text and isinstance(output_text, str):
            lines = output_text.strip().split("\n")
            current_speaker = None
            current_texts = []

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                m = re.match(
                    r"^(?:\[?(spk[_:]?\d+|Speaker\s*\d+)\]?|\b(spk[_:]\d+)\b):\s*(.*)$",
                    line_str,
                    re.I,
                )
                if m:
                    raw_spk = m.group(1) or m.group(2)
                    rest_text = m.group(3).strip()
                    spk = normalize_speaker_label(raw_spk)

                    if spk != current_speaker:
                        if current_speaker is not None and current_texts:
                            turns.append((current_speaker, current_texts))
                        current_speaker = spk
                        current_texts = [rest_text] if rest_text else []
                    else:
                        if rest_text:
                            current_texts.append(rest_text)
                else:
                    if current_speaker is None:
                        current_speaker = "Speaker 1"
                    current_texts.append(line_str)

            if current_speaker is not None and current_texts:
                turns.append((current_speaker, current_texts))

    result_lines: list[str] = []
    for spk, texts in turns:
        combined_text = " ".join(texts).strip()
        if not combined_text:
            continue
        formatted_turn = format_transcript_lines(
            combined_text,
            max_chars_per_line=max_chars_per_line,
            max_sentences_per_line=max_sentences_per_line,
        )
        if not formatted_turn:
            continue
        turn_lines = formatted_turn.split("\n")
        result_lines.append(f"{spk}: {turn_lines[0]}")
        result_lines.extend(turn_lines[1:])

    return "\n".join(result_lines)


def format_gemini_plain_transcript(
    response: Any,
    mode: str = "verbatim",
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
) -> str:
    """
    Extracts and formats plain (non-diarized) transcript from a Gemini response.
    Supports verbatim and smart modes.
    """
    if not response:
        return ""

    raw_text = ""
    if isinstance(response, str):
        raw_text = response
    elif hasattr(response, "output_text") and response.output_text:
        raw_text = response.output_text
    elif isinstance(response, dict) and "output_text" in response:
        raw_text = response["output_text"]
    elif hasattr(response, "text") and response.text:
        raw_text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        parts_texts: list[str] = []
        for cand in response.candidates:
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                for p in cand.content.parts:
                    t = getattr(p, "text", "") or ""
                    if t:
                        parts_texts.append(t)
        raw_text = " ".join(parts_texts)
    elif isinstance(response, dict) and "candidates" in response:
        parts_texts = []
        for cand in response.get("candidates", []):
            content = cand.get("content") or {}
            for p in content.get("parts", []):
                t = p.get("text", "")
                if t:
                    parts_texts.append(t)
        raw_text = " ".join(parts_texts)
    else:
        raw_text = str(response)

    if mode == "smart":
        return format_smart_transcript(raw_text, max_chars_per_line=max_chars_per_line)
    return format_transcript_lines(
        raw_text,
        max_chars_per_line=max_chars_per_line,
        max_sentences_per_line=max_sentences_per_line,
    )


def stitch_diarized_transcripts(
    responses: list[Any],
) -> str:
    """
    Stitch together diarized transcripts from multiple chunked files.

    Inserts chunk boundary markers and explicit speaker non-alignment warning notices
    between segments. Speaker numbering resets per chunk — cross-chunk speaker
    reconciliation is deferred to v0.4.

    Args:
        responses: Ordered list of BatchRecognizeResponse or Gemini responses.

    Returns:
        Full stitched transcript with chunk boundary markers and speaker notices.
    """
    parts: list[str] = []
    total_chunks = len(responses)

    for i, response in enumerate(responses):
        if hasattr(response, "results") and hasattr(response, "_pb"):
            results = _extract_results_from_response(response)
            chunk_text = format_diarized_transcript(results)
        elif isinstance(response, str):
            chunk_text = response
        else:
            chunk_text = format_gemini_diarized_transcript(response)

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
    responses: list[Any],
    mode: str = "verbatim",
) -> str:
    """
    Stitch together plain (non-diarized) transcripts.

    Inserts chunk boundary markers if multiple chunks are present.

    Args:
        responses: Ordered list of BatchRecognizeResponse or Gemini responses.
        mode: Transcription mode ('verbatim' or 'smart') for Gemini plain text.

    Returns:
        Full stitched transcript text.
    """
    parts: list[str] = []
    total_chunks = len(responses)

    for i, response in enumerate(responses):
        if hasattr(response, "results") and hasattr(response, "_pb"):
            results = _extract_results_from_response(response)
            chunk_text = format_plain_transcript(results)
        elif isinstance(response, str):
            chunk_text = response
        else:
            chunk_text = format_gemini_plain_transcript(response, mode=mode)

        if chunk_text:
            if i > 0 and total_chunks > 1:
                parts.append(f"\n\n--- Chunk Boundary (Chunk {i + 1} of {total_chunks}) ---\n\n")
            parts.append(chunk_text)

    return ''.join(parts) if total_chunks > 1 else ' '.join(parts)


def save_transcript(
    transcript_text: str,
    raw_responses: list[Any],
    output_dir: str,
    base_name: str,
    lang: str,
) -> tuple[str, str]:
    """
    Save the formatted transcript and raw API response to disk.

    Args:
        transcript_text: Formatted transcript string.
        raw_responses: Raw response objects (BatchRecognizeResponse or Gemini responses) for JSON dump.
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
