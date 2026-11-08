"""Transcript formatting, chunk stitching, Markdown generation, and file output."""

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict

from src.config import (
    REPOSITORY_NAME,
    TRANSCRIPTION_MAX_CHARS_PER_LINE,
    TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
    TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)


def parse_duration_seconds(offset_val: Any) -> float | None:
    """
    Parses various duration representations into a float number of seconds.
    Supports protobuf Duration, dicts with seconds/nanos, strings ('12.5s', '01:23'),
    and numeric types.
    """
    if offset_val is None:
        return None
    # Avoid MagicMock auto-attributes unless explicitly configured
    if "Mock" in type(offset_val).__name__:
        if hasattr(offset_val, "total_seconds") and callable(offset_val.total_seconds):
            res = offset_val.total_seconds()
            if isinstance(res, (int, float)) and "Mock" not in type(res).__name__:
                return float(res)
        sec = getattr(offset_val, "seconds", None)
        if isinstance(sec, (int, float)) and "Mock" not in type(sec).__name__:
            nano = getattr(offset_val, "nanos", 0)
            nano_f = float(nano) if isinstance(nano, (int, float)) and "Mock" not in type(nano).__name__ else 0.0
            return float(sec) + nano_f * 1e-9
        return None

    if type(offset_val) in (int, float):
        return float(offset_val)
    if hasattr(offset_val, "total_seconds") and callable(offset_val.total_seconds):
        try:
            res = offset_val.total_seconds()
            if isinstance(res, (int, float)):
                return float(res)
        except Exception:
            pass
    if hasattr(offset_val, "seconds") or hasattr(offset_val, "nanos"):
        sec = getattr(offset_val, "seconds", 0) or 0
        nano = getattr(offset_val, "nanos", 0) or 0
        if type(sec) in (int, float) and type(nano) in (int, float):
            return float(sec) + float(nano) * 1e-9
    if isinstance(offset_val, dict):
        sec = offset_val.get("seconds", 0) or 0
        nano = offset_val.get("nanos", 0) or 0
        if isinstance(sec, (int, float)) and isinstance(nano, (int, float)):
            return float(sec) + float(nano) * 1e-9
    if isinstance(offset_val, str):
        s = offset_val.strip()
        if not s:
            return None
        if s.endswith("s"):
            try:
                return float(s[:-1])
            except ValueError:
                pass
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                pass
        try:
            return float(s)
        except ValueError:
            pass
    return None


def format_timestamp(seconds: float, include_hours: bool = False) -> str:
    """
    Formats a duration in seconds into a bracketed timestamp string [MM:SS] or [HH:MM:SS].
    """
    total_sec = max(0, int(round(seconds)))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if include_hours or h > 0:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def format_duration_human(seconds: float | None) -> str:
    """
    Formats seconds into a concise human-readable duration (e.g. '45m 12s', '1h 05m 20s').
    """
    if seconds is None or seconds < 0:
        return "unknown"
    total_sec = int(round(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def generate_frontmatter(
    source_file: str = "",
    duration: float | None = None,
    repository: str = REPOSITORY_NAME,
    model: str = "",
    mode: str = "",
    lang: str = "",
    languages: list[str] | None = None,
    diarize: bool | None = None,
    speakers: int | str | None = None,
    generated_at: str | None = None,
    extra_settings: dict[str, Any] | None = None,
) -> str:
    """
    Generates standard YAML frontmatter containing metadata about the transcript.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = ["---", f"generated_at: {json.dumps(generated_at)}"]
    if source_file:
        lines.append(f"source_file: {json.dumps(os.path.basename(source_file))}")
    if duration is not None:
        lines.append(f"duration: {json.dumps(format_duration_human(duration))}")
        lines.append(f"duration_seconds: {round(duration, 2)}")
    lines.append(f"repository: {json.dumps(repository)}")
    lines.append("settings:")
    if model:
        lines.append(f"  model: {json.dumps(model)}")
    if mode and model != "chirp_3":
        lines.append(f"  mode: {json.dumps(mode)}")
    if lang:
        lines.append(f"  language: {json.dumps(lang)}")
    if languages:
        lines.append("  languages:")
        for l in languages:
            lines.append(f"    - {json.dumps(l)}")
    if diarize is not None:
        lines.append(f"  diarization: {'true' if diarize else 'false'}")
    if diarize:
        if speakers is not None:
            if isinstance(speakers, int):
                lines.append(f"  speakers: {speakers}")
            else:
                lines.append(f"  speakers: {json.dumps(str(speakers))}")
        else:
            lines.append('  speakers: "auto-detect"')
    elif speakers is not None:
        if isinstance(speakers, int):
            lines.append(f"  speakers: {speakers}")
        else:
            lines.append(f"  speakers: {json.dumps(str(speakers))}")

    if extra_settings:
        for k, v in extra_settings.items():
            if isinstance(v, bool):
                lines.append(f"  {k}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"  {k}: {v}")
            elif isinstance(v, list):
                lines.append(f"  {k}:")
                for item in v:
                    lines.append(f"    - {json.dumps(str(item))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")

    lines.append("---")
    return "\n".join(lines)


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

    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars_per_line:
            sentences.append(s)
        else:
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


def _render_turns_with_milestones(
    turns: list[tuple[str | None, list[str], float | None]],
    time_offset: float = 0.0,
    duration: float | None = None,
    total_audio_duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
) -> str:
    """
    Renders speaker turns (or plain paragraphs) into lines while inserting
    milestone indicators (e.g. _**[05:00]**_) when milestone_interval is reached.
    """
    if not turns:
        return ""

    total_words_list = [sum(len(t.split()) for t in texts) for _, texts, _ in turns]
    total_words = sum(total_words_list)

    assigned_turns: list[tuple[str | None, list[str], float | None, int]] = []
    cursor_words = 0
    for (spk, texts, explicit_t), w_count in zip(turns, total_words_list):
        if explicit_t is not None:
            t = explicit_t
        elif duration and duration > 0 and total_words > 0:
            t = (cursor_words / total_words) * duration
        else:
            t = None
        cursor_words += w_count
        assigned_turns.append((spk, texts, t, w_count))

    effective_total_duration = total_audio_duration if total_audio_duration is not None else (
        (time_offset + duration) if duration else None
    )
    include_hours = (effective_total_duration is not None and effective_total_duration >= 3600) or (
        (time_offset + (duration or 0.0)) >= 3600
    )

    milestone_step = float(milestone_interval) if milestone_interval > 0 else 0.0
    next_milestone = None
    if milestone_step > 0:
        first_ms = math.ceil(time_offset / milestone_step) * milestone_step
        next_milestone = first_ms if first_ms > 0 else milestone_step

    lines: list[str] = []
    for spk, texts, turn_t, w_count in assigned_turns:
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
        turn_dur = (w_count / total_words) * duration if (duration and total_words > 0) else 0.0
        global_turn_start = (time_offset + turn_t) if turn_t is not None else None
        global_turn_end = (global_turn_start + turn_dur) if global_turn_start is not None else None

        # Check milestone insertion before this turn
        if milestone_step > 0 and next_milestone is not None and global_turn_start is not None:
            if global_turn_start >= next_milestone:
                while next_milestone <= global_turn_start:
                    lines.append(f"_**{format_timestamp(next_milestone, include_hours)}**_")
                    next_milestone += milestone_step

        # If turn has multiple lines and spans across next_milestone, split inside turn
        if (
            milestone_step > 0
            and next_milestone is not None
            and global_turn_start is not None
            and global_turn_end is not None
            and len(turn_lines) > 1
            and global_turn_end >= next_milestone
        ):
            for l_idx, line in enumerate(turn_lines):
                line_t = global_turn_start + (l_idx / len(turn_lines)) * turn_dur
                if line_t >= next_milestone:
                    while next_milestone <= line_t:
                        lines.append(f"_**{format_timestamp(next_milestone, include_hours)}**_")
                        next_milestone += milestone_step
                if spk and l_idx == 0:
                    lines.append(f"{spk}: {line}")
                else:
                    lines.append(line)
        else:
            if spk:
                prefix = f"{spk}: "
                lines.append(f"{prefix}{turn_lines[0]}")
                lines.extend(turn_lines[1:])
            else:
                lines.extend(turn_lines)

        # Check if milestone was reached during/by end of this turn
        if milestone_step > 0 and next_milestone is not None and global_turn_end is not None:
            if global_turn_end >= next_milestone:
                while next_milestone <= global_turn_end:
                    lines.append(f"_**{format_timestamp(next_milestone, include_hours)}**_")
                    next_milestone += milestone_step

    return "\n".join(lines)


def format_diarized_transcript(
    results: list[cloud_speech.SpeechRecognitionResult],
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
    time_offset: float = 0.0,
    duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    total_audio_duration: float | None = None,
) -> str:
    """
    Process word-level speaker labels into speaker turns with milestone timestamps.
    Groups consecutive words by the same speaker into paragraphs, wrapping
    long speaker turns onto multiple lines at sentence boundaries.

    Output format:
        Speaker 1: Good morning everyone, let's start with the quarterly review.
        Speaker 2: Thanks. So looking at the numbers from last quarter...

    Args:
        results: List of SpeechRecognitionResult from the API response.
        max_chars_per_line: Target maximum character length per line.
        max_sentences_per_line: Target maximum sentences per line.
        time_offset: Audio offset in seconds for this chunk.
        duration: Duration of this chunk in seconds.
        milestone_interval: Interval in seconds for milestone timestamp indicators.
        total_audio_duration: Total duration of the recording across all chunks.

    Returns:
        Formatted transcript string with speaker labels and milestone indicators.
    """
    if not results:
        return ""

    turns_data: list[tuple[str | None, list[str], float | None]] = []
    current_speaker: int | None = None
    current_words: list[str] = []
    turn_start_time: float | None = None

    def _flush_turn() -> None:
        nonlocal current_speaker, current_words, turn_start_time
        if not current_words:
            return
        spk_label = f"Speaker {current_speaker}" if current_speaker is not None else None
        turns_data.append((spk_label, list(current_words), turn_start_time))
        current_words = []
        turn_start_time = None

    for result in results:
        if not result.alternatives:
            continue

        alternative = result.alternatives[0]

        # If no word-level info, fall back to transcript text
        if not alternative.words:
            transcript = alternative.transcript.strip()
            if transcript:
                res_time = parse_duration_seconds(getattr(result, "result_end_offset", None))
                turns_data.append((None, [transcript], res_time))
            continue

        for word_info in alternative.words:
            speaker_tag = word_info.speaker_label
            try:
                speaker_num = int(speaker_tag) if speaker_tag else None
            except (ValueError, TypeError):
                speaker_num = None

            word = word_info.word
            w_time = parse_duration_seconds(getattr(word_info, "start_offset", None))

            if speaker_num != current_speaker:
                _flush_turn()
                current_words = [word]
                current_speaker = speaker_num
                turn_start_time = w_time
            else:
                current_words.append(word)
                if turn_start_time is None and w_time is not None:
                    turn_start_time = w_time

    # Flush remaining words
    _flush_turn()

    return _render_turns_with_milestones(
        turns=turns_data,
        time_offset=time_offset,
        duration=duration,
        total_audio_duration=total_audio_duration,
        milestone_interval=milestone_interval,
        max_chars_per_line=max_chars_per_line,
        max_sentences_per_line=max_sentences_per_line,
    )


def format_plain_transcript(
    results: list[cloud_speech.SpeechRecognitionResult],
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
    time_offset: float = 0.0,
    duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    total_audio_duration: float | None = None,
) -> str:
    """
    Concatenate transcript segments into plain text formatted at sentence boundaries,
    with milestone indicators.

    Args:
        results: List of SpeechRecognitionResult from the API response.
        max_chars_per_line: Target maximum character length per line.
        max_sentences_per_line: Target maximum sentences per line.
        time_offset: Audio offset in seconds for this chunk.
        duration: Duration of this chunk in seconds.
        milestone_interval: Interval in seconds for milestone timestamp indicators.
        total_audio_duration: Total duration of the recording across all chunks.

    Returns:
        Plain transcript text split cleanly into readable lines.
    """
    if not results:
        return ""

    turns_data: list[tuple[str | None, list[str], float | None]] = []
    for result in results:
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript.strip()
        if transcript:
            res_time = parse_duration_seconds(getattr(result, "result_end_offset", None))
            turns_data.append((None, [transcript], res_time))

    if not turns_data:
        return ""

    segments = [" ".join(texts) for _, texts, _ in turns_data]
    raw_text = " ".join(segments)

    # If duration and milestone interval are active, render with milestones
    if milestone_interval > 0 and duration and duration > 0:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text.strip()) if s.strip()]
        sentence_turns = [(None, [s], None) for s in sentences]
        return _render_turns_with_milestones(
            turns=sentence_turns,
            time_offset=time_offset,
            duration=duration,
            total_audio_duration=total_audio_duration,
            milestone_interval=milestone_interval,
            max_chars_per_line=max_chars_per_line,
            max_sentences_per_line=max_sentences_per_line,
        )

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
    time_offset: float = 0.0,
    duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    total_audio_duration: float | None = None,
) -> str:
    """
    Formats smart transcription text while preserving structured markdown
    (bullets, numbered lists, headings), applying line-length wrapping,
    and inserting milestone timestamp indicators.
    """
    if not text or not text.strip():
        return ""

    raw_blocks = text.strip().split("\n")
    formatted_blocks: list[tuple[str, int]] = []

    for line in raw_blocks:
        stripped = line.strip()
        if not stripped:
            formatted_blocks.append(("", 0))
            continue

        list_match = re.match(r"^([*+-]|\d+\.)\s+(.*)$", stripped)
        if list_match:
            prefix = list_match.group(1) + " "
            item_text = list_match.group(2)
            if len(stripped) <= max_chars_per_line:
                formatted_blocks.append((stripped, len(stripped.split())))
            else:
                wrapped = format_transcript_lines(item_text, max_chars_per_line=max(30, max_chars_per_line - len(prefix)))
                wrapped_lines = wrapped.split("\n")
                first = f"{prefix}{wrapped_lines[0]}"
                indent = " " * len(prefix)
                rest = [f"{indent}{wl}" for wl in wrapped_lines[1:]]
                formatted_blocks.append(("\n".join([first] + rest), len(stripped.split())))
        elif stripped.startswith("#"):
            formatted_blocks.append((stripped, len(stripped.split())))
        else:
            if len(stripped) <= max_chars_per_line:
                formatted_blocks.append((stripped, len(stripped.split())))
            else:
                wrapped_text = format_transcript_lines(stripped, max_chars_per_line=max_chars_per_line)
                formatted_blocks.append((wrapped_text, len(stripped.split())))

    effective_total_duration = total_audio_duration if total_audio_duration is not None else (
        (time_offset + duration) if duration else None
    )
    include_hours = (effective_total_duration is not None and effective_total_duration >= 3600) or (
        (time_offset + (duration or 0.0)) >= 3600
    )

    milestone_step = float(milestone_interval) if milestone_interval > 0 else 0.0
    next_milestone = None
    if milestone_step > 0 and duration and duration > 0:
        first_ms = math.ceil(time_offset / milestone_step) * milestone_step
        next_milestone = first_ms if first_ms > 0 else milestone_step

    total_words = sum(w_count for _, w_count in formatted_blocks)
    result_lines: list[str] = []
    cursor_words = 0

    for block_text, w_count in formatted_blocks:
        if not block_text:
            result_lines.append("")
            continue

        if milestone_step > 0 and next_milestone is not None and total_words > 0 and duration:
            t = (cursor_words / total_words) * duration
            global_t = time_offset + t
            if global_t >= next_milestone:
                while next_milestone <= global_t:
                    result_lines.append("")
                    result_lines.append(f"_**{format_timestamp(next_milestone, include_hours)}**_")
                    result_lines.append("")
                    next_milestone += milestone_step

        cursor_words += w_count
        result_lines.append(block_text)

    # Clean up consecutive empty lines
    cleaned_lines: list[str] = []
    consecutive_empty = 0
    for line in result_lines:
        if not line.strip():
            consecutive_empty += 1
            if consecutive_empty <= 1:
                cleaned_lines.append("")
        else:
            consecutive_empty = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def format_gemini_diarized_transcript(
    response: Any,
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
    time_offset: float = 0.0,
    duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    total_audio_duration: float | None = None,
) -> str:
    """
    Processes a Gemini transcription response with diarization into standard speaker turns:
        Speaker 1: ...
        Speaker 2: ...
    with milestone timestamp indicators.
    """
    if not response:
        return ""

    turns: list[tuple[str, list[str], float | None]] = []

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
            turn_start_time: float | None = None

            for part in parts:
                audio_tx = getattr(part, "audio_transcription", None)
                if audio_tx is None and isinstance(part, dict):
                    audio_tx = part.get("audio_transcription")

                speaker_raw = None
                words_list = None
                if audio_tx:
                    speaker_raw = (
                        getattr(audio_tx, "speaker_label", None)
                        or getattr(audio_tx, "speakerLabel", None)
                    )
                    if speaker_raw is None and isinstance(audio_tx, dict):
                        speaker_raw = audio_tx.get("speaker_label") or audio_tx.get("speakerLabel")
                    words_list = getattr(audio_tx, "words", None)
                    if words_list is None and isinstance(audio_tx, dict):
                        words_list = audio_tx.get("words")

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

                first_w_time = None
                if words_list and len(words_list) > 0:
                    w0 = words_list[0]
                    first_w_time = parse_duration_seconds(
                        getattr(w0, "start_offset", None)
                        if not isinstance(w0, dict)
                        else w0.get("start_offset")
                    )

                if spk != current_speaker:
                    if current_speaker is not None and current_texts:
                        turns.append((current_speaker, current_texts, turn_start_time))
                    current_speaker = spk
                    current_texts = [text_str]
                    turn_start_time = first_w_time
                else:
                    current_texts.append(text_str)
                    if turn_start_time is None and first_w_time is not None:
                        turn_start_time = first_w_time

            if current_speaker is not None and current_texts:
                turns.append((current_speaker, current_texts, turn_start_time))

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
                            turns.append((current_speaker, current_texts, None))
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
                turns.append((current_speaker, current_texts, None))

    turns_typed: list[tuple[str | None, list[str], float | None]] = [
        (spk, texts, t) for spk, texts, t in turns
    ]

    return _render_turns_with_milestones(
        turns=turns_typed,
        time_offset=time_offset,
        duration=duration,
        total_audio_duration=total_audio_duration,
        milestone_interval=milestone_interval,
        max_chars_per_line=max_chars_per_line,
        max_sentences_per_line=max_sentences_per_line,
    )


def format_gemini_plain_transcript(
    response: Any,
    mode: str = "verbatim",
    max_chars_per_line: int = TRANSCRIPTION_MAX_CHARS_PER_LINE,
    max_sentences_per_line: int = TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
    time_offset: float = 0.0,
    duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    total_audio_duration: float | None = None,
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
        return format_smart_transcript(
            raw_text,
            max_chars_per_line=max_chars_per_line,
            time_offset=time_offset,
            duration=duration,
            milestone_interval=milestone_interval,
            total_audio_duration=total_audio_duration,
        )

    # Verbatim mode
    if milestone_interval > 0 and duration and duration > 0:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text.strip()) if s.strip()]
        turns_data: list[tuple[str | None, list[str], float | None]] = [
            (None, [s], None) for s in sentences
        ]
        return _render_turns_with_milestones(
            turns=turns_data,
            time_offset=time_offset,
            duration=duration,
            total_audio_duration=total_audio_duration,
            milestone_interval=milestone_interval,
            max_chars_per_line=max_chars_per_line,
            max_sentences_per_line=max_sentences_per_line,
        )

    return format_transcript_lines(
        raw_text,
        max_chars_per_line=max_chars_per_line,
        max_sentences_per_line=max_sentences_per_line,
    )


def stitch_diarized_transcripts(
    responses: list[Any],
    chunk_durations: list[float] | None = None,
    total_duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
) -> str:
    """
    Stitch together diarized transcripts from multiple chunked files.

    Inserts Markdown chunk indicators and explicit speaker non-alignment warning notices
    between segments, informing readers that speaker numbers reset per chunk.

    Args:
        responses: Ordered list of BatchRecognizeResponse or Gemini responses.
        chunk_durations: Optional list of durations for each chunk in seconds.
        total_duration: Total duration of the recording in seconds.
        milestone_interval: Interval for milestone timestamp indicators.

    Returns:
        Full stitched transcript with chunk boundary headers and speaker notices.
    """
    parts: list[str] = []
    total_chunks = len(responses)
    cumulative_time = 0.0

    effective_total_duration = total_duration
    if effective_total_duration is None and chunk_durations:
        effective_total_duration = sum(chunk_durations)

    include_hours = effective_total_duration is not None and effective_total_duration >= 3600

    for i, response in enumerate(responses):
        dur = None
        if chunk_durations and i < len(chunk_durations):
            dur = chunk_durations[i]
        elif total_duration and total_chunks > 0:
            dur = total_duration / total_chunks

        if hasattr(response, "results") and hasattr(response, "_pb"):
            results = _extract_results_from_response(response)
            chunk_text = format_diarized_transcript(
                results,
                time_offset=cumulative_time,
                duration=dur,
                milestone_interval=milestone_interval,
                total_audio_duration=effective_total_duration,
            )
        elif isinstance(response, str):
            chunk_text = response
        else:
            chunk_text = format_gemini_diarized_transcript(
                response,
                time_offset=cumulative_time,
                duration=dur,
                milestone_interval=milestone_interval,
                total_audio_duration=effective_total_duration,
            )

        if chunk_text:
            if total_chunks > 1:
                part_letter = chr(ord('A') + i) if i < 26 else str(i + 1)
                time_range = ""
                if dur and dur > 0:
                    start_t = cumulative_time
                    end_t = cumulative_time + dur
                    time_range = f" ({format_timestamp(start_t, include_hours)[1:-1]} - {format_timestamp(end_t, include_hours)[1:-1]})"

                if i == 0:
                    header = (
                        f"## Part A (Chunk 1 of {total_chunks}){time_range}\n\n"
                        f"> ℹ️ **Speaker Attribution Scope:** This recording was split into {total_chunks} parts for processing. "
                        f"Speaker labels (e.g. `Speaker 1`, `Speaker 2`) are local to Part A and might not align with Part B, C, D, etc.\n\n"
                    )
                    parts.append(header)
                else:
                    prev_letter = chr(ord('A') + i - 1) if i - 1 < 26 else str(i)
                    boundary_marker = (
                        f"\n\n---\n\n"
                        f"## Part {part_letter} (Chunk {i + 1} of {total_chunks}){time_range}\n\n"
                        f"> ⚠️ **Speaker Attribution Notice:** Speaker numbers in Chunk {i + 1} (Part {part_letter}) "
                        f"are local to this chunk and DO NOT correspond to speaker numbers in previous chunks (Part {prev_letter}). "
                        f"Speaker labels from Part {prev_letter} might not align with Part {part_letter}, etc.\n\n"
                    )
                    parts.append(boundary_marker)

            parts.append(chunk_text)

        if dur:
            cumulative_time += dur

    return "".join(parts)


def stitch_plain_transcripts(
    responses: list[Any],
    mode: str = "verbatim",
    chunk_durations: list[float] | None = None,
    total_duration: float | None = None,
    milestone_interval: float = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
) -> str:
    """
    Stitch together plain (non-diarized) transcripts.

    Inserts Markdown chunk boundary markers if multiple chunks are present.

    Args:
        responses: Ordered list of BatchRecognizeResponse or Gemini responses.
        mode: Transcription mode ('verbatim' or 'smart') for Gemini plain text.
        chunk_durations: Optional list of durations for each chunk in seconds.
        total_duration: Total duration of the recording in seconds.
        milestone_interval: Interval for milestone timestamp indicators.

    Returns:
        Full stitched transcript text.
    """
    parts: list[str] = []
    total_chunks = len(responses)
    cumulative_time = 0.0

    effective_total_duration = total_duration
    if effective_total_duration is None and chunk_durations:
        effective_total_duration = sum(chunk_durations)

    include_hours = effective_total_duration is not None and effective_total_duration >= 3600

    for i, response in enumerate(responses):
        dur = None
        if chunk_durations and i < len(chunk_durations):
            dur = chunk_durations[i]
        elif total_duration and total_chunks > 0:
            dur = total_duration / total_chunks

        if hasattr(response, "results") and hasattr(response, "_pb"):
            results = _extract_results_from_response(response)
            chunk_text = format_plain_transcript(
                results,
                time_offset=cumulative_time,
                duration=dur,
                milestone_interval=milestone_interval,
                total_audio_duration=effective_total_duration,
            )
        elif isinstance(response, str):
            chunk_text = response
        else:
            chunk_text = format_gemini_plain_transcript(
                response,
                mode=mode,
                time_offset=cumulative_time,
                duration=dur,
                milestone_interval=milestone_interval,
                total_audio_duration=effective_total_duration,
            )

        if chunk_text:
            if total_chunks > 1:
                part_letter = chr(ord('A') + i) if i < 26 else str(i + 1)
                time_range = ""
                if dur and dur > 0:
                    start_t = cumulative_time
                    end_t = cumulative_time + dur
                    time_range = f" ({format_timestamp(start_t, include_hours)[1:-1]} - {format_timestamp(end_t, include_hours)[1:-1]})"

                if i == 0:
                    header = f"## Part A (Chunk 1 of {total_chunks}){time_range}\n\n"
                    parts.append(header)
                else:
                    boundary_marker = f"\n\n---\n\n## Part {part_letter} (Chunk {i + 1} of {total_chunks}){time_range}\n\n"
                    parts.append(boundary_marker)

            parts.append(chunk_text)

        if dur:
            cumulative_time += dur

    return "".join(parts) if total_chunks > 1 else (" ".join(parts) if parts else "")


def save_transcript(
    transcript_text: str,
    raw_responses: list[Any],
    output_dir: str,
    base_name: str,
    lang: str,
    duration: float | None = None,
    model: str | None = None,
    diarize: bool | None = None,
    speaker_count: int | str | None = None,
    mode: str | None = None,
    languages: list[str] | None = None,
    source_file: str | None = None,
    repository: str = REPOSITORY_NAME,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Save the formatted Markdown transcript (with frontmatter) and raw API response JSON to disk.

    Args:
        transcript_text: Formatted transcript string.
        raw_responses: Raw response objects for JSON dump.
        output_dir: Directory to save output files.
        base_name: Base filename (without extension).
        lang: Primary language code used.
        duration: Source file duration in seconds.
        model: Model name used (e.g. 'gemini-3.5-transcribe', 'chirp_3').
        diarize: Whether speaker diarization was enabled.
        speaker_count: Speaker count hint or detected count.
        mode: Transcription mode ('verbatim' or 'smart').
        languages: Full list of language codes configured (e.g. for code-switching).
        source_file: Original source media filename or path.
        repository: Repository name for frontmatter.
        metadata: Additional settings dictionary to include in frontmatter.

    Returns:
        Tuple of (transcript_md_path, json_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure Markdown content has YAML frontmatter
    if transcript_text.startswith("---\n") or transcript_text.startswith("---\r\n"):
        full_content = transcript_text
    else:
        frontmatter = generate_frontmatter(
            source_file=source_file or base_name,
            duration=duration,
            repository=repository,
            model=model or "",
            mode=mode or "",
            lang=lang,
            languages=languages,
            diarize=diarize,
            speakers=speaker_count,
            extra_settings=metadata,
        )
        display_title = os.path.basename(source_file) if source_file else base_name
        full_content = f"{frontmatter}\n\n# Transcript: {display_title}\n\n{transcript_text.strip()}\n"

    # Save human/LLM-readable transcript as Markdown (.md)
    md_path = os.path.join(output_dir, f"{base_name}_{lang}_transcript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_content)
    logger.info("Transcript saved to: %s", md_path)

    # Save raw API response as JSON
    json_path = os.path.join(output_dir, f"{base_name}_{lang}_raw.json")
    raw_data = _responses_to_dict(raw_responses)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    logger.info("Raw API response saved to: %s", json_path)

    return md_path, json_path
