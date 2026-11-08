import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

from src.config import (
    CHUNK_MAX_LENGTH_SECONDS_DUTCH,
    CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
    CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
    CHUNK_MAX_LENGTH_SECONDS_GEMINI_PLAIN,
    CHUNK_MAX_LENGTH_SECONDS_GEMINI_SMART,
    CHUNK_MIN_LENGTH_SECONDS,
    CHUNK_SILENCE_MIN_DURATION_SECONDS,
    CHUNK_SILENCE_NOISE_THRESHOLD_DB,
    CHUNK_SILENCE_SEARCH_WINDOW_SECONDS,
    GCP_PROJECT_ID,
    GCP_REGION,
    GCS_BUCKET_NAME,
    GCS_UPLOAD_TIMEOUT_SECONDS,
    GEMINI_API_KEY,
    GEMINI_BACKEND,
    LANGUAGE_CODES_ALL,
    LANGUAGE_CODES_DUTCH,
    LANGUAGE_CODES_ENGLISH,
    OUTPUT_DIR,
    TRANSCRIPT_OUTPUT_DIR,
    TRANSCRIPTION_CONCURRENCY,
    TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
    TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
    TRANSCRIPTION_MODEL,
    resolve_gemini_language_codes,
    validate_backend_config,
    validate_extension,
    validate_gcp_config,
)
from src.audio import (
    ProcessingError,
    compute_chunk_boundaries,
    detect_silences,
    extract_audio_to_flac,
    get_audio_duration,
    split_flac,
)
from src.auth import check_auth, get_gemini_backend_mode
from src.gcs import delete_gcs_prefix, upload_files_to_gcs, upload_to_gcs
from src.logger import setup_logging
from src.transcribe import transcribe_files, transcribe_single_file
from src.transcribe_gcs import transcribe_from_gcs
from src.transcribe_gemini import transcribe_gemini_file, transcribe_gemini_files
from src.transcript_output import (
    save_transcript,
    stitch_diarized_transcripts,
    stitch_plain_transcripts,
)

logger = logging.getLogger(__name__)

EXIT_DECISION_REQUIRED = 2

# Duration safe ceilings in seconds
GEMINI_DIARIZATION_MAX_SECONDS = CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED  # 1680s (28 min)
CHIRP_MAX_SECONDS = CHUNK_MAX_LENGTH_SECONDS_ENGLISH  # 2700s (45 min)
GEMINI_PLAIN_MAX_SECONDS = CHUNK_MAX_LENGTH_SECONDS_GEMINI_PLAIN  # 3480s (58 min)
GEMINI_SMART_MAX_SECONDS = CHUNK_MAX_LENGTH_SECONDS_GEMINI_SMART  # 1200s (20 min)


def _get_gemini_plain_chunk_limit(mode: str) -> int:
    """Returns safe chunk length for non-diarized Gemini transcription depending on mode."""
    if mode == "smart":
        return CHUNK_MAX_LENGTH_SECONDS_GEMINI_SMART
    return CHUNK_MAX_LENGTH_SECONDS_GEMINI_PLAIN


class TranscriptionPlan:
    """Encapsulates the resolved model, diarization mode, and chunking strategy."""

    def __init__(
        self,
        model: str,
        diarize: bool,
        mode: str = "verbatim",
        speaker_count: int | None = None,
        max_chunk_length: int = 2700,
        target_parallelism: int | None = None,
    ):
        self.model = model
        self.diarize = diarize
        self.mode = mode
        self.speaker_count = speaker_count
        self.max_chunk_length = max_chunk_length
        self.target_parallelism = target_parallelism

    def __repr__(self) -> str:
        return (
            f"TranscriptionPlan(model='{self.model}', diarize={self.diarize}, mode='{self.mode}', "
            f"speaker_count={self.speaker_count}, max_chunk_length={self.max_chunk_length}, "
            f"target_parallelism={self.target_parallelism})"
        )


def _has_gemini_backend() -> bool:
    """Checks whether Gemini transcription can be initialized from config."""
    if GEMINI_BACKEND == "api_key":
        return bool(GEMINI_API_KEY)
    if GEMINI_BACKEND == "vertex":
        return bool(GCP_PROJECT_ID)
    # auto
    return bool(GEMINI_API_KEY or GCP_PROJECT_ID)


def _has_chirp_backend() -> bool:
    """Checks whether Speech-to-Text V2 (Chirp 3) is configured."""
    return bool(GCP_PROJECT_ID and GCS_BUCKET_NAME)


def _prompt_speaker_count() -> int | None:
    """
    Interactively prompt the user for the number of speakers in the recording.
    Returns:
        int if the user provides a valid positive integer, None for auto-detect.
    """
    if not sys.stdin.isatty():
        logger.info("Non-interactive mode detected. Using auto-detect for speaker count.")
        return None

    while True:
        try:
            raw = input("How many speakers were in this recording? (press Enter to auto-detect): ")
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Using auto-detect for speaker count.")
            return None

        raw = raw.strip()
        if not raw:
            logger.info("Speaker count: auto-detect (empty input).")
            return None

        try:
            count = int(raw)
            if count < 1:
                print("Please enter a positive integer (1 or greater), or press Enter to auto-detect.")
                continue
            logger.info("Speaker count: %d (user-provided).", count)
            return count
        except ValueError:
            print("Invalid input. Please enter a number or press Enter to auto-detect.")


def _format_decision_block(
    input_file: str,
    lang: str,
    duration: float,
    reason: str,
    options: list[dict[str, Any]],
) -> str:
    """Renders human-readable decision text block for terminal or logs."""
    mm = int(duration // 60)
    ss = int(duration % 60)
    duration_str = f"{mm}m{ss:02d}s"

    lines = [
        "",
        "=" * 78,
        "[DECISION REQUIRED] Audio duration requires a transcription routing decision.",
        f"  Recording: {input_file}",
        f"  Duration:  {duration_str} ({duration:.2f}s)",
        f"  Language:  {lang}",
        f"  Reason:    {reason}",
        "",
        "Available options to re-run (or pass -i / --interactive):",
    ]

    for idx, opt in enumerate(options, 1):
        lines.append(f"  {idx}. {opt['title']}:")
        lines.append(f"     Description: {opt['description']}")
        lines.append(f"     Command:     python -m src.main {input_file} --lang {lang} {opt['flags']}")
        if "tradeoffs" in opt and opt["tradeoffs"]:
            lines.append(f"     Trade-offs:  {opt['tradeoffs']}")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


def resolve_transcription_plan(
    duration: float,
    lang: str,
    requested_model: str = "auto",
    requested_diarize: bool | None = None,
    mode: str = "verbatim",
    speaker_count: int | None = None,
    interactive: bool = False,
    input_file: str = "",
) -> TranscriptionPlan:
    """
    Evaluates duration, language, and backend capabilities to determine the transcription plan.
    If an ambiguous trade-off boundary is reached and flags were not explicitly provided:
      - In interactive mode (-i): prompts user interactively.
      - In non-interactive mode: logs structured decision payload and exits with code 2.

    Returns:
        Configured TranscriptionPlan.

    Raises:
        ProcessingError: On invalid language/model combinations.
        SystemExit(2): When non-interactive decision is required.
    """
    lang_lower = lang.lower()
    is_dutch = lang_lower in LANGUAGE_CODES_DUTCH
    is_english = lang_lower in LANGUAGE_CODES_ENGLISH
    has_gemini = _has_gemini_backend()
    has_chirp = _has_chirp_backend()

    # 1. Explicit Model Selection
    if requested_model == "chirp_3":
        if is_dutch and requested_diarize is True:
            raise ProcessingError(
                "Chirp 3 does not support speaker diarization for Dutch (nl-NL). "
                "Please use --model gemini-3.5-transcribe or --no-diarize."
            )
        if lang_lower == "auto":
            raise ProcessingError(
                "Chirp 3 does not support automatic language detection ('--lang auto'). "
                "Please specify an explicit language code or use --model gemini-3.5-transcribe."
            )

        diarize = (is_english) if requested_diarize is None else requested_diarize
        if diarize:
            return TranscriptionPlan(
                model="chirp_3",
                diarize=True,
                speaker_count=speaker_count,
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
                target_parallelism=None,
            )
        else:
            return TranscriptionPlan(
                model="chirp_3",
                diarize=False,
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                target_parallelism=TRANSCRIPTION_CONCURRENCY,
            )

    if requested_model == "gemini-3.5-transcribe":
        if requested_diarize is True:
            if duration > GEMINI_DIARIZATION_MAX_SECONDS:
                logger.warning(
                    "Recording is %.2fs (> 28m). Audio will be chunked into <= 28m parts; "
                    "speaker numbers (Speaker 1, Speaker 2) will reset per chunk.",
                    duration,
                )
            return TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=True,
                mode="verbatim",
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                target_parallelism=None,
            )
        elif requested_diarize is False:
            return TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=False,
                mode=mode,
                max_chunk_length=_get_gemini_plain_chunk_limit(mode),
                target_parallelism=None,
            )
        else:
            # requested_diarize is None with gemini-3.5-transcribe
            if duration <= GEMINI_DIARIZATION_MAX_SECONDS:
                return TranscriptionPlan(
                    model="gemini-3.5-transcribe",
                    diarize=True,
                    mode="verbatim",
                    max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                    target_parallelism=None,
                )
            else:
                reason = (
                    f"Recording is {int(duration // 60)}m{int(duration % 60):02d}s, exceeding "
                    "the 28-minute single-chunk diarization limit for gemini-3.5-transcribe."
                )
                options = [
                    {
                        "title": "Chunked Diarization",
                        "description": "Split audio into <= 28m chunks with speaker labels.",
                        "flags": "--model gemini-3.5-transcribe --diarize",
                        "tradeoffs": "Speaker numbers reset per chunk and will not align across chunk cuts.",
                        "plan": TranscriptionPlan(
                            model="gemini-3.5-transcribe",
                            diarize=True,
                            mode="verbatim",
                            max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                            target_parallelism=None,
                        ),
                    },
                    {
                        "title": "Disable Diarization",
                        "description": f"Transcribe without speaker labels (mode: {mode}, chunks up to {int(_get_gemini_plain_chunk_limit(mode) // 60)}m).",
                        "flags": f"--model gemini-3.5-transcribe --no-diarize --mode {mode}",
                        "tradeoffs": "Continuous text without speaker attribution labels.",
                        "plan": TranscriptionPlan(
                            model="gemini-3.5-transcribe",
                            diarize=False,
                            mode=mode,
                            max_chunk_length=_get_gemini_plain_chunk_limit(mode),
                            target_parallelism=None,
                        ),
                    },
                ]
                return _handle_decision(input_file, lang, duration, reason, options, interactive)

    # 2. Model is "auto"
    # Case 2A: Short Recordings (<= 28m / 1680s)
    if duration <= GEMINI_DIARIZATION_MAX_SECONDS:
        if has_gemini:
            diarize = True if requested_diarize is None else requested_diarize
            return TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=diarize,
                mode=mode if not diarize else "verbatim",
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED if diarize else _get_gemini_plain_chunk_limit(mode),
                target_parallelism=None,
            )
        else:
            # Fall back to Chirp 3
            diarize = is_english if requested_diarize is None else requested_diarize
            return TranscriptionPlan(
                model="chirp_3",
                diarize=diarize,
                speaker_count=speaker_count,
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH if diarize else CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                target_parallelism=None if diarize else TRANSCRIPTION_CONCURRENCY,
            )

    # Case 2B: Medium Recordings (1680s < duration <= 2700s) (28–45 min)
    if duration <= CHIRP_MAX_SECONDS:
        if is_english:
            if has_chirp:
                if requested_diarize is False:
                    model = "gemini-3.5-transcribe" if has_gemini else "chirp_3"
                    return TranscriptionPlan(
                        model=model,
                        diarize=False,
                        mode=mode,
                        max_chunk_length=_get_gemini_plain_chunk_limit(mode) if model == "gemini-3.5-transcribe" else CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                        target_parallelism=None if model == "gemini-3.5-transcribe" else TRANSCRIPTION_CONCURRENCY,
                    )
                else:
                    # English 28-45m auto-switches to chirp_3 with diarization=True for monolithic pass
                    logger.info(
                        "Audio duration (%.2fs) is between 28m and 45m. Auto-selecting chirp_3 with diarization "
                        "to transcribe monolithically without chunk cuts.",
                        duration,
                    )
                    return TranscriptionPlan(
                        model="chirp_3",
                        diarize=True,
                        speaker_count=speaker_count,
                        max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
                        target_parallelism=None,
                    )
            else:
                # Only Gemini configured (no GCS/Chirp)
                if requested_diarize is True:
                    return TranscriptionPlan(
                        model="gemini-3.5-transcribe",
                        diarize=True,
                        mode="verbatim",
                        max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                        target_parallelism=None,
                    )
                elif requested_diarize is False:
                    return TranscriptionPlan(
                        model="gemini-3.5-transcribe",
                        diarize=False,
                        mode=mode,
                        max_chunk_length=_get_gemini_plain_chunk_limit(mode),
                        target_parallelism=None,
                    )
                else:
                    reason = (
                        f"Recording is {int(duration // 60)}m{int(duration % 60):02d}s, exceeding the 28m single-chunk "
                        "diarization limit for gemini-3.5-transcribe, and GCP/GCS is not configured for chirp_3 monolithic transcription."
                    )
                    effective_plain_mode = mode if mode != "verbatim" else "verbatim"
                    options = [
                        {
                            "title": "Chunked Diarization (Gemini 3.5 Transcribe)",
                            "description": "Split audio into ~14–22 min chunks with speaker labels.",
                            "flags": "--model gemini-3.5-transcribe --diarize",
                            "tradeoffs": "Speaker numbers reset per chunk and will not align across chunk cuts.",
                            "plan": TranscriptionPlan(
                                model="gemini-3.5-transcribe",
                                diarize=True,
                                mode="verbatim",
                                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                                target_parallelism=None,
                            ),
                        },
                        {
                        "title": "Plain Transcription (Gemini 3.5 Transcribe)",
                        "description": f"Transcription without speaker diarization (mode: {effective_plain_mode}, chunks up to {int(_get_gemini_plain_chunk_limit(effective_plain_mode) // 60)}m).",
                        "flags": f"--model gemini-3.5-transcribe --no-diarize --mode {effective_plain_mode}",
                        "tradeoffs": "No speaker attribution labels, but produces continuous transcript.",
                        "plan": TranscriptionPlan(
                            model="gemini-3.5-transcribe",
                            diarize=False,
                            mode=effective_plain_mode,
                            max_chunk_length=_get_gemini_plain_chunk_limit(effective_plain_mode),
                            target_parallelism=None,
                        ),
                        },
                    ]
                    return _handle_decision(input_file, lang, duration, reason, options, interactive)
        else:
            # Dutch or other non-English medium recording
            if requested_diarize is True:
                return TranscriptionPlan(
                    model="gemini-3.5-transcribe",
                    diarize=True,
                    mode="verbatim",
                    max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                    target_parallelism=None,
                )
            elif requested_diarize is False:
                model = "chirp_3" if (has_chirp and not has_gemini) else "gemini-3.5-transcribe"
                return TranscriptionPlan(
                    model=model,
                    diarize=False,
                    mode=mode,
                    max_chunk_length=_get_gemini_plain_chunk_limit(mode) if model == "gemini-3.5-transcribe" else CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                    target_parallelism=None if model == "gemini-3.5-transcribe" else TRANSCRIPTION_CONCURRENCY,
                )
            else:
                reason = (
                    f"Recording is {int(duration // 60)}m{int(duration % 60):02d}s. Chirp 3 does not support Dutch diarization, "
                    "and gemini-3.5-transcribe cannot diarize recordings over 28m in a single pass."
                )
                effective_plain_mode = mode if mode != "verbatim" else "verbatim"
                options = [
                    {
                        "title": "Chunked Diarization (Gemini 3.5 Transcribe)",
                        "description": "Split into <= 28 min chunks with speaker diarization.",
                        "flags": "--model gemini-3.5-transcribe --diarize",
                        "tradeoffs": "Speaker numbers reset per chunk and will not align across chunk cuts.",
                        "plan": TranscriptionPlan(
                            model="gemini-3.5-transcribe",
                            diarize=True,
                            mode="verbatim",
                            max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                            target_parallelism=None,
                        ),
                    },
                    {
                        "title": "Plain Transcription (Gemini 3.5 Transcribe)",
                        "description": f"Transcription without diarization (mode: {effective_plain_mode}, chunks up to {int(_get_gemini_plain_chunk_limit(effective_plain_mode) // 60)}m).",
                        "flags": f"--model gemini-3.5-transcribe --no-diarize --mode {effective_plain_mode}",
                        "tradeoffs": "No speaker attribution, but produces continuous transcript.",
                        "plan": TranscriptionPlan(
                            model="gemini-3.5-transcribe",
                            diarize=False,
                            mode=effective_plain_mode,
                            max_chunk_length=_get_gemini_plain_chunk_limit(effective_plain_mode),
                            target_parallelism=None,
                        ),
                    },
                ]
                if has_chirp:
                    options.append({
                        "title": "Fast Parallel Wave Plain (Chirp 3)",
                        "description": "Adaptive parallel waves on Chirp 3 for fastest wall-clock completion.",
                        "flags": "--model chirp_3 --no-diarize",
                        "tradeoffs": "No speaker attribution; plain text output.",
                        "plan": TranscriptionPlan(
                            model="chirp_3",
                            diarize=False,
                            max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                            target_parallelism=TRANSCRIPTION_CONCURRENCY,
                        ),
                    })
                return _handle_decision(input_file, lang, duration, reason, options, interactive)

    # Case 2C: Long Recordings (> 2700s / > 45 min)
    if requested_diarize is True:
        if not is_english or not has_chirp:
            return TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=True,
                mode="verbatim",
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                target_parallelism=None,
            )
        else:
            return TranscriptionPlan(
                model="chirp_3",
                diarize=True,
                speaker_count=speaker_count,
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
                target_parallelism=None,
            )
    elif requested_diarize is False:
        if has_chirp:
            return TranscriptionPlan(
                model="chirp_3",
                diarize=False,
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                target_parallelism=TRANSCRIPTION_CONCURRENCY,
            )
        else:
            return TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=False,
                mode=mode,
                max_chunk_length=_get_gemini_plain_chunk_limit(mode),
                target_parallelism=None,
            )
    else:
        # Long recording with no requested_diarize preference
        reason = (
            f"Recording is {int(duration // 60)}m{int(duration % 60):02d}s (> 45 min). "
            "Diarization without chunk boundaries is impossible on any model."
        )
        options = [
            {
                "title": "Keep Diarization via Gemini 3.5 Transcribe",
                "description": "Chunk audio into <= 28 min chunks with speaker labels (Dutch & English).",
                "flags": "--model gemini-3.5-transcribe --diarize",
                "tradeoffs": "Speaker numbers reset per chunk. Best for <= 2 speakers or multilingual audio.",
                "plan": TranscriptionPlan(
                    model="gemini-3.5-transcribe",
                    diarize=True,
                    mode="verbatim",
                    max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_DIARIZED,
                    target_parallelism=None,
                ),
            },
        ]
        if is_english and has_chirp:
            options.append({
                "title": "Keep Diarization via Chirp 3 (Fewer Chunks)",
                "description": "Chunk audio into <= 45 min chunks with speaker labels (fewer chunk cuts than Gemini).",
                "flags": "--model chirp_3 --diarize" + (f" --speakers {speaker_count}" if speaker_count else ""),
                "tradeoffs": "Speaker numbers reset per chunk. Better attribution for 3+ speakers.",
                "plan": TranscriptionPlan(
                    model="chirp_3",
                    diarize=True,
                    speaker_count=speaker_count,
                    max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
                    target_parallelism=None,
                ),
            })
        options.append({
            "title": "Disable Diarization via Gemini 3.5 Transcribe (Verbatim Plain)",
            "description": "Chunk into <= 58 min parts without diarization, preserving word-for-word transcript without omissions.",
            "flags": "--model gemini-3.5-transcribe --no-diarize --mode verbatim",
            "tradeoffs": "No speaker attribution labels, but guarantees completeness across long recordings.",
            "plan": TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=False,
                mode="verbatim",
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_PLAIN,
                target_parallelism=None,
            ),
        })
        options.append({
            "title": "Disable Diarization via Gemini 3.5 Transcribe (Smart Mode)",
            "description": "Chunk into safe <= 20 min parts without diarization, with intelligent formatting and disfluency cleanup.",
            "flags": "--model gemini-3.5-transcribe --no-diarize --mode smart",
            "tradeoffs": "No speaker attribution labels; chunked at <= 20m to prevent mid-context compression/skipping.",
            "plan": TranscriptionPlan(
                model="gemini-3.5-transcribe",
                diarize=False,
                mode="smart",
                max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_GEMINI_SMART,
                target_parallelism=None,
            ),
        })
        if has_chirp:
            options.append({
                "title": "Disable Diarization via Chirp 3 (Fast Parallel Waves)",
                "description": "Adaptive parallel wave chunking on Chirp 3 for fastest wall-clock time.",
                "flags": "--model chirp_3 --no-diarize",
                "tradeoffs": "No speaker attribution; plain text output.",
                "plan": TranscriptionPlan(
                    model="chirp_3",
                    diarize=False,
                    max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_DUTCH,
                    target_parallelism=TRANSCRIPTION_CONCURRENCY,
                ),
            })

        return _handle_decision(input_file, lang, duration, reason, options, interactive)


def _handle_decision(
    input_file: str,
    lang: str,
    duration: float,
    reason: str,
    options: list[dict[str, Any]],
    interactive: bool,
) -> TranscriptionPlan:
    """Prompts interactively or logs structured decision JSON and exits with code 2."""
    if interactive:
        print(_format_decision_block(input_file, lang, duration, reason, options))
        while True:
            try:
                choice_str = input(f"Please select an option (1-{len(options)}): ").strip()
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(options):
                    chosen_opt = options[choice_idx]
                    logger.info("Selected option %d: %s", choice_idx + 1, chosen_opt["title"])
                    return chosen_opt["plan"]
                print(f"Invalid selection. Please enter a number between 1 and {len(options)}.")
            except (ValueError, IndexError):
                print(f"Invalid input. Please enter a number between 1 and {len(options)}.")
            except (KeyboardInterrupt, EOFError):
                print()
                logger.info("Decision cancelled by user. Exiting.")
                sys.exit(1)

    # Non-interactive mode: emit human-readable warning and structured JSON to stderr
    logger.warning(_format_decision_block(input_file, lang, duration, reason, options))
    decision_json = {
        "status": "decision_required",
        "input_file": input_file,
        "language": lang,
        "duration_seconds": round(duration, 2),
        "duration_formatted": f"{int(duration // 60)}m{int(duration % 60):02d}s",
        "reason": reason,
        "options": [
            {
                "option": i + 1,
                "title": opt["title"],
                "description": opt["description"],
                "flags": opt["flags"],
                "command": f"python -m src.main {input_file} --lang {lang} {opt['flags']}".strip(),
                "tradeoffs": opt.get("tradeoffs", ""),
            }
            for i, opt in enumerate(options)
        ],
    }
    sys.stderr.write(json.dumps(decision_json, indent=2) + "\n")
    sys.exit(EXIT_DECISION_REQUIRED)


def execute_transcription_plan(
    input_path: str,
    final_output: str,
    base_name: str,
    lang: str,
    duration: float,
    plan: TranscriptionPlan,
    prepare_only: bool = False,
    milestone_interval: int = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
) -> None:
    """Executes chunking, upload, transcription, stitching, saving, and cleanup per plan."""
    silences = detect_silences(
        final_output,
        noise_threshold=CHUNK_SILENCE_NOISE_THRESHOLD_DB,
        min_duration=CHUNK_SILENCE_MIN_DURATION_SECONDS,
    )
    boundaries = compute_chunk_boundaries(
        duration=duration,
        max_chunk_length=plan.max_chunk_length,
        silences=silences,
        window=CHUNK_SILENCE_SEARCH_WINDOW_SECONDS,
        min_chunk_length=CHUNK_MIN_LENGTH_SECONDS,
        target_parallelism=plan.target_parallelism,
    )

    is_chunked = len(boundaries) > 0
    chunk_paths: list[str] = []

    if is_chunked:
        logger.info(
            "Audio duration (%.2fs) divided into %d chunks with adaptive pause snapping (max chunk length: %ds).",
            duration, len(boundaries) + 1, plan.max_chunk_length,
        )
        chunk_paths = split_flac(final_output, OUTPUT_DIR, base_name, boundaries)

        if prepare_only:
            logger.info("-------------------------------------------------------------------")
            logger.info(
                "READY FOR GCP: Files generated in %s/ matching %s_chunk_*.flac",
                OUTPUT_DIR, base_name,
            )
            diarize_msg = "ENABLE" if plan.diarize else "Do NOT enable"
            logger.info("Upload these chunked files to GCP. %s Speaker Diarization.", diarize_msg)
            logger.info(
                "Note: You will need to manually stitch the resulting transcripts "
                "until the full pipeline is used (without --prepare-only)."
            )
            logger.info("-------------------------------------------------------------------")
            if os.path.exists(final_output):
                os.remove(final_output)
            return
    else:
        logger.info(
            "Audio duration (%.2fs) within monolithic limit (%ds). Proceeding with monolithic file.",
            duration, plan.max_chunk_length,
        )
        if prepare_only:
            logger.info("-------------------------------------------------------------------")
            logger.info("READY FOR GCP: %s", final_output)
            diarize_msg = "ENABLE" if plan.diarize else "Do NOT enable"
            logger.info("Upload this file to GCP. %s Speaker Diarization.", diarize_msg)
            logger.info("-------------------------------------------------------------------")
            return

    # Validate backend configuration
    validate_backend_config(model=plan.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_prefix = f"{base_name}_{timestamp}"
    uploaded_to_gcs = False

    try:
        if plan.model == "gemini-3.5-transcribe":
            backend = get_gemini_backend_mode()
            if backend == "api_key":
                targets = chunk_paths if is_chunked else [final_output]
                logger.info(
                    "Starting Gemini transcription (model: %s, backend: %s, lang: %s, diarization: %s, mode: %s)...",
                    plan.model, backend, lang, "enabled" if plan.diarize else "disabled", plan.mode,
                )
                responses = transcribe_gemini_files(
                    audio_targets=targets,
                    lang=lang,
                    diarize=plan.diarize,
                    mode=plan.mode,
                    backend="api_key",
                    concurrency=TRANSCRIPTION_CONCURRENCY,
                    total_audio_seconds=duration,
                    chunk_duration=duration / len(targets) if targets else duration,
                )
            else:  # vertex
                uploaded_to_gcs = True
                assert GCS_BUCKET_NAME is not None
                if is_chunked:
                    gcs_uris = upload_files_to_gcs(
                        chunk_paths, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS
                    )
                else:
                    gcs_uris = [
                        upload_to_gcs(
                            final_output, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS
                        )
                    ]
                logger.info(
                    "Starting Gemini transcription (model: %s, backend: vertex, lang: %s, diarization: %s, mode: %s)...",
                    plan.model, lang, "enabled" if plan.diarize else "disabled", plan.mode,
                )
                responses = transcribe_gemini_files(
                    audio_targets=gcs_uris,
                    lang=lang,
                    diarize=plan.diarize,
                    mode=plan.mode,
                    backend="vertex",
                    concurrency=TRANSCRIPTION_CONCURRENCY,
                    total_audio_seconds=duration,
                    chunk_duration=duration / len(gcs_uris) if gcs_uris else duration,
                )

            logger.info("Transcription complete. Processing results...")
            chunk_durs = (
                [boundaries[0]] + [boundaries[k] - boundaries[k - 1] for k in range(1, len(boundaries))] + [duration - boundaries[-1]]
                if is_chunked and boundaries
                else [duration]
            )
            if plan.diarize:
                transcript_text = stitch_diarized_transcripts(
                    responses,
                    chunk_durations=chunk_durs,
                    total_duration=duration,
                    milestone_interval=milestone_interval,
                )
            else:
                transcript_text = stitch_plain_transcripts(
                    responses,
                    mode=plan.mode,
                    chunk_durations=chunk_durs,
                    total_duration=duration,
                    milestone_interval=milestone_interval,
                )

            save_transcript(
                transcript_text=transcript_text,
                raw_responses=responses,
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
                duration=duration,
                model=plan.model,
                diarize=plan.diarize,
                speaker_count=plan.speaker_count,
                mode=plan.mode,
                languages=resolve_gemini_language_codes(lang) if plan.model == "gemini-3.5-transcribe" else [lang],
                source_file=input_path,
            )

        elif plan.model == "chirp_3":
            uploaded_to_gcs = True
            assert GCS_BUCKET_NAME is not None
            assert GCP_PROJECT_ID is not None

            if is_chunked:
                gcs_uris = upload_files_to_gcs(
                    chunk_paths, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS
                )
                speaker_str = str(plan.speaker_count) if plan.speaker_count else "auto-detect"
                logger.info(
                    "Starting transcription (model: chirp_3, lang: %s, diarization: %s, speakers: %s)...",
                    lang, "enabled" if plan.diarize else "disabled", speaker_str,
                )
                responses = transcribe_files(
                    gcs_uris=gcs_uris,
                    lang=lang,
                    enable_diarization=plan.diarize,
                    speaker_count=plan.speaker_count,
                    project_id=GCP_PROJECT_ID,
                    region=GCP_REGION,
                    model="chirp_3",
                    max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
                    concurrency=TRANSCRIPTION_CONCURRENCY,
                    total_audio_seconds=duration,
                    chunk_duration=duration / len(chunk_paths),
                )
            else:
                gcs_uri = upload_to_gcs(
                    final_output, GCS_BUCKET_NAME, gcs_prefix, timeout=GCS_UPLOAD_TIMEOUT_SECONDS
                )
                speaker_str = str(plan.speaker_count) if plan.speaker_count else "auto-detect"
                logger.info(
                    "Starting transcription (model: chirp_3, lang: %s, diarization: %s, speakers: %s, monolithic)...",
                    lang, "enabled" if plan.diarize else "disabled", speaker_str,
                )
                response = transcribe_single_file(
                    gcs_uri=gcs_uri,
                    lang=lang,
                    enable_diarization=plan.diarize,
                    speaker_count=plan.speaker_count,
                    project_id=GCP_PROJECT_ID,
                    region=GCP_REGION,
                    model="chirp_3",
                    max_timeout=TRANSCRIPTION_MAX_TIMEOUT_SECONDS,
                    total_audio_seconds=duration,
                    chunk_duration=duration,
                )
                responses = [response]

            logger.info("Transcription complete. Processing results...")
            chunk_durs = (
                [boundaries[0]] + [boundaries[k] - boundaries[k - 1] for k in range(1, len(boundaries))] + [duration - boundaries[-1]]
                if is_chunked and boundaries
                else [duration]
            )
            if plan.diarize:
                transcript_text = stitch_diarized_transcripts(
                    responses,
                    chunk_durations=chunk_durs,
                    total_duration=duration,
                    milestone_interval=milestone_interval,
                )
            else:
                transcript_text = stitch_plain_transcripts(
                    responses,
                    chunk_durations=chunk_durs,
                    total_duration=duration,
                    milestone_interval=milestone_interval,
                )

            save_transcript(
                transcript_text=transcript_text,
                raw_responses=responses,
                output_dir=TRANSCRIPT_OUTPUT_DIR,
                base_name=base_name,
                lang=lang,
                duration=duration,
                model="chirp_3",
                diarize=plan.diarize,
                speaker_count=plan.speaker_count,
                mode=None,
                languages=[lang],
                source_file=input_path,
            )
        else:
            raise ProcessingError(f"Unsupported transcription model: {plan.model}")

    finally:
        if uploaded_to_gcs:
            logger.info("Cleaning up temporary GCS files...")
            try:
                assert GCS_BUCKET_NAME is not None
                delete_gcs_prefix(GCS_BUCKET_NAME, gcs_prefix)
            except Exception as e:
                logger.warning("Failed to clean up GCS prefix %s: %s", gcs_prefix, e)

        if is_chunked:
            for chunk_path in chunk_paths:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            logger.info("Removed %d local chunk file(s).", len(chunk_paths))

        if os.path.exists(final_output):
            os.remove(final_output)
            logger.info("Removed local FLAC: %s", final_output)


def _process_dutch(
    input_path: str,
    base_name: str,
    lang: str,
    prepare_only: bool,
    milestone_interval: int = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
) -> None:
    """Backward-compatibility route for Dutch processing using chirp_3 without diarization."""
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")
    extract_audio_to_flac(input_path, final_output)
    duration = get_audio_duration(final_output)
    plan = TranscriptionPlan(
        model="chirp_3",
        diarize=False,
        max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_DUTCH,
        target_parallelism=TRANSCRIPTION_CONCURRENCY,
    )
    execute_transcription_plan(
        input_path=input_path,
        final_output=final_output,
        base_name=base_name,
        lang=lang,
        duration=duration,
        plan=plan,
        prepare_only=prepare_only,
        milestone_interval=milestone_interval,
    )


def _process_english(
    input_path: str,
    base_name: str,
    lang: str,
    prepare_only: bool,
    milestone_interval: int = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
) -> None:
    """Backward-compatibility route for English processing using chirp_3 with diarization."""
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")
    extract_audio_to_flac(input_path, final_output)
    duration = get_audio_duration(final_output)
    speaker_count = _prompt_speaker_count()
    plan = TranscriptionPlan(
        model="chirp_3",
        diarize=True,
        speaker_count=speaker_count,
        max_chunk_length=CHUNK_MAX_LENGTH_SECONDS_ENGLISH,
        target_parallelism=None,
    )
    execute_transcription_plan(
        input_path=input_path,
        final_output=final_output,
        base_name=base_name,
        lang=lang,
        duration=duration,
        plan=plan,
        prepare_only=prepare_only,
        milestone_interval=milestone_interval,
    )


def process_file(
    input_path: str,
    lang: str,
    prepare_only: bool = False,
    model: str = "auto",
    diarize: bool | None = None,
    mode: str = "verbatim",
    speaker_count: int | None = None,
    interactive: bool = False,
    milestone_interval: int = TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
) -> None:
    """
    Orchestrates the full audio processing and transcription pipeline for a single file.

    Args:
        input_path: Path to the source media file.
        lang: Language code ('nl-NL', 'en-US', 'auto').
        prepare_only: If True, stop after FLAC preparation.
        model: 'auto', 'gemini-3.5-transcribe', or 'chirp_3'.
        diarize: Force True, False, or None (auto-resolve).
        mode: Transcription mode for Gemini ('verbatim' or 'smart').
        speaker_count: Speaker count hint.
        interactive: Whether interactive prompts are permitted.

    Raises:
        ProcessingError: On validation failures or processing errors.
        SystemExit(2): When non-interactive decision is required.
    """
    if not os.path.exists(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")

    if not validate_extension(input_path):
        raise ProcessingError(f"Unsupported file extension for: {input_path}")

    lang_lower = lang.lower()
    if lang_lower not in [c.lower() for c in LANGUAGE_CODES_ALL] and lang_lower != "auto":
        raise ProcessingError(
            f"Unsupported language route: {lang}. "
            f"Configured languages: {', '.join(LANGUAGE_CODES_ALL)}"
        )

    # Validate config early if not prepare_only
    if not prepare_only:
        validate_backend_config(model=model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_{lang}.flac")

    logger.info("Extracting audio to FLAC: %s", final_output)
    extract_audio_to_flac(input_path, final_output)
    duration = get_audio_duration(final_output)

    try:
        plan = resolve_transcription_plan(
            duration=duration,
            lang=lang,
            requested_model=model,
            requested_diarize=diarize,
            mode=mode,
            speaker_count=speaker_count,
            interactive=interactive,
            input_file=input_path,
        )

        if plan.model == "chirp_3" and plan.diarize and plan.speaker_count is None and interactive:
            plan.speaker_count = _prompt_speaker_count()
        elif plan.model == "gemini-3.5-transcribe" and plan.diarize and plan.speaker_count and plan.speaker_count > 2:
            logger.warning(
                "Gemini 3.5 Transcribe speaker attribution for 3+ speakers is experimental (requested: %d).",
                plan.speaker_count,
            )
    except SystemExit:
        # If resolving the plan exits (e.g. exit code 2 when a trade-off decision is required),
        # clean up the extracted FLAC so temporary files are not left orphaned on disk.
        if os.path.exists(final_output):
            try:
                os.remove(final_output)
                logger.info("Cleaned up temporary FLAC on exit: %s", final_output)
            except OSError as e:
                logger.warning("Failed to remove temporary FLAC on exit %s: %s", final_output, e)
        raise

    execute_transcription_plan(
        input_path=input_path,
        final_output=final_output,
        base_name=base_name,
        lang=lang,
        duration=duration,
        plan=plan,
        prepare_only=prepare_only,
        milestone_interval=milestone_interval,
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Marketing Engineers: Transcription Automator v0.3",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to the input media file (e.g., input/meeting.mp4)",
    )
    parser.add_argument(
        "--lang",
        required=False,
        default=None,
        choices=[code.lower() for code in LANGUAGE_CODES_ALL] + ["auto"],
        type=str.lower,
        help="Language of the recording (determines processing route, supports 'auto' on gemini-3.5-transcribe)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        default=False,
        help="Enable interactive terminal prompts for language, diarization, and model decisions.",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "gemini-3.5-transcribe", "chirp_3"],
        default=TRANSCRIPTION_MODEL,
        help="Model to use for transcription ('auto', 'gemini-3.5-transcribe', 'chirp_3'). Default from TRANSCRIPTION_MODEL.",
    )
    parser.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        default=None,
        help="Force enable speaker diarization.",
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Force disable speaker diarization.",
    )
    parser.add_argument(
        "--mode",
        choices=["verbatim", "smart"],
        default="verbatim",
        help="Transcription mode for gemini-3.5-transcribe ('verbatim' or 'smart'). Diarization requires verbatim.",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        default=False,
        help="Validate credentials, configuration, and bucket permissions, then exit.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        default=False,
        help="Stop after generating FLAC files (v0.1 behavior).",
    )
    parser.add_argument(
        "--from-gcs",
        metavar="GCS_TARGET",
        default=None,
        help="Transcribe audio files already uploaded to GCS (prefix or gs://bucket/prefix).",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Optional speaker count hint for diarization.",
    )
    parser.add_argument(
        "--milestone-interval",
        type=int,
        default=TRANSCRIPTION_MILESTONE_INTERVAL_SECONDS,
        help="Milestone timestamp interval in seconds (default: 300, set to 0 to disable).",
    )

    args = parser.parse_args()

    logger.info("Starting Marketing Engineers Transcription Automator v0.3")

    if args.check_auth:
        ok = check_auth()
        sys.exit(0 if ok else 1)

    if not args.lang:
        if args.interactive:
            while True:
                try:
                    prompted_lang = input(f"Select language ({', '.join(LANGUAGE_CODES_ALL)}, auto) [default: auto]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    sys.exit(1)
                if not prompted_lang:
                    # Default to 'auto' after explicit confirmation
                    try:
                        confirm = input("No language selected. Default to automatic language detection ('auto')? [Y/n]: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        sys.exit(1)
                    if confirm in ("", "y", "yes"):
                        args.lang = "auto"
                        logger.info("Language defaulted to 'auto' (automatic detection enabled).")
                        break
                    continue
                if prompted_lang in [c.lower() for c in LANGUAGE_CODES_ALL] or prompted_lang == "auto":
                    args.lang = prompted_lang
                    break
                print(f"Invalid language '{prompted_lang}'. Supported: {', '.join(LANGUAGE_CODES_ALL)}, auto")
        else:
            parser.error("--lang is required when processing audio (or pass -i / --interactive).")

    if not args.input_file and not args.from_gcs:
        parser.error("Either input_file or --from-gcs must be specified.")

    if args.input_file and args.from_gcs:
        parser.error("Cannot specify both input_file and --from-gcs.")

    try:
        if args.from_gcs:
            speaker_count = args.speakers
            if speaker_count is None and args.lang in LANGUAGE_CODES_ENGLISH and args.interactive:
                speaker_count = _prompt_speaker_count()
            md_path, json_path = transcribe_from_gcs(
                target=args.from_gcs,
                lang=args.lang,
                speaker_count=speaker_count,
                model=args.model,
                diarize=args.diarize,
                mode=args.mode,
                interactive=args.interactive,
                milestone_interval=args.milestone_interval,
            )
            logger.info("Transcript saved to: %s", md_path)
            logger.info("Raw API response saved to: %s", json_path)
        else:
            process_file(
                input_path=args.input_file,
                lang=args.lang,
                prepare_only=args.prepare_only,
                model=args.model,
                diarize=args.diarize,
                mode=args.mode,
                speaker_count=args.speakers,
                interactive=args.interactive,
                milestone_interval=args.milestone_interval,
            )
    except ProcessingError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
