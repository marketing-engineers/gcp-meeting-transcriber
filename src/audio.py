import logging
import math
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Raised when audio processing fails for a known, handled reason."""


def _check_ffmpeg_available() -> None:
    """Verify ffmpeg is available on PATH. Raises ProcessingError if not found."""
    if shutil.which("ffmpeg") is None:
        raise ProcessingError(
            "ffmpeg not found on PATH. Install it via your package manager "
            "(e.g., 'sudo apt install ffmpeg' or 'brew install ffmpeg')."
        )


def _check_ffprobe_available() -> None:
    """Verify ffprobe is available on PATH. Raises ProcessingError if not found."""
    if shutil.which("ffprobe") is None:
        raise ProcessingError(
            "ffprobe not found on PATH. Install it via your package manager "
            "(e.g., 'sudo apt install ffmpeg' or 'brew install ffmpeg')."
        )


def extract_audio_to_flac(input_path: str, output_path: str) -> None:
    """
    Extracts audio from a media file and transcodes it losslessly to FLAC using FFmpeg.
    Drops any video streams.

    Args:
        input_path: The path to the input media file.
        output_path: The path where the FLAC file will be saved.

    Raises:
        ProcessingError: If FFmpeg is not found or exits with a non-zero status.
    """
    _check_ffmpeg_available()
    logger.info("Extracting audio to FLAC: %s -> %s", input_path, output_path)

    command = [
        "ffmpeg",
        "-y",               # Overwrite output files without asking
        "-i", input_path,   # Input file
        "-vn",              # Disable video
        "-c:a", "flac",     # Audio codec: flac
        output_path,        # Output file
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Successfully generated FLAC: %s", output_path)
    except subprocess.CalledProcessError as e:
        raise ProcessingError(
            f"FFmpeg failed during audio extraction.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr: {e.stderr}"
        ) from e


def detect_silences(
    file_path: str,
    noise_threshold: int = -30,
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Detects silent intervals in an audio file using FFmpeg's silencedetect filter.

    Args:
        file_path: Path to the audio file (e.g. extracted FLAC).
        noise_threshold: Noise threshold in dB (default -30).
        min_duration: Minimum duration of silence in seconds (default 0.5).

    Returns:
        List of (silence_start, silence_end) timestamps in seconds.
        Returns empty list on parse failure or if no silences are found.
    """
    _check_ffmpeg_available()
    logger.info(
        "Detecting silences in %s (noise=%ddB, min_dur=%.1fs)...",
        file_path, noise_threshold, min_duration,
    )

    command = [
        "ffmpeg",
        "-i", file_path,
        "-af", f"silencedetect=noise={noise_threshold}dB:d={min_duration}",
        "-f", "null",
        "-",
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as e:
        logger.warning("silencedetect execution failed for %s: %s. Degrading gracefully.", file_path, e)
        return []

    re_start = re.compile(r"silence_start:\s*([0-9.]+)")
    re_end = re.compile(r"silence_end:\s*([0-9.]+)")

    silences: list[tuple[float, float]] = []
    current_start: float | None = None

    try:
        for line in result.stderr.splitlines():
            start_match = re_start.search(line)
            if start_match:
                current_start = float(start_match.group(1))
                continue
            end_match = re_end.search(line)
            if end_match:
                end_val = float(end_match.group(1))
                start_val = current_start if current_start is not None else 0.0
                silences.append((start_val, end_val))
                current_start = None
    except Exception as e:
        logger.warning("Failed to parse silencedetect output for %s: %s. Degrading gracefully.", file_path, e)
        return []

    logger.info("Detected %d silence interval(s) in %s", len(silences), file_path)
    return silences


def compute_chunk_boundaries(
    duration: float,
    max_chunk_length: float,
    silences: list[tuple[float, float]] | None = None,
    window: float = 15.0,
    min_chunk_length: float = 600.0,
    target_parallelism: int | None = None,
) -> list[float]:
    """
    Computes optimal chunk boundary timestamps (in seconds) respecting API ceilings,
    target concurrency, minimum chunk length floors, and snapping to natural pauses.

    Args:
        duration: Total audio duration in seconds.
        max_chunk_length: Maximum allowed chunk duration (ceiling).
        silences: List of detected (start, end) silence intervals.
        window: Maximum +/- deviation from nominal boundary to search for a pause.
        min_chunk_length: Minimum chunk duration floor (default 600s).
        target_parallelism: Target concurrency (Dutch), or None (English / ceiling-driven).

    Returns:
        List of strictly increasing cut timestamps (e.g. [1489.6, 2979.2, ...]).
        Returns empty list [] if N == 1 (monolithic).
    """
    if silences is None:
        silences = []

    # 1. Determine chunk count N
    # Subtract window in n_ceiling to reserve headroom so a forward snap never breaches max_chunk_length
    n_ceiling = math.ceil(duration / (max_chunk_length - window))
    if target_parallelism is not None:
        n_parallel = min(target_parallelism, math.floor(duration / min_chunk_length))
        N = max(1, n_ceiling, n_parallel)
    else:
        N = max(1, n_ceiling)

    if N == 1:
        return []

    nominal = duration / N
    boundaries: list[float] = []
    prev = 0.0

    for i in range(1, N):
        target = i * nominal
        qualifying = []
        for s_start, s_end in silences:
            m = (s_start + s_end) / 2.0
            if abs(m - target) <= window:
                qualifying.append((abs(m - target), m))

        if qualifying:
            qualifying.sort(key=lambda x: x[0])
            candidate = qualifying[0][1]
        else:
            candidate = target

        # Clamping
        # Lower bound:
        # 1. Respect min_chunk_length from prev if possible
        min_from_prev = prev + min_chunk_length
        # 2. Ceiling on remaining audio: duration - candidate <= (N - i) * max_chunk_length
        min_for_remaining_ceiling = duration - (N - i) * max_chunk_length
        lower_bound = max(prev + 1e-3, min_for_remaining_ceiling)
        if lower_bound <= min_from_prev:
            lower_bound = min_from_prev

        # Upper bound:
        # 1. At most max_chunk_length from prev
        max_from_prev = prev + max_chunk_length
        # 2. Leave room for remaining chunks
        max_for_remaining = duration - (N - i) * min_chunk_length
        upper_bound = max_from_prev
        if max_for_remaining >= lower_bound:
            upper_bound = min(upper_bound, max_for_remaining)

        candidate = max(candidate, lower_bound)
        candidate = min(candidate, upper_bound)

        boundaries.append(round(candidate, 3))
        prev = candidate

    assert all(b1 < b2 for b1, b2 in zip([0.0] + boundaries, boundaries + [duration])), (
        "Boundaries must be strictly increasing"
    )
    for b1, b2 in zip([0.0] + boundaries, boundaries + [duration]):
        chunk_len = b2 - b1
        assert chunk_len <= max_chunk_length + 1e-6, (
            f"Chunk length {chunk_len} exceeds max {max_chunk_length}"
        )

    return boundaries


def split_flac(input_path: str, output_dir: str, base_name: str, segment_times: list[float]) -> list[str]:
    """
    Splits a FLAC file at explicit boundary timestamps using FFmpeg's segment muxer (lossless copy).
    Cleans up any pre-existing chunk files for the base_name before splitting.

    Args:
        input_path: The path to the input FLAC file.
        output_dir: The directory where chunks will be saved.
        base_name: The base name to use for naming output chunk files.
        segment_times: List of cut timestamps in seconds (e.g. [600.0, 1200.0]).

    Returns:
        List of paths to generated chunk files, sorted alphanumerically.

    Raises:
        ProcessingError: If FFmpeg is not found or exits with a non-zero status.
    """
    _check_ffmpeg_available()

    if not segment_times:
        return []

    # Clean up any pre-existing chunk files for this base_name
    os.makedirs(output_dir, exist_ok=True)
    stale_files = [
        f for f in os.listdir(output_dir)
        if f.startswith(f"{base_name}_chunk_") and f.endswith(".flac")
    ]
    if stale_files:
        logger.info(
            "Removing %d pre-existing stale chunk file(s) for %s in %s...",
            len(stale_files), base_name, output_dir,
        )
        for f in stale_files:
            try:
                os.remove(os.path.join(output_dir, f))
            except OSError as e:
                logger.warning("Failed to remove stale chunk file %s: %s", f, e)

    output_pattern = os.path.join(output_dir, f"{base_name}_chunk_%03d.flac")
    times_str = ",".join(str(t) for t in segment_times)

    logger.info(
        "Splitting FLAC at %d boundary point(s) (%s): %s -> %s",
        len(segment_times), times_str, input_path, output_pattern,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-f", "segment",
        "-segment_times", times_str,
        "-c", "copy",       # Copy stream directly (lossless)
        output_pattern,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise ProcessingError(
            f"FFmpeg failed during FLAC chunking.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr: {e.stderr}"
        ) from e

    # Discover generated chunk files
    chunk_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith(f"{base_name}_chunk_") and f.endswith(".flac")
    )
    chunk_paths = [os.path.join(output_dir, f) for f in chunk_files]
    logger.info("Successfully chunked FLAC into %d files in: %s", len(chunk_paths), output_dir)
    return chunk_paths


def get_audio_duration(file_path: str) -> float:
    """
    Retrieves the duration of a media file in seconds using ffprobe.

    Args:
        file_path: The path to the media file.

    Returns:
        The duration in seconds as a float.

    Raises:
        ProcessingError: If ffprobe fails or the duration cannot be parsed.
    """
    _check_ffprobe_available()
    logger.info("Retrieving duration for: %s", file_path)

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration_str = result.stdout.strip()
        if not duration_str:
            raise ProcessingError(f"ffprobe returned empty output for: {file_path}")
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        raise ProcessingError(
            f"ffprobe failed to retrieve audio duration.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr: {e.stderr}"
        ) from e
    except ValueError as e:
        raise ProcessingError(
            f"Failed to parse audio duration from ffprobe output: '{duration_str}'.\n"
            f"Original error: {e}"
        ) from e
