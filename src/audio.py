import logging
import os
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


def split_flac(input_path: str, output_dir: str, base_name: str, chunk_length: int) -> list[str]:
    """
    Splits a FLAC file into chunks of a specified length using FFmpeg's segment muxer (lossless copy).

    Args:
        input_path: The path to the input FLAC file.
        output_dir: The directory where chunks will be saved.
        base_name: The base name to use for naming output chunk files.
        chunk_length: The length of each chunk in seconds.

    Returns:
        List of paths to generated chunk files.

    Raises:
        ProcessingError: If FFmpeg is not found or exits with a non-zero status.
    """
    _check_ffmpeg_available()
    output_pattern = os.path.join(output_dir, f"{base_name}_chunk_%03d.flac")

    logger.info("Splitting FLAC into %ds chunks: %s -> %s", chunk_length, input_path, output_pattern)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
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
