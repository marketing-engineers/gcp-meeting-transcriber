import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def extract_audio_to_flac(input_path: str, output_path: str) -> None:
    """
    Extracts audio from a media file and transcodes it losslessly to FLAC using FFmpeg.
    Drops any video streams.

    Args:
        input_path: The path to the input media file.
        output_path: The path where the FLAC file will be saved.

    Raises:
        subprocess.CalledProcessError: If FFmpeg exits with a non-zero status.
    """
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
        logger.error("FFmpeg failed during extraction.\n%s", e.stderr)
        raise


def split_flac(input_path: str, output_dir: str, base_name: str, chunk_length: int) -> None:
    """
    Splits a FLAC file into chunks of a specified length using FFmpeg's segment muxer (lossless copy).

    Args:
        input_path: The path to the input FLAC file.
        output_dir: The directory where chunks will be saved.
        base_name: The base name to use for naming output chunk files.
        chunk_length: The length of each chunk in seconds.

    Raises:
        subprocess.CalledProcessError: If FFmpeg exits with a non-zero status.
    """
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
        logger.info("Successfully chunked FLAC into: %s", output_dir)
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg failed during chunking.\n%s", e.stderr)
        raise
