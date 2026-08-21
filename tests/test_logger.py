"""Tests for src/logger.py — logging setup, CleanFileFormatter, and TerminalProgressHandler."""

import logging
from io import StringIO
from unittest.mock import MagicMock

from src.logger import CleanFileFormatter, TerminalProgressHandler, setup_logging


def test_clean_file_formatter_strips_ascii_matrix():
    """CleanFileFormatter strips ASCII visual matrix from multi-line log records."""
    formatter = CleanFileFormatter("%(levelname)s - %(message)s")

    matrix = "[TRANSCRIPTION MATRIX] Batch 1/1\n┌───┐\n│01 │\n└───┘\nLegend: ..."
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Wave 1/1: 2/5 chunks completed (Total progress: 2/5 - %d%%)\n%s",
        args=(40, matrix),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert formatted == "INFO - Wave 1/1: 2/5 chunks completed (Total progress: 2/5 - 40%)"


def test_clean_file_formatter_preserves_normal_messages():
    """CleanFileFormatter preserves standard log messages without ASCII art."""
    formatter = CleanFileFormatter("%(levelname)s - %(message)s")

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Normal message with %s",
        args=("params",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert formatted == "INFO - Normal message with params"


def test_terminal_progress_handler_tty_in_place():
    """TerminalProgressHandler outputs in-place escape sequence when isatty is True."""
    stream = MagicMock()
    stream.isatty.return_value = True

    handler = TerminalProgressHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    record1 = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Upload progress: 50%",
        args=(),
        exc_info=None,
    )
    record1.in_place = True

    handler.emit(record1)
    stream.write.assert_called_with("\r\x1b[2KUpload progress: 50%")
    stream.flush.assert_called_once()

    record2 = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Uploaded successfully",
        args=(),
        exc_info=None,
    )
    handler.emit(record2)
    # Clears line and appends newline terminator
    stream.write.assert_any_call("\r\x1b[2K")
    stream.write.assert_any_call("Uploaded successfully\n")


def test_terminal_progress_handler_non_tty_standard_newlines():
    """TerminalProgressHandler outputs standard newlines when isatty is False."""
    stream = StringIO()

    handler = TerminalProgressHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Upload progress: 50%",
        args=(),
        exc_info=None,
    )
    record.in_place = True

    handler.emit(record)
    assert stream.getvalue() == "Upload progress: 50%\n"


def test_setup_logging_configures_handlers():
    """setup_logging creates root handlers for stdout and rotating file handlers."""
    setup_logging()
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 2
