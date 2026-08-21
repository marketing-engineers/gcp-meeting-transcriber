import logging
import logging.handlers
import os
import sys
from typing import Any


class CleanFileFormatter(logging.Formatter):
    """
    Formatter for app.log that strips multi-line ASCII visual matrices,
    keeping file log entries clean, formal, and single-line.
    """

    def format(self, record: logging.LogRecord) -> str:
        orig_message = record.getMessage()
        if "\n[TRANSCRIPTION MATRIX]" in orig_message or "\n┌" in orig_message:
            clean_msg = orig_message.split("\n[TRANSCRIPTION MATRIX]")[0].split("\n┌")[0].strip()
            saved_message = getattr(record, "message", None)
            record.message = clean_msg
            if self.usesTime():
                record.asctime = self.formatTime(record, self.datefmt)
            s = self.formatMessage(record)
            if record.exc_info:
                if not record.exc_text:
                    record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                if s[-1:] != "\n":
                    s = s + "\n"
                s = s + record.exc_text
            if record.stack_info:
                if s[-1:] != "\n":
                    s = s + "\n"
                s = s + self.formatStack(record.stack_info)
            if saved_message is not None:
                record.message = saved_message
            elif hasattr(record, "message"):
                delattr(record, "message")
            return s

        return super().format(record)


class TerminalProgressHandler(logging.StreamHandler):
    """
    StreamHandler for stdout that renders in-place single-line updates for transient progress
    (when extra={'in_place': True}) in interactive TTY sessions, while maintaining standard
    logging output for all other messages.
    """

    def __init__(self, stream: Any = None) -> None:
        super().__init__(stream or sys.stdout)
        self._last_was_in_place = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            is_in_place = getattr(record, "in_place", False)
            is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()

            if is_tty and is_in_place:
                self.stream.write("\r\x1b[2K" + msg)
                self.stream.flush()
                self._last_was_in_place = True
            else:
                if is_tty and self._last_was_in_place:
                    self.stream.write("\r\x1b[2K")
                self.stream.write(msg + self.terminator)
                self.flush()
                self._last_was_in_place = False
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    """
    Configures logging for the entire application.

    Creates the logs/ directory if it doesn't exist.
    Configures stdout (with in-place TTY terminal updates) and file logging (app.log and gcp_api.log).
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Standard formatter for terminal output
    stdout_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Clean single-line formatter for app.log
    file_formatter = CleanFileFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers to prevent duplicates (especially in tests)
    root_logger.handlers.clear()

    # Stdout handler (INFO) - includes full ASCII matrix updates and in-place TTY progress
    stdout_handler = TerminalProgressHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(stdout_formatter)
    root_logger.addHandler(stdout_handler)

    # App file handler (INFO) - strips ASCII matrix to clean single-line logs
    app_log_path = os.path.join(log_dir, "app.log")
    app_file_handler = logging.handlers.RotatingFileHandler(
        app_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(app_file_handler)

    # Google SDK logger setup
    google_logger = logging.getLogger("google")
    google_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers on the google logger to prevent duplicates
    google_logger.handlers.clear()

    # GCP API file handler (DEBUG)
    gcp_log_path = os.path.join(log_dir, "gcp_api.log")
    gcp_file_handler = logging.handlers.RotatingFileHandler(
        gcp_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    gcp_file_handler.setLevel(logging.DEBUG)
    gcp_file_handler.setFormatter(file_formatter)
    google_logger.addHandler(gcp_file_handler)
