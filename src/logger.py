import logging
import logging.handlers
import os
import sys


def setup_logging() -> None:
    """
    Configures logging for the entire application.

    Creates the logs/ directory if it doesn't exist.
    Configures stdout and file logging (app.log and gcp_api.log).
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Common formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers to prevent duplicates (especially in tests)
    root_logger.handlers.clear()

    # Stdout handler (INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # App file handler (INFO)
    app_log_path = os.path.join(log_dir, "app.log")
    app_file_handler = logging.handlers.RotatingFileHandler(
        app_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(formatter)
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
    gcp_file_handler.setFormatter(formatter)
    google_logger.addHandler(gcp_file_handler)
