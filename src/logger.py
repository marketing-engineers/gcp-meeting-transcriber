import logging
import sys


def setup_logging() -> None:
    """
    Configures logging for the entire application.

    Must be called once at the application entry point (main.py)
    before any logger is used. All modules should simply use
    logging.getLogger(__name__) to obtain their logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
