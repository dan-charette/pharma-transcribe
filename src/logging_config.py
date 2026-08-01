"""Centralized logging configuration for PharmaTranscribe AI.

setup_logging() is idempotent so it can be called at the top of app.py,
which Streamlit re-executes on every rerun, without duplicating handlers.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "transcriber.log"

_LOGGER_NAME = "transcriber"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger.

    Safe to call repeatedly (e.g., on every Streamlit rerun): handlers are
    only attached the first time.

    Args:
        level: Logging level for the logger (default: INFO)

    Returns:
        The configured "transcriber" logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Return the app logger or a named child of it.

    Args:
        name: Optional child name (e.g., "storage" -> "transcriber.storage")

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
