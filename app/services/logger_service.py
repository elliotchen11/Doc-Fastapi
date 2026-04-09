import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_fmt, datefmt=_datefmt)


def _make_rotating(filename: str, level: int) -> TimedRotatingFileHandler:
    handler = TimedRotatingFileHandler(
        LOG_DIR / filename,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(_formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call once per module: logger = get_logger(__name__)"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_formatter)

    # fastapi_info.log — DEBUG and above (all requests + general logs)
    info_handler = _make_rotating("fastapi_info.log", logging.DEBUG)

    # fastapi_errors.log — WARNING and above (exceptions and errors only)
    error_handler = _make_rotating("fastapi_errors.log", logging.WARNING)

    logger.addHandler(console)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.propagate = False

    return logger
