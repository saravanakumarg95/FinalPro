"""src/utils/logger.py — Centralised loguru-based logging setup."""

import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """
    Configure the application logger.

    Removes the default loguru handler and adds a formatted console
    handler. Call this once at application startup.

    Args:
        level: Log level string — DEBUG | INFO | WARNING | ERROR.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )


def get_logger(name: str):
    """
    Return a bound logger with the given name context.

    Usage:
        log = get_logger(__name__)
        log.info("Starting pipeline...")
    """
    return logger.bind(name=name)


# Expose logger directly for simple usage
__all__ = ["logger", "setup_logger", "get_logger"]
