"""
src/monitoring/logger.py
------------------------
Logging initialization and formatting for Real-Time Voice Recognition.
Provides structured formatting, console output, and optional file logging.
"""

import logging
import os
import pathlib
import sys
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | pathlib.Path] = None,
    console: bool = True,
    logger_name: Optional[str] = "rtos_voice",
) -> logging.Logger:
    """
    Initializes application logging.

    Parameters:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file: Optional file path to write log output.
        console: Whether to log to standard error / output.
        logger_name: Logger namespace name.

    Returns:
        Configured root or package logger instance.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: '{level}'")

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    logger.handlers.clear()

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_path = pathlib.Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid propagation to root logger duplicating messages if package logger is used
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger inside the application namespace.

    Parameters:
        name: Sub-module name (e.g. 'audio', 'dsp', 'rts').

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"rtos_voice.{name}")
    return logging.getLogger("rtos_voice")

