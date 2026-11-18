"""
tests/test_logging.py
---------------------
Tests for logging setup, handler configuration, and structured formatting.
"""

import logging
import pathlib
import pytest
from src.monitoring.logger import setup_logging, get_logger


def test_setup_logging_defaults():
    """Verifies default logging setup with INFO level."""
    logger = setup_logging(level="INFO", console=True)
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_setup_logging_file_output(tmp_path: pathlib.Path):
    """Verifies that log messages are correctly written to file."""
    log_file = tmp_path / "test_run.log"
    logger = setup_logging(level="DEBUG", log_file=log_file, console=False, logger_name="test_logger_file")

    test_message = "Test log message for verification"
    logger.info(test_message)

    # Flush handlers to disk
    for h in logger.handlers:
        h.flush()

    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert test_message in content
    assert "INFO" in content


def test_setup_logging_invalid_level():
    """Verifies that an invalid log level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(level="NOT_A_LEVEL")


def test_get_logger_subnames():
    """Verifies that get_logger properly prefixes module namespaces."""
    audio_logger = get_logger("audio")
    assert audio_logger.name == "rtos_voice.audio"

    dsp_logger = get_logger("dsp")
    assert dsp_logger.name == "rtos_voice.dsp"

    root_app_logger = get_logger()
    assert root_app_logger.name == "rtos_voice"

