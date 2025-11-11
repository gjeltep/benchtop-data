"""Unit tests for logging configuration module."""

import logging
import pytest

from data_pipeline.logging import DEFAULT_LOGGER_NAME, setup_logging, get_logger


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging(self):
        """Test setup_logging with defaults, custom params, and override behavior."""
        # Test defaults
        logger = setup_logging()
        root_logger = logging.getLogger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == DEFAULT_LOGGER_NAME
        assert root_logger.level == logging.INFO

        # Test custom parameters
        logger = setup_logging(logger_name="custom_logger", level=logging.DEBUG)
        root_logger = logging.getLogger()
        assert logger.name == "custom_logger"
        assert root_logger.level == logging.DEBUG

        # Test override behavior (force=True)
        setup_logging(level=logging.DEBUG, logger_name="test1")
        logger = setup_logging(level=logging.WARNING, logger_name="test2")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
        assert logger.name == "test2"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting logger with and without explicit name."""
        # With explicit name
        logger = get_logger(name="my_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my_module"

        # Without name (inferred)
        logger = get_logger(name=None)
        assert isinstance(logger, logging.Logger)
        assert logger.name is not None
