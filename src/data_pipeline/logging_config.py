"""
Logging configuration for the data pipeline.
"""

import logging
import sys
from typing import Optional

# Default logger name
DEFAULT_LOGGER_NAME = "data_pipeline"

# Log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
# TODO: Add other log levels

def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    logger_name: str = DEFAULT_LOGGER_NAME
) -> logging.Logger:
    """
    Set up logging configuration for the data pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        logger_name: Name of the logger

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%H:%M:%S',
        stream=sys.stderr,
        force=True  # Override any existing configuration
    )

    logger = logging.getLogger(logger_name)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a module.

    Automatically configures logging if not already configured.
    Logs go to sys.stderr by default.

    Args:
        name: Logger name (defaults to module name if None)

    Returns:
        Logger instance
    """
    # Ensure logging is configured (only if not already configured)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt='%H:%M:%S',
            stream=sys.stderr,
            force=False  # Don't override existing config
        )

    if name is None:
        # Use caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame is not None:
            caller_frame = frame.f_back
            if caller_frame is not None:
                module = caller_frame.f_globals.get('__name__', DEFAULT_LOGGER_NAME)
                name = module
        else:
            name = DEFAULT_LOGGER_NAME

    return logging.getLogger(name)

