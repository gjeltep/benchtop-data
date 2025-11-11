import logging
import sys
from typing import Optional

# Default logger name
DEFAULT_LOGGER_NAME = "data_pipeline"

# Log levels
DEBUG = logging.DEBUG
INFO = logging.INFO

# Third-party loggers that are too verbose at INFO level
NOISY_LOGGERS = [
    "httpx"  # HTTP client used by LlamaIndex/Ollama - logs every request
]


def _suppress_noisy_loggers():
    """
    Suppress verbose logging from third-party libraries.

    Sets noisy loggers to WARNING level to reduce log noise while keeping
    errors and warnings visible.
    """
    for logger_name in NOISY_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    logger_name: str = DEFAULT_LOGGER_NAME,
    suppress_http_logs: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the data pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        logger_name: Name of the logger
        suppress_http_logs: If True, suppresses verbose HTTP request logs from httpx/httpcore

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    logging.basicConfig(
        level=level, format=format_string, datefmt="%H:%M:%S", stream=sys.stderr, force=True
    )

    # Suppress noisy third-party loggers
    if suppress_http_logs:
        _suppress_noisy_loggers()

    logger = logging.getLogger(logger_name)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Automatically configures logging if not already configured.
    Logs go to sys.stderr by default.

    Args:
        name: Logger name (typically pass __name__ explicitly)

    Returns:
        Logger instance
    """
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)
