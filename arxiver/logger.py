import logging
import os


def setup_logging():
    """Set up logging configuration based on environment variable."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate log level
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        log_level = "INFO"

    logging.basicConfig(level=getattr(logging, log_level))
