"""Logging setup. See the "Logging" section of AGENTS.md for conventions."""

import os
import sys
from typing import Literal

from loguru import logger


def configure_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None) -> None:
    """Configure loguru once, at an entry point.

    Args:
        level: The log level to use. If not provided, the level
            will be read from the ``LOG_LEVEL`` environment
            variable, defaulting to ``INFO``.
    """
    logger.remove()
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    logger.add(sys.stderr, level=level)
