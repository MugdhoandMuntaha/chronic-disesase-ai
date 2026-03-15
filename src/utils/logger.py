"""Logging utility for clinical ML pipeline."""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "chronic_disease_ml", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger with clean console formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
