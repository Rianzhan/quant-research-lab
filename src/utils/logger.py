"""Minimal standard-library logging; no log files are written by config checks."""

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure this application's stderr logger exactly once per process."""
    logger = logging.getLogger("quant_lab")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
