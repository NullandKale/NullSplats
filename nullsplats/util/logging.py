"""Logging helper for NullSplats.

Configures both console and file logging without relying on environment
variables. The default log file lives under ``logs/app.log`` relative to the
project root.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    *,
    log_dir: str | Path = "logs",
    log_name: str = "app.log",
    level: int = logging.DEBUG,
    console_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure application logging for console and file outputs."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("nullsplats")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()  # ensure console output is always configured

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s [%(threadName)s:%(thread)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = log_dir_path / log_name
    # Line-buffered stream to flush quickly for debugging slow GPU operations.
    file_stream = file_path.open("a", encoding="utf-8", buffering=1)
    file_handler = logging.StreamHandler(file_stream)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger under the ``nullsplats`` namespace."""
    full_name = "nullsplats" if not name else f"nullsplats.{name}"
    logger = logging.getLogger(full_name)
    logger.setLevel(logging.DEBUG)
    return logger


__all__ = ["setup_logging", "get_logger"]
