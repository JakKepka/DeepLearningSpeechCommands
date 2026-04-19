"""Project-wide logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    name: str = "speech_commands",
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir) / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "speech_commands") -> logging.Logger:
    return logging.getLogger(name)
