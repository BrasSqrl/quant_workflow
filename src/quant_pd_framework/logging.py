"""Package logging helpers for CLI and background execution."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

PACKAGE_LOGGER_NAME = "quant_pd_framework"


def get_logger(name: str | None = None) -> logging.Logger:
    """Returns a package-scoped logger."""

    if not name:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(PACKAGE_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_cli_logging(
    *,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> None:
    """Configures concise CLI logging once without disrupting host applications."""

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

