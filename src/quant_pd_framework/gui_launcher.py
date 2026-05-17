"""Convenience launcher for the Streamlit GUI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .config import PerformanceConfig
from .logging import configure_cli_logging, get_logger

MAX_UPLOAD_SIZE_MB = 51_200
LOGGER = get_logger(__name__)


def resolve_project_root() -> Path:
    """Resolves the repository root so project config is always picked up."""

    return Path(__file__).resolve().parents[2]


def resolve_gui_app_path() -> Path:
    """Resolves the repository's Streamlit app file."""

    return resolve_project_root() / "app" / "streamlit_app.py"


def build_streamlit_command() -> list[str]:
    """Builds the Streamlit launch command."""

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(resolve_gui_app_path()),
        "--server.maxUploadSize",
        str(MAX_UPLOAD_SIZE_MB),
        "--server.maxMessageSize",
        str(MAX_UPLOAD_SIZE_MB),
    ]


def main() -> int:
    """Launches the GUI using the current Python environment."""

    configure_cli_logging()
    try:
        timeout_seconds = int(
            os.environ.get(
                "QUANT_STUDIO_GUI_TIMEOUT_SECONDS",
                PerformanceConfig().gui_launch_timeout_seconds,
            )
        )
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid QUANT_STUDIO_GUI_TIMEOUT_SECONDS; using default timeout."
        )
        timeout_seconds = PerformanceConfig().gui_launch_timeout_seconds
    try:
        completed = subprocess.run(
            build_streamlit_command(),
            check=False,
            cwd=resolve_project_root(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        LOGGER.error("Streamlit launcher exceeded timeout of %s seconds.", timeout_seconds)
        return 124
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
