"""Convenience launcher for the Streamlit GUI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_UPLOAD_SIZE_MB = 51_200


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

    completed = subprocess.run(
        build_streamlit_command(),
        check=False,
        cwd=resolve_project_root(),
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
