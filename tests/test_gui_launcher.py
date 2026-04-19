"""Tests for the GUI launcher helpers."""

from __future__ import annotations

import sys
from subprocess import CompletedProcess

from quant_pd_framework import gui_launcher
from quant_pd_framework.gui_launcher import (
    MAX_UPLOAD_SIZE_MB,
    build_streamlit_command,
    resolve_gui_app_path,
    resolve_project_root,
)


def test_resolve_gui_app_path_points_to_streamlit_app() -> None:
    resolved_path = resolve_gui_app_path()

    assert resolved_path.name == "streamlit_app.py"
    assert resolved_path.exists()


def test_build_streamlit_command_uses_current_python_and_streamlit() -> None:
    command = build_streamlit_command()

    assert command[0] == sys.executable
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[4].endswith("streamlit_app.py")
    assert command[5:] == [
        "--server.maxUploadSize",
        str(MAX_UPLOAD_SIZE_MB),
        "--server.maxMessageSize",
        str(MAX_UPLOAD_SIZE_MB),
    ]


def test_main_returns_subprocess_exit_code(monkeypatch) -> None:
    def fake_run(command, check, cwd):
        assert command == build_streamlit_command()
        assert check is False
        assert cwd == resolve_project_root()
        return CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(gui_launcher.subprocess, "run", fake_run)

    assert gui_launcher.main() == 0
