"""Regression checks for the Windows batch launchers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_launch_gui_script_rebuilds_invalid_venv_and_uses_python_module() -> None:
    script_text = (PROJECT_ROOT / "launch_gui.bat").read_text(encoding="utf-8")

    assert 'set "VENV_CONFIG=%VENV_DIR%\\pyvenv.cfg"' in script_text
    assert 'if not exist "%VENV_CONFIG%" goto bootstrap' in script_text
    assert 'set "PYTHONPATH=%CD%\\src;%PYTHONPATH%"' in script_text
    assert 'call "%VENV_PYTHON%" -m quant_pd_framework.gui_launcher' in script_text


def test_setup_gui_script_rebuilds_incomplete_venv() -> None:
    script_text = (PROJECT_ROOT / "setup_gui.bat").read_text(encoding="utf-8")

    assert 'set "VENV_CONFIG=%VENV_DIR%\\pyvenv.cfg"' in script_text
    assert 'set "BOOTSTRAP_TEMP=%CD%\\.setup_tmp"' in script_text
    assert 'set "BOOTSTRAP_PIP_WHEEL="' in script_text
    assert 'if not exist "%VENV_CONFIG%" goto rebuild_venv' in script_text
    assert (
        'call %BOOTSTRAP_PYTHON% -m pip --python "%VENV_PYTHON%" install '
        '"%BOOTSTRAP_PIP_WHEEL%"' in script_text
    )
    assert "quant_pd_launcher_fallback.pth" in script_text
    assert 'rmdir /s /q "%VENV_DIR%"' in script_text
