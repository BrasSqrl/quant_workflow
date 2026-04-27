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


def test_macos_bootstrap_script_creates_local_venv_and_installs_gui_extra() -> None:
    script_text = (PROJECT_ROOT / "scripts" / "bootstrap_macos.sh").read_text(encoding="utf-8")

    assert 'VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"' in script_text
    assert '"$PYTHON_BIN" -m venv "$VENV_DIR"' in script_text
    assert '"$VENV_PYTHON" -m pip install -e ".[gui]"' in script_text
    assert "Streamlit" in script_text


def test_macos_run_script_launches_streamlit_with_upload_limits() -> None:
    script_text = (PROJECT_ROOT / "scripts" / "run_macos_streamlit.sh").read_text(encoding="utf-8")

    assert 'PYTHON_BIN="$VENV_DIR/bin/python"' in script_text
    assert 'HOST="${HOST:-localhost}"' in script_text
    assert 'MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-51200}"' in script_text
    assert '-m streamlit run "$PROJECT_ROOT/app/streamlit_app.py"' in script_text
