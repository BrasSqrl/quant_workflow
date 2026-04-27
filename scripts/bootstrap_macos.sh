#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This script is intended for macOS. Continuing anyway because bash is available." >&2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  cat >&2 <<EOF
Python executable '$PYTHON_BIN' was not found on PATH.

Install Python 3.11 or newer, then rerun this script. Common macOS options:
- python.org installer
- Homebrew: brew install python
- pyenv-managed Python
EOF
  exit 1
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        f"Quant Studio requires Python 3.11 or newer. Found {sys.version.split()[0]}."
    )

print(f"Using Python {sys.version.split()[0]}")
PY

if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
  echo "Creating macOS virtual environment at $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Upgrading pip/setuptools/wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "Installing Quant Studio with GUI dependencies..."
if ! "$VENV_PYTHON" -m pip install -e ".[gui]"; then
  cat >&2 <<'EOF'
Dependency installation failed.

If the error mentions compiling native packages, install Xcode Command Line Tools:
  xcode-select --install

If you are on Apple Silicon, make sure Terminal and Python use the same architecture.
See docs/MACOS_SETUP.md for detailed troubleshooting.
EOF
  exit 1
fi

echo "Verifying Streamlit import..."
"$VENV_PYTHON" - <<'PY'
import streamlit

print(f"Streamlit {streamlit.__version__} is available.")
PY

cat <<'EOF'

Quant Studio is installed for local macOS use.

Start the app with:
  bash scripts/run_macos_streamlit.sh

Then open:
  http://localhost:8501
EOF
