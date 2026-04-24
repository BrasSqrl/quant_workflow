#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.sagemaker_venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' was not found on PATH." >&2
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
  echo "Creating isolated SageMaker virtual environment at $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Upgrading pip/setuptools/wheel where possible..."
if ! "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel; then
  echo "Build-tool upgrade failed. Continuing with the current environment." >&2
fi

echo "Installing SageMaker runtime dependencies..."
if ! "$VENV_PYTHON" -m pip install -r requirements-sagemaker.txt; then
  cat >&2 <<'EOF'
Dependency installation failed.

In SageMaker this usually means one of the following:
- the Studio domain is running without outbound internet access to PyPI
- the environment is configured with a private package index
- a lifecycle configuration or custom image is required instead of ad hoc installs

See docs/SAGEMAKER_SETUP.md for the non-interactive lifecycle-config and custom-image paths.
EOF
  exit 1
fi

echo "Installing Quant Studio from the local source tree..."
"$VENV_PYTHON" -m pip install -e . --no-deps --no-build-isolation

cat <<'EOF'

Quant Studio is installed for the current SageMaker environment.

Start the app with:
  bash scripts/run_sagemaker_streamlit.sh
EOF
