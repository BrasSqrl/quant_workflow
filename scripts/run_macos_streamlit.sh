#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if [ -z "${PYTHON_BIN:-}" ] && [ -x "$VENV_DIR/bin/python" ]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

HOST="${HOST:-localhost}"
PORT="${PORT:-8501}"
MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-51200}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  cat >&2 <<EOF
Python executable '$PYTHON_BIN' was not found.

Run the setup script first:
  bash scripts/bootstrap_macos.sh
EOF
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cat <<EOF
Starting Quant Studio on macOS...
Open http://localhost:$PORT if the browser does not open automatically.
EOF

exec "$PYTHON_BIN" -m streamlit run "$PROJECT_ROOT/app/streamlit_app.py" \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.maxUploadSize "$MAX_UPLOAD_MB" \
  --server.maxMessageSize "$MAX_UPLOAD_MB"
