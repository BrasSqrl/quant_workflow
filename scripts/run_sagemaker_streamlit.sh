#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"
MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-51200}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' was not found on PATH." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m streamlit run "$PROJECT_ROOT/app/streamlit_app.py" \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.maxUploadSize "$MAX_UPLOAD_MB" \
  --server.maxMessageSize "$MAX_UPLOAD_MB"
