#!/usr/bin/env bash

set -euo pipefail

# Starter lifecycle-configuration script for SageMaker Code Editor.
# Typical usage is to attach this script as a Code Editor lifecycle configuration
# after replacing REPO_URL or setting it as an environment variable in the LCC.

REPO_URL="${REPO_URL:-https://github.com/BrasSqrl/quant_workflow.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_PARENT="${REPO_PARENT:-/home/sagemaker-user}"
REPO_DIR="${REPO_DIR:-$REPO_PARENT/quant_workflow}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$REPO_PARENT"

if [ ! -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_PARENT" clone --branch "$REPO_BRANCH" "$REPO_URL" "$(basename "$REPO_DIR")"
else
  git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
  git -C "$REPO_DIR" checkout "$REPO_BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
fi

cd "$REPO_DIR"

PYTHON_BIN="$PYTHON_BIN" bash "$REPO_DIR/scripts/bootstrap_sagemaker.sh"

cat <<EOF

Quant Studio is prepared in:
  $REPO_DIR

To launch the GUI from a terminal:
  bash $REPO_DIR/scripts/run_sagemaker_streamlit.sh

Open the forwarded/previewed port for 8501 in SageMaker to view the app in your browser.
EOF
