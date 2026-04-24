# SageMaker Setup

This guide explains how to run Quant Studio from a browser-based Amazon
SageMaker environment such as Code Editor or JupyterLab, using SageMaker's
compute rather than a local Windows machine.

## Goal

The intended SageMaker flow is:

1. Clone this repository into a SageMaker space.
2. Install the application into the SageMaker runtime.
3. Launch the Streamlit app on the SageMaker instance.
4. Open the app through SageMaker's browser-access path for local ports.

## Supported SageMaker Paths

Quant Studio can be run in three SageMaker-friendly ways:

1. Direct install in a running Code Editor or JupyterLab terminal.
2. Automatic bootstrap through a SageMaker Code Editor lifecycle configuration.
3. Prebaked custom image with the dependencies already installed.

The first option is simplest. The second and third are more reliable when the
domain does not allow ad hoc internet-based package installation.

## Files Added For SageMaker Support

- `requirements-sagemaker.txt`
- `scripts/bootstrap_sagemaker.sh`
- `scripts/run_sagemaker_streamlit.sh`
- `scripts/sagemaker_code_editor_lifecycle.sh`
- `SAGEMAKER_SETUP.txt`

## Prerequisites

Quant Studio expects:

- Python `3.11+`
- a cloned copy of the repo
- either:
  - outbound access to PyPI, or
  - a lifecycle configuration/custom image that preinstalls dependencies

If your Studio domain runs with restricted outbound internet access, direct
`pip install` commands may fail. In that case, use the lifecycle-configuration
or custom-image path.

## Recommended Interactive Setup In SageMaker Code Editor

From a terminal inside the cloned repo:

```bash
cd /home/sagemaker-user/quant_workflow
bash scripts/bootstrap_sagemaker.sh
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

What these do:

- `bootstrap_sagemaker.sh`
  - verifies Python `3.11+`
  - creates a repo-local `.sagemaker_venv` virtual environment
  - upgrades `pip`, `setuptools`, and `wheel` when possible
  - installs the runtime dependencies listed in `requirements-sagemaker.txt`
    into `.sagemaker_venv`
  - installs the local project in editable mode with `--no-build-isolation`
    inside `.sagemaker_venv`

- `run_sagemaker_streamlit.sh`
  - uses `.sagemaker_venv/bin/python` automatically when the venv exists
  - sets `PYTHONPATH` to the local `src/` tree
  - runs Streamlit on `0.0.0.0:8501`
  - accepts `BASE_URL_PATH` for SageMaker proxy paths
  - keeps the same large-upload settings used in the Windows launcher

The virtual environment is intentional. SageMaker JupyterLab and Code Editor
images include their own Jupyter and Notebook packages. Installing application
packages directly into that base environment can expose resolver conflicts in
SageMaker's IDE stack, for example:

```text
notebook 7.4.4 requires jupyterlab<4.5,>=4.4.4, but you have jupyterlab 4.2.5
```

Quant Studio runs as a Streamlit app and does not need to modify SageMaker's
Jupyter runtime, so the project-local `.sagemaker_venv` keeps those two
environments separate.

## Opening The App In SageMaker

The Streamlit server runs on port `8501`.

In a SageMaker browser IDE, you typically need one of:

- a forwarded-port panel
- an "Open Preview" or "Open in Browser" action
- a JupyterLab proxy URL

Bind the app to `0.0.0.0`, not `127.0.0.1`, so the SageMaker browser access
layer can reach it. The run script already does this.

### SageMaker JupyterLab Proxy Command

For SageMaker Studio JupyterLab, use the absolute proxy base path:

```bash
cd /home/sagemaker-user/quant_workflow
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

Then open this URL pattern in your browser, using the same host and region from
your current JupyterLab tab:

```text
https://<your-space>.studio.<region>.sagemaker.aws/jupyterlab/default/proxy/absolute/8501/
```

### SageMaker Notebook Instance Proxy Command

For a SageMaker Notebook Instance, use the shorter notebook proxy path:

```bash
cd /home/sagemaker-user/quant_workflow
BASE_URL_PATH=/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

Then open:

```text
https://<notebook-name>.notebook.<region>.sagemaker.aws/proxy/absolute/8501/
```

## Environment Variables For The Run Script

You can override the defaults:

```bash
HOST=0.0.0.0 PORT=8502 MAX_UPLOAD_MB=1024 bash scripts/run_sagemaker_streamlit.sh
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

Default values:

- `HOST=0.0.0.0`
- `PORT=8501`
- `MAX_UPLOAD_MB=51200`
- `BASE_URL_PATH=`
- `PYTHON_BIN=python3`

## Lifecycle Configuration Starter

If you want SageMaker Code Editor to prepare the repo automatically on startup,
use:

```bash
scripts/sagemaker_code_editor_lifecycle.sh
```

That starter script:

1. clones the repo into `/home/sagemaker-user/quant_workflow` if it does not exist
2. otherwise pulls the configured branch
3. runs `scripts/bootstrap_sagemaker.sh`

Useful variables for the lifecycle script:

```bash
REPO_URL=https://github.com/BrasSqrl/quant_workflow.git
REPO_BRANCH=main
REPO_PARENT=/home/sagemaker-user
REPO_DIR=/home/sagemaker-user/quant_workflow
PYTHON_BIN=python3
```

## Custom Image Path

If your Studio domain cannot install from PyPI at runtime, the best long-term
solution is a custom Code Editor or JupyterLab image that already contains:

- Python `3.11+`
- the packages in `requirements-sagemaker.txt`
- optionally the project itself

In that setup, your interactive run path becomes:

```bash
cd /home/sagemaker-user/quant_workflow
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

## Troubleshooting

### `could not find a version that satisfies the requirement setuptools>=68`

This usually means the environment cannot fetch packages from PyPI or is using
an incompatible package source. Use:

- a lifecycle configuration with approved package access
- or a custom image with dependencies preinstalled

### `file ... does not appear to be a python project`

You are not in the repo root. Run:

```bash
pwd
ls
```

Then `cd` into the folder containing `pyproject.toml`.

### `notebook ... requires jupyterlab ... but you have jupyterlab ...`

This means pip is detecting a dependency conflict in SageMaker's preinstalled
Jupyter environment. With the current scripts, Quant Studio installs into the
repo-local `.sagemaker_venv`, so this warning should not block the app.

If you previously ran an older bootstrap script, rebuild the isolated venv:

```bash
rm -rf .sagemaker_venv
bash scripts/bootstrap_sagemaker.sh
```

### Streamlit starts but the app is not visible in the browser

This is usually a port-access issue, not an app issue. Confirm:

1. Streamlit is bound to `0.0.0.0`
2. it is running on port `8501`
3. SageMaker exposes that port through preview or forwarding
4. JupyterLab runs use `BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501`
5. Notebook Instance runs use `BASE_URL_PATH=/proxy/absolute/8501`

### The app works locally but not in SageMaker

Check:

- Python version
- outbound internet / PyPI access
- Streamlit port access
- whether the SageMaker space is Code Editor or JupyterLab
- whether your domain is `VpcOnly`

## Recommended SageMaker Commands

If the repo has already been cloned:

```bash
cd /home/sagemaker-user/quant_workflow
bash scripts/bootstrap_sagemaker.sh
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

If dependencies are already available in the image:

```bash
cd /home/sagemaker-user/quant_workflow
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```
