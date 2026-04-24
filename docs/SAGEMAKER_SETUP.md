# SageMaker Setup

This guide explains how to run Quant Studio on SageMaker compute while using
your local VS Code desktop IDE through VS Code Remote access.

## Recommended Path: Local VS Code Remote

This is the recommended SageMaker workflow for Quant Studio:

1. Launch the SageMaker space and connect to it from your local VS Code desktop IDE.
2. Open a VS Code terminal that is connected to the SageMaker machine.
3. Clone or pull the repo on the SageMaker filesystem.
4. Bootstrap the app once.
5. Start Streamlit from the remote VS Code terminal.
6. Forward remote port `8501` from VS Code's Ports panel.
7. Open the forwarded local URL, usually `http://localhost:8501`.

## Commands

From the VS Code terminal connected to SageMaker:

```bash
cd /home/sagemaker-user/quant_workflow
bash scripts/bootstrap_sagemaker.sh
bash scripts/run_sagemaker_streamlit.sh
```

Leave the Streamlit terminal running. In local VS Code:

1. Open the `Ports` or `Forwarded Ports` panel.
2. Add or confirm forwarding for remote port `8501`.
3. Open the forwarded URL in your local browser.

The local browser URL is usually:

```text
http://localhost:8501
```

## Why Localhost Works With VS Code Remote

Streamlit is running on the SageMaker machine. VS Code Remote forwards the
remote SageMaker port back to your laptop. After port `8501` is forwarded,
`localhost:8501` in your local browser is routed through VS Code to the remote
SageMaker Streamlit process.

Without VS Code port forwarding, `localhost:8501` means your laptop only and
will not reach SageMaker.

## Do Not Use `BASE_URL_PATH` For VS Code Remote

When using local VS Code Remote plus forwarded ports, use the plain launcher:

```bash
bash scripts/run_sagemaker_streamlit.sh
```

Do not use this JupyterLab proxy form:

```bash
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

`BASE_URL_PATH` is only needed when you are opening Streamlit through a
SageMaker browser proxy URL instead of VS Code port forwarding.

## What The Scripts Do

- `bootstrap_sagemaker.sh`
  - verifies Python `3.11+`
  - creates a repo-local `.sagemaker_venv` virtual environment
  - upgrades `pip`, `setuptools`, and `wheel` when possible
  - installs the runtime dependencies listed in `requirements-sagemaker.txt`
    into `.sagemaker_venv`
  - installs the local project in editable mode with `--no-build-isolation`
    inside `.sagemaker_venv`
  - checks Plotly static chart export support and attempts to install Chrome
    for Kaleido when the SageMaker image does not already provide it

- `run_sagemaker_streamlit.sh`
  - uses `.sagemaker_venv/bin/python` automatically when the venv exists
  - sets `PYTHONPATH` to the local `src/` tree
  - runs Streamlit on `0.0.0.0:8501`
  - accepts `BASE_URL_PATH` only for browser proxy fallback scenarios
  - keeps the same large-upload settings used in the Windows launcher

## Why The Virtual Environment Matters

SageMaker JupyterLab and Code Editor images include their own Jupyter and
Notebook packages. Installing application packages directly into that base
environment can expose resolver conflicts in SageMaker's IDE stack, for
example:

```text
notebook 7.4.4 requires jupyterlab<4.5,>=4.4.4, but you have jupyterlab 4.2.5
```

Quant Studio runs as a Streamlit app and does not need to modify SageMaker's
Jupyter runtime, so the project-local `.sagemaker_venv` keeps those two
environments separate.

## Downloaded HTML Reports And Chart Rendering

The exported `interactive_report.html` includes dynamic Plotly charts and
static SVG fallback charts. On Linux, Kaleido v1 requires Chrome for static
image export. Many SageMaker images do not include Chrome by default, so
`bootstrap_sagemaker.sh` checks static export support and attempts to install
Chrome automatically through Plotly when needed.

If downloaded reports show chart fallback messages instead of charts, rerun:

```bash
bash scripts/bootstrap_sagemaker.sh
```

Then rerun the Quant Studio workflow so a new `interactive_report.html` is
generated.

If your SageMaker environment blocks Chrome downloads and you want to skip the
attempt explicitly, use:

```bash
INSTALL_PLOTLY_CHROME=0 bash scripts/bootstrap_sagemaker.sh
```

## Environment Variables For The Run Script

You can override the defaults:

```bash
HOST=0.0.0.0 PORT=8502 MAX_UPLOAD_MB=1024 bash scripts/run_sagemaker_streamlit.sh
```

Default values:

- `HOST=0.0.0.0`
- `PORT=8501`
- `MAX_UPLOAD_MB=51200`
- `BASE_URL_PATH=`
- `PYTHON_BIN=python3`

## Fallback: SageMaker JupyterLab Browser Proxy

Use this only if you are not using local VS Code Remote port forwarding and are
instead opening the app directly from a SageMaker JupyterLab browser session.

Start Streamlit with the JupyterLab absolute proxy base path:

```bash
cd /home/sagemaker-user/quant_workflow
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

Then open this URL pattern in your browser, using the same host and region from
your current JupyterLab tab:

```text
https://<your-space>.studio.<region>.sagemaker.aws/jupyterlab/default/proxy/absolute/8501/
```

## Fallback: SageMaker Notebook Instance Proxy

For a SageMaker Notebook Instance, use the shorter notebook proxy path:

```bash
cd /home/sagemaker-user/quant_workflow
BASE_URL_PATH=/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

Then open:

```text
https://<notebook-name>.notebook.<region>.sagemaker.aws/proxy/absolute/8501/
```

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
bash scripts/run_sagemaker_streamlit.sh
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

### Streamlit starts but the browser cannot open it

Confirm:

1. The Streamlit terminal is still running.
2. Health check returns `ok`:

```bash
curl http://127.0.0.1:8501/_stcore/health
```

3. VS Code is forwarding remote port `8501`.
4. You are opening the forwarded local URL, usually `http://localhost:8501`.
5. You did not set `BASE_URL_PATH` for the VS Code Remote path.

### The app works locally but not in SageMaker

Check:

- Python version
- outbound internet / PyPI access
- Streamlit port access
- whether VS Code is connected to the SageMaker remote machine
- whether port `8501` is forwarded in local VS Code
- whether your SageMaker domain is `VpcOnly`
