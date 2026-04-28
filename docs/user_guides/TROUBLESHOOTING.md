# Troubleshooting Guide

This guide lists common Quant Studio problems, likely causes, and practical
fixes.

## Launch Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Double-click launcher appears to do nothing | Virtual environment or dependency setup failed before the browser opened | Run `.\setup_gui.bat`, then run `.\launch_gui.bat` from PowerShell so errors stay visible. |
| Streamlit command not found | GUI dependencies are not installed in the active environment | Run `.\setup_gui.bat` or install with `python -m pip install -e .[gui]`. |
| `pyproject.toml` not found | Command was run outside repo root | `cd` to the folder that contains `pyproject.toml`. |
| Git says dubious ownership | Repo was created or modified by another Windows identity | Run the safe-directory command shown by Git from your own terminal. |
| macOS script says permission denied | Shell script is not executable or was downloaded with restrictive flags | Run with `bash scripts/bootstrap_macos.sh`, or use `chmod +x scripts/*.sh`. |
| macOS cannot find Python | Python 3.11+ is not on PATH | Install Python, then run `PYTHON_BIN=/path/to/python3 bash scripts/bootstrap_macos.sh`. |

## Data Load Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Uploaded file does not appear | Unsupported format or browser upload friction | Use CSV, Excel, or Parquet. For large files, place the file in `Data_Load/`. |
| `Data_Load/` file is missing from dropdown | File extension is unsupported or app needs refresh | Confirm suffix is CSV, XLSX, XLS, Parquet, or PQ, then refresh the app. |
| App becomes slow while loading | File is too large for eager browser upload | Use `Data_Load/`, Parquet, and Large Data Mode. |
| Arrow serialization warning | Mixed Python object types in a displayed dataframe column | Usually display-related. Convert mixed object columns to strings if it affects UI display. |

## Readiness Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No target source selected | No active column has role `target_source` | Open Column Designer and mark the raw outcome column. |
| Target has only one class | Positive target values are wrong or filtered data contains one class | Check target mapping and source data. |
| Date required | Time-series, panel, or time diagnostics need a date role | Mark a valid date column in Column Designer. |
| Feature count is zero | Features are disabled or all columns are non-feature roles | Enable appropriate feature columns. |
| Feature subset cap warning | Candidate combinations exceed configured cap | Lower maximum subset size, reduce candidate features, or increase cap only if runtime is acceptable. |

## Model Fit Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Model convergence warning | Iterations too low, scaling issue, collinearity, or separation | Increase max iterations, simplify features, use regularization, or review outliers. |
| Perfect or near-perfect performance | Leakage or target-like feature included | Review enabled features and disable post-event or ID fields. |
| Existing model scoring fails | New data does not match saved model feature contract | Provide prior `config/run_config.json` and ensure raw features exist. |
| XGBoost unavailable | Optional package may not be installed | Install GUI/dev dependencies or choose another model family. |

## Output And Report Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Cannot find outputs | Artifact root changed or run did not finish | Check Step 4 output locations, Step 5 evidence links, and `artifacts/` timestamped folders. |
| `reports/interactive_report.html` opens blank | Browser security or incomplete downloaded file | Open in Chrome or Edge, confirm file size, and try serving from a local HTTP server if needed. |
| Charts show loading message | Browser blocks local dynamic chart rendering | Use Chrome or serve the file from a local HTTP server. Static chart fallbacks are no longer embedded because they slow report generation. |
| Separate figure files are missing | Individual figure export is off by default | Turn on `Export individual figure HTML and PNG files` before running. |
| Advanced Visual Analytics charts are missing | The optional advanced layer is off by default | Turn on `Advanced Visual Analytics` in Step 2, then rerun the workflow. |
| Missing `model_bundle_for_monitoring/` | Mode was not `fit_new_model` or export did not complete | This bundle is created for new fitted models only. |

## SageMaker Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `localhost:8501` does not work | Port is not forwarded from SageMaker to local machine | Use local VS Code Remote and forward port `8501`. |
| Browser proxy returns 404 | Wrong SageMaker proxy path or app not running | Prefer VS Code port forwarding. Otherwise follow [SageMaker Setup](../SAGEMAKER_SETUP.md). |
| Pip dependency resolver warning about JupyterLab | Base SageMaker environment has its own Jupyter packages | Current bootstrap uses `.sagemaker_venv`; warning should not block app if install completes. |
| Cannot install from PyPI | Network or private package restrictions | Use approved package source, lifecycle config, or custom SageMaker image. |

## macOS Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Native package build fails | Xcode Command Line Tools are missing | Run `xcode-select --install`, then rerun `bash scripts/bootstrap_macos.sh`. |
| SSL certificate errors | Python.org certificates were not installed | Run `Install Certificates.command` from the Python installer folder. |
| Scripts are blocked after downloading zip | macOS quarantine attribute is present | From repo root, run `xattr -dr com.apple.quarantine .`. |
| Port 8501 is already in use | Another local process is using Streamlit's default port | Run `PORT=8502 bash scripts/run_macos_streamlit.sh`. |
| Apple Silicon install behaves inconsistently | Mixed x86/Rosetta and arm64 Python environments | Use a native arm64 terminal and Python, for example `/opt/homebrew/bin/python3`. |

## Large Data Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Machine freezes or swaps heavily | Pandas load exceeds available RAM | Stop run if possible, use Parquet, Large Data Mode, governed sample, and larger compute. |
| Full CSV export is too large | Export policy writes full tables for a non-Parquet input | Use sampled or metadata-only policy, or use a Parquet Step 1 input when Parquet artifacts are required. |
| Run takes too long | Expensive diagnostics, advanced visuals, figure export, or subset search | Use fast export profile, keep Advanced Visual Analytics and individual figure files off, reduce diagnostic scope, or use governed sample. |

## What To Capture When Asking For Help

Provide:

- screenshot or exact error
- execution mode
- model type
- target mode
- whether Large Data Mode is on
- data shape and file type
- latest timestamped artifact folder
- `run_debug_trace.json` if exported
- terminal output around the failure
