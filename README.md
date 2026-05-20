# Quant Studio

Quant Studio is a Python/Streamlit application for quantitative model development,
validation, documentation, and export. The primary focus is PD/LGD and forecasting
workflows for credit-risk model development, with audit-ready artifacts that allow a
completed run to be reviewed, rerun, packaged for ongoing monitoring, or used as
evidence for model methodology documentation.

## What The App Does

- Builds new models, scores existing models, and runs feature subset search.
- Can fit governed segmented model builds, where one router artifact contains a
  global fallback model plus eligible segment-level models.
- Supports common credit-risk and forecasting model families, including logistic and
  scorecard logistic regression, linear/regularized models, Tobit/probit-style options,
  XGBoost, panel/time-series choices, and SAS-equivalent Python model paths where
  available.
- Provides a five-step workflow: `Dataset & Schema`, `Model Configuration`,
  `Readiness Check & Run`, `Results & Artifacts`, and `Decision Summary`.
- Exports organized run folders with configuration, tests, metrics, charts, manifests,
  generated Python rerun code, model artifacts, LLM documentation packages, and
  on-demand ongoing-monitoring bundles.
- Supports small-data in-memory runs and Large Data Mode with file-backed profiling,
  staging, S3/local path intake, checkpointed execution, background jobs, and large-data
  certification evidence.

## Quick Start

Install GUI dependencies manually, if you are not using the launcher scripts:

```powershell
python -m pip install -r requirements-gui.txt
python -m pip install -e . --no-deps --no-build-isolation
```

Launch on Windows. If `.venv` is missing, `launch_gui.bat` runs `setup_gui.bat`
and uses the same requirements-first install path:

```powershell
.\launch_gui.bat
```

Launch directly with Python:

```powershell
python -m streamlit run app/streamlit_app.py --server.maxUploadSize 51200
```

Launch on macOS/Linux:

```bash
bash scripts/bootstrap_macos.sh
bash scripts/run_macos_streamlit.sh
```

SageMaker setup and launch guidance is in [SageMaker setup](SAGEMAKER_SETUP.txt).

## Typical Workflow

1. Load data from bundled sample data, `Data_Load/`, upload, local path, or S3 path.
2. Review schema, roles, target, identifiers, feature dictionary, and Transformation
   Studio.
3. Configure execution mode, model family/type, optional segmented modeling, split
   strategy, diagnostics, large-data behavior, outputs, and governance settings.
4. Run readiness checks, review resource warnings, then execute the workflow.
5. Review results, artifacts, decision summary, downloads, registry entries, and audit
   events.

## Key Documentation

- Documentation index: [docs/README.md](docs/README.md)
- Detailed project guide: [docs/DETAILED_PROJECT_GUIDE.md](docs/DETAILED_PROJECT_GUIDE.md)
- Quick start: [QUICK_START.md](QUICK_START.md)
- Data requirements: [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)
- Execution modes: [EXECUTION_MODE_DECISION_GUIDE.md](EXECUTION_MODE_DECISION_GUIDE.md)
- Model selection: [MODEL_SELECTION_GUIDE.md](MODEL_SELECTION_GUIDE.md)
- Model type reference: [MODEL_TYPE_REFERENCE_GUIDE.md](MODEL_TYPE_REFERENCE_GUIDE.md)
- Artifact map: [ARTIFACT_MAP.md](ARTIFACT_MAP.md)
- Large Data Playbook: [LARGE_DATA_PLAYBOOK.md](LARGE_DATA_PLAYBOOK.md)
- Large data certification: [docs/user_guides/LARGE_DATA_CERTIFICATION_GUIDE.md](docs/user_guides/LARGE_DATA_CERTIFICATION_GUIDE.md)
- PD logistic walkthrough: [docs/user_guides/PD_LOGISTIC_REGRESSION_WALKTHROUGH.html](docs/user_guides/PD_LOGISTIC_REGRESSION_WALKTHROUGH.html)
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## Developer Commands

Install development dependencies:

```powershell
python -m pip install -e ".[dev,gui]"
```

Run quality checks:

```powershell
python -m ruff check src tests scripts
python -m mypy
python scripts/check_module_size.py
python -m pytest
```

Run the large-data certification smoke harness:

```powershell
quant-pd-certify-large-data --preset smoke --model-scope small
```

## Security Notes

Quant Studio writes SHA-256 sidecars for `.joblib` model, checkpoint, and background
snapshot artifacts. Externally supplied `.joblib` files are blocked when the sidecar is
missing or mismatched because joblib/pickle files can execute code when loaded. See
[SECURITY.md](SECURITY.md) for the trust model and reporting process.

Do not commit secrets, proprietary data, customer files, or private artifact output.
Use IAM roles, AWS profiles, or environment-level configuration for S3 access.

## Repository Layout

- `app/`: Streamlit entrypoint.
- `src/quant_pd_framework/`: application package and workflow engine.
- `src/quant_pd_framework/streamlit_ui/`: Streamlit UI modules.
- `src/quant_pd_framework/steps/`: workflow pipeline steps.
- `src/quant_pd_framework/exporting/`: export and artifact helpers.
- `src/quant_pd_framework/large_data_support/`: large-data handles, profiles, staging,
  and certification helpers.
- `docs/`: detailed user, technical, and governance documentation.
- `scripts/`: setup, benchmark, and maintenance scripts.
- `tests/`: regression and contract tests.

## License

This project is licensed for noncommercial use only.

Commercial use requires explicit written permission. See the [LICENSE](LICENSE) file for
details.
