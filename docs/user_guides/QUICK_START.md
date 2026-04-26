# User Quick Start Guide

This guide gets a user from launch to a completed first model run. It uses the
bundled sample data so the workflow can be learned before moving to a real
dataset.

## What You Will Produce

A successful first run writes a timestamped folder under `artifacts/` with:

- `START_HERE.md`
- `reports/interactive_report.html`
- `model/quant_model.joblib`
- `config/run_config.json`
- `metadata/metrics.json`
- `data/predictions/predictions.csv` or `data/predictions/predictions.parquet`
- `reports/model_documentation_pack.md`
- `reports/validation_pack.md`
- `artifact_manifest.json`
- `metadata/run_debug_trace.json`

## Launch The App

On Windows, use:

```powershell
.\launch_gui.bat
```

If dependencies need to be prepared first, use:

```powershell
.\setup_gui.bat
```

In SageMaker or Linux, follow [SageMaker Setup](../SAGEMAKER_SETUP.md).

## The Four-Step Workflow

Quant Studio opens to Step 1 by default.

| Step | Purpose |
| --- | --- |
| `1 Dataset & Schema` | Load data, preview rows, define column roles, edit feature dictionary, configure transformations, and exchange the governance workbook. |
| `2 Model Configuration` | Choose execution mode, preset, model family, split strategy, diagnostics, export profile, governance options, explainability, scenarios, and documentation settings. |
| `3 Readiness Check` | Review issues, preflight summary, guardrails, configuration diffs, and run the workflow. |
| `4 Results & Artifacts` | Review outputs, charts, tables, artifact locations, reviewer notes, and model-card downloads. |

## First Successful Run

1. Open Step 1, `Dataset & Schema`.
2. Leave bundled sample data selected.
3. Open `Column Designer`.
4. Confirm `default_status` is the target source.
5. Confirm `as_of_date` is the date column.
6. Disable obvious non-modeling fields such as identifiers or placeholder text.
7. Open Step 2, `Model Configuration`.
8. Use `Execution mode = fit_new_model`.
9. Use `Model type = logistic_regression`.
10. Use `Target mode = binary`.
11. Use `Data structure = cross_sectional`.
12. Keep the default train, validation, and test split for the first run.
13. Keep individual figure HTML/PNG export off unless separate chart files are needed.
14. Keep `Include enhanced report visuals` on for a polished first report, or turn it off for faster iteration runs.
15. Open Step 3, `Readiness Check`.
16. Resolve blocking readiness issues if any appear.
17. Click `Run Quant Model Workflow`.
18. Open Step 4, `Results & Artifacts`.
19. Review the overview, model performance, calibration, governance, and artifact explorer sections.

## First Real-Data Run

For real data, keep the same workflow but change Step 1:

- upload CSV, Excel, or Parquet, or place the file in `Data_Load/` and choose `Select from Data_Load`
- rebuild the column roles around the real target, date, identifier, and feature fields
- update feature dictionary definitions and documentation fields before export
- use [Data Requirements Guide](./DATA_REQUIREMENTS.md) if the data does not load or roles are unclear

## Save A Profile

After a good setup in Step 2, save a configuration profile. Profiles preserve
the setup choices without storing raw data rows. They are useful when a user
wants the same modeling setup after closing and reopening the app.

## Where To Go Next

- Use [Execution Mode Decision Guide](./EXECUTION_MODE_DECISION_GUIDE.md) if you are not sure whether to fit, score, or search feature subsets.
- Use [Artifact Map](./ARTIFACT_MAP.md) to understand output files.
- Use [Configuration Cookbook](./CONFIGURATION_COOKBOOK.md) for common PD, LGD, CECL, CCAR, existing-model, and large-data setups.
