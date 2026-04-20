# CCAR Forecasting Reference

Canonical macro-linked CCAR forecasting workflow with panel regression and challengers.

## What This Workflow Is For

- Use this workflow when the task is macro-linked panel forecasting for CCAR development rather than binary default classification.
- Use it when segment-level quarterly history and macro drivers need to be documented, stress tested, and compared across challengers.

## How To Run

Run the workflow from the repository root:

```powershell
python examples/reference_workflows/ccar_forecasting.py
```

The script writes a full artifact bundle under:

- `artifacts/reference_workflows/ccar_forecasting/`

## Configuration Snapshot

- Preset: `ccar_forecasting`
- Model family: `panel_regression`
- Target mode: `continuous`
- Data structure: `panel`
- Challenger comparison enabled: `True`
- Diagnostics export root: `artifacts\reference_workflows\ccar_forecasting`

## Key Review Questions

- Does the incumbent forecasting model outperform its challengers on held-out panel periods?
- Do time-series diagnostics and macro-sensitivity outputs support the selected specification?
- Are the scenario outputs directionally sensible for a stress-testing workflow?

## How To Read The Artifact Bundle

1. Start with `committee_report.pdf` for the high-level forecasting story and the core held-out metrics.
2. Then read `validation_report.pdf` for assumption checks, numerical diagnostics, and the detailed appendix trail.
3. Open `interactive_report.html` for macro-sensitivity, quantile backtesting, and forecasting diagnostics.
4. Use `analysis_workbook.xlsx`, `metrics.json`, and `tables/` when you need exact values behind the report narrative.

## Key Artifact Deliverables

- `model`: Serialized fitted model artifact used for score-existing-model workflows and reruns.
- `metrics`: JSON metric snapshot suitable for regression checks and exact numeric review.
- `input_snapshot`: CSV snapshot of the dataset used for the reference run.
- `predictions`: Per-row scored output exported from the completed run.
- `feature_importance`: Model-specific coefficient or importance summary for the retained features.
- `backtest`: Risk-band or quantile backtest summary exported from the workflow.
- `report`: Narrative run report that summarizes metrics, diagnostics, and exported files.
- `documentation_pack`: Development-oriented markdown narrative with purpose, feature scope, and performance summary.
- `validation_pack`: Validator-oriented markdown package with assumptions, exclusions, and review trail.
- `committee_report_docx`: Editable committee-ready DOCX for internal markup or meeting preparation.
- `validation_report_docx`: Editable validation-ready DOCX for detailed review and commentary.
- `committee_report_pdf`: Committee-facing PDF with the concise narrative, key metrics, and appendix map.
- `validation_report_pdf`: Validation-facing PDF with guardrails, suitability, diagnostics, and appendix detail.
- `interactive_report`: Interactive HTML report for chart-first review of diagnostics and exports.
- `config`: Resolved run configuration saved for reproducibility and reruns.
- `tests`: Structured statistical-test output for audit and supporting detail.
- `reproducibility_manifest`: Run fingerprint with hashes, package versions, and environment metadata.
- `configuration_template`: Offline governance workbook for schema, dictionary, transforms, and review tables.
- `manifest`: Artifact manifest that indexes the run bundle.
- `step_manifest`: Exact ordered pipeline stack used by the workflow.
- `runner_script`: Standalone Python entrypoint for replaying the run outside the GUI.
- `rerun_readme`: Instructions for replaying the saved run bundle outside the GUI.
- `workbook`: Excel workbook containing the major exported tables.
- `reference_example_pack.md`: Workflow-specific walkthrough and reading guide.

## Expected Tables And Figures

### Core Tables

- `assumption_checks`
- `feature_dictionary`
- `workflow_guardrails`
- `reproducibility_manifest`
- `variable_selection`
- `feature_policy_checks`
- `scenario_summary`
- `model_comparison`
- `forecasting_statistical_tests`
- `macro_sensitivity`
- `vintage_summary`
- `migration_matrix`
- `roll_rate_summary`

### Core Figures

- `quantile_backtest`
- `model_comparison_chart`
- `scenario_summary_chart`
- `time_backtest_test`
- `macro_sensitivity_chart`
- `vintage_curve`
- `migration_heatmap`

## How To Interpret The Outputs

- Read performance diagnostics together with time-series statistical tests; strong fit without stable residual behavior is not enough.
- Use macro-sensitivity and scenario outputs to confirm the forecast moves in the expected stress direction.
- Check challenger rankings and quantile views before settling on the incumbent forecasting specification.

## What To Change First When Adapting This Example

- Swap in the real forecast target, entity grain, and macro series before revisiting model-family choices.
- Tune lag, rolling, and percent-change transformations to reflect the real macro specification.
- Document forecast horizon, macro assumptions, and stress-scenario interpretation in more detail for CCAR review.

## Regression Anchors

### Test

- `rmse` expected near `0.0088860264` with tolerance `0.003`
- `mae` expected near `0.0068951804` with tolerance `0.003`
- `r2` expected near `0.7784440405` with tolerance `0.08`
