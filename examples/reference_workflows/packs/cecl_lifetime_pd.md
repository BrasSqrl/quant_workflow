# Lifetime PD / CECL Reference

Canonical person-period lifetime PD workflow for CECL-style development.

## What This Workflow Is For

- Use this workflow when the development question is about default timing or lifetime PD rather than one-period default incidence.
- Use it when a CECL-style person-period structure is available and macro sensitivity across time matters.

## How To Run

Run the workflow from the repository root:

```powershell
python examples/reference_workflows/cecl_lifetime_pd.py
```

The script writes a full artifact bundle under:

- `artifacts/reference_workflows/cecl_lifetime_pd/`

## Configuration Snapshot

- Preset: `lifetime_pd_cecl`
- Model family: `discrete_time_hazard_model`
- Target mode: `binary`
- Data structure: `panel`
- Challenger comparison enabled: `True`
- Diagnostics export root: `artifacts\reference_workflows\cecl_lifetime_pd`

## Key Review Questions

- Do the lifetime PD curves remain directionally sensible over the modeled horizon?
- Do calibration, backtesting, and macro-sensitivity outputs remain coherent on held-out periods?
- Are roll-rate, migration, and cohort outputs consistent with the intended lifetime-PD story?

## How To Read The Artifact Bundle

1. Start with `validation_report.pdf` to review the use case, guardrails, calibration, and lifetime-PD diagnostics in one place.
2. Open `interactive_report.html` next to inspect lifetime curves, time diagnostics, migration, and macro-sensitivity visuals.
3. Read `model_documentation_pack.md` for the development narrative and `validation_pack.md` for assumptions and exclusions.
4. Use `analysis_workbook.xlsx`, `tables/`, and `statistical_tests.json` for the supporting detail behind the published views.

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
- `calibration_summary`
- `feature_dictionary`
- `lifetime_pd_curve`
- `workflow_guardrails`
- `reproducibility_manifest`
- `variable_selection`
- `feature_policy_checks`
- `scenario_summary`
- `woe_iv_summary`
- `vintage_summary`
- `cohort_pd_summary`
- `migration_matrix`
- `roll_rate_summary`
- `macro_sensitivity`

### Core Figures

- `lifetime_pd_curve`
- `quantile_backtest`
- `calibration_curve`
- `threshold_analysis`
- `time_backtest_test`
- `vintage_curve`
- `cohort_pd_curve`
- `migration_heatmap`
- `macro_sensitivity_chart`

## How To Interpret The Outputs

- Prioritize the lifetime-PD curve, calibration summary, and time-backtest outputs before reading secondary diagnostics.
- Use migration, roll-rate, and cohort views together; none of them should be interpreted in isolation.
- Check macro-sensitivity outputs to confirm that stress relationships are directionally consistent with the use case.

## What To Change First When Adapting This Example

- Replace the person-period target and entity/date columns with the real CECL structure first.
- Review hazard-model challengers and feature-policy thresholds before widening the feature set.
- Tighten documentation around horizon definition, default timing rules, and macro-driver assumptions.

## Regression Anchors

### Test

- `roc_auc` expected near `0.4444444444` with tolerance `0.06`
- `brier_score` expected near `0.221514553` with tolerance `0.04`
- `ks_statistic` expected near `0.2380952381` with tolerance `0.07`
