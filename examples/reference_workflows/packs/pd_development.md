# PD Development Reference

Canonical binary PD development workflow with challengers and calibration.

## What This Workflow Is For

- Use this workflow when the goal is one-period binary PD development with interpretable challengers and calibration review.
- Use it as the baseline pattern for retail or portfolio-level default modeling workflows built from tabular cross-sectional data.

## How To Run

Run the workflow from the repository root:

```powershell
python examples/reference_workflows/pd_development.py
```

The script writes a full artifact bundle under:

- `artifacts/reference_workflows/pd_development/`

## Configuration Snapshot

- Preset: `pd_development`
- Model family: `logistic_regression`
- Target mode: `binary`
- Data structure: `cross_sectional`
- Challenger comparison enabled: `True`
- Diagnostics export root: `artifacts\reference_workflows\pd_development`

## Key Review Questions

- Does the incumbent logistic model clear the held-out performance and calibration threshold for development use?
- How do challengers, scorecard views, and threshold outputs compare with the incumbent choice?
- Are the selected features, assumptions, and scenario responses defensible for the target definition?

## How To Read The Artifact Bundle

1. Start with `committee_report.pdf` for the concise story of fit, scope, and decision-ready highlights.
2. Then open `validation_report.pdf` to review guardrails, suitability, calibration, and appendix detail.
3. Use `interactive_report.html` to inspect ROC, PR, calibration, threshold, and governance views interactively.
4. Read `model_documentation_pack.md` and `validation_pack.md` for the development and validator narratives, then use `analysis_workbook.xlsx` for the detailed tables.

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
- `workflow_guardrails`
- `reproducibility_manifest`
- `variable_selection`
- `feature_policy_checks`
- `scenario_summary`
- `model_comparison`
- `woe_iv_summary`
- `vintage_summary`

### Core Figures

- `calibration_curve`
- `quantile_backtest`
- `model_comparison_chart`
- `scenario_summary_chart`
- `threshold_analysis`
- `vintage_curve`

## How To Interpret The Outputs

- Start with held-out ROC AUC, average precision, Brier score, and the calibration summary before reviewing secondary diagnostics.
- Use scorecard and challenger outputs as supporting evidence, not substitutes for the main held-out metrics and guardrail checks.
- Review scenario outputs and feature-policy findings before treating the run as ready for broader circulation.

## What To Change First When Adapting This Example

- Replace the target definition and positive-class mapping first when adapting this workflow to a real portfolio.
- Adjust feature dictionary entries, imputation policy, and feature-policy thresholds before widening the feature set.
- If the incumbent model changes, update the challenger list and calibration strategy together.

## Regression Anchors

### Test

- `roc_auc` expected near `0.6400996264` with tolerance `0.03`
- `average_precision` expected near `0.4047811983` with tolerance `0.04`
- `brier_score` expected near `0.2035245716` with tolerance `0.03`
