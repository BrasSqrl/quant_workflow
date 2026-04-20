# LGD Severity Reference

Canonical bounded continuous LGD severity workflow.

## What This Workflow Is For

- Use this workflow when the development target is bounded loss severity rather than binary default incidence.
- Use it as the baseline LGD pattern when residual diagnostics, segment views, and recovery segmentation matter.

## How To Run

Run the workflow from the repository root:

```powershell
python examples/reference_workflows/lgd_severity.py
```

The script writes a full artifact bundle under:

- `artifacts/reference_workflows/lgd_severity/`

## Configuration Snapshot

- Preset: `lgd_severity`
- Model family: `two_stage_lgd_model`
- Target mode: `continuous`
- Data structure: `cross_sectional`
- Challenger comparison enabled: `True`
- Diagnostics export root: `artifacts\reference_workflows\lgd_severity`

## Key Review Questions

- Does the incumbent LGD model outperform challengers on held-out RMSE, MAE, and R-squared?
- Do residual, QQ, and segment views support the chosen model form?
- Are recovery and LGD segment views directionally consistent with the intended severity story?

## How To Read The Artifact Bundle

1. Start with `validation_report.pdf` for the model form, guardrails, and residual-focused validation view.
2. Then inspect `interactive_report.html` for residual plots, QQ behavior, segment views, and scenario outputs.
3. Use `model_documentation_pack.md` for the development narrative and `analysis_workbook.xlsx` for the exact tables.
4. Open `committee_report.pdf` when you need the concise committee-facing framing after the validation readout.

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
- `residual_summary`
- `qq_plot_data`
- `variable_selection`
- `feature_policy_checks`
- `scenario_summary`
- `lgd_stage_two_coefficients`
- `vintage_summary`
- `lgd_segment_summary`
- `recovery_segmentation`

### Core Figures

- `quantile_backtest`
- `residuals_vs_predicted`
- `qq_plot`
- `scenario_summary_chart`
- `model_comparison_chart`
- `vintage_curve`
- `lgd_segment_chart`
- `recovery_segment_chart`

## How To Interpret The Outputs

- Read RMSE, MAE, and R-squared together with residual and QQ diagnostics; fit metrics alone are not enough for severity models.
- Use LGD segment and recovery views to confirm the model is not hiding unstable segment behavior.
- Scenario outputs should be directionally sensible for the stressed features before relying on them in narrative material.

## What To Change First When Adapting This Example

- Replace the bounded severity target and segment definitions before changing model family choices.
- Review beta, two-stage LGD, and quantile challengers together when the residual pattern changes materially.
- Document loss definition, exclusions, and recovery assumptions early because they shape the whole workflow.

## Regression Anchors

### Test

- `rmse` expected near `0.0569009983` with tolerance `0.01`
- `mae` expected near `0.0455352697` with tolerance `0.01`
- `r2` expected near `0.7854174063` with tolerance `0.05`
