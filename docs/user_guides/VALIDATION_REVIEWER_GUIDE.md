# Validation Reviewer Guide

This guide helps validation and risk reviewers inspect a completed Quant Studio
run without needing to understand every implementation detail first.

## Review Starting Point

Start with these files:

- `reports/interactive_report.html`
- `reports/validation_pack.md`
- `reports/model_documentation_pack.md`
- `config/run_config.json`
- `artifact_manifest.json`
- `metadata/metrics.json`
- `metadata/statistical_tests.json`
- `metadata/reproducibility_manifest.json`

Use Step 4, `Results & Artifacts`, in the GUI when reviewing immediately after a
run. Use the exported run folder when reviewing offline.

## Recommended Review Sequence

1. Confirm execution mode.
2. Confirm dataset, target, and scope.
3. Review schema, feature dictionary, and excluded fields.
4. Review missing-value treatment and transformations.
5. Review train/validation/test split design.
6. Review model suitability and assumption checks.
7. Review performance metrics and backtests.
8. Review calibration and threshold behavior for binary models.
9. Review stability, drift, and segmentation outputs.
10. Review explainability, feature effects, and scenario tests.
11. Review warnings, run debug trace, and diagnostic registry.
12. Record reviewer notes and exceptions.

## Execution Mode Review

| Mode | Review focus |
| --- | --- |
| `fit_new_model` | Whether the fitted model is appropriate and sufficiently documented. |
| `score_existing_model` | Whether the saved model was reused correctly and whether new data is compatible. |
| `search_feature_subsets` | Whether feature-selection evidence supports a later final model run. |

Feature-subset-search output should not be treated as the final model package.

## Data And Schema Review

Review:

- target source and positive target mapping
- enabled feature list
- disabled identifiers and leakage fields
- date/entity columns for time-series or panel workflows
- feature dictionary definitions
- missing-value treatment
- governed transformations
- manual bins or interaction terms

Important evidence:

- `config/run_config.json`
- `config/configuration_template.xlsx`
- diagnostic tables for schema, imputation, and transformations
- `reports/model_documentation_pack.md`

## Model Performance Review

For binary models, review:

- ROC AUC
- KS
- average precision
- confusion-matrix metrics
- threshold analysis
- lift/gain
- calibration curve
- Brier score

For continuous models, review:

- RMSE
- MAE
- R-squared where applicable
- residual diagnostics
- actual-versus-predicted views
- segment-level error

Use [Metric Catalog](../METRIC_CATALOG.md) for detailed definitions.

## Suitability Check Review

In Step 4, open the Results overview and review the `Suitability Checks` panel.
Failed checks are shown first with the observed value, configured threshold,
plain-English interpretation, why the issue matters, and a recommended action.

Offline, review `tables/diagnostics/assumption_checks.csv` for the same fields.
A `Fail` status means the run completed but the condition should be resolved or
explicitly accepted before the model evidence is relied on. A `Watch` status
means the condition is not a hard failure but still needs reviewer attention.

## Statistical Test Review

Focus on tests that match the model and data:

- distribution-shift tests for train/test or segment instability
- residual tests for continuous models
- outlier and influence diagnostics
- dependency and collinearity diagnostics
- stationarity and structural-break diagnostics for time-aware workflows
- paired comparison tests for challenger review
- cross-validation metric and feature stability when enabled

Use [Statistical Test Catalog](../STATISTICAL_TEST_CATALOG.md) for code and
interpretation detail.

## Explainability Review

Review:

- coefficients or feature importance
- permutation importance
- PDP, ICE, centered ICE, and ALE for top features
- monotonicity diagnostics
- segmented effects
- scenario testing results
- interaction strength and two-way effect plots when enabled

For XGBoost or other less transparent challengers, explainability evidence
should be stronger than for a simple logistic baseline.

## Red Flags

Escalate when:

- target mapping is unclear
- identifiers or leakage fields are enabled as features
- training, validation, and test results diverge sharply
- calibration is poor and not addressed
- feature effects contradict business expectations without explanation
- model convergence warnings are ignored
- cross-validation or robustness results are unstable
- large-data sampled evidence is presented without clear sample documentation
- score-only existing-model runs are described as observed performance reviews

## Reviewer Workspace

The GUI includes reviewer status, notes, exceptions, and model-card download
surfaces in Step 4. These are development-stage governance aids. They do not
replace formal validation policy, but they keep review context near the model
evidence.
