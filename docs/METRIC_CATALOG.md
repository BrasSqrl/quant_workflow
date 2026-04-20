# Metric Catalog

This document describes the major metrics and analytical outputs produced by the
framework. It complements:

- [STATISTICAL_TEST_CATALOG.md](./STATISTICAL_TEST_CATALOG.md)
- [MODEL_CATALOG.md](./MODEL_CATALOG.md)

Primary implementation files:

- `src/quant_pd_framework/steps/evaluation.py`
- `src/quant_pd_framework/steps/backtesting.py`
- `src/quant_pd_framework/steps/diagnostics.py`

## Metric Families

The framework reports metrics in five broad families:

1. split-level model performance metrics
2. calibration metrics
3. backtest and rank-order metrics
4. regression residual metrics
5. explainability and governance diagnostics
6. robustness and scorecard-development diagnostics

## 1. Split-Level Binary Metrics

These are calculated in `EvaluationStep._score_binary_split(...)`.

| Metric | Meaning | Better direction | Notes |
| --- | --- | --- | --- |
| `roc_auc` | Ranking/discrimination quality across thresholds | higher | Independent of chosen threshold |
| `average_precision` | Precision-recall summary under class imbalance | higher | Often more informative than ROC AUC when defaults are rare |
| `brier_score` | Mean squared probability error | lower | Sensitive to calibration and discrimination |
| `log_loss` | Negative log-likelihood style penalty | lower | Strongly penalizes confident wrong predictions |
| `ks_statistic` | Max separation between positive and negative score CDFs | higher | Widely used in PD review |
| `accuracy` | Share of correct thresholded classifications | higher | Threshold-dependent |
| `precision` | Positive predictive value | higher | Threshold-dependent |
| `recall` | True positive rate | higher | Threshold-dependent |
| `f1_score` | Harmonic mean of precision and recall | higher | Threshold-dependent |
| `matthews_correlation` | Balanced classification correlation statistic | higher | Useful with class imbalance |
| `true_negative`, `false_positive`, `false_negative`, `true_positive` | Confusion-matrix counts | context-specific | Threshold-dependent |

## 2. Split-Level Continuous Metrics

These are calculated in `EvaluationStep._score_continuous_split(...)`.

| Metric | Meaning | Better direction | Notes |
| --- | --- | --- | --- |
| `rmse` | Root mean squared error | lower | Penalizes large errors more heavily |
| `mae` | Mean absolute error | lower | More robust to extreme residuals |
| `r2` | Proportion of variance explained | higher | Can be negative out of sample |
| `explained_variance` | Variance explained without full `R^2` framing | higher | Similar to but not identical to `r2` |
| `mean_actual` | Average realized target | context-specific | Descriptive |
| `mean_predicted` | Average prediction | context-specific | Descriptive |

## 3. Calibration Metrics

These are calculated inside `DiagnosticsStep._add_calibration_outputs(...)`.

| Metric | Meaning | Better direction | Output |
| --- | --- | --- | --- |
| `brier_score` | Mean squared probability error | lower | `calibration_summary` |
| `log_loss` | Logarithmic probability loss | lower | `calibration_summary` |
| `expected_calibration_error` | Weighted average bin-level absolute gap | lower | `calibration_summary` |
| `maximum_calibration_error` | Worst bin-level absolute gap | lower | `calibration_summary` |
| `calibration_intercept` | Average probability bias | closer to 0 | `calibration_summary` |
| `calibration_slope` | Probability dispersion quality | closer to 1 | `calibration_summary` |
| `hosmer_lemeshow_statistic` | Bin-based fit statistic | lower | `calibration_summary` |
| `hosmer_lemeshow_p_value` | Tail probability for HL statistic | contextual | `calibration_summary` |

The framework evaluates these across:

- base probabilities
- Platt scaling
- isotonic calibration

It then records:

- `recommended_calibration_method`
- `recommended_calibration_score_column`
- `calibration_ranking_metric`

## 4. Threshold Metrics

Threshold analysis is produced in `DiagnosticsStep._add_threshold_outputs(...)`.

It computes the following across thresholds from `0.05` to `0.95`:

- precision
- recall
- accuracy
- `f1_score`

This output is useful when:

- the default threshold is not fixed yet
- risk appetite needs to be visualized
- the user wants a threshold documentation exhibit

Primary output:

- table `threshold_analysis`
- figure `threshold_analysis`

## 5. Rank-Order and Backtest Metrics

### Quantile backtest

Implemented in `BacktestStep.run(...)` and visualized in
`DiagnosticsStep._add_quantile_outputs(...)` or
`DiagnosticsStep._add_regression_quantile_outputs(...)`.

Binary outputs include:

- `observation_count`
- `default_count`
- `average_predicted_pd`
- `observed_default_rate`
- `min_predicted_pd`
- `max_predicted_pd`

Continuous outputs include:

- `observation_count`
- `average_predicted_value`
- `observed_average`
- `min_predicted_value`
- `max_predicted_value`

### Lift and gain

Implemented in `DiagnosticsStep._add_lift_gain_outputs(...)`.

Key outputs:

- `capture_rate`
- `cumulative_capture_rate`
- `lift`

These are appropriate for binary ranking review and early-bucket capture
analysis.

## 6. Curve Outputs

The framework exports both tables and figures for common diagnostic curves.

| Curve | Purpose | Main output names |
| --- | --- | --- |
| ROC | Classification ranking curve | `roc_curve`, `roc_curve` figure |
| Precision-recall | Imbalanced-class tradeoff | `precision_recall_curve`, `precision_recall_curve` figure |
| Calibration curve | Observed vs predicted by bin | `calibration`, `calibration_curve` figure |
| Gain curve | Cumulative capture of defaults | `gain_chart` |
| Lift chart | Relative lift by ranked bucket | `lift_chart` |
| Residuals vs predicted | Regression error shape | `residuals_vs_predicted` |
| Actual vs predicted | Regression fit sanity check | `actual_vs_predicted` |
| QQ plot | Residual distribution review | `qq_plot_data`, `qq_plot` |

## 7. Explainability Metrics

These live in `DiagnosticsStep._add_explainability_outputs(...)`.

### Feature importance

Returned by each model adapter via `get_feature_importance()`.

Common columns:

- `feature_name`
- `importance_value`
- `importance_type`
- `coefficient`
- `abs_coefficient`
- `std_error`
- `p_value`
- `odds_ratio`

### Permutation importance

Produced by `_build_permutation_importance(...)`.

Binary mode:

- baseline metric is `roc_auc`
- importance is `baseline_metric - permuted_metric`

Continuous mode:

- baseline metric is `rmse`
- importance is `permuted_metric - baseline_metric`

### Feature effect curves

Produced by `_build_feature_effect_curves(...)`.

These are partial-dependence-style average prediction curves over:

- numeric quantile grids
- most common categorical levels

## 8. Governance and Policy Metrics

Feature policy checks live in `DiagnosticsStep._add_feature_policy_outputs(...)`.

The exported table `feature_policy_checks` records:

- `policy_name`
- `feature_name`
- `status`
- `observed_value`
- `threshold`

Potential policy types include:

- `required_feature`
- `excluded_feature`
- `max_missing_pct`
- `max_vif`
- `minimum_information_value`
- `expected_sign`
- `monotonicity`

## 9. Segment and Scenario Metrics

### Segment metrics

Segment outputs live in `DiagnosticsStep._add_segment_outputs(...)`.

Typical fields:

- `observation_count`
- `average_score`
- `average_actual`

### Scenario metrics

Scenario outputs live in `DiagnosticsStep._add_scenario_outputs(...)`.

Typical fields:

- `mean_baseline_score`
- `mean_scenario_score`
- `mean_delta`
- `baseline_positive_rate`
- `scenario_positive_rate`

These are not classical model metrics, but they are important development
artifacts for stress-style documentation.

## 10. Where Metrics Are Exported

The same metric family often appears in more than one place:

| Artifact | Content |
| --- | --- |
| `metrics.json` | split-level metric dictionary |
| `analysis_workbook.xlsx` | diagnostic tables |
| `interactive_report.html` | grouped visual presentation |
| `tables/*.csv` | individual diagnostic tables |
| `run_report.md` | run narrative summary |
| `model_documentation_pack.md` | development documentation summary |

## 11. Robustness And Stability Metrics

These are calculated in `DiagnosticsStep._add_robustness_outputs(...)`.

Primary exported tables:

- `robustness_metric_distribution`
- `robustness_metric_summary`
- `robustness_feature_distribution`
- `robustness_feature_stability`

Key fields include:

- `metric_name`
- `metric_value`
- `mean_value`
- `std_value`
- `p05_value`
- `median_value`
- `p95_value`
- `selection_frequency`
- `mean_effect`
- `std_effect`
- `mean_abs_effect`
- `sign_consistency`

These are useful when the user needs to know whether:

- held-out performance is stable across repeated train resamples
- important features retain similar influence across resamples
- coefficient direction remains consistent for interpretable models

Primary figures:

- `robustness_metric_boxplot`
- `robustness_metric_summary_chart`
- `robustness_feature_stability`

## 12. Scorecard Workbench Outputs

These are calculated in `DiagnosticsStep._add_scorecard_workbench_outputs(...)`
for `scorecard_logistic_regression` runs.

Primary exported tables:

- `scorecard_feature_summary`
- `scorecard_woe_table`
- `scorecard_points_table`
- `scorecard_scaling_summary`
- `scorecard_reason_code_frequency`

Primary exported figures:

- `scorecard_feature_iv`
- `scorecard_score_distribution`
- `scorecard_reason_code_frequency_chart`
- per-feature scorecard bucket charts such as `scorecard_bad_rate_*`,
  `scorecard_woe_*`, and `scorecard_points_*`

These outputs are intended for scorecard-development review rather than generic
model ranking. They help the user inspect:

- information value by feature
- monotonicity and bad-rate shape by bucket
- partial points by bucket
- total score distribution
- which features most often drive the exported reason codes

## Audit Notes

- Split metrics are computed before the broader diagnostics step.
- Diagnostics add interpretation surfaces and richer tables, but they do not
  replace the base split metrics.
- A metric being present in the UI depends on both the model type and whether
  labels are available.
