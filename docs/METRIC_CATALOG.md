# Metric Catalog

This document describes the major metrics and analytical outputs produced by the
framework. It complements:

- [STATISTICAL_TEST_CATALOG.md](./STATISTICAL_TEST_CATALOG.md)
- [MODEL_CATALOG.md](./MODEL_CATALOG.md)

Primary implementation files:

- `src/quant_pd_framework/steps/evaluation.py`
- `src/quant_pd_framework/steps/backtesting.py`
- `src/quant_pd_framework/steps/diagnostics.py`
- `src/quant_pd_framework/diagnostics/assets.py`
- `src/quant_pd_framework/diagnostics/registry.py`
- `src/quant_pd_framework/diagnostic_frameworks.py`

## Metric Families

Each run also exports a `diagnostic_registry` table. It lists major diagnostic
families, their controlling config paths, expected output tables and figures,
target-mode or label restrictions, large-data behavior, and whether each item
was emitted, disabled, or skipped.

The framework reports metrics in nine broad families:

1. split-level model performance metrics
2. calibration metrics
3. backtest and rank-order metrics
4. regression residual metrics
5. explainability and governance diagnostics
6. robustness and scorecard-development diagnostics
7. credit-risk development diagnostics
8. distribution, dependency, and outlier diagnostics
9. time-series, econometric, and structural-break diagnostics

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
| Annotated ROC | ROC curve with random-model reference and AUC annotation | `roc_curve_annotated` |
| Precision-recall | Imbalanced-class tradeoff | `precision_recall_curve`, `precision_recall_curve` figure |
| Annotated precision-recall | Precision-recall curve with average-precision annotation | `precision_recall_curve_annotated` |
| KS separation | Event and non-event cumulative distribution gap | `ks_curve_annotated` |
| Calibration curve | Observed vs predicted by bin | `calibration`, `calibration_curve` figure |
| Calibration residual bars | Observed-minus-predicted calibration residual by bin | `calibration_residual_bars` |
| Gain curve | Cumulative capture of defaults | `gain_chart` |
| Lift chart | Relative lift by ranked bucket | `lift_chart` |
| Score distribution violin | Score distribution shape by development split | `score_distribution_violin` |
| Split metric slope chart | Train, validation, and test metric movement | `split_metric_slope_chart` |
| Segment dumbbell | Observed-versus-predicted segment gaps | `segment_performance_dumbbell` |
| Feature-importance waterfall | Signed coefficient or importance ranking | `feature_importance_waterfall` |
| PSI threshold bars | Stability review bands for PSI values | `psi_threshold_bars` |
| VIF threshold bars | Multicollinearity review bands for VIF values | `vif_threshold_bars` |
| Scenario tornado | Ranked average score impact by scenario | `scenario_tornado` |
| Cross-validation violin | Fold-metric distribution shape | `cross_validation_metric_violin` |
| Feature-effect small multiples | Split-level feature-effect stability | `feature_effect_stability_small_multiples` |
| Residuals vs predicted | Regression error shape | `residuals_vs_predicted` |
| Actual vs predicted | Regression fit sanity check | `actual_vs_predicted` |
| QQ plot | Residual distribution review | `qq_plot_data`, `qq_plot` |

The annotated and companion chart outputs are presentation-layer additions:
they reuse existing run tables, metrics, and predictions and do not change the
model object, fitted coefficients, statistical test calculations, or exported
source tables. They are controlled by `ArtifactConfig.include_enhanced_report_visuals`
and the GUI toggle `Include enhanced report visuals`.

`Advanced Visual Analytics` is a separate optional presentation layer controlled
by `ArtifactConfig.include_advanced_visual_analytics`. It adds exploratory chart
families such as contribution beeswarms, interaction heatmaps, PDP/ICE matrices,
segment calibration small multiples, score ridgelines, temporal score streams,
correlation networks, lift/gain heatmaps, risk treemaps, model-comparison radar
charts, scenario waterfalls, and feature-importance lollipop charts. These
views reuse existing diagnostics, predictions, and metrics; they do not create
new fitted models or change statistical test outputs.

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

### Partial dependence

Produced by `_build_feature_effect_curves(...)` and exported as
`partial_dependence`.

Partial dependence shows the average predicted response when one feature is
varied across a grid and all other features stay at their observed values.

### ICE and centered ICE

Produced by `_build_ice_curves(...)`.

`ice_curves` shows individual conditional expectation curves for sampled rows.
`centered_ice_curves` subtracts each row's baseline response so heterogeneous
response shapes are easier to compare.

### Accumulated local effects

Produced by `_build_accumulated_local_effects(...)`.

`accumulated_local_effects` estimates local prediction changes within feature
intervals and accumulates those local effects. ALE is useful when predictors are
correlated and PDPs may be harder to interpret.

### Two-way feature effects and interaction strength

Produced by `_build_two_way_effects(...)` and
`_build_interaction_strength_table(...)`.

`two_way_feature_effects` provides response surfaces for top numeric feature
pairs. `interaction_strength` summarizes the non-additive part of those
surfaces, which helps decide whether an interaction term is materially useful.

### Feature effect confidence bands

Produced by `_build_effect_confidence_bands(...)`.

`feature_effect_confidence_bands` bootstraps the sampled predictions behind PDP
points and exports lower and upper 90% bands.

### Monotonicity, segmentation, and stability

Produced by `_build_effect_monotonicity_table(...)`,
`_build_segmented_feature_effects(...)`, and
`_build_effect_stability_table(...)`.

These tables review whether feature effects follow expected directions, whether
effect curves differ across important segments, and whether train, validation,
and test effects stay aligned.

### Average marginal effects

Produced by `_build_marginal_effects(...)`.

`average_marginal_effects` uses finite differences around each numeric feature
to estimate the average change in prediction for a small feature movement.

### Feature effect calibration

Produced by `_build_effect_calibration_table(...)`.

`feature_effect_calibration` compares actual and predicted outcomes by feature
bucket so reviewers can see whether a feature's learned response is also
calibrated across its range.

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

## 9. Distribution, Dependency, and Outlier Outputs

Quant Studio includes several framework-level diagnostics that are not single
model-performance metrics but are still exported as first-class review tables.

Common outputs now include:

- `distribution_tests`
- `distribution_shift_tests`
- `missingness_predictive_power`
- `littles_mcar_test`
- `dependency_cluster_summary`
- `condition_index_detail`
- `outlier_flags`
- `multiple_imputation_pooling_summary`
- `multiple_imputation_pooled_coefficients`

These are useful when a reviewer is asking:

- whether feature distributions are stable across splits
- whether missingness itself is predictive
- whether the numeric design is redundant or clustered
- whether a small set of observations is dominating the fit

## 10. Time-Series, Econometric, and Structural-Break Outputs

Time-aware workflows can now export:

- `time_series_extension_tests`
- `structural_break_tests`
- `seasonality_profile`
- `structural_break_profile`

These complement the existing:

- `adf_tests`
- `forecasting_statistical_tests`
- `cointegration_tests`
- `granger_causality_tests`

The intent is to give CECL and CCAR users a broader econometric review surface
without changing the narrow scope of the application away from model
development and documentation.

Newer rows inside those exports include:

- `kpss`
- `phillips_perron`
- `cusum`
- `cusum_squares`

## 11. Comparison-Significance Outputs

Challenger runs can now export:

- `model_comparison_significance_tests`

Typical rows include:

- `delong_auc_difference`
- `mcnemar_threshold_difference`
- `diebold_mariano`

## 12. Segment and Scenario Metrics

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

## 13. Where Metrics Are Exported

The same metric family often appears in more than one place:

| Artifact | Content |
| --- | --- |
| `metadata/metrics.json` | split-level metric dictionary |
| `workbooks/analysis_workbook.xlsx` | diagnostic tables |
| `reports/interactive_report.html` | grouped visual presentation |
| `tables/*.csv` | individual diagnostic tables |
| `reports/run_report.md` | run narrative summary |

## 14. Workflow Guardrails And Governance Readiness

Workflow guardrail outputs live in `workflow_guardrails.py` and are exported by
`DiagnosticsStep._add_workflow_guardrail_outputs(...)`.

Primary fields:

- `severity`
- `code`
- `field_path`
- `message`

This is not a model metric in the usual statistical sense. It is a governed
readiness output that records whether the selected preset was configured in a
way that fits its intended development use case.

Primary output:

- `workflow_guardrails`

## 15. Credit-Risk Development Diagnostics

These are calculated in `DiagnosticsStep._add_credit_risk_outputs(...)`.

Primary outputs include:

- `vintage_summary` / `vintage_curve`
- `cohort_pd_summary` / `cohort_pd_curve`
- `migration_matrix` / `migration_heatmap`
- `roll_rate_summary`
- `lgd_segment_summary` / `lgd_segment_chart`
- `recovery_segmentation` / `recovery_segment_chart`
- `macro_sensitivity` / `macro_sensitivity_chart`

These outputs are intended for development and validation review in PD, LGD,
CCAR, and CECL workflows when the available columns support them.
Migration outputs require an explicitly selected low-cardinality migration state
column, such as delinquency bucket, rating grade, or stage. Leaving the GUI
control set to `(none)` skips the migration matrix so continuous score fields are
not accidentally treated as states.

## 15. Numerical Stability And Estimation Health

These outputs are assembled from normalized warning capture in the model
adapters and exported through `DiagnosticsStep._add_model_artifact_outputs(...)`.

Primary outputs:

- `numerical_warning_summary`
- `model_numerical_diagnostics`

Typical fields include:

- `source`
- `stage`
- `warning_code`
- `category`
- `message`
- `occurrence_count`
- `diagnostic_name`
- `value`
- `status`

These are governance and auditability outputs rather than business metrics.
They show whether the underlying estimator reached its iteration cap, returned
finite standard errors, emitted runtime warnings, or required numerical
normalization.

## 16. Regulator-Ready Report Exports

Regulator-ready report generation is assembled in `reporting.py` and exported
by `ArtifactExportStep`.

Primary artifacts:

- `committee_report.docx`
- `committee_report.pdf`
- `validation_report.docx`
- `validation_report.pdf`

These files summarize and package the metrics above into committee-ready and
validation-ready delivery formats without changing the underlying quantitative
results. The current report exports include a cover page, a report map, and
explicit numerical-stability content alongside the existing development and
validation sections.

## 17. Robustness And Stability Metrics

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

## 18. Cross-Validation Diagnostics

These are calculated in `CrossValidationStep` after the standard diagnostics
finish. They fit temporary fold-level models and do not replace the final model
artifact saved as `model/quant_model.joblib`.

Primary exported tables:

- `cross_validation_fold_metrics`
- `cross_validation_metric_distribution`
- `cross_validation_metric_summary`
- `cross_validation_feature_distribution`
- `cross_validation_feature_stability`

Key fields include:

- `fold_id`
- `validation_method`
- `train_rows`
- `validation_rows`
- `metric_name`
- `metric_value`
- `mean_value`
- `std_value`
- `selection_frequency`
- `mean_effect`
- `mean_abs_effect`
- `sign_consistency`

The automatic strategy uses stratified k-fold validation for binary
cross-sectional data, regular k-fold validation for continuous cross-sectional
data, and expanding-window validation for time-series or panel data.

Primary figures:

- `cross_validation_metric_boxplot`
- `cross_validation_metric_summary_chart`
- `cross_validation_feature_stability`

## 19. Scorecard Workbench Outputs

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
