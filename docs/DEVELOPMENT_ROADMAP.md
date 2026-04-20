# Quant Studio Development Roadmap

This roadmap is now implemented. The current development focus strengthened
imputation governance, richer feature construction, and deeper statistical
review without broadening the platform beyond model development.

## 1. Advanced Imputation Framework

Status: implemented

Delivered:

- per-column missingness-indicator feature generation
- train-fit grouped imputation using user-selected grouping columns
- exported grouped fill rules for auditability
- metadata capture of pre-imputation splits and resolved feature-level rules

## 2. Imputation Governance And Sensitivity Testing

Status: implemented

Delivered:

- alternative mean/median/mode sensitivity scoring on a selected split
- score-delta and primary-metric-delta comparison by feature and policy
- exported tables `imputation_sensitivity_summary` and
  `imputation_sensitivity_detail`
- GUI-backed controls through `ImputationSensitivityConfig`

## 3. Richer Governed Transformation Library

Status: implemented

Delivered:

- `yeo_johnson`
- `capped_zscore`
- `lag`
- `rolling_mean`
- `pct_change`
- persisted train-fit parameters through the existing governed-transformation
  audit surface

## 4. Interaction-Term Engine

Status: implemented

Delivered:

- numeric-numeric interaction screening
- categorical-numeric interaction screening
- persisted generated interaction specs so reruns and existing-model scoring
  replay the same engineered features
- exported interaction rankings through `interaction_candidates`

## 5. Missingness Diagnostics Upgrade

Status: implemented

Delivered:

- train/validation/test missingness stability views
- missingness-to-target association diagnostics
- pairwise missingness-indicator correlation
- exported tables `missingness_by_split`, `missingness_target_association`, and
  `missingness_indicator_correlation`
- exported figure `missingness_indicator_heatmap`

## 6. Additional Model Specification Tests

Status: implemented

Delivered:

- condition index
- Box-Tidwell tests
- link test
- leverage and Cook's-distance influence summaries
- exported tables `model_specification_tests` and `model_influence_summary`
- exported figure `model_influence_plot`

## 7. Forecasting And Time-Series Statistical Tests

Status: implemented

Delivered:

- Durbin-Watson
- Ljung-Box
- ARCH LM
- cointegration tests
- Granger-causality tests
- exported tables `forecasting_statistical_tests`, `cointegration_tests`, and
  `granger_causality_tests`
- split fallback logic so forecasting tests use the first split with enough time
  depth

## Notes

- The roadmap is fully delivered in the current codebase.
- Future work should continue to favor governed, exportable, and reviewable
  model-development utilities rather than monitoring or operational tooling.
