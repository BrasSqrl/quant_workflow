# Configuration Cookbook

This cookbook gives practical starting configurations for common Quant Studio
workflows. Treat these as starting points, then adjust based on data, policy,
and validation requirements.

## Recipe 1: Baseline PD Logistic Model

Use when building a first-pass binary PD model.

Recommended setup:

- Preset: `PD Development`
- Execution mode: `fit_new_model`
- Model type: `logistic_regression`
- Target mode: `binary`
- Data structure: `cross_sectional`
- Split: train/validation/test with stratification on. Validation and test can be set to `0` when a workflow intentionally omits that holdout split.
- Feature policy: start with warnings, not hard stops
- Explainability: coefficient breakdown, permutation importance, PDP/ALE for top features
- Export profile: `standard`

Key checks:

- target source and positive values are correct
- identifiers and leakage fields are disabled
- calibration and threshold outputs are reviewed
- `model/quant_model.joblib` and `config/run_config.json` are preserved

## Recipe 2: Transparent Scorecard PD Model

Use when implementation transparency, points, bins, and reason codes matter.

Recommended setup:

- Preset: `PD Development`
- Execution mode: `fit_new_model`
- Model type: `scorecard_logistic_regression`
- Target mode: `binary`
- Enable scorecard workbench outputs
- Review monotonicity and minimum bin share settings
- Use manual bin override only when justified

Key checks:

- WoE/binning outputs make business sense
- bin overrides are documented
- reason-code outputs are reviewed
- validation pack includes scorecard evidence

## Recipe 3: Out-Of-Time Validation Or Test Split

Use when validation or test performance must come from specific calendar
periods instead of random or row-percentage holdouts.

Recommended setup:

- Data structure: `time_series` or `panel`
- Column Designer: mark the observation/as-of field as `date`
- Split strategy: `Date cutoff` when train, validation, and test are sequential
  time blocks
- Date cutoff example: validation starts `2023-01-01`, test starts
  `2024-01-01`; rows before validation become train
- Split strategy: `Explicit date windows` when each split must be restricted to
  named start/end dates
- Split strategy: `Custom split column` when the data already contains
  `train`, `validation` / `val`, or `test` / `oot` labels

Key checks:

- no target, recovery, collection, or post-default fields are available before
  the prediction date
- validation and test windows are large enough to contain events
- the exported `split_summary` date ranges match the intended development
  policy
- custom split columns are reviewed as metadata only; Quant Studio excludes the
  custom split column from model features automatically

## Recipe 4: Existing Model Scoring

Use when the model already exists and must not be rebuilt.

Recommended setup:

- Execution mode: `score_existing_model`
- Existing model path: prior `model/quant_model.joblib`
- Existing config path: prior `config/run_config.json` when available
- New input data: same raw feature contract as original model
- Labels: optional

Key checks:

- feature compatibility messages are clean
- score-only runs do not imply observed performance
- labeled scoring runs produce validation metrics
- exported predictions and report are preserved

## Recipe 5: Feature-Subset Search

Use before final model development when the candidate feature set is uncertain.

Recommended setup:

- Execution mode: `search_feature_subsets`
- Target mode: `binary`, `multiclass`, or `continuous`
- Model family: choose a feature-dependent family that matches the target mode
- Candidate features: restrict to business-plausible fields
- Max subset size: keep low enough to avoid combinatorial explosion
- Ranking metric: use AUC/ROC or KS for binary, accuracy or F1 for
  multiclass, and RMSE/MAE/R-squared for continuous targets

Key checks:

- review the ranked candidate table
- compare target-appropriate metrics and model simplicity
- review winning coefficients where available
- run `fit_new_model` after choosing the final features

## Recipe 6: Lifetime PD / CECL

Use for person-period or time-aware lifetime PD development.

Recommended setup:

- Preset: `Lifetime PD / CECL`
- Model type: `discrete_time_hazard_model`
- Target mode: `binary`
- Data structure: `time_series` or `panel`
- Date column: required
- Entity column: recommended for panel data
- Migration state column: leave as `(none)` unless a true low-cardinality state
  field exists, such as delinquency bucket, rating grade, or stage
- Backtesting and time diagnostics: on

Key checks:

- date ordering is valid
- period construction makes business sense
- lifetime PD outputs are reviewed
- time-aware validation is documented

## Recipe 7: LGD Severity

Use for continuous loss severity modeling.

Recommended setup:

- Preset: `LGD Severity`
- Target mode: `continuous`
- Model type: `two_stage_lgd_model`, `beta_regression`,
  `fractional_logit`, `zero_one_inflated_beta`, or `tobit_regression`
- Data structure: usually `cross_sectional` or `panel`
- Scenario testing: useful for macro or collateral shocks

Key checks:

- target bounds, boundary values, and censoring are understood
- zero-loss and positive-loss behavior are reviewed
- residual diagnostics are reviewed
- calibration and segment error are reviewed

## Recipe 8: CCAR Forecasting

Use for stress or macro-linked forecasting workflows.

Recommended setup:

- Preset: `CCAR Forecasting`
- Target mode: usually `continuous`
- Model type: `panel_regression`, `linear_regression`, regularized
  regression, `quantile_regression`, survival-style models for duration
  targets, or tree-based challengers
- Data structure: `time_series` or `panel`
- Date column: required
- Scenario testing: on
- Time-series diagnostics: on

Key checks:

- macro variables are available at forecast time
- time splits are appropriate
- structural-break and stationarity outputs are reviewed
- scenario outputs are documented

## Recipe 9: Multiclass Grade Or Stage Model

Use when the target is a three-or-more-class outcome rather than a binary event
or continuous value.

Recommended setup:

- Execution mode: `fit_new_model`
- Target mode: `multiclass`
- Model type: `multinomial_logistic_regression` for unordered classes
- Model type: `ordinal_logistic_regression` for ordered grades, stages, or
  bands
- Model type: `decision_tree` when transparent segmentation is the primary
  objective
- Stratified split: on when class sizes are sufficient

Key checks:

- the class order is documented when ordinal logistic is used
- the target mapping exported in metadata is reviewed
- class accuracy, macro F1, weighted F1, and log loss are reviewed
- binary PD diagnostics are not expected for multiclass runs

## Recipe 10: Count Or Severity GLM

Use when the target resembles a SAS-style GLM use case.

Recommended setup:

- Execution mode: `fit_new_model`
- Target mode: `continuous`
- Model type: `poisson_regression` for non-negative counts
- Model type: `negative_binomial_regression` for overdispersed counts
- Model type: `gamma_regression` for strictly positive skewed severities
- Model type: `tweedie_regression` for zero-plus-positive severity or loss
  targets
- Tweedie variance power: document the selected value

Key checks:

- target support matches the selected family
- residual and actual-versus-predicted diagnostics are reviewed
- segment-level error is reviewed for material bias
- model assumptions are documented in the decision summary

## Recipe 11: Smooth Nonlinear Spline Model

Use when a linear model underfits a smooth relationship but a black-box model is
hard to justify.

Recommended setup:

- Execution mode: `fit_new_model`
- Target mode: `binary` for `gam_spline_logistic`
- Target mode: `continuous` for `gam_spline_regression`
- Spline knots: start low and increase only when supported by validation
- Spline degree: keep the default unless there is a documented reason
- Explainability: PDP, ICE, ALE, and feature-effect monotonicity on

Key checks:

- feature-effect curves are smooth and business-sensible
- train/test divergence does not indicate overfitting
- spline settings are documented as model assumptions

## Recipe 12: Mixed Effects Or Forecasting Challenger

Use when repeated-observation structure or time-series behavior is central to
the business question.

Recommended setup:

- `mixed_effects_regression` for continuous repeated-observation data with a
  meaningful random-intercept group
- `sarimax_forecast` for ARIMA-style forecasts with optional exogenous drivers
- `exponential_smoothing_forecast` for trend and optional seasonality baseline
- `unobserved_components_forecast` for state-space trend or seasonality
  decomposition
- Date roles and time-aware split: required for forecasting workflows

Key checks:

- group column is defensible for mixed effects
- future scoring does not require unavailable group-level random effects
- forecast order, trend, and seasonality assumptions are documented
- forecasting statistical tests and structural-break diagnostics are reviewed

## Recipe 13: Large Data Run

Use when data is too large for comfortable browser upload or full in-memory
diagnostics.

Recommended setup:

- Place file in `Data_Load/`
- Prefer Parquet
- Enable `Large Data Mode`
- Enable CSV-to-Parquet staging if starting from CSV
- Use governed sample fitting
- Use chunked full-data scoring
- Use Parquet or sampled tabular exports
- Keep individual figure export off unless required
- Leave `Keep all checkpoints` off unless support needs post-run context files
- Keep `Advanced Visual Analytics` off unless the added exploratory report
  visuals are worth the extra rendering time

Key checks:

- memory estimate is acceptable
- sample-development folder is exported
- full-data scoring folder is exported
- sampled exports are clearly documented

## Recipe 14: Memory-Optimized Full-Data Fit

Use when the model must be fit on the full train split, but the source file is
large enough that duplicated pandas dataframes can exhaust RAM.

Recommended setup:

- Prefer Parquet input from `Data_Load/`
- Keep `Execution mode = fit_new_model`
- Leave `Optimize dtypes during ingestion` on
- Leave `Compact prediction exports` on
- Leave `Retain full diagnostic working dataframe` off
- Let tabular outputs follow the Step 1 file type; use Parquet input when
  Parquet artifact outputs are required
- Keep Excel workbook export off unless specifically needed
- Keep individual figure files and Advanced Visual Analytics off while tuning
- Leave `Keep all checkpoints` off to prune large context files during the run
- Review high-cardinality categorical warnings before fitting

Key checks:

- final model is still fit on the full training split
- `working_data_snapshot` documents diagnostic snapshot row counts
- `categorical_cardinality_profile` has no unapproved high-cardinality features
- `metadata/run_debug_trace.json` shows reasonable split and prediction memory
- prediction files contain scores and audit identifiers, not a duplicated full
  feature matrix

## Recipe 15: Fast Iteration Run

Use when testing setup before creating a full evidence package.

Recommended setup:

- Export profile: `fast`
- Individual figure files: off
- Advanced Visual Analytics: off
- Large optional diagnostics: off unless needed
- Use a smaller feature set or governed sample when appropriate

Key checks:

- do not treat a fast run as the final audit package unless the skipped assets are acceptable
- switch to `standard` or `audit` when preparing formal review evidence
