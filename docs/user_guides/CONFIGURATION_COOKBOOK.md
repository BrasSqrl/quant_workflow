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

## Recipe 3: Existing Model Scoring

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

## Recipe 4: Feature-Subset Search

Use before final model development when the candidate feature set is uncertain.

Recommended setup:

- Execution mode: `search_feature_subsets`
- Target mode: `binary`
- Model family: start with logistic regression unless testing another specific family
- Candidate features: restrict to business-plausible fields
- Max subset size: keep low enough to avoid combinatorial explosion
- Ranking metric: include AUC/ROC and KS review

Key checks:

- review the ranked candidate table
- compare AUC/ROC, KS, and model simplicity
- review winning coefficients where available
- run `fit_new_model` after choosing the final features

## Recipe 5: Lifetime PD / CECL

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

## Recipe 6: LGD Severity

Use for continuous loss severity modeling.

Recommended setup:

- Preset: `LGD Severity`
- Target mode: `continuous`
- Model type: `two_stage_lgd_model`, `beta_regression`, or `tobit_regression`
- Data structure: usually `cross_sectional` or `panel`
- Scenario testing: useful for macro or collateral shocks

Key checks:

- target bounds and censoring are understood
- zero-loss and positive-loss behavior are reviewed
- residual diagnostics are reviewed
- calibration and segment error are reviewed

## Recipe 7: CCAR Forecasting

Use for stress or macro-linked forecasting workflows.

Recommended setup:

- Preset: `CCAR Forecasting`
- Target mode: usually `continuous`
- Model type: `panel_regression`, `linear_regression`, `quantile_regression`, or XGBoost challenger
- Data structure: `time_series` or `panel`
- Date column: required
- Scenario testing: on
- Time-series diagnostics: on

Key checks:

- macro variables are available at forecast time
- time splits are appropriate
- structural-break and stationarity outputs are reviewed
- scenario outputs are documented

## Recipe 8: Large Data Run

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
- Keep `Advanced Visual Analytics` off unless the added exploratory report
  visuals are worth the extra rendering time

Key checks:

- memory estimate is acceptable
- sample-development folder is exported
- full-data scoring folder is exported
- sampled exports are clearly documented

## Recipe 9: Memory-Optimized Full-Data Fit

Use when the model must be fit on the full train split, but the source file is
large enough that duplicated pandas dataframes can exhaust RAM.

Recommended setup:

- Prefer Parquet input from `Data_Load/`
- Keep `Execution mode = fit_new_model`
- Leave `Optimize dtypes during ingestion` on
- Leave `Compact prediction exports` on
- Leave `Retain full diagnostic working dataframe` off
- Use Parquet tabular outputs, or `both` with sampled CSV review files
- Keep Excel workbook export off unless specifically needed
- Keep individual figure files and Advanced Visual Analytics off while tuning
- Review high-cardinality categorical warnings before fitting

Key checks:

- final model is still fit on the full training split
- `working_data_snapshot` documents diagnostic snapshot row counts
- `categorical_cardinality_profile` has no unapproved high-cardinality features
- `metadata/run_debug_trace.json` shows reasonable split and prediction memory
- prediction files contain scores and audit identifiers, not a duplicated full
  feature matrix

## Recipe 10: Fast Iteration Run

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
