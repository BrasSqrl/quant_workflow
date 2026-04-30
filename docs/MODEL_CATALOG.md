# Model Catalog

This document catalogs the model families currently available in Quant Studio,
their intended use cases, implementation classes, constraints, and major audit
outputs.

Primary implementation files:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/models.py`
- `src/quant_pd_framework/steps/training.py`
- `src/quant_pd_framework/steps/comparison.py`

Factory entry point:

- `build_model_adapter(...)` in `src/quant_pd_framework/models.py`

## Quick Reference

| Model type | Target mode | Typical use | Main implementation |
| --- | --- | --- | --- |
| `logistic_regression` | binary | Baseline PD development | `LogisticRegressionAdapter` |
| `discrete_time_hazard_model` | binary | Lifetime PD / CECL | `DiscreteTimeHazardModelAdapter` |
| `elastic_net_logistic_regression` | binary | Sparse or collinear PD challenger | `ElasticNetLogisticRegressionAdapter` |
| `scorecard_logistic_regression` | binary | Transparent PD scorecard development | `ScorecardLogisticRegressionAdapter` |
| `probit_regression` | binary | Interpretable binary challenger | `ProbitRegressionAdapter` |
| `gee_logistic_regression` | binary | Cluster-aware panel PD challenger | `GEELogisticRegressionAdapter` |
| `multinomial_logistic_regression` | multiclass | Unordered class outcomes | `MultinomialLogisticRegressionAdapter` |
| `ordinal_logistic_regression` | multiclass | Ordered class outcomes | `OrdinalLogisticRegressionAdapter` |
| `linear_regression` | continuous or binary | Simple forecast baseline or linear-probability fallback | `LinearRegressionAdapter` |
| `ridge_regression` | continuous | Shrunk continuous baseline for collinear features | `RidgeRegressionAdapter` |
| `lasso_regression` | continuous | Sparse continuous baseline | `LassoRegressionAdapter` |
| `elastic_net_regression` | continuous | Continuous shrinkage and selection blend | `ElasticNetRegressionAdapter` |
| `beta_regression` | continuous | Bounded LGD severity in `(0, 1)` | `BetaRegressionAdapter` |
| `fractional_logit` | continuous | Bounded rate/proportion model in `[0, 1]` | `FractionalLogitAdapter` |
| `poisson_regression` | continuous | Count outcome GLM | `GLMRegressionAdapter` |
| `negative_binomial_regression` | continuous | Overdispersed count GLM | `GLMRegressionAdapter` |
| `gamma_regression` | continuous | Positive skewed severity GLM | `GLMRegressionAdapter` |
| `tweedie_regression` | continuous | Compound zero-plus-positive GLM | `GLMRegressionAdapter` |
| `gam_spline_regression` | continuous | Explainable nonlinear continuous effects | `GAMSplineAdapter` |
| `gam_spline_logistic` | binary | Explainable nonlinear PD effects | `GAMSplineAdapter` |
| `zero_one_inflated_beta` | continuous | LGD with true mass at 0, 1, and interior values | `ZeroOneInflatedBetaAdapter` |
| `two_stage_lgd_model` | continuous | Positive-loss probability times severity | `TwoStageLGDModelAdapter` |
| `mixed_effects_regression` | continuous | Random-intercept repeated-observation regression | `MixedEffectsRegressionAdapter` |
| `panel_regression` | continuous | Panel forecasting / CCAR-style modeling | `PanelRegressionAdapter` |
| `quantile_regression` | continuous | Tail or percentile forecasting | `QuantileRegressionAdapter` |
| `tobit_regression` | continuous | Censored continuous targets | `TobitRegressionAdapter` |
| `cox_proportional_hazards` | continuous | Time-to-event risk ranking | `CoxProportionalHazardsAdapter` |
| `aft_survival_model` | continuous | Log-duration time-to-event baseline | `AFTSurvivalModelAdapter` |
| `decision_tree` | binary, multiclass, or continuous | Transparent tree segmentation | `DecisionTreeAdapter` |
| `sarimax_forecast` | continuous | ARIMA-style forecast with exogenous features | `SARIMAXForecastAdapter` |
| `exponential_smoothing_forecast` | continuous | ETS-style trend/seasonal forecast | `ExponentialSmoothingForecastAdapter` |
| `unobserved_components_forecast` | continuous | State-space trend/seasonal forecast | `UnobservedComponentsForecastAdapter` |
| `random_forest` | binary or continuous | Non-linear bagged-tree challenger | `RandomForestAdapter` |
| `extra_trees` | binary or continuous | Randomized-tree sensitivity challenger | `ExtraTreesAdapter` |
| `explainable_boosting_machine` | binary or continuous | EBM-style shallow boosted-tree challenger | `ExplainableBoostingMachineAdapter` |
| `xgboost` | binary or continuous | Non-linear challenger | `XGBoostAdapter` |

## Common Design Pattern

All models conform to the common adapter contract defined by
`BaseModelAdapter` in `src/quant_pd_framework/models.py`. That contract
requires:

- `fit(...)`
- `predict_score(...)`
- `get_feature_importance()`

Optional extensions also support auditability:

- `get_model_artifacts()`
- `get_prediction_outputs()`

This is why the rest of the pipeline can remain model-agnostic.

## 1. Logistic Regression

### When to use it

Use as the default PD baseline when interpretability, coefficient review, and
validator familiarity matter.

### Strengths

- widely understood
- stable baseline for challenger comparisons
- supports standard discrimination and calibration review

### Constraints

- requires `TargetMode.BINARY`

### Key configuration

- `ModelConfig.max_iter`
- `ModelConfig.C`
- `ModelConfig.solver`
- `ModelConfig.class_weight`
- `ModelConfig.threshold`

### Outputs

- coefficients and odds ratios in `feature_importance`
- standard binary metrics
- full calibration workflow

## 2. Discrete-Time Hazard Model

### When to use it

Use for lifetime PD and CECL-style person-period modeling where timing matters,
not just one-period default incidence.

### Implementation note

The implementation is a pooled logistic regression over person-period rows. The
feature engineering step adds:

- `hazard_period_index`
- `hazard_period_index_sq`

### Constraints

- requires `TargetMode.BINARY`
- requires `DataStructure.TIME_SERIES` or `DataStructure.PANEL`
- requires a configured date column

### Outputs

- all binary metrics
- lifetime PD table `lifetime_pd_curve`
- lifetime PD figure `lifetime_pd_curve`

## 3. Elastic-Net Logistic Regression

### When to use it

Use as a challenger to plain logistic regression when:

- the feature set is large
- predictors are correlated
- coefficient shrinkage is desirable

### Strengths

- combines L1 and L2 regularization
- useful for candidate reduction pressure

### Key configuration

- `ModelConfig.l1_ratio`
- `ModelConfig.C`
- `ModelConfig.max_iter`

### Constraints

- requires `TargetMode.BINARY`

## 4. Scorecard Logistic Regression

### When to use it

Use when the goal is a highly transparent PD model with binning, WoE
transformation, scaled points, and reason codes.

### Distinguishing behavior

- numeric features are binned
- categorical features are grouped directly
- each bucket is converted to WoE
- logistic regression is fit on WoE-transformed features
- point scaling and reason codes are derived after fitting

### Key configuration

- `ModelConfig.scorecard_bins`
- `ScorecardConfig.monotonicity`
- `ScorecardConfig.min_bin_share`
- `ScorecardConfig.base_score`
- `ScorecardConfig.points_to_double_odds`
- `ScorecardConfig.odds_reference`
- `ScorecardConfig.reason_code_count`

### Special artifacts

- `scorecard_woe_table`
- `scorecard_points_table`
- `scorecard_scaling_summary`
- prediction-side outputs such as `scorecard_points` and `reason_code_1...n`

### When not to use it

Do not treat it as a black-box performance model. It is optimized for
governance and transparency.

## 5. Probit Regression

### When to use it

Use as a binary challenger when you want a latent-normal-link alternative to
logistic regression.

### Strengths

- interpretable parametric binary model
- often useful as a challenger in documentation-heavy workflows

### Constraints

- requires `TargetMode.BINARY`

## 6. Linear Regression

### When to use it

Use as:

- a continuous-target baseline
- a simple forecast baseline
- a linear-probability fallback when binary mode is intentionally chosen

### Important note

When used in binary mode, predictions are clipped into `[0, 1]` for downstream
diagnostics. This is convenient, but it does not make linear regression a true
probability model.

## 6A. Regularized Linear Regression

### When to use it

Use `ridge_regression`, `lasso_regression`, or `elastic_net_regression` for
continuous LGD or forecasting targets when ordinary linear regression is too
unstable because features are numerous or correlated.

### Key configuration

- `ModelConfig.regularization_alpha`
- `ModelConfig.l1_ratio` for elastic net only

### Constraints

- requires `TargetMode.CONTINUOUS`

## 7. Beta Regression

### When to use it

Use for bounded continuous targets such as LGD severity when the target is meant
to stay within `(0, 1)`.

### Strengths

- better aligned to bounded targets than plain OLS
- returns coefficient standard errors and p-values from statsmodels

### Key configuration

- `ModelConfig.beta_clip_epsilon`

### Constraints

- requires `TargetMode.CONTINUOUS`
- target values are clipped away from exact `0` and `1` before fitting

## 7A. Fractional Logit

### When to use it

Use `fractional_logit` for bounded continuous rates or proportions, including
LGD-style targets in `[0, 1]`, when a GLM-style bounded mean model is more
appropriate than OLS.

### Constraints

- requires `TargetMode.CONTINUOUS`
- target values should represent a bounded rate or proportion

## 7B. Zero-One Inflated Beta

### When to use it

Use `zero_one_inflated_beta` when LGD or another bounded severity target has
meaningful true observations at exactly `0` and exactly `1`, plus interior
values between the boundaries.

### Implementation note

The adapter fits separate boundary classifiers for exact zero and exact one,
then fits beta regression on interior observations. The final estimate combines
boundary probabilities with conditional interior severity.

### Constraints

- requires `TargetMode.CONTINUOUS`
- requires enough interior observations to fit the beta component

## 8. Two-Stage LGD Model

### When to use it

Use when LGD behavior is better modeled as:

1. probability of a positive loss
2. severity conditional on positive loss

### Implementation note

Stage one is logistic regression on a positive-loss indicator. Stage two is beta
regression on the positive-loss subset. Final prediction is:

- `positive_loss_probability * conditional_severity`

### Strengths

- closer to many real LGD workflows than one-step regression

### Constraints

- requires `TargetMode.CONTINUOUS`
- requires at least 10 positive-loss observations

### Special artifacts

- `lgd_stage_one_coefficients`
- `lgd_stage_two_coefficients`

## 9. Panel Regression

### When to use it

Use for CCAR-style or other panel forecasting problems where panel structure is
important and the workflow wants encoded panel effects inside a documented OLS
process.

### Constraints

- requires `TargetMode.CONTINUOUS`
- most appropriate with `DataStructure.PANEL`

### Important note

The current implementation is dense OLS over the engineered design matrix. It is
not a full econometric panel package with random-effects or within estimators.

## 10. Quantile Regression

### When to use it

Use when you want to model a conditional quantile rather than a conditional
mean, such as:

- downside forecast views
- stressed percentile estimates
- tail-sensitive challenger models

### Key configuration

- `ModelConfig.quantile_alpha`

### Constraints

- requires `TargetMode.CONTINUOUS`

## 11. Tobit Regression

### When to use it

Use when the target is continuous but censored at a lower bound, upper bound, or
both.

### Implementation note

The framework uses a custom maximum-likelihood Tobit implementation with BFGS
optimization rather than a wrapper around an external black-box class.

### Key configuration

- `ModelConfig.tobit_left_censoring`
- `ModelConfig.tobit_right_censoring`

### Constraints

- requires `TargetMode.CONTINUOUS`

## 12. XGBoost

### When to use it

Use as the main non-linear challenger for both binary and continuous workflows.

### Strengths

- handles non-linear interactions
- serves as a useful challenger against interpretable baselines

### Tradeoff

It is less transparent than the linear, probit, or scorecard families. In this
platform it is best used as a challenger rather than as the only documented
view.

### Key configuration

- `ModelConfig.xgboost_n_estimators`
- `ModelConfig.xgboost_learning_rate`
- `ModelConfig.xgboost_max_depth`
- `ModelConfig.xgboost_subsample`
- `ModelConfig.xgboost_colsample_bytree`

## 13. GEE Logistic Regression

### When to use it

Use `gee_logistic_regression` for binary panel PD workflows where rows are
clustered by borrower, account, loan, or another repeated-observation unit.
The model estimates population-average effects with robust covariance.

### Key configuration

- `ModelConfig.gee_group_column`
- `ModelConfig.max_iter`

### Constraints

- requires `TargetMode.BINARY`
- if no group column is selected, rows are treated as independent groups

## 14. Survival-Style Models

### When to use them

Use `cox_proportional_hazards` or `aft_survival_model` when the target is a
positive duration or time-to-event value.

### Implementation notes

- `cox_proportional_hazards` uses statsmodels `PHReg` and reports hazard ratios.
- `aft_survival_model` fits a log-duration regression and predicts positive
  durations.

### Constraints

- requires `TargetMode.CONTINUOUS`
- the current Cox implementation assumes observed events because no censoring
  indicator is currently part of `ModelConfig`

## 15. Random Forest, Extra Trees, and EBM-Style Models

### When to use them

Use these as non-linear challenger or sensitivity models when XGBoost is not
the only tree-based view you want.

### Implementation notes

- `random_forest` uses sklearn random forests.
- `extra_trees` uses sklearn extremely randomized trees.
- `explainable_boosting_machine` is an EBM-style shallow boosted-tree adapter
  built with existing sklearn dependencies. It is not the external
  `interpret` package implementation.

### Key configuration

- `ModelConfig.tree_n_estimators`
- `ModelConfig.tree_max_depth`
- `ModelConfig.xgboost_learning_rate` for the EBM-style adapter

### Dependency note

True LightGBM is intentionally not exposed because this project currently does
not add a new third-party dependency. XGBoost remains the installed
gradient-boosted-tree implementation.

## 16. SAS-Equivalent Model Families

These families were added for normal `fit_new_model` workflows to improve
parity with common SAS/STAT and SAS/ETS development patterns. They are not
enabled for feature subset search yet.

### Multiclass and Ordinal Classification

- `multinomial_logistic_regression` supports unordered class outcomes.
- `ordinal_logistic_regression` supports ordered class outcomes.
- `decision_tree` can also run against multiclass targets.

Use `TargetMode.MULTICLASS` for these models. The framework exports class
predictions, class probabilities where available, accuracy, macro F1, weighted
F1, log loss, feature importance, and model artifacts. Binary PD calibration,
threshold, lift/gain, and WoE/IV diagnostics are intentionally skipped.

### Count and Severity GLMs

- `poisson_regression` for non-negative count outcomes.
- `negative_binomial_regression` for overdispersed counts.
- `gamma_regression` for strictly positive skewed outcomes.
- `tweedie_regression` for compound zero-plus-positive severity outcomes.

These are implemented through statsmodels GLM families and use the existing
continuous-target evaluation path.

### GAM-Style Spline Models

- `gam_spline_logistic` for binary PD-style targets.
- `gam_spline_regression` for continuous targets.

These are lightweight spline-expanded models using existing sklearn
dependencies, not an external GAM package.

### Mixed Effects and Forecasting

- `mixed_effects_regression` fits a random-intercept MixedLM with an optional
  grouping column.
- `sarimax_forecast` fits a SARIMAX model with the configured `(p, d, q)` order
  and exogenous feature matrix.
- `exponential_smoothing_forecast` fits an ETS-style trend/seasonal baseline.
- `unobserved_components_forecast` fits a state-space level/trend/seasonal
  model.

Forecasting models require a continuous target and are best used with
time-aware splits.

## Preset Alignment

Preset defaults are defined in `src/quant_pd_framework/presets.py`.

| Preset | Default model |
| --- | --- |
| `PD Development` | `logistic_regression` |
| `Lifetime PD / CECL` | `discrete_time_hazard_model` |
| `LGD Severity` | `two_stage_lgd_model` |
| `CCAR Forecasting` | `panel_regression` |

## Comparison Mode

Model comparison is controlled by `ComparisonConfig` and implemented in:

- `src/quant_pd_framework/steps/comparison.py`

The framework can train challenger models alongside the primary model and rank
them using:

- binary defaults such as `roc_auc` or `average_precision`
- multiclass defaults such as `accuracy`, macro F1, weighted F1, or log loss
- continuous metrics such as `rmse` or `mae`

## Audit Notes

- The authoritative model dispatch is `build_model_adapter(...)`.
- Model eligibility is enforced in `ModelConfig.validate(...)`.
- The exported `model/model_summary.txt` and `reports/model_documentation_pack.md` should be
  read together for audit review.
