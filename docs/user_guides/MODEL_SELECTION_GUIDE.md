# Model Selection Guide

This guide helps users choose an initial model family. It is practical guidance,
not a substitute for model governance judgment.

## Recommended Starting Point

Start with the simplest model that can support the business question and review
standard. Add challengers only when they answer a specific modeling need.

| Use case | Recommended starting model |
| --- | --- |
| One-period PD | `logistic_regression` |
| Transparent PD scorecard | `scorecard_logistic_regression` |
| Sparse or collinear PD feature set | `elastic_net_logistic_regression` |
| Clustered or repeated-observation PD | `gee_logistic_regression` |
| Lifetime PD / CECL person-period data | `discrete_time_hazard_model` |
| LGD severity with bounded target | `fractional_logit`, `beta_regression`, or `two_stage_lgd_model` |
| LGD with true 0 and 1 outcomes | `zero_one_inflated_beta` |
| Censored LGD or recovery target | `tobit_regression` |
| CCAR or macro forecasting panel | `panel_regression` |
| Tail or percentile forecasting | `quantile_regression` |
| Duration or time-to-event target | `cox_proportional_hazards` or `aft_survival_model` |
| Non-linear challenger | `xgboost`, `random_forest`, `extra_trees`, or `explainable_boosting_machine` |
| Simple continuous baseline | `linear_regression` |

## Binary Target Models

| Model | Best use | Main caution |
| --- | --- | --- |
| `logistic_regression` | Default interpretable PD baseline. | Assumes mostly linear log-odds after transformations. |
| `scorecard_logistic_regression` | Scorecard, WoE, reason-code, and points-based development. | Requires careful bin review and monotonicity judgment. |
| `elastic_net_logistic_regression` | Large or correlated feature sets. | Coefficients are regularized and need careful interpretation. |
| `probit_regression` | Interpretable binary challenger. | Similar use to logistic, but less common operationally. |
| `gee_logistic_regression` | Cluster-aware binary panel or repeated-observation PD model. | Requires a defensible group column for clustered inference. |
| `discrete_time_hazard_model` | Lifetime PD / CECL period-level modeling. | Requires time or panel structure. |
| `random_forest` | Bagged-tree challenger for non-linear binary patterns. | Less transparent than coefficient models. |
| `extra_trees` | Randomized-tree sensitivity challenger. | Randomized split logic can be harder to govern. |
| `explainable_boosting_machine` | Shallow boosted-tree challenger with PDP/ALE review. | This is an sklearn-based EBM-style adapter, not the external `interpret` EBM package. |
| `xgboost` | Non-linear challenger for binary outcomes. | Stronger performance may reduce interpretability. |

## Continuous Target Models

| Model | Best use | Main caution |
| --- | --- | --- |
| `linear_regression` | Simple baseline for continuous targets and forecasting. | Sensitive to non-linearity and outliers. |
| `ridge_regression` | Continuous baseline with coefficient shrinkage. | Keeps correlated features rather than selecting among them. |
| `lasso_regression` | Continuous baseline with sparse feature selection pressure. | Can remove features that are still business-relevant. |
| `elastic_net_regression` | Continuous shrinkage plus feature-selection pressure. | Requires deliberate tuning of alpha and l1 ratio. |
| `fractional_logit` | Bounded rates or proportions in `[0, 1]`. | Boundary-heavy targets may need a zero-one model. |
| `beta_regression` | Bounded severity rates in `(0, 1)`. | Target must be appropriate for beta-style modeling. |
| `zero_one_inflated_beta` | Bounded severity with true mass at zero and one. | Needs enough interior observations for the beta component. |
| `two_stage_lgd_model` | LGD with many zero or no-loss observations plus positive severities. | Requires interpretation of two stages. |
| `panel_regression` | Entity-time panel forecasting and CCAR-style development. | Requires stable entity/date structure. |
| `quantile_regression` | Tail, downside, or percentile outcome modeling. | Does not target the conditional mean. |
| `tobit_regression` | Left- or right-censored continuous outcomes. | Censoring points must be meaningful. |
| `cox_proportional_hazards` | Time-to-event risk ranking. | Current implementation assumes observed events without a censoring indicator. |
| `aft_survival_model` | Log-duration time-to-event baseline. | Target must be a positive duration. |
| `random_forest` | Bagged-tree continuous challenger. | Requires explainability support. |
| `extra_trees` | Randomized-tree continuous sensitivity challenger. | Randomized split logic can be harder to govern. |
| `explainable_boosting_machine` | EBM-style shallow boosted-tree challenger. | Not the external `interpret` package implementation. |
| `xgboost` | Non-linear continuous challenger. | Requires stronger explainability review. |

## Model Choice By Governance Need

| Governance priority | Prefer |
| --- | --- |
| Maximum interpretability | logistic, scorecard, linear, panel |
| Strong challenger performance | elastic-net, XGBoost, random forest, extra trees |
| Scorecard implementation | scorecard logistic |
| CECL lifetime structure | discrete-time hazard |
| Clustered binary panel inference | GEE logistic |
| LGD with zero/positive split | two-stage LGD or zero-one inflated beta |
| Bounded severity | fractional logit or beta regression |
| Censored outcomes | Tobit |
| Tail risk | quantile regression |
| Time-to-event modeling | Cox PH or AFT survival |

LightGBM is not currently exposed because Quant Studio is not adding new
third-party dependencies in this implementation pass. Use XGBoost when an
installed gradient-boosted-tree model is needed.

## Suggested Development Pattern

1. Start with a transparent baseline.
2. Add one or two challengers only if they test a meaningful alternative.
3. Compare discrimination, calibration, stability, feature behavior, and documentation quality.
4. Use explainability outputs for non-linear models.
5. Use feature-subset search before final development only when feature selection is uncertain.
6. Produce the final evidence package with `fit_new_model`.

For implementation details, see [Model Catalog](../MODEL_CATALOG.md).
