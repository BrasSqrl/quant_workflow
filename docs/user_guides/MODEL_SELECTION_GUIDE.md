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
| Lifetime PD / CECL person-period data | `discrete_time_hazard_model` |
| LGD severity with bounded target | `beta_regression` or `two_stage_lgd_model` |
| Censored LGD or recovery target | `tobit_regression` |
| CCAR or macro forecasting panel | `panel_regression` |
| Tail or percentile forecasting | `quantile_regression` |
| Non-linear challenger | `xgboost` |
| Simple continuous baseline | `linear_regression` |

## Binary Target Models

| Model | Best use | Main caution |
| --- | --- | --- |
| `logistic_regression` | Default interpretable PD baseline. | Assumes mostly linear log-odds after transformations. |
| `scorecard_logistic_regression` | Scorecard, WoE, reason-code, and points-based development. | Requires careful bin review and monotonicity judgment. |
| `elastic_net_logistic_regression` | Large or correlated feature sets. | Coefficients are regularized and need careful interpretation. |
| `probit_regression` | Interpretable binary challenger. | Similar use to logistic, but less common operationally. |
| `discrete_time_hazard_model` | Lifetime PD / CECL period-level modeling. | Requires time or panel structure. |
| `xgboost` | Non-linear challenger for binary outcomes. | Stronger performance may reduce interpretability. |

## Continuous Target Models

| Model | Best use | Main caution |
| --- | --- | --- |
| `linear_regression` | Simple baseline for continuous targets and forecasting. | Sensitive to non-linearity and outliers. |
| `beta_regression` | Bounded severity rates in `(0, 1)`. | Target must be appropriate for beta-style modeling. |
| `two_stage_lgd_model` | LGD with many zero or no-loss observations plus positive severities. | Requires interpretation of two stages. |
| `panel_regression` | Entity-time panel forecasting and CCAR-style development. | Requires stable entity/date structure. |
| `quantile_regression` | Tail, downside, or percentile outcome modeling. | Does not target the conditional mean. |
| `tobit_regression` | Left- or right-censored continuous outcomes. | Censoring points must be meaningful. |
| `xgboost` | Non-linear continuous challenger. | Requires stronger explainability review. |

## Model Choice By Governance Need

| Governance priority | Prefer |
| --- | --- |
| Maximum interpretability | logistic, scorecard, linear, panel |
| Strong challenger performance | elastic-net, XGBoost |
| Scorecard implementation | scorecard logistic |
| CECL lifetime structure | discrete-time hazard |
| LGD with zero/positive split | two-stage LGD |
| Bounded severity | beta regression |
| Censored outcomes | Tobit |
| Tail risk | quantile regression |

## Suggested Development Pattern

1. Start with a transparent baseline.
2. Add one or two challengers only if they test a meaningful alternative.
3. Compare discrimination, calibration, stability, feature behavior, and documentation quality.
4. Use explainability outputs for non-linear models.
5. Use feature-subset search before final development only when feature selection is uncertain.
6. Produce the final evidence package with `fit_new_model`.

For implementation details, see [Model Catalog](../MODEL_CATALOG.md).

