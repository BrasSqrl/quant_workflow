# Model Type Reference Guide

This guide explains how to choose and configure each model type available in
Quant Studio. It is written as a practical walkthrough for a GUI user who wants
to build a model, understand the required data setup, and know which outputs to
review after the run.

For a shorter model-selection table, see [Model Selection Guide](./MODEL_SELECTION_GUIDE.md).
For implementation-level details, see [Model Catalog](../MODEL_CATALOG.md).

## Common Setup For Any Model

Every model run starts the same way:

1. Open Step 1, `Dataset & Schema`.
2. Load bundled sample data, choose a file from `Data_Load/`, or upload a CSV,
   Excel, or Parquet file.
3. In the Column Designer, assign the target, feature, identifier, date, and
   excluded roles.
4. Open Step 2, `Model Configuration`.
5. Choose `fit_new_model` unless you are scoring a saved model or running
   feature-subset search.
6. Choose the model type.
7. Read the Model Type Story Card that appears beside the selector. It explains
   best-use cases, avoid conditions, key settings, outputs to review, and
   validation questions for the selected model.
8. Choose the correct target mode: `binary` or `continuous`.
9. Configure split strategy, model settings, diagnostics, exports, and optional
   challenger or policy settings.
10. Open Step 3, `Readiness Check`, resolve blocking issues, and run the
   workflow.
11. Review Step 4 outputs and the Step 5 Decision Room / decision summary.

The most important setup decision is target mode:

- Use `binary` when the target is an event flag such as default / no default,
  cure / no cure, fraud / no fraud, or approval / decline.
- Use `continuous` when the target is a numeric amount, rate, loss severity,
  balance, forecast value, or bounded severity.

## Quick Model Map

| Model type | Target mode | Best use |
| --- | --- | --- |
| `logistic_regression` | binary | Standard PD baseline and primary interpretable classifier. |
| `discrete_time_hazard_model` | binary | Lifetime PD / CECL timing model on time-series or panel data. |
| `elastic_net_logistic_regression` | binary | Binary model with shrinkage for correlated or larger feature sets. |
| `scorecard_logistic_regression` | binary | WoE-binned scorecard with points and reason codes. |
| `probit_regression` | binary | Parametric challenger to logistic regression. |
| `gee_logistic_regression` | binary | Cluster-aware panel PD model with robust GEE estimates. |
| `linear_regression` | continuous or binary | Simple continuous baseline, forecast baseline, or linear-probability fallback. |
| `ridge_regression` | continuous | Regularized continuous model that shrinks correlated coefficients. |
| `lasso_regression` | continuous | Regularized continuous model with sparse feature-selection pressure. |
| `elastic_net_regression` | continuous | Continuous model blending ridge shrinkage and lasso selection. |
| `beta_regression` | continuous | LGD or severity model where the target is bounded in `(0, 1)`. |
| `fractional_logit` | continuous | Bounded rate/proportion model for targets in `[0, 1]`. |
| `zero_one_inflated_beta` | continuous | LGD/severity model with true mass at 0, 1, and interior values. |
| `two_stage_lgd_model` | continuous | LGD model separating positive-loss likelihood from severity. |
| `panel_regression` | continuous | Panel-style forecasting or CCAR-style development. |
| `quantile_regression` | continuous | Tail, downside, or percentile forecasting. |
| `tobit_regression` | continuous | Censored continuous outcomes. |
| `cox_proportional_hazards` | continuous | Time-to-event risk ranking with hazard ratios. |
| `aft_survival_model` | continuous | Log-duration time-to-event baseline. |
| `random_forest` | binary or continuous | Bagged-tree non-linear challenger. |
| `extra_trees` | binary or continuous | Randomized-tree non-linear challenger. |
| `explainable_boosting_machine` | binary or continuous | EBM-style shallow boosted-tree challenger using existing dependencies. |
| `xgboost` | binary or continuous | Non-linear challenger for complex relationships. |

## 1. Logistic Regression

### What It Does

Logistic regression models the probability of a binary event. In PD work, that
usually means estimating the probability that a loan or account defaults.

The model produces coefficients that are relatively easy to explain. Positive
coefficients increase the log-odds of the event, and negative coefficients
decrease the log-odds of the event.

### When To Use It

Use `logistic_regression` when:

- you are building a standard PD-style model
- the target is binary
- interpretability matters
- you want a strong baseline before testing challengers
- validators expect coefficient review, calibration, ROC/AUC, KS, and threshold
  diagnostics

### GUI Walkthrough

1. In Step 1, mark the default or event field as the target column.
2. Mark usable predictors as features.
3. Mark IDs as identifiers and leakage fields as excluded.
4. In Step 2, choose `fit_new_model`.
5. Set `Target mode` to `Binary`.
6. Set `Model type` to `Logistic Regression`.
7. In `Model Settings`, review:
   - `Classification threshold`
   - `Max iterations`
   - `Inverse regularization (C)`
   - `Solver`
   - `Class weight`
8. Keep calibration, threshold analysis, ROC/AUC, KS, VIF, and feature policy
   diagnostics enabled unless there is a specific reason to turn them off.
9. Run Step 3.
10. Review Step 4 model performance, calibration, feature importance, and
    governance outputs.

### Outputs To Review

- `metadata/metrics.json`
- `model/feature_importance.csv`
- `tables/model_performance/split_metrics.*`
- `tables/calibration/calibration.*`
- `tables/calibration/threshold_analysis.*`
- `tables/model_performance/roc_curve.*`
- `tables/model_performance/lift_gain.*`

### Watchouts

- Logistic regression assumes a stable linear relationship on the log-odds
  scale.
- Highly correlated features can make coefficient signs unstable.
- A high AUC does not guarantee good calibration.
- If defaults are rare, review class weighting and precision-recall metrics.

## 2. Discrete-Time Hazard Model

### What It Does

The discrete-time hazard model estimates event probability over time. In this
project, it is implemented as a pooled logistic model over time-indexed rows.
The framework adds time-period terms such as `hazard_period_index` and
`hazard_period_index_sq` so the model can learn how event risk changes over
the life of an account or exposure.

### When To Use It

Use `discrete_time_hazard_model` when:

- the target is binary
- timing matters, not just whether default occurs
- the data has repeated observations over time
- you are developing lifetime PD, CECL, or vintage-style PD evidence

### GUI Walkthrough

1. In Step 1, mark the event/default field as the target.
2. Mark the date field as a date column.
3. Mark entity/account/loan ID as an identifier.
4. Mark time-varying drivers as features.
5. In Step 2, choose `fit_new_model`.
6. Set `Target mode` to `Binary`.
7. Set `Data structure` to `Time Series` or `Panel`.
8. Set `Model type` to `Discrete-Time Hazard Model`.
9. In `Split Strategy`, prefer a time-aware split when observations are ordered
   by period.
10. Keep credit-risk development diagnostics enabled when lifetime PD evidence
    is needed.
11. Run Step 3 and review the lifetime PD outputs.

### Outputs To Review

- `lifetime_pd_curve`
- vintage and cohort diagnostics when enabled
- binary performance metrics
- calibration and backtest outputs
- `model/feature_importance.csv`

### Watchouts

- This model needs meaningful time structure.
- Cross-sectional data without a reliable date field is not a good fit.
- Make sure the target definition is period-specific, not accidentally
  cumulative across all periods unless that is intended.

## 3. Elastic-Net Logistic Regression

### What It Does

Elastic-net logistic regression is a logistic model with combined L1 and L2
regularization. L1 pressure can shrink weaker coefficients toward zero, while
L2 pressure helps stabilize correlated predictors.

### When To Use It

Use `elastic_net_logistic_regression` when:

- the target is binary
- you have many candidate features
- predictors are correlated
- you want a more regularized challenger to plain logistic regression
- feature selection pressure is useful but you still want an interpretable
  linear-style model

### GUI Walkthrough

1. Configure Step 1 like a normal binary PD model.
2. In Step 2, choose `fit_new_model`.
3. Set `Target mode` to `Binary`.
4. Set `Model type` to `Elastic-Net Logistic Regression`.
5. In `Model Settings`, review:
   - `Elastic-net l1 ratio`
   - `Inverse regularization (C)`
   - `Max iterations`
   - `Class weight`
6. Use model comparison mode if you want to compare it against plain logistic
   regression.
7. Run Step 3.
8. Review coefficient stability, VIF, feature importance, ROC/AUC, KS, and
   calibration.

### Outputs To Review

- `model/feature_importance.csv`
- split metrics
- calibration outputs
- model comparison outputs if challenger mode is enabled
- coefficient stability outputs if robustness or cross-validation is enabled

### Watchouts

- Strong regularization can improve stability but reduce intuitive coefficient
  interpretation.
- Tune `l1_ratio` and `C` deliberately; defaults are a starting point, not a
  validation conclusion.

## 4. Scorecard Logistic Regression

### What It Does

Scorecard logistic regression bins numeric features, converts each bucket to
Weight of Evidence, fits logistic regression on the WoE-transformed features,
and then produces scaled scorecard points and reason codes.

Weight of Evidence is:

```text
WoE bucket = ln(% good observations in bucket / % bad observations in bucket)
```

This turns continuous variables into interpretable risk buckets. For example,
`utilization` may be converted into several ranges, each range gets a WoE
value, and the model is fit on those WoE values.

### When To Use It

Use `scorecard_logistic_regression` when:

- the target is binary
- transparency is more important than black-box flexibility
- users need scorecard points, bins, reason codes, or WoE tables
- the model will be reviewed by credit-risk, validation, audit, or governance
  teams

### GUI Walkthrough

1. In Step 1, use the bundled sample for a complete demo, or load your own
   binary default/event dataset.
2. Mark your default/event column as the target.
3. For panel data, mark the observation date as `date` and the loan/account ID
   as `identifier`.
4. Mark continuous financial ratios, loan behavior fields, and categorical
   borrower descriptors as features.
5. In Step 2, choose `fit_new_model`.
6. Set `Target mode` to `Binary`.
7. Set `Data structure` to `Panel` when using the bundled sample.
8. Set `Model type` to `Scorecard Logistic Regression`.
9. In `Model Settings`, configure:
   - `Scorecard bins`
   - `Scorecard monotonicity`
   - `Scorecard min bin share`
   - `Scorecard base score`
   - `Scorecard PDO`
   - `Scorecard odds reference`
   - `Reason code count`
10. In `Diagnostics & Exports`, keep `Enable scorecard workbench` on.
11. If you want manual bin control, enable `Manual Review Workflow` and enter
   scorecard bin overrides such as `0.20, 0.35, 0.50, 0.75`.
12. Run Step 3.
13. Review the Scorecard / Binning Workbench in Step 4.
14. Open `Binning Theater` for the selected feature when you want a focused
    review of bucket quality, IV, largest-bin share, WoE span, selected WoE
    rows, selected point rows, and a copyable manual-bin override candidate.

The Binning Theater is review-only. It does not change bins or refit the model
inside the completed run. If you copy override edges from it, paste them back
into Step 2 and rerun the workflow so the exported model, tests, and artifacts
all reflect the selected bin policy.

### Outputs To Review

- `tables/scorecard/scorecard_woe_table.*`
- `tables/scorecard/scorecard_points_table.*`
- `tables/scorecard/scorecard_feature_summary.*`
- `tables/scorecard/scorecard_scaling_summary.*`
- `tables/scorecard/scorecard_reason_code_frequency.*`
- prediction outputs with `scorecard_points` and `reason_code_1...n`

### Watchouts

- WoE requires a binary target.
- Bins should be reviewed for monotonicity, minimum size, and business meaning.
- Manual bin overrides should be documented with rationale.
- Scorecards are meant for transparency; they may not beat XGBoost on raw
  performance.

## 5. Probit Regression

### What It Does

Probit regression is a binary response model like logistic regression, but it
uses a normal cumulative distribution link instead of the logistic link.

### When To Use It

Use `probit_regression` when:

- the target is binary
- you want an interpretable parametric challenger
- the modeling or validation team wants to compare logistic-link and
  normal-link behavior

### GUI Walkthrough

1. Configure Step 1 like a binary model.
2. In Step 2, choose `fit_new_model`.
3. Set `Target mode` to `Binary`.
4. Set `Model type` to `Probit Regression`.
5. Review `Max iterations`, `Solver`, and `Class weight` where applicable.
6. Enable model comparison if you want probit to run beside logistic.
7. Run Step 3.
8. Review metrics, calibration, and feature importance.

### Outputs To Review

- split metrics
- calibration outputs
- `model/feature_importance.csv`
- model comparison outputs if enabled

### Watchouts

- Coefficients are not on the same scale as logistic regression coefficients.
- Interpret signs and relative direction more than raw coefficient magnitude.

## 6. Linear Regression

### What It Does

Linear regression estimates a continuous value using a linear relationship
between features and target. In this project, it can also be used in binary
mode as a linear-probability fallback, but that should be treated cautiously.

### When To Use It

Use `linear_regression` when:

- the target is continuous
- you need a simple benchmark
- you are building a forecast baseline
- interpretability matters
- you want a challenger against more complex continuous models

### GUI Walkthrough

1. In Step 1, mark the numeric outcome as the target.
2. Mark forecast drivers or explanatory variables as features.
3. Mark date/entity columns appropriately if the data has time or panel
   structure.
4. In Step 2, choose `fit_new_model`.
5. Set `Target mode` to `Continuous` for normal regression use.
6. Set `Model type` to `Linear Regression`.
7. Configure split strategy and diagnostics.
8. Run Step 3.
9. Review residual diagnostics, actual-versus-predicted views, segment bias,
   and coefficient outputs.

### Outputs To Review

- RMSE, MAE, and R-squared metrics
- residual summary
- residual diagnostics
- actual-versus-predicted charts
- feature importance / coefficient outputs

### Watchouts

- Outliers can strongly affect coefficients.
- Relationships may be non-linear.
- If used in binary mode, predictions are clipped into `[0, 1]`; that does not
  make it a true probability model.

## 7. Beta Regression

### What It Does

Beta regression models continuous outcomes bounded between 0 and 1. This is
often useful for LGD severity rates, recovery-adjusted loss rates, or other
proportions.

### When To Use It

Use `beta_regression` when:

- the target is continuous
- values are proportions or rates
- the target should stay between 0 and 1
- LGD severity is better modeled as a bounded outcome than an unbounded OLS
  value

### GUI Walkthrough

1. In Step 1, mark the bounded severity/rate field as the target.
2. Confirm values are mostly within `[0, 1]`.
3. Mark usable drivers as features.
4. In Step 2, choose `fit_new_model`.
5. Set `Target mode` to `Continuous`.
6. Set `Model type` to `Beta Regression`.
7. Review diagnostics and export options.
8. Run Step 3.
9. Review residual diagnostics, actual-versus-predicted views, and coefficient
   outputs.

### Outputs To Review

- continuous metrics such as RMSE and MAE
- residual diagnostics
- model summary
- feature importance / coefficient outputs

### Watchouts

- Exact `0` and `1` values are clipped away from the boundary before fitting.
- If many observations are exactly zero, consider `two_stage_lgd_model`.
- If the target can be negative or above 1, beta regression is usually the
  wrong model.

## 8. Two-Stage LGD Model

### What It Does

The two-stage LGD model separates LGD into two parts:

1. the probability that loss is positive
2. the severity amount conditional on positive loss

Final prediction is:

```text
predicted LGD = probability of positive loss * conditional severity
```

### When To Use It

Use `two_stage_lgd_model` when:

- the target is continuous
- many records have zero or near-zero loss
- positive-loss severity behaves differently from the event of having any loss
- you are building LGD evidence for credit-risk development

### GUI Walkthrough

1. In Step 1, mark LGD or loss severity as the target.
2. Mark recovery, collateral, account, macro, or borrower fields as features.
3. In Step 2, choose `fit_new_model`.
4. Set `Target mode` to `Continuous`.
5. Set `Model type` to `Two-Stage LGD Model`.
6. Keep LGD segment and recovery diagnostics enabled when relevant.
7. Run Step 3.
8. Review stage-one and stage-two coefficients, LGD segment outputs, and
   actual-versus-predicted diagnostics.

### Outputs To Review

- `lgd_stage_one_coefficients`
- `lgd_stage_two_coefficients`
- LGD segment summary
- recovery segmentation
- continuous performance metrics

### Watchouts

- The model needs enough positive-loss observations.
- If almost all observations are positive and bounded, beta regression may be
  simpler.
- If zero loss means missing or unobserved rather than true zero, fix the data
  definition before modeling.

## 9. Panel Regression

### What It Does

Panel regression is used when observations are organized by entity and time.
This project implements a documented dense regression over the engineered panel
design matrix rather than a full econometric random-effects or fixed-effects
package.

### When To Use It

Use `panel_regression` when:

- the target is continuous
- observations repeat across entities over time
- you are building CCAR-style, portfolio, account, or regional forecast
  evidence
- panel structure should be explicitly documented

### GUI Walkthrough

1. In Step 1, mark the continuous outcome as the target.
2. Mark entity/account/portfolio ID as an identifier.
3. Mark the reporting date as a date column.
4. Mark explanatory drivers as features.
5. In Step 2, choose `fit_new_model`.
6. Set `Target mode` to `Continuous`.
7. Set `Data structure` to `Panel`.
8. Set `Model type` to `Panel Regression`.
9. Keep time-aware diagnostics, residual diagnostics, and structural-break
   diagnostics enabled when relevant.
10. Run Step 3.

### Outputs To Review

- continuous metrics
- residual diagnostics
- time-series diagnostics
- structural-break tests if enabled
- actual-versus-predicted and segment views

### Watchouts

- Do not assume this is a full fixed-effects or random-effects econometric
  estimator.
- Date and entity roles need to be assigned correctly.
- Watch for leakage from future-period fields.

## 10. Quantile Regression

### What It Does

Quantile regression models a selected conditional percentile rather than the
conditional mean. For example, `alpha = 0.75` estimates the 75th percentile of
the target conditional on the features.

### When To Use It

Use `quantile_regression` when:

- the target is continuous
- tail behavior matters
- you need downside, upside, stressed, or percentile-specific forecasts
- average behavior is less important than a selected part of the distribution

### GUI Walkthrough

1. In Step 1, mark the continuous outcome as the target.
2. Mark explanatory variables as features.
3. In Step 2, choose `fit_new_model`.
4. Set `Target mode` to `Continuous`.
5. Set `Model type` to `Quantile Regression`.
6. In `Model Settings`, set `Quantile alpha`.
   - Use `0.50` for median.
   - Use higher values for upper-tail forecasts.
   - Use lower values for downside or lower-tail forecasts.
7. Run Step 3.
8. Review residual and forecast diagnostics.

### Outputs To Review

- continuous metrics
- residual summary
- actual-versus-predicted outputs
- model summary
- coefficient outputs

### Watchouts

- Quantile predictions are not mean predictions.
- The selected `alpha` must match the business question.
- Comparing quantile regression directly to OLS requires clear explanation of
  what each model estimates.

## 11. Tobit Regression

### What It Does

Tobit regression models continuous outcomes that are censored. Censoring means
the observed target is clipped at a lower or upper bound. A common example is a
loss variable that cannot go below zero.

### When To Use It

Use `tobit_regression` when:

- the target is continuous
- observations are censored at a known lower or upper value
- ordinary linear regression would be biased by a floor or ceiling

### GUI Walkthrough

1. In Step 1, mark the censored continuous outcome as the target.
2. Mark explanatory variables as features.
3. In Step 2, choose `fit_new_model`.
4. Set `Target mode` to `Continuous`.
5. Set `Model type` to `Tobit Regression`.
6. In `Model Settings`, set:
   - `Tobit left censor`
   - `Tobit right censor`, or leave blank if there is no right censor
7. Run Step 3.
8. Review residual diagnostics, actual-versus-predicted outputs, and model
   summary.

### Outputs To Review

- continuous metrics
- model summary
- residual diagnostics
- actual-versus-predicted charts

### Watchouts

- Censoring is not the same as missing data.
- Set censoring bounds based on the data-generating process, not just observed
  min/max values.
- If the target is bounded between 0 and 1 but not censored, beta regression may
  be more appropriate.

## 12. XGBoost

### What It Does

XGBoost is a gradient-boosted tree model. It can capture non-linear effects,
thresholds, and interactions that linear-style models may miss.

### When To Use It

Use `xgboost` when:

- the target is binary or continuous
- non-linear relationships are likely
- you want a powerful challenger model
- predictive performance is important enough to justify additional
  explainability work

### GUI Walkthrough

1. In Step 1, mark the target and feature roles.
2. In Step 2, choose `fit_new_model`.
3. Set `Target mode` to `Binary` or `Continuous`, depending on the target.
4. Set `Model type` to `XGBoost`.
5. In `Model Settings`, configure:
   - `XGBoost estimators`
   - `XGBoost learning rate`
   - `XGBoost max depth`
   - `XGBoost subsample`
   - `XGBoost colsample`
6. Keep explainability diagnostics enabled.
7. Use model comparison mode when XGBoost is a challenger to an interpretable
   baseline.
8. Run Step 3.
9. Review performance, feature importance, permutation importance, PDP/ICE/ALE,
   and scenario outputs.

### Outputs To Review

- split metrics
- feature importance
- permutation importance
- partial dependence and ICE outputs
- scenario outputs
- model comparison outputs if enabled

### Watchouts

- XGBoost is less transparent than logistic, probit, linear, or scorecard
  models.
- Strong performance should be paired with stronger explainability evidence.
- Deep trees can overfit; review train/test divergence and cross-validation.

## 13. GEE Logistic Regression

### What It Does

Generalized Estimating Equations logistic regression models a binary event while
accounting for repeated observations in groups such as borrower, loan, or
account. It estimates population-average effects and robust standard errors.

### When To Use It

Use `gee_logistic_regression` when:

- the target is binary
- the same entity appears multiple times
- clustered standard errors are important
- you want a panel-aware challenger to standard logistic regression

### GUI Walkthrough

1. In Step 1, mark the default/event field as the target.
2. Mark the borrower/account/loan identifier as an identifier.
3. In Step 2, set `Target mode` to `Binary`.
4. Set `Model type` to `GEE Logistic Regression`.
5. In `Model Settings`, set `GEE group column` to the repeated-observation ID
   when available.
6. Run Step 3 and compare results against regular logistic regression.

### Outputs To Review

- model summary
- robust coefficient table
- calibration, ROC/AUC, and KS outputs
- model comparison outputs if enabled

### Watchouts

- If no group column is selected, the adapter treats rows as independent.
- GEE effects are population-average effects, not subject-specific random
  effects.

## 14. Ridge Regression

### What It Does

Ridge regression is linear regression with L2 shrinkage. It stabilizes
coefficients when continuous predictors are correlated, but it usually keeps all
features in the model.

### When To Use It

Use `ridge_regression` for continuous LGD or forecasting targets when ordinary
linear regression has unstable coefficients caused by collinearity.

### GUI Walkthrough

1. In Step 1, mark the continuous target and feature roles.
2. In Step 2, set `Target mode` to `Continuous`.
3. Set `Model type` to `Ridge Regression`.
4. In `Model Settings`, tune `Regularization alpha`.
5. Run Step 3 and compare against linear regression.

### Outputs To Review

- continuous split metrics
- feature importance / coefficient magnitude
- residual diagnostics
- stability and cross-validation outputs when enabled

### Watchouts

- Ridge shrinks coefficients but does not remove features.
- High alpha values can make coefficients more stable but less responsive.

## 15. Lasso Regression

### What It Does

Lasso regression is linear regression with L1 regularization. It can shrink some
coefficients to zero and therefore acts as a feature-selection pressure.

### When To Use It

Use `lasso_regression` when the target is continuous and you want a simpler
linear model from a larger candidate feature set.

### GUI Walkthrough

1. Set the target to `Continuous`.
2. Choose `Lasso Regression`.
3. Tune `Regularization alpha`.
4. Review which coefficients remain non-zero.
5. Use challenger comparison before deciding whether to replace a simpler
   baseline.

### Outputs To Review

- selected coefficient table
- residual diagnostics
- RMSE, MAE, and R-squared
- feature stability outputs

### Watchouts

- A zero coefficient is a statistical result, not automatic business approval
  to remove the feature.
- Lasso can be unstable when predictors are highly correlated.

## 16. Elastic-Net Regression

### What It Does

Elastic-net regression combines L1 and L2 regularization for continuous targets.
It balances lasso-style feature selection with ridge-style coefficient
stability.

### When To Use It

Use `elastic_net_regression` when continuous predictors are numerous and
correlated, and you want both shrinkage and selection pressure.

### GUI Walkthrough

1. Set the target to `Continuous`.
2. Choose `Elastic-Net Regression`.
3. Tune `Regularization alpha`.
4. Tune `Elastic-net l1 ratio`.
5. Run Step 3 and review holdout performance and selected features.

### Outputs To Review

- coefficient table
- residual diagnostics
- continuous metrics
- stability and cross-validation outputs

### Watchouts

- `Regularization alpha` controls shrinkage strength.
- `Elastic-net l1 ratio` controls how much behavior is lasso-like versus
  ridge-like.

## 17. Fractional Logit

### What It Does

Fractional logit is a GLM-style bounded-response model for rates and
proportions in `[0, 1]`. It is often useful for LGD-style severity rates when
predictions should remain bounded.

### When To Use It

Use `fractional_logit` when:

- the target is continuous
- the values are rates or proportions
- predictions outside `[0, 1]` would be invalid

### GUI Walkthrough

1. Mark the bounded rate/proportion field as the target.
2. Set `Target mode` to `Continuous`.
3. Choose `Fractional Logit`.
4. Keep residual and actual-versus-predicted diagnostics enabled.
5. Run Step 3 and verify bounded prediction behavior.

### Outputs To Review

- model summary
- actual-versus-predicted charts
- residual diagnostics
- segment-level error tables

### Watchouts

- If exact zeros and ones are common and meaningful, consider
  `zero_one_inflated_beta`.
- The target should represent a true bounded proportion, not an arbitrary scaled
  number.

## 18. Zero-One Inflated Beta

### What It Does

Zero-one inflated beta modeling handles bounded severity targets with true mass
at exactly `0`, exactly `1`, and interior values between `0` and `1`.

### When To Use It

Use `zero_one_inflated_beta` for LGD data where:

- zero means a true no-loss or full-recovery outcome
- one means a true full-loss outcome
- interior values represent partial losses

### GUI Walkthrough

1. Mark the LGD/severity field as the target.
2. Set `Target mode` to `Continuous`.
3. Choose `Zero-One Inflated Beta`.
4. Keep LGD, residual, and actual-versus-predicted diagnostics enabled.
5. Run Step 3 and review boundary and interior model behavior.

### Outputs To Review

- boundary-model feature importance
- interior beta-model coefficients
- LGD segment diagnostics
- actual-versus-predicted outputs

### Watchouts

- The model needs enough interior observations.
- Only use it when zero and one values are real economic states, not coding
  artifacts.

## 19. Cox Proportional Hazards

### What It Does

Cox proportional hazards estimates the relationship between features and
time-to-event risk. The framework reports hazard-ratio style feature evidence.

### When To Use It

Use `cox_proportional_hazards` when:

- the target is a positive duration
- the goal is time-to-event risk ranking
- hazard-ratio interpretation is useful

### GUI Walkthrough

1. Create or select a positive duration target.
2. Set `Target mode` to `Continuous`.
3. Choose `Cox Proportional Hazards`.
4. Run Step 3 and review hazard ratios and ranking evidence.

### Outputs To Review

- hazard-ratio feature importance
- duration/risk score outputs
- residual and segment diagnostics

### Watchouts

- The current implementation assumes observed events because censoring
  indicators are not yet part of the model configuration.
- Do not use it when the target is not a positive duration.

## 20. AFT Survival Model

### What It Does

The AFT survival model fits a log-duration regression. It estimates how features
accelerate or decelerate expected time to event.

### When To Use It

Use `aft_survival_model` when:

- the target is a positive duration
- expected duration is easier to explain than hazard ratios
- a transparent time-to-event baseline is desired

### GUI Walkthrough

1. Create or select a positive duration target.
2. Set `Target mode` to `Continuous`.
3. Choose `AFT Survival Model`.
4. Run Step 3 and review predicted durations.

### Outputs To Review

- continuous metrics
- feature importance
- residual diagnostics
- actual-versus-predicted duration charts

### Watchouts

- The target is clipped to avoid invalid log durations.
- This is a practical log-duration baseline, not a full censored survival
  package.

## 21. Random Forest

### What It Does

Random forest fits many decision trees on bootstrapped samples and averages
their predictions. It can capture non-linear effects and interactions.

### When To Use It

Use `random_forest` as a non-linear challenger for binary or continuous targets
when interpretability is still supported through feature importance and
explainability outputs.

### GUI Walkthrough

1. Set the correct target mode.
2. Choose `Random Forest`.
3. In `Model Settings`, tune `Tree / EBM estimators` and `Tree / EBM max depth`.
4. Keep explainability diagnostics enabled.
5. Run Step 3 and compare against simpler baselines.

### Outputs To Review

- split metrics
- feature importance
- permutation importance
- PDP, ICE, and ALE outputs

### Watchouts

- It is less transparent than coefficient or scorecard models.
- Review train/test divergence and explainability evidence before using it as a
  preferred model.

## 22. Extra Trees

### What It Does

Extra Trees is a randomized tree ensemble. It is similar to random forest but
adds more randomness to split selection, often making it useful as a sensitivity
challenger.

### When To Use It

Use `extra_trees` when you want a fast non-linear sensitivity check or a second
tree-ensemble view beside random forest or XGBoost.

### GUI Walkthrough

1. Set the correct target mode.
2. Choose `Extra Trees`.
3. Tune `Tree / EBM estimators` and `Tree / EBM max depth`.
4. Keep explainability diagnostics enabled.
5. Run Step 3 and compare against the baseline and other challengers.

### Outputs To Review

- split metrics
- feature importance
- permutation importance
- PDP, ICE, and ALE outputs

### Watchouts

- Randomized split behavior can be harder to explain to validators.
- Use it as challenger evidence unless there is a strong governance reason to
  promote it.

## 23. Explainable Boosting Machine

### What It Does

The current `explainable_boosting_machine` option is an EBM-style shallow
boosted-tree adapter built with existing sklearn dependencies. It is intended to
provide a more transparent non-linear challenger when paired with PDP/ALE
outputs.

### When To Use It

Use it when:

- the target is binary or continuous
- non-linear shape effects are expected
- a shallow boosted-tree challenger is more useful than a deeper black-box model

### GUI Walkthrough

1. Set the correct target mode.
2. Choose `Explainable Boosting Machine`.
3. Tune `Tree / EBM estimators`, `Tree / EBM max depth`, and
   `XGBoost / EBM learning rate`.
4. Keep PDP, ICE, ALE, and feature-importance diagnostics enabled.
5. Run Step 3 and review whether the shape effects are business-sensible.

### Outputs To Review

- feature importance
- PDP, ICE, and ALE outputs
- calibration or residual diagnostics
- split metrics and model comparison outputs

### Watchouts

- This is not the external `interpret` package's native EBM implementation.
- True LightGBM is not included because this implementation pass does not add
  third-party packages.

## Choosing Between Similar Models

| If you are deciding between... | Usually start with... | Why |
| --- | --- | --- |
| Logistic vs scorecard | Logistic, then scorecard if points/WoE are required | Logistic is simpler; scorecard is better for governed points-based implementation. |
| Logistic vs elastic-net | Logistic, then elastic-net if features are numerous or correlated | Elastic-net helps when coefficient stability is weak. |
| Logistic vs probit | Logistic | Probit is usually a challenger, not the default. |
| Logistic vs GEE logistic | Logistic for independent rows, GEE for repeated entities | GEE is useful when clustered inference matters. |
| Linear vs beta | Beta if the target is a true bounded proportion | Linear can predict outside valid bounds. |
| Linear vs regularized linear | Linear first, then ridge/lasso/elastic-net if stability is weak | Regularization can improve stability but adds tuning judgment. |
| Fractional logit vs beta | Fractional logit when boundary handling matters less | Beta is best for mostly interior `(0, 1)` outcomes. |
| Beta vs zero-one inflated beta | Zero-one inflated beta if exact zero and one are true economic states | It models boundary mass explicitly. |
| Beta vs two-stage LGD | Two-stage LGD if there are many true zeros | Beta is better for mostly positive bounded severity. |
| Linear vs quantile | Quantile if the business question is percentile-specific | Linear estimates the mean; quantile estimates a percentile. |
| Linear vs Tobit | Tobit if the target is censored | Tobit explicitly handles censoring. |
| Panel vs linear | Panel if entity-time structure matters | Panel setup documents repeated observations over time. |
| Cox vs AFT | Cox for hazard ratios, AFT for duration estimates | They answer different time-to-event questions. |
| Interpretable model vs tree ensemble | Interpretable model first, tree models as challengers | This keeps governance evidence cleaner. |

## Recommended Review Pattern

After any run, review:

1. Step 5 decision summary.
2. `metadata/metrics.json`.
3. `reports/interactive_report.html`.
4. `model/feature_importance.csv`.
5. model-specific tables listed in the relevant section above.
6. `tables/governance/validation_checklist.*`.
7. warnings and `metadata/run_debug_trace.json`.

For high-stakes model-development work, use challenger comparison, cross
validation, robustness testing, feature policy checks, and explainability
outputs rather than relying on one metric from one model.
