# Glossary

This glossary defines common terms used in Quant Studio.

## App And Workflow Terms

| Term | Meaning |
| --- | --- |
| Quant Studio | Streamlit application and Python framework for model development, validation support, and documentation. |
| Workflow preset | Starting configuration for a use case such as PD Development, LGD Severity, Lifetime PD / CECL, or CCAR Forecasting. |
| Guided mode | Compact UI mode that keeps advanced controls on preset defaults. |
| Advanced mode | UI mode that exposes deeper configuration, governance, explainability, and documentation controls. |
| Execution mode | The run type: fit a new model, score an existing model, or search feature subsets. |
| Readiness Check | Step 3 review surface that consolidates validation issues before execution. |
| Artifact | Any output file or folder written by a run. |
| Configuration profile | Saved GUI setup that can be loaded in a later session without storing raw source data rows. |

## Data Terms

| Term | Meaning |
| --- | --- |
| Target source | Raw input column used to build the final modeled target. |
| Positive target value | Source value mapped to event/default `1` in binary modeling. |
| Feature | Predictor column used by the model. |
| Identifier | Record, account, customer, or loan ID used for traceability but usually excluded from modeling. |
| Date column | Time field used for splits, backtesting, time diagnostics, and panel/time-series workflows. |
| Panel data | Data with repeated observations for entities over time. |
| Time-series data | Data ordered by time where temporal validation matters. |
| Governed sample | Repeatable and documented sample used for development when full data is too large for interactive fitting. |
| Parquet | Columnar typed file format that is usually smaller and faster than CSV for large tabular data. |

## Modeling Terms

| Term | Meaning |
| --- | --- |
| PD | Probability of default. |
| LGD | Loss given default. |
| CECL | Current Expected Credit Losses. |
| CCAR | Comprehensive Capital Analysis and Review. |
| Challenger model | Alternative model compared against a baseline. |
| Scorecard | Transparent points-based model usually built from binned variables and WoE-style evidence. |
| Tobit | Regression approach for censored continuous outcomes. |
| Quantile regression | Model that estimates a conditional percentile instead of the conditional mean. |
| Discrete-time hazard model | Period-level binary model used for lifetime event timing. |
| XGBoost | Gradient-boosted tree model used as a non-linear challenger. |

## Metric And Test Terms

| Term | Meaning |
| --- | --- |
| AUC / ROC AUC | Discrimination metric measuring how well scores rank positives above negatives across thresholds. |
| ROC curve | Plot of true positive rate against false positive rate across thresholds. |
| KS | Maximum separation between positive and negative score distributions. |
| Average precision | Precision-recall ranking metric useful for imbalanced binary outcomes. |
| Brier score | Probability calibration error for binary predictions. |
| Calibration | Agreement between predicted probabilities and observed rates. |
| PSI | Population Stability Index, used to compare distribution shifts. |
| IV | Information Value, often used to summarize univariate predictive strength. |
| WoE | Weight of Evidence, a bin-level transformation common in scorecard development. |
| VIF | Variance Inflation Factor, used to detect multicollinearity. |
| ADF | Augmented Dickey-Fuller stationarity test. |
| Backtest | Review of model behavior on held-out or later-period data. |
| Cross-validation | Temporary fold-based refits used to assess performance stability. |

## Explainability Terms

| Term | Meaning |
| --- | --- |
| Coefficient breakdown | Signed contribution summary for linear-style models. |
| Feature importance | Ranking of features by model-specific importance or contribution. |
| Permutation importance | Importance measured by performance drop after shuffling a feature. |
| PDP | Partial Dependence Plot, showing average prediction effect as a feature changes. |
| ICE | Individual Conditional Expectation, showing feature effect curves for individual records. |
| Centered ICE | ICE curves centered to make shape comparison easier. |
| ALE | Accumulated Local Effects, an alternative effect plot that can behave better with correlated features. |
| Marginal effect | Estimated local change in prediction associated with a feature change. |
| Scenario test | Prediction comparison after applying defined feature shocks. |

## Artifact Terms

| Term | Meaning |
| --- | --- |
| `model/quant_model.joblib` | Saved model object exported by a fitted model run. |
| `config/run_config.json` | Effective configuration used for a run. |
| `reports/interactive_report.html` | Standalone formal visual report with grouped charts, companion diagnostics, interpretation badges, and table previews. |
| `artifact_manifest.json` | Machine-readable index of exported files and folders. |
| `metadata/run_debug_trace.json` | Step timing, status, shape snapshots, and failure details. |
| `reports/model_documentation_pack.md` | Model-development documentation summary. |
| `reports/validation_pack.md` | Validator-facing review summary and evidence index. |
| `model_bundle_for_monitoring/` | Handoff bundle for the separate ongoing-monitoring application. |
