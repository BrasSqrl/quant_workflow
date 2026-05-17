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
| Data Contract Scorecard | Step 1 review table that checks target role, date role, identifier role, missingness, duplicates, target distribution, and high-cardinality fields before configuration. |
| Potential Leakage Flags | Step 1 name-based warning table for fields that may describe default, loss, recovery, collections, or other post-outcome information. |
| Schema Fingerprint | Step 1 reproducibility table with deterministic hashes for the selected data source shape, column signature, and a small content sample. |
| Transformation Studio | Step 1 governed-transformation workspace with recommendations, recipe cards, a custom builder, pipeline validation, advanced generation controls, and the workbook-compatible advanced table editor. |
| Transformation Preview | Step 1 before/after preview for one configured or ad hoc transformation on a capped sample. |
| Readiness Check & Run | Step 3 surface that consolidates validation issues and starts execution. |
| Model Type Story Card | Step 2 guidance card that explains the selected model type, best-use cases, avoid conditions, key settings, outputs, and validation questions. |
| Model Suitability Explainer | Step 2 table that explains whether the selected model, target mode, data structure, sample size, event density, and transformation load look reasonable. |
| Configuration Risk Score | Step 2 pre-run complexity score that flags settings likely to increase instability, memory use, or review burden. |
| Runtime / Artifact Size Estimate | Step 2 directional planning panel for expected runtime band and output size based on current data and export settings. |
| Resource Planner / Run Cost Estimate | Step 3 memory, disk, Large Data Mode, checkpoint, high-cost-option, and report-visual review shown before execution. |
| Explain this output | Collapsed explanation panel on selected high-value charts and tables that explains what an output shows, how to read it, good/bad signals, and recommended action. |
| Binning Theater | Step 4 scorecard review surface for WoE buckets, IV, points, bin quality, and manual-bin override candidates. |
| Decision Room | Default Step 5 landing view that summarizes recommendation, attention items, top drivers, key artifacts, and next actions for a review meeting. |
| Glossary hover badge | Small in-app term badge with a hover definition for common modeling and validation terms. |
| Decision Summary | Step 5 synthesis surface that converts completed-run metrics, issues, feature drivers, and artifacts into a decision-ready scorecard. |
| LLM documentation package | Step 5 on-demand zip download with curated non-row-level evidence, model facts digest, section evidence maps, approved claims, documentation gaps, regulatory crosswalks, target document schema, evidence-strength policy, citable-evidence and do-not-cite indexes, completion rules, controlled vocabulary, draft-validation rules, operator prompt variants, citation rules, interpretation briefs, DOCX build instructions, figure placement manifest, section-specific evidence folders, capped HTML plus document-ready PNG chart assets, package build profile, quality rubric, table-of-contents drop zone, and review checklist for LLM-assisted model methodology drafting. |
| Individual images package | Step 5 on-demand zip download containing standalone chart PNG files, lightweight HTML chart files, one shared Plotly JavaScript file, and a figure manifest. |
| Feature lineage map | Step 5 and exported table that maps model terms back to source features, transformations, imputation, selection rationale, importance, and documentation fields. |
| Model development dossier | Exported Markdown narrative that ties purpose, data, target, feature governance, methodology, validation evidence, limitations, and key artifacts together. |
| Validation checklist | Step 5 and exported table that summarizes whether major review evidence areas are complete, attention-needed, or not applicable. |
| Evidence traceability map | Exported question-to-artifact map that tells reviewers which file or table answers each common review question. |
| Artifact | Any output file or folder written by a run. |
| Configuration profile | Saved GUI setup that can be loaded in a later session without storing raw source data rows. |
| Split strategy | Step 2 control that determines how rows are assigned to train, validation, and test splits. |
| Out-of-time split | Time-based validation or test design where holdout rows come from later calendar periods than training rows. |
| Date cutoff split | Split strategy where validation and/or test begin at explicit cutoff dates. |
| Explicit date-window split | Split strategy where train, validation, and test are defined by inclusive start/end date windows. |
| Custom split column | Split strategy where the input data already contains train, validation/val, or test/oot labels. |

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
| `reports/decision_summary.md` | Portable Markdown version of the Step 5 decision scorecard. |
| `artifact_manifest.json` | Machine-readable index of exported files and folders. |
| `metadata/run_debug_trace.json` | Step timing, status, shape snapshots, and failure details. |
| `reports/model_documentation_pack.md` | Model-development documentation summary. |
| `reports/model_development_dossier.md` | Audit-ready model-development narrative package. |
| `reports/validation_pack.md` | Validator-facing review summary and evidence index. |
| `tables/governance/validation_checklist.*` | Exported validation-review checklist in the input-driven tabular format. |
| `tables/governance/evidence_traceability_map.*` | Exported map from review questions to artifacts and diagnostic tables in the input-driven tabular format. |
| `tables/governance/feature_lineage_map.*` | Exported model-term to source-feature lineage map in the input-driven tabular format. |
| `model/feature_lineage_map.csv` | Direct CSV copy of the feature lineage map in the model folder. |
| `tables/governance/report_payload_audit.*` | Exported record of report charts kept, downsampled, or skipped by HTML payload limits in the input-driven tabular format. |
| `Download OM Package` | Step 5 on-demand zip download that creates the `model_bundle_for_monitoring` handoff bundle for the separate ongoing-monitoring application. |
