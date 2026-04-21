# GUI-to-Code Traceability Guide

This document maps the main Quant Studio GUI controls to:

1. the config fields they populate
2. the pipeline steps they influence
3. the code modules that implement the behavior
4. the exported artifacts that preserve the decision

The purpose is auditability. The GUI is intentionally thin. It does not own the
modeling logic.

Primary implementation files:

- `app/streamlit_app.py`
- `src/quant_pd_framework/gui_support.py`
- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/orchestrator.py`
- `src/quant_pd_framework/steps/assumption_checks.py`
- `src/quant_pd_framework/steps/transformations.py`

## Core Principle

The GUI collects user choices. `build_framework_config_from_editor(...)` turns
those choices into a `FrameworkConfig`. The orchestrator then runs the same
Python pipeline that a code-only user would run.

Traceability chain:

`Streamlit widget -> GUIBuildInputs / editor table -> FrameworkConfig -> pipeline step -> artifact`

## 1. Data Source Controls

| GUI location | Main controls | Config or runtime target | Code path | Audit surface |
| --- | --- | --- | --- | --- |
| `Data Source` expander | bundled sample toggle, CSV/Excel upload | runtime dataframe only | `select_input_dataframe`, `load_uploaded_dataframe` | `input_snapshot.csv`, `input_shape` |

Notes:

- The upload control does not modify model config directly.
- It determines the dataframe consumed by the orchestrator.

## 2. Column Designer

The column designer is the most important traceability surface because it
defines the modeled schema.

| Column designer field | Config field | Main downstream step(s) |
| --- | --- | --- |
| `enabled` | `ColumnSpec.enabled` | `SchemaManagementStep` |
| `source_name` | `ColumnSpec.source_name` | `SchemaManagementStep` |
| `name` | `ColumnSpec.name` | `SchemaManagementStep`, all later steps |
| `role` | `ColumnSpec.role` | `TargetConstructionStep`, `FeatureEngineeringStep`, `SplitStep` |
| `dtype` | `ColumnSpec.dtype` | `SchemaManagementStep` |
| `missing_value_policy` | `ColumnSpec.missing_value_policy` | `ImputationStep` |
| `missing_value_fill_value` | `ColumnSpec.missing_value_fill_value` | `ImputationStep` |
| `missing_value_group_columns` | `ColumnSpec.missing_value_group_columns` | `ImputationStep` |
| `create_missing_indicator` | `ColumnSpec.create_missing_indicator` | `ImputationStep` |
| `create_if_missing` | `ColumnSpec.create_if_missing` | `SchemaManagementStep` |
| `default_value` | `ColumnSpec.default_value` | `SchemaManagementStep` |
| `keep_source` | `ColumnSpec.keep_source` | `SchemaManagementStep` |

Main builder functions:

- `build_column_editor_frame(...)`
- `normalize_editor_frame(...)`
- `build_column_specs_from_editor(...)`
- `build_framework_config_from_editor(...)`

Current notable option families:

- `missing_value_policy` now includes `knn` and `iterative` in addition to the
  scalar and directional options
- the same column-designer surface still drives grouped scalar imputation and
  missingness-indicator generation

## 3. Build-Workspace Editors

The main workspace now includes dedicated tabs beyond the column designer.

| Workspace tab | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Feature Dictionary` | `FeatureDictionaryConfig.entries` | `parse_feature_dictionary_frame(...)`, diagnostics feature-dictionary output | `feature_dictionary`, `validation_pack.md` |
| `Transformations` | `TransformationConfig.transformations` | `parse_transformation_frame(...)`, `TransformationStep` | `governed_transformations`, `interaction_candidates` |
| `Template Workbook` | none directly; imports/exports editor tables | `build_template_workbook_bytes(...)`, `load_template_workbook(...)` | `configuration_template.xlsx` |

The transformation editor now supports the expanded roadmap families as values
of `TransformationSpec.transform_type`, including `box_cox`,
`natural_spline`, `piecewise_linear`, `difference`, `ewma`,
`rolling_median`, `rolling_min`, `rolling_max`, and `rolling_std`.

## 4. Workspace Mode

The GUI now includes a workspace-mode toggle that changes how much of the
configuration surface is editable.

| GUI control | Config or runtime target | Main implementation | Audit surface |
| --- | --- | --- | --- |
| `Workspace mode` | runtime-only UI state | `app/streamlit_app.py` sidebar control gating advanced expanders | indirect; preserved by the resolved preset-backed `run_config.json` |

Notes:

- `guided` mode keeps advanced controls on the current preset defaults.
- `advanced` mode unlocks comparison, review, explainability, and documentation
  controls directly in the sidebar.
- The authoritative audit surface is still the resolved `FrameworkConfig`,
  not the UI mode itself.

## 5. Sidebar Group: Core Setup

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Execution mode` | `ExecutionConfig.mode` | `orchestrator._resolve_execution_config`, `ModelTrainingStep` |
| `Existing model artifact path` | `ExecutionConfig.existing_model_path` | `ModelTrainingStep` |
| `Existing run config path` | `ExecutionConfig.existing_config_path` | `orchestrator._resolve_execution_config` |
| `Model type` | `ModelConfig.model_type` | `build_model_adapter(...)` |
| `Target mode` | `TargetConfig.mode` | `TargetConstructionStep`, `ModelConfig.validate(...)` |
| `Data structure` | `SplitConfig.data_structure` | `SplitStep` |
| `Output target name` | `TargetConfig.output_column` | `TargetConstructionStep` |
| `Positive target values` | `TargetConfig.positive_values` | `TargetConstructionStep` |
| `Drop source target column` | `TargetConfig.drop_source_column` | `TargetConstructionStep` |

Execution-mode meaning:

- `fit_new_model`
  Runs the standard development workflow.
- `score_existing_model`
  Loads a prior exported model artifact and scores new data.
- `search_feature_subsets`
  Runs the dedicated feature-subset-search workflow and exports only
  comparison-ready ranking evidence.

## 6. Sidebar Group: Split Strategy

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Train size` | `SplitConfig.train_size` | `SplitStep` | `split_summary` |
| `Validation size` | `SplitConfig.validation_size` | `SplitStep` | `split_summary` |
| `Test size` | `SplitConfig.test_size` | `SplitStep` | `split_summary` |
| `Random state` | `SplitConfig.random_state` | `SplitStep`, sampling helpers | `run_config.json` |
| `Stratify cross-sectional split` | `SplitConfig.stratify` | `SplitStep._split_cross_sectional` | `run_config.json` |

Date and identifier columns are not collected here. They are derived from the
column designer role assignments.

## 7. Sidebar Group: Model Settings

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Classification threshold` | `ModelConfig.threshold` | `EvaluationStep`, `DiagnosticsStep._add_threshold_outputs` |
| `Max iterations` | `ModelConfig.max_iter` | model adapters |
| `Inverse regularization (C)` | `ModelConfig.C` | logistic, elastic-net, LGD stage one |
| `Solver` | `ModelConfig.solver` | logistic-family and LGD stage one |
| `Class weight` | `ModelConfig.class_weight` | binary models where applicable |
| `Elastic-net l1 ratio` | `ModelConfig.l1_ratio` | `ElasticNetLogisticRegressionAdapter` |
| `Scorecard bins` | `ModelConfig.scorecard_bins` | `ScorecardLogisticRegressionAdapter` |
| `Scorecard monotonicity` | `ScorecardConfig.monotonicity` | scorecard bin optimization |
| `Scorecard min bin share` | `ScorecardConfig.min_bin_share` | scorecard bin optimization |
| `Scorecard base score` | `ScorecardConfig.base_score` | score scaling |
| `Scorecard PDO` | `ScorecardConfig.points_to_double_odds` | score scaling |
| `Scorecard odds reference` | `ScorecardConfig.odds_reference` | score scaling |
| `Reason code count` | `ScorecardConfig.reason_code_count` | prediction-side reason codes |
| `Quantile alpha` | `ModelConfig.quantile_alpha` | `QuantileRegressionAdapter` |
| `XGBoost ...` controls | `ModelConfig.xgboost_*` | `XGBoostAdapter` |
| `Tobit ...` controls | `ModelConfig.tobit_*` | `TobitRegressionAdapter` |

## 8. Sidebar Group: Feature Subset Search

These controls only matter when `ExecutionConfig.mode` is
`search_feature_subsets`.

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Candidate features` | `FeatureSubsetSearchConfig.candidate_feature_names` | `FeatureSubsetSearchStep` | `subset_search_scope`, `subset_search_candidates` |
| `Locked include features` | `FeatureSubsetSearchConfig.locked_include_features` | `FeatureSubsetSearchStep` | `subset_search_candidates` |
| `Locked exclude features` | `FeatureSubsetSearchConfig.locked_exclude_features` | `FeatureSubsetSearchStep` | `subset_search_candidates` |
| `Minimum subset size` | `FeatureSubsetSearchConfig.min_subset_size` | `FeatureSubsetSearchStep` | `subset_search_scope` |
| `Maximum subset size` | `FeatureSubsetSearchConfig.max_subset_size` | `FeatureSubsetSearchStep` | `subset_search_scope` |
| `Maximum candidate features` | `FeatureSubsetSearchConfig.max_candidate_features` | `FeatureSubsetSearchStep` | `subset_search_scope` |
| `Ranking split` | `FeatureSubsetSearchConfig.ranking_split` | `FeatureSubsetSearchStep` | `subset_search_candidates`, `subset_search_frontier` |
| `Ranking metric` | `FeatureSubsetSearchConfig.ranking_metric` | `FeatureSubsetSearchStep` | `subset_search_candidates`, `subset_search_frontier` |
| `Top candidates to retain` | `FeatureSubsetSearchConfig.top_candidate_count` | `FeatureSubsetSearchStep` | `subset_search_candidates`, `subset_search_feature_frequency`, `subset_search_significance_tests` |
| `Include paired significance tests ...` | `FeatureSubsetSearchConfig.include_significance_tests` | `FeatureSubsetSearchStep` | `subset_search_significance_tests` |

## 9. Sidebar Group: Data Preparation

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Trim string columns` | `CleaningConfig.trim_string_columns` | `CleaningStep` |
| `Treat blank strings as null` | `CleaningConfig.blank_strings_as_null` | `CleaningStep` |
| `Drop duplicate rows` | `CleaningConfig.drop_duplicate_rows` | `CleaningStep` |
| `Drop rows with missing target` | `CleaningConfig.drop_rows_with_missing_target` | `CleaningStep` |
| `Drop fully null feature columns` | `CleaningConfig.drop_all_null_feature_columns` | `CleaningStep` |
| `Create date-part features` | `FeatureEngineeringConfig.derive_date_parts` | `FeatureEngineeringStep` |
| `Drop raw date columns ...` | `FeatureEngineeringConfig.drop_raw_date_columns` | `FeatureEngineeringStep` |
| `Date parts` | `FeatureEngineeringConfig.date_parts` | `FeatureEngineeringStep` |

## 10. Sidebar Group: Diagnostics & Exports

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Default segment column` | `DiagnosticConfig.default_segment_column` | `DiagnosticsStep._add_segment_outputs` |
| `Diagnostic suites` | `DiagnosticConfig.*` booleans | `DiagnosticsStep.run(...)` |
| `Export surfaces` | `DiagnosticConfig.interactive_visualizations`, `static_image_exports`, `export_excel_workbook` | `ArtifactExportStep` |
| `Top features for analysis` | `DiagnosticConfig.top_n_features` | diagnostics feature ranking |
| `Top categories per chart` | `DiagnosticConfig.top_n_categories` | segment/effect chart limits |
| `Max rows rendered in plots` | `DiagnosticConfig.max_plot_rows` | sampling helpers |
| `Quantile bucket count` | `DiagnosticConfig.quantile_bucket_count` | backtesting, lift/gain, WoE/IV |
| `Calibration bin count` | `CalibrationConfig.bin_count` | calibration tables |
| `Calibration binning strategy` | `CalibrationConfig.strategy` | calibration tables |
| `Fit Platt scaling challenger` | `CalibrationConfig.platt_scaling` | calibration workflow |
| `Fit isotonic challenger` | `CalibrationConfig.isotonic_calibration` | calibration workflow |
| `Calibration ranking metric` | `CalibrationConfig.ranking_metric` | calibration method recommendation |
| `Model specification tests` | `DiagnosticConfig.model_specification_tests` | `DiagnosticsStep._add_model_specification_outputs` |
| `Forecasting statistical tests` | `DiagnosticConfig.forecasting_statistical_tests` | `DiagnosticsStep._add_forecasting_test_outputs` |
| `Enable robustness testing` | `RobustnessConfig.enabled` | `DiagnosticsStep._add_robustness_outputs` |
| `Robustness resamples` | `RobustnessConfig.resample_count` | robustness diagnostics |
| `Robustness sample fraction` | `RobustnessConfig.sample_fraction` | robustness diagnostics |
| `Sample with replacement` | `RobustnessConfig.sample_with_replacement` | robustness diagnostics |
| `Robustness evaluation split` | `RobustnessConfig.evaluation_split` | robustness diagnostics |
| `Export metric-stability views` | `RobustnessConfig.metric_stability` | robustness diagnostics |
| `Export coefficient-stability views` | `RobustnessConfig.coefficient_stability` | robustness diagnostics |
| `Enable scorecard workbench` | `ScorecardWorkbenchConfig.enabled` | `DiagnosticsStep._add_scorecard_workbench_outputs` |
| `Scorecard workbench features` | `ScorecardWorkbenchConfig.max_features` | scorecard workbench asset selection |
| `Include scorecard points distribution` | `ScorecardWorkbenchConfig.include_score_distribution` | scorecard workbench diagnostics |
| `Include reason-code frequency view` | `ScorecardWorkbenchConfig.include_reason_code_analysis` | scorecard workbench diagnostics |
| `Enable credit-risk development diagnostics` | `CreditRiskDiagnosticConfig.enabled` | `DiagnosticsStep._add_credit_risk_outputs` |
| `Vintage analysis` | `CreditRiskDiagnosticConfig.vintage_analysis` | credit-risk diagnostics |
| `Migration and delinquency transitions` | `CreditRiskDiagnosticConfig.migration_analysis`, `delinquency_transition_analysis` | credit-risk diagnostics |
| `Cohort PD analysis` | `CreditRiskDiagnosticConfig.cohort_pd_analysis` | credit-risk diagnostics |
| `LGD segment and recovery views` | `CreditRiskDiagnosticConfig.lgd_segment_analysis`, `recovery_analysis` | credit-risk diagnostics |
| `Macro sensitivity` | `CreditRiskDiagnosticConfig.macro_sensitivity_analysis` | credit-risk diagnostics |
| `Macro features to stress` | `CreditRiskDiagnosticConfig.top_macro_features` | macro sensitivity diagnostics |
| `Top credit-risk segments` | `CreditRiskDiagnosticConfig.top_segments` | segment-limited credit diagnostics |
| `Macro shock std multiplier` | `CreditRiskDiagnosticConfig.shock_std_multiplier` | macro sensitivity diagnostics |

## 11. Sidebar Group: Challengers & Policies

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Enable model comparison mode` | `ComparisonConfig.enabled` | `ModelComparisonStep` | `model_comparison`, `model_comparison_significance_tests` |
| `Challenger model families` | `ComparisonConfig.challenger_model_types` | `ModelComparisonStep` | `model_comparison`, `model_comparison_significance_tests` |
| `Comparison ranking metric` | `ComparisonConfig.ranking_metric` | `ModelComparisonStep` | `model_comparison`, `model_comparison_significance_tests` |
| `Enable feature policy checks` | `FeaturePolicyConfig.enabled` | `DiagnosticsStep._add_feature_policy_outputs` | `feature_policy_checks` |
| `Required features` | `FeaturePolicyConfig.required_features` | policy checks | `feature_policy_checks` |
| `Excluded features` | `FeaturePolicyConfig.excluded_features` | policy checks | `feature_policy_checks` |
| `Expected signs` | `FeaturePolicyConfig.expected_signs` | policy checks | `feature_policy_checks` |
| `Monotonic features` | `FeaturePolicyConfig.monotonic_features` | policy checks | `feature_policy_checks` |
| `Max missing %` | `FeaturePolicyConfig.max_missing_pct` | policy checks | `feature_policy_checks` |
| `Max VIF` | `FeaturePolicyConfig.max_vif` | policy checks | `feature_policy_checks` |
| `Minimum IV` | `FeaturePolicyConfig.minimum_information_value` | policy checks | `feature_policy_checks` |
| `Fail run on policy violation` | `FeaturePolicyConfig.error_on_violation` | policy checks | run failure if violated |

## 12. Sidebar Group: Selection & Documentation

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Enable variable selection` | `VariableSelectionConfig.enabled` | `VariableSelectionStep` |
| `Max selected features` | `VariableSelectionConfig.max_features` | `VariableSelectionStep` |
| `Minimum univariate score` | `VariableSelectionConfig.min_univariate_score` | `VariableSelectionStep` |
| `Correlation threshold` | `VariableSelectionConfig.correlation_threshold` | `VariableSelectionStep` |
| `Locked include features` | `VariableSelectionConfig.locked_include_features` | `VariableSelectionStep` |
| `Locked exclude features` | `VariableSelectionConfig.locked_exclude_features` | `VariableSelectionStep` |
| `Auto-screen interaction terms` | `TransformationConfig.auto_interactions_enabled` | `TransformationStep` |
| `Numeric-numeric interactions` | `TransformationConfig.include_numeric_numeric_interactions` | `TransformationStep` |
| `Categorical-numeric interactions` | `TransformationConfig.include_categorical_numeric_interactions` | `TransformationStep` |
| `Max auto interactions` | `TransformationConfig.max_auto_interactions` | `TransformationStep` |
| `Max categorical levels per feature` | `TransformationConfig.max_categorical_levels` | `TransformationStep` |
| `Min interaction score` | `TransformationConfig.min_interaction_score` | `TransformationStep` |
| `Imputation sensitivity testing` | `ImputationSensitivityConfig.enabled` | `DiagnosticsStep._add_imputation_sensitivity_outputs` |
| `Imputation sensitivity split` | `ImputationSensitivityConfig.evaluation_split` | `DiagnosticsStep._add_imputation_sensitivity_outputs` |
| `Alternative imputation policies` | `ImputationSensitivityConfig.alternative_policies` | `DiagnosticsStep._add_imputation_sensitivity_outputs` |
| `Sensitivity features` | `ImputationSensitivityConfig.max_features` | `DiagnosticsStep._add_imputation_sensitivity_outputs` |
| `Min train missing count` | `ImputationSensitivityConfig.min_missing_count` | `DiagnosticsStep._add_imputation_sensitivity_outputs` |
| `Multiple imputation with pooling` | `AdvancedImputationConfig.multiple_imputation_enabled` | `diagnostic_frameworks._add_imputation_framework_extensions` |
| `Multiple-imputation datasets` | `AdvancedImputationConfig.multiple_imputation_datasets` | `diagnostic_frameworks._add_imputation_framework_extensions` |
| `Multiple-imputation evaluation split` | `AdvancedImputationConfig.multiple_imputation_evaluation_split` | `diagnostic_frameworks._add_imputation_framework_extensions` |
| `Multiple-imputation feature cap` | `AdvancedImputationConfig.multiple_imputation_top_features` | `diagnostic_frameworks._add_imputation_framework_extensions` |
| `Export documentation pack` | `DocumentationConfig.enabled` | `ArtifactExportStep` |
| documentation text fields | `DocumentationConfig.*` | diagnostics metadata and documentation pack |
| `Export regulator-ready reports` | `RegulatoryReportConfig.enabled` | `ArtifactExportStep`, `reporting.py` |
| `Export DOCX reports` | `RegulatoryReportConfig.export_docx` | regulator-ready report export |
| `Export PDF reports` | `RegulatoryReportConfig.export_pdf` | regulator-ready report export |
| `Committee template name` | `RegulatoryReportConfig.committee_template_name` | committee-ready report export |
| `Validation template name` | `RegulatoryReportConfig.validation_template_name` | validation-ready report export |
| report section toggles | `RegulatoryReportConfig.include_*` | regulator-ready report assembly |

## 13. Sidebar Group: Governance & Review

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Enable suitability checks` | `SuitabilityCheckConfig.enabled` | `AssumptionCheckStep` | `assumption_checks` |
| `Fail run on suitability failure` | `SuitabilityCheckConfig.error_on_failure` | `AssumptionCheckStep` | run failure if violated |
| `Min events per feature` | `SuitabilityCheckConfig.min_events_per_feature` | `AssumptionCheckStep` | `assumption_checks` |
| `Min/Max class rate` | `SuitabilityCheckConfig.min_class_rate`, `max_class_rate` | `AssumptionCheckStep` | `assumption_checks` |
| `Max dominant category share` | `SuitabilityCheckConfig.max_dominant_category_share` | `AssumptionCheckStep` | `assumption_checks` |
| `Enable workflow guardrails` | `WorkflowGuardrailConfig.enabled` | `build_framework_config_from_editor(...)`, `FrameworkConfig.validate`, `DiagnosticsStep._add_workflow_guardrail_outputs` | `workflow_guardrails` |
| `Block run on guardrail errors` | `WorkflowGuardrailConfig.fail_on_error` | `FrameworkConfig.validate` | run failure if violated |
| `Require preset documentation fields` | `WorkflowGuardrailConfig.enforce_documentation_requirements` | `workflow_guardrails.py` | `workflow_guardrails` |
| `Enable manual review workflow` | `ManualReviewConfig.enabled` | `VariableSelectionStep`, scorecard training path | `manual_review_feature_decisions` |
| `Require review decisions ...` | `ManualReviewConfig.require_review_complete` | `VariableSelectionStep` | run failure if incomplete |
| `Reviewer name` | `ManualReviewConfig.reviewer_name` | `VariableSelectionStep`, diagnostics | review tables |
| feature review editor rows | `ManualReviewConfig.feature_decisions` | `parse_manual_review_frames(...)`, `VariableSelectionStep` | `manual_review_feature_decisions` |
| scorecard override rows | `ManualReviewConfig.scorecard_bin_overrides` | `parse_manual_review_frames(...)`, `ScorecardLogisticRegressionAdapter` | `scorecard_bin_overrides` |

## 14. Sidebar Group: Explainability & Scenarios

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Enable explainability outputs` | `ExplainabilityConfig.enabled` | `DiagnosticsStep._add_explainability_outputs` |
| `Permutation importance` | `ExplainabilityConfig.permutation_importance` | permutation importance |
| `Feature effect curves` | `ExplainabilityConfig.feature_effect_curves` | feature effect curves |
| `Coefficient breakdown` | `ExplainabilityConfig.coefficient_breakdown` | coefficient table |
| `Explainability top features` | `ExplainabilityConfig.top_n_features` | explainability ranking |
| `Effect curve grid points` | `ExplainabilityConfig.grid_points` | numeric effect grids |
| `Explainability sample size` | `ExplainabilityConfig.sample_size` | explainability sampling |
| `Scenario evaluation split` | `ScenarioTestConfig.evaluation_split` | scenario testing |
| scenario editor rows | `ScenarioTestConfig.scenarios` | scenario testing |

## 15. Sidebar Group: Output Options

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Keep unconfigured columns` | `SchemaConfig.pass_through_unconfigured_columns` | `SchemaManagementStep` |
| `Export reproducibility manifest` | `ReproducibilityConfig.enabled` | `ArtifactExportStep` |
| `Capture git commit metadata` | `ReproducibilityConfig.capture_git_metadata` | `ArtifactExportStep` |
| `Tracked package names` | `ReproducibilityConfig.package_names` | `ArtifactExportStep` |
| `Artifact root` | `ArtifactConfig.output_root` | `ArtifactExportStep` |

The current performance safeguards are preset-backed and serialized through
`PerformanceConfig`, but they are not directly edited in the sidebar yet.
Their main audit surface is `run_config.json`, plus the exported
`performance_hardening_actions` table when large-run limits are applied.

## 16. Run Button

The `Run Quant Model Workflow` button performs this chain:

1. collects widget state into `GUIBuildInputs`
2. converts the schema editor and inputs into `FrameworkConfig`
3. renders the pre-run readiness summary from that same resolved config
4. instantiates `QuantModelOrchestrator`
5. runs the full pipeline
6. stores a snapshot for the result viewer

Relevant code path:

- `build_framework_config_from_editor(...)`
- `QuantModelOrchestrator(config=config).run(dataframe)`
- `build_run_snapshot(...)`

## 17. Result-Viewer Filters

The controls under `Interactive Filters` do not change the model or rerun the
pipeline. They only change the live display of exported run outputs.

Feature-subset-search runs intentionally use a different result viewer. That
viewer emphasizes candidate ranking, ROC/KS comparison, frontier charts, and
comparison-only governance exports rather than prediction filtering.

Examples:

- split selector
- feature lens
- segment filter
- view depth
- chart/table display selection

These are presentation controls, not modeling controls.

## 18. Authoritative Export Files for Traceability

For an audit review, the most important files are:

- `run_config.json`
- `step_manifest.json`
- `artifact_manifest.json`
  This now includes `core_artifacts`, `directories`, interactive figure paths,
  regulator-ready report paths, and the rerun bundle map.
- `metrics.json`
- `run_report.md`
- `model_documentation_pack.md`
- `validation_pack.md`
- `reproducibility_manifest.json`
- `configuration_template.xlsx`
- `interactive_report.html`
- `subset_search_candidates.csv`, `subset_search_frontier.csv`,
  `subset_search_feature_frequency.csv`, and
  `subset_search_significance_tests.csv` when the third execution mode is used

Together these create the formal record of what the GUI settings actually
became in code.
