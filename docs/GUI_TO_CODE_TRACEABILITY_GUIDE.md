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
- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `src/quant_pd_framework/streamlit_ui/artifact_summary.py`
- `src/quant_pd_framework/streamlit_ui/config_builder.py`
- `src/quant_pd_framework/streamlit_ui/config_profiles.py`
- `src/quant_pd_framework/streamlit_ui/enterprise_workflow.py`
- `src/quant_pd_framework/streamlit_ui/data.py`
- `src/quant_pd_framework/streamlit_ui/error_guidance.py`
- `src/quant_pd_framework/streamlit_ui/results.py`
- `src/quant_pd_framework/streamlit_ui/run_execution.py`
- `src/quant_pd_framework/streamlit_ui/state.py`
- `src/quant_pd_framework/streamlit_ui/workflow_feedback.py`
- `src/quant_pd_framework/streamlit_ui/workspace.py`
- `src/quant_pd_framework/gui_support.py`
- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/config_serialization.py`
- `src/quant_pd_framework/checkpointing.py`
- `src/quant_pd_framework/stage_runner.py`
- `src/quant_pd_framework/run_stage.py`
- `src/quant_pd_framework/export_layout.py`
- `src/quant_pd_framework/diagnostics/assets.py`
- `src/quant_pd_framework/diagnostics/registry.py`
- `src/quant_pd_framework/decision_summary.py`
- `src/quant_pd_framework/orchestrator.py`
- `src/quant_pd_framework/steps/assumption_checks.py`
- `src/quant_pd_framework/steps/cross_validation.py`
- `src/quant_pd_framework/steps/transformations.py`

## Core Principle

The GUI collects user choices. `build_framework_config_from_editor(...)` turns
those choices into a `FrameworkConfig`. The GUI then uses
`CheckpointedWorkflowRunner` to run restartable stages; each stage delegates to
the same `QuantModelOrchestrator` step logic that a code-only user can run
directly.

Traceability chain:

`Streamlit widget -> GUIBuildInputs / editor table -> FrameworkConfig -> pipeline step -> artifact`

Execution and recovery chain:

`Run button -> run_execution.py -> CheckpointedWorkflowRunner -> run_stage.py -> QuantModelOrchestrator.run_context(...) -> build_run_snapshot -> workflow_feedback.py`

If execution fails, `error_guidance.py` classifies the exception into a likely
cause, recommended recovery action, and expandable technical traceback.

Enterprise UX chain:

`FrameworkConfig / run snapshot -> enterprise_workflow.py -> status, issues, preflight, diffs, model card, artifact explorer`

## 1. Data Source Controls

| GUI location | Main controls | Config or runtime target | Code path | Audit surface |
| --- | --- | --- | --- | --- |
| Step 1 `Dataset & Schema` data-source section | bundled sample, `Data_Load/` file selection, CSV/Excel/Parquet upload, `Large Data Mode` | runtime dataframe or file-backed `DatasetHandle` plus input-source metadata | `select_input_dataframe`, `list_data_load_files`, `load_data_load_dataframe`, `load_uploaded_dataframe_bytes`, `build_dataset_handle` | `data/input/input_snapshot.csv`, `data/input/input_snapshot.parquet`, `data/sample_development/`, `data/full_data_scoring/`, `metadata/large_data/`, `input_shape`, `input_source::*` rows in `metadata/reproducibility_manifest.json` |

Notes:

- The data-source controls do not modify model config directly.
- They determine the dataframe, file path, or `DatasetHandle` consumed by the
  checkpointed runner and downstream orchestrator stages.
- `Data_Load/` is a git-ignored landing-zone directory for CSV, Excel, and
  Parquet files.
  The app scans it on demand and exposes supported files in a dropdown.
- Data_Load CSV files can be converted to Parquet from the UI before selection
  when a more efficient local file format is preferred.
- When `Large Data Mode` is enabled, Data_Load files are previewed from disk
  and the full file is not loaded into pandas before execution.

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

The main workspace now includes dedicated sections beyond the column designer.

| Workspace tab | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Feature Dictionary` | `FeatureDictionaryConfig.entries` | `parse_feature_dictionary_frame(...)`, diagnostics feature-dictionary output | `feature_dictionary`, `reports/validation_pack.md` |
| `Transformations` | `TransformationConfig.transformations` | `parse_transformation_frame(...)`, `TransformationStep` | `governed_transformations`, `interaction_candidates` |
| `Template Workbook` | none directly; imports/exports editor tables | `build_template_workbook_bytes(...)`, `load_template_workbook(...)`, `streamlit_ui/workspace.py` | `config/configuration_template.xlsx` |

Implementation note:

- `app/streamlit_app.py` is now only the entrypoint.
- The live GUI flow is coordinated by `streamlit_ui/app_controller.py`.
- The build workspace and result surfaces now render one active section at a
  time instead of rendering every tab body on every rerun.

The transformation editor supports the expanded transformation families as values
of `TransformationSpec.transform_type`, including `box_cox`,
`natural_spline`, `piecewise_linear`, `difference`, `ewma`,
`rolling_median`, `rolling_min`, `rolling_max`, and `rolling_std`.

## 4. Workspace Mode

The GUI now includes a workspace-mode toggle that changes how much of the
configuration surface is editable.

| GUI control | Config or runtime target | Main implementation | Audit surface |
| --- | --- | --- | --- |
| `Workspace mode` | runtime-only UI state | `streamlit_ui/app_controller.py` Step 2 control gating advanced sections | indirect; preserved by the resolved preset-backed `config/run_config.json` |

Notes:

- `guided` mode keeps advanced controls on the current preset defaults.
- `advanced` mode unlocks comparison, review, explainability, and documentation
  controls directly in the Step 2 model-configuration workspace.
- The authoritative audit surface is still the resolved `FrameworkConfig`,
  not the UI mode itself.

## 5. Configuration Profiles

Step 2 includes a reusable profile manager for saving and restoring GUI setup
decisions across app launches.

| GUI control | Config or runtime target | Main implementation | Audit surface |
| --- | --- | --- | --- |
| `Save profile locally` | current validated `FrameworkConfig` plus workspace editor tables | `build_configuration_profile(...)`, `save_configuration_profile(...)` | `configs/saved_profiles/*.json` |
| `Download profile JSON` | same profile payload as the local save path | `profile_to_download_bytes(...)` | downloaded JSON profile |
| `Load selected saved profile` | session defaults and editor tables for the current workspace | `load_configuration_profile(...)`, `apply_configuration_profile_to_workspace(...)` | active profile message and pre-run readiness |
| `Load imported profile` | same as saved-profile loading, from uploaded JSON bytes | `load_configuration_profile(...)`, `apply_configuration_profile_to_workspace(...)` | active profile message and pre-run readiness |
| `Search profiles` | local profile metadata filter | `build_profile_library_frame(...)` | local profile JSON metadata |
| `Duplicate selected profile` | copied profile JSON with refreshed metadata | `duplicate_configuration_profile(...)` | new file under `configs/saved_profiles/` |
| `Delete selected profile` | local saved profile removal | `delete_configuration_profile(...)` | removed local profile JSON |
| `Compare selected profile` | current config vs selected profile config | `build_config_diff_frame(...)` | on-screen diff table |
| `Clear active profile defaults` | runtime-only profile state | `st.session_state` key `active_configuration_profile` | none; subsequent config preview uses current controls |

Profile contents:

- `framework_config` is the resolved configuration produced by the same builder
  path used for execution.
- `workspace_tables` stores the column designer, feature dictionary,
  transformations, manual review, and scorecard override editor tables.
- `dataset_fingerprint` stores source label, row count, columns, dtypes, schema
  hash, and input metadata.
- `metadata` stores profile name, tags, model purpose, target mode, model type,
  execution mode, notes, and creation timestamp.
- Raw input data rows are intentionally not stored in the profile.

Loading behavior:

- loading a profile applies the saved editor tables to the current dataset
  workspace
- loading clears the associated data-editor widget state so the restored tables
  render cleanly on the next rerun
- dataset differences are non-blocking but visible through missing-column,
  new-column, and row-count warnings
- saved profile JSON files are git-ignored under `configs/saved_profiles/`

## 6. Enterprise Workflow Status And Guidance

| GUI surface | Config or runtime target | Main implementation | Audit surface |
| --- | --- | --- | --- |
| `Workflow status` | current dataset availability, config preview, guardrail findings, and last-run snapshot | `build_workflow_step_states(...)`, `render_workflow_status_strip(...)` | runtime-only status table |
| `Guidance Library` | runtime-only help topics | `GUIDANCE_TOPICS`, `render_guidance_center(...)` | none |
| `Readiness Issue Center` | config build errors, guardrail findings, and profile mismatch warnings | `collect_readiness_issues(...)`, `render_issue_center(...)` | visible pre-run issue table |
| `Run Preflight Summary` | resolved `FrameworkConfig`, editor tables, and dataset shape | `build_preflight_summary(...)`, `render_preflight_summary(...)` | visible pre-run summary |
| `Configuration Diff Viewer` | current config vs active profile and last completed run | `build_config_diff_frame(...)` | visible diff table |

Notes:

- These surfaces do not change the model or pipeline.
- They make the current workflow state, remaining issues, and configuration
  changes more transparent before execution.
- The workflow status model has five stages: Dataset & Schema, Model
  Configuration, Readiness Check, Results & Artifacts, and Decision Summary.

## 7. Step 2 Group: Core Setup

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

## 8. Step 2 Group: Split Strategy

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Train size` | `SplitConfig.train_size` | `SplitStep` | `split_summary` |
| `Validation size` | `SplitConfig.validation_size` | `SplitStep` | `split_summary` |
| `Test size` | `SplitConfig.test_size` | `SplitStep` | `split_summary` |
| `Random state` | `SplitConfig.random_state` | `SplitStep`, sampling helpers | `config/run_config.json` |
| `Stratify cross-sectional split` | `SplitConfig.stratify` | `SplitStep._split_cross_sectional` | `config/run_config.json` |

Date and identifier columns are not collected here. They are derived from the
column designer role assignments.

## 9. Step 2 Group: Model Settings

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

## 10. Step 2 Group: Feature Subset Search

These controls only matter when `ExecutionConfig.mode` is
`search_feature_subsets`.

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Candidate features` | `FeatureSubsetSearchConfig.candidate_feature_names` | `build_subset_search_feature_options(...)`, `FeatureSubsetSearchStep` | `subset_search_scope`, `subset_search_candidates` |
| `Locked include features` | `FeatureSubsetSearchConfig.locked_include_features` | `FeatureSubsetSearchStep` | `subset_search_candidates` |
| `Locked exclude features` | `FeatureSubsetSearchConfig.locked_exclude_features` | `FeatureSubsetSearchStep` | `subset_search_candidates` |
| `Minimum subset size` | `FeatureSubsetSearchConfig.min_subset_size` | `FeatureSubsetSearchStep` | `subset_search_scope` |
| `Maximum subset size` | `FeatureSubsetSearchConfig.max_subset_size` | `FeatureSubsetSearchStep` | `subset_search_scope` |
| `Maximum candidate features` | `FeatureSubsetSearchConfig.max_candidate_features` | `FeatureSubsetSearchStep` | `subset_search_scope` |
| `Ranking split` | `FeatureSubsetSearchConfig.ranking_split` | `FeatureSubsetSearchStep` | `subset_search_candidates`, `subset_search_frontier` |
| `Ranking metric` | `FeatureSubsetSearchConfig.ranking_metric` | `FeatureSubsetSearchStep` | `subset_search_candidates`, `subset_search_frontier` |
| `Top candidates to retain` | `FeatureSubsetSearchConfig.top_candidate_count` | `FeatureSubsetSearchStep` | `subset_search_candidates`, `subset_search_feature_frequency`, `subset_search_significance_tests` |
| `Include paired significance tests ...` | `FeatureSubsetSearchConfig.include_significance_tests` | `FeatureSubsetSearchStep` | `subset_search_significance_tests` |

## 11. Step 2 Group: Data Preparation

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

## 12. Step 2 Group: Diagnostics & Exports

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Default segment column` | `DiagnosticConfig.default_segment_column` | `DiagnosticsStep._add_segment_outputs` |
| `Large Data Mode` | `PerformanceConfig.large_data_mode` | Data Source file-backed intake, safer GUI defaults, sampled diagnostics, and chunked full-data scoring |
| `Large-data diagnostic sample rows` | `PerformanceConfig.diagnostic_sample_rows` | diagnostics sampling helpers |
| `Memory warning threshold` | `PerformanceConfig.memory_limit_gb` | `IngestionStep._record_memory_estimate` |
| `Optimize dtypes during ingestion` | `PerformanceConfig.optimize_dtypes` | `IngestionStep._apply_large_data_controls` |
| `Capture memory profile in debug trace` | `PerformanceConfig.capture_memory_profile` | `QuantModelOrchestrator._memory_profile` |
| `Retain full diagnostic working dataframe` | `PerformanceConfig.retain_full_working_data` | `ImputationStep._build_working_data_snapshot` |
| `Max categorical levels before approval` | `PerformanceConfig.max_categorical_cardinality` | `FeatureEngineeringStep._profile_categorical_cardinality` |
| `Allow high-cardinality categorical model features` | `PerformanceConfig.allow_high_cardinality_categoricals` | `FeatureEngineeringStep._profile_categorical_cardinality` |
| `Convert CSV file-path inputs to Parquet before ingestion` | `PerformanceConfig.convert_csv_to_parquet` | `IngestionStep._read_file`, `convert_csv_to_parquet` |
| `CSV-to-Parquet chunk rows` | `PerformanceConfig.csv_conversion_chunk_rows` | chunked conversion helper |
| `Large-data training sample rows` | `PerformanceConfig.large_data_training_sample_rows` | sample-fit row cap for file-backed runs |
| `Full-data scoring chunk rows` | `PerformanceConfig.large_data_score_chunk_rows` | chunk size for `LargeDataFullScoringStep` |
| `Export individual figure HTML and PNG files` | `ArtifactConfig.export_individual_figure_files`; default `False` | `ArtifactExportStep._export_visualizations`; mirrors the report-grade visualization set when enabled |
| `Include enhanced report visuals` | `ArtifactConfig.include_enhanced_report_visuals`; default `True` | `presentation.enhance_report_visualizations`, live Results & Artifacts view, `reports/interactive_report.html`, optional individual figure exports |
| `Advanced Visual Analytics` | `ArtifactConfig.include_advanced_visual_analytics`; default `False` | `presentation.apply_advanced_visual_analytics`, live Results & Artifacts view, `reports/interactive_report.html`, optional individual figure exports |
| `Max points per report chart` | `PerformanceConfig.html_max_points_per_figure` | `report_payload.optimize_report_visualizations`, `ArtifactExportStep._build_report_visualizations`, `tables/governance/report_payload_audit.*` |
| `Max MB per report chart` | `PerformanceConfig.html_max_figure_payload_mb` | `report_payload.optimize_report_visualizations`, `ArtifactExportStep._build_report_visualizations`, `tables/governance/report_payload_audit.*` |
| `Max total report chart MB` | `PerformanceConfig.html_max_total_figure_payload_mb` | `report_payload.optimize_report_visualizations`, `ArtifactExportStep._build_report_visualizations`, `tables/governance/report_payload_audit.*` |
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
| `Enable cross-validation diagnostics` | `CrossValidationConfig.enabled` | `CrossValidationStep` |
| `Cross-validation folds` | `CrossValidationConfig.fold_count` | fold count for cross-validation diagnostics |
| `Cross-validation strategy` | `CrossValidationConfig.strategy` | stratified k-fold, k-fold, or expanding-window selection |
| `Shuffle cross-sectional folds` | `CrossValidationConfig.shuffle` | cross-sectional fold construction |
| `Export cross-validation metric views` | `CrossValidationConfig.metric_stability` | fold metric tables and charts |
| `Export cross-validation feature-stability views` | `CrossValidationConfig.coefficient_stability` | fold feature/coefficient stability tables and charts |
| `Enable scorecard workbench` | `ScorecardWorkbenchConfig.enabled` | `DiagnosticsStep._add_scorecard_workbench_outputs` |
| `Scorecard workbench features` | `ScorecardWorkbenchConfig.max_features` | scorecard workbench asset selection |
| `Include scorecard points distribution` | `ScorecardWorkbenchConfig.include_score_distribution` | scorecard workbench diagnostics |
| `Include reason-code frequency view` | `ScorecardWorkbenchConfig.include_reason_code_analysis` | scorecard workbench diagnostics |
| `Enable credit-risk development diagnostics` | `CreditRiskDiagnosticConfig.enabled` | `DiagnosticsStep._add_credit_risk_outputs` |
| `Vintage analysis` | `CreditRiskDiagnosticConfig.vintage_analysis` | credit-risk diagnostics |
| `Migration and delinquency transitions` | `CreditRiskDiagnosticConfig.migration_analysis`, `delinquency_transition_analysis` | credit-risk diagnostics |
| `Migration state column` | `CreditRiskDiagnosticConfig.migration_state_column` | explicit low-cardinality state field used for migration matrices; `(none)` skips transition diagnostics |
| `Cohort PD analysis` | `CreditRiskDiagnosticConfig.cohort_pd_analysis` | credit-risk diagnostics |
| `LGD segment and recovery views` | `CreditRiskDiagnosticConfig.lgd_segment_analysis`, `recovery_analysis` | credit-risk diagnostics |
| `Macro sensitivity` | `CreditRiskDiagnosticConfig.macro_sensitivity_analysis` | credit-risk diagnostics |
| `Macro features to stress` | `CreditRiskDiagnosticConfig.top_macro_features` | macro sensitivity diagnostics |
| `Top credit-risk segments` | `CreditRiskDiagnosticConfig.top_segments` | segment-limited credit diagnostics |
| `Macro shock std multiplier` | `CreditRiskDiagnosticConfig.shock_std_multiplier` | macro sensitivity diagnostics |

## 13. Step 2 Group: Challengers & Policies

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

## 14. Step 2 Group: Selection & Documentation

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

## 15. Step 2 Group: Governance & Review

| GUI control | Config field(s) | Main implementation | Export evidence |
| --- | --- | --- | --- |
| `Enable suitability checks` | `SuitabilityCheckConfig.enabled` | `AssumptionCheckStep` | `assumption_checks` |
| `Fail run on suitability failure` | `SuitabilityCheckConfig.error_on_failure` | `AssumptionCheckStep` | run failure if violated |
| `Min events per feature` | `SuitabilityCheckConfig.min_events_per_feature` | `AssumptionCheckStep` | `assumption_checks` |
| `Min/Max class rate` | `SuitabilityCheckConfig.min_class_rate`, `max_class_rate` | `AssumptionCheckStep` | `assumption_checks` |
| `Max dominant category share` | `SuitabilityCheckConfig.max_dominant_category_share` | `AssumptionCheckStep` | `assumption_checks` |
| Results `Suitability Checks` panel | `diagnostics_tables["assumption_checks"]` | `render_suitability_checks_panel(...)` | failure-first reviewer table with `interpretation`, `why_it_matters`, and `recommended_action` |
| `Enable workflow guardrails` | `WorkflowGuardrailConfig.enabled` | `build_framework_config_from_editor(...)`, `FrameworkConfig.validate`, `DiagnosticsStep._add_workflow_guardrail_outputs` | `workflow_guardrails` |
| `Block run on guardrail errors` | `WorkflowGuardrailConfig.fail_on_error` | `FrameworkConfig.validate` | run failure if violated |
| `Require preset documentation fields` | `WorkflowGuardrailConfig.enforce_documentation_requirements` | `workflow_guardrails.py` | `workflow_guardrails` |
| `Enable manual review workflow` | `ManualReviewConfig.enabled` | `VariableSelectionStep`, scorecard training path | `manual_review_feature_decisions` |
| `Require review decisions ...` | `ManualReviewConfig.require_review_complete` | `VariableSelectionStep` | run failure if incomplete |
| `Reviewer name` | `ManualReviewConfig.reviewer_name` | `VariableSelectionStep`, diagnostics | review tables |
| feature review editor rows | `ManualReviewConfig.feature_decisions` | `parse_manual_review_frames(...)`, `VariableSelectionStep` | `manual_review_feature_decisions` |
| scorecard override rows | `ManualReviewConfig.scorecard_bin_overrides` | `parse_manual_review_frames(...)`, `ScorecardLogisticRegressionAdapter` | `scorecard_bin_overrides` |

## 16. Step 2 Group: Explainability & Scenarios

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Enable explainability outputs` | `ExplainabilityConfig.enabled` | `DiagnosticsStep._add_explainability_outputs` |
| `Permutation importance` | `ExplainabilityConfig.permutation_importance` | permutation importance |
| `Feature effect curves` | `ExplainabilityConfig.feature_effect_curves` | feature effect curves |
| `Partial dependence plots` | `ExplainabilityConfig.partial_dependence` | PDP tables and figures |
| `ICE and centered ICE curves` | `ExplainabilityConfig.ice_curves`, `centered_ice_curves` | ICE tables and figures |
| `Accumulated local effects` | `ExplainabilityConfig.accumulated_local_effects` | ALE tables and figures |
| `2D feature effect heatmaps` | `ExplainabilityConfig.two_way_effects` | two-way effect surfaces |
| `Feature effect confidence bands` | `ExplainabilityConfig.effect_confidence_bands` | PDP bootstrap bands |
| `Feature effect monotonicity diagnostics` | `ExplainabilityConfig.monotonicity_diagnostics` | effect monotonicity table |
| `Segmented feature effects` | `ExplainabilityConfig.segmented_effects` | segment-level PDP curves |
| `Feature effect stability by split` | `ExplainabilityConfig.effect_stability` | split-level effect curves |
| `Average marginal effects` | `ExplainabilityConfig.marginal_effects` | finite-difference marginal effects |
| `Interaction strength tests` | `ExplainabilityConfig.interaction_strength` | interaction-strength summary |
| `Calibration by feature bucket` | `ExplainabilityConfig.effect_calibration` | feature-bucket calibration |
| `Coefficient breakdown` | `ExplainabilityConfig.coefficient_breakdown` | coefficient table |
| `Explainability top features` | `ExplainabilityConfig.top_n_features` | explainability ranking |
| `Effect curve grid points` | `ExplainabilityConfig.grid_points` | numeric effect grids |
| `Explainability sample size` | `ExplainabilityConfig.sample_size` | explainability sampling |
| `ICE sample size` | `ExplainabilityConfig.ice_sample_size` | ICE sampling |
| `Effect confidence resamples` | `ExplainabilityConfig.effect_band_resamples` | confidence-band resampling |
| `2D effect grid points` | `ExplainabilityConfig.two_way_grid_points` | two-way effect grid |
| `Max feature-effect segments` | `ExplainabilityConfig.max_effect_segments` | segmented effect limits |
| `Scenario evaluation split` | `ScenarioTestConfig.evaluation_split` | scenario testing |
| scenario editor rows | `ScenarioTestConfig.scenarios` | scenario testing |

## 17. Step 2 Group: Output Options

| GUI control | Config field(s) | Main implementation |
| --- | --- | --- |
| `Export profile` | `ArtifactConfig.export_profile` | `ArtifactExportStep`, `export_profiles.py` |
| `Export input snapshot` | `ArtifactConfig.export_input_snapshot` | `ArtifactExportStep._export_tabular_dataframe` |
| `Export code snapshot` | `ArtifactConfig.export_code_snapshot` | `ArtifactExportStep._export_code_snapshot` |
| `Compact prediction exports` | `ArtifactConfig.compact_prediction_exports` | `EvaluationStep._build_scored_frame`, `ArtifactExportStep._compact_prediction_frame` |
| `Tabular artifact format` | `ArtifactConfig.tabular_output_format`; effective format resolved by original Step 1 input suffix | CSV for non-Parquet inputs; Parquet for `.parquet` / `.pq` inputs in `ArtifactExportStep` |
| `Large tabular export policy` | `ArtifactConfig.large_data_export_policy` | full, sampled, or metadata-only tabular export policy |
| `Rows in sampled tabular exports` | `ArtifactConfig.large_data_sample_rows` | sampled CSV or Parquet output size, depending on the Step 1 input suffix |
| `Parquet compression` | `ArtifactConfig.parquet_compression` | Parquet writer compression |
| `Keep unconfigured columns` | `SchemaConfig.pass_through_unconfigured_columns` | `SchemaManagementStep` |
| `Export reproducibility manifest` | `ReproducibilityConfig.enabled` | `ArtifactExportStep` |
| `Capture git commit metadata` | `ReproducibilityConfig.capture_git_metadata` | `ArtifactExportStep` |
| `Tracked package names` | `ReproducibilityConfig.package_names` | `ArtifactExportStep` |
| `Artifact root` | `ArtifactConfig.output_root` | `ArtifactExportStep` |

Each run writes to a readable UTC timestamped folder under the artifact root,
for example `run_2026-04-24_15-42-10_UTC`.

The large-run audit surfaces are `config/run_config.json`,
`metadata/run_debug_trace.json`, `large_data_memory_estimate`,
`dtype_optimization`, optional
`csv_to_parquet_conversion`, `large_data_full_scoring_summary`,
`large_data_full_score_distribution`, and the `tabular_export_policy` metadata
recorded in the run context. The Streamlit results viewer may keep sampled
in-memory previews for large runs when `PerformanceConfig.lazy_streamlit_results`
is enabled; full-data predictions are written separately under
`data/full_data_scoring/`.

Compact prediction exports are applied during scoring, not only during final
file export. This keeps GUI session snapshots, diagnostics that use scored
outputs, and exported prediction files from carrying a duplicate copy of the
full feature matrix. Diagnostics that require model features read from
`context.split_frames` instead.

The GUI output-location table is assembled by
`streamlit_ui/artifact_summary.py`. It prioritizes the run folder, interactive
report, model object, run config, reproducibility manifest, debug trace,
predictions, monitoring bundle, and Large Data Mode folders before listing
secondary artifacts.

Artifact filenames and subdirectories are resolved through
`export_layout.py`, which keeps the export path contract centralized before
`ArtifactExportStep` writes files.

The diagnostics step also exports `diagnostic_registry`, built from
`diagnostics/registry.py`. This table records each major diagnostic family,
the config path that controls it, expected tables and figures, target-mode
limits, label requirements, large-data behavior, and whether the diagnostic was
emitted, disabled, or skipped for the run.

## 18. Run Button

The `Run Quant Model Workflow` button performs this chain:

1. collects widget state into `GUIBuildInputs`
2. converts the schema editor and inputs into `FrameworkConfig`
3. renders the pre-run readiness summary from that same resolved config
4. renders the execution-plan summary from `streamlit_ui/run_execution.py`
5. chooses the correct dataframe or file-backed `DatasetHandle`
6. creates a checkpointed run through `CheckpointedWorkflowRunner`
7. runs each major stage in a subprocess-backed checkpoint sequence
8. publishes live stage-flow events to `render_runtime_status(...)`
9. records per-stage and per-step debug timing, shape snapshots, and memory estimates
10. prunes checkpoint context files when `Keep all checkpoints` is off
11. copies the checkpoint manifest into the final metadata folder
12. stores a bounded snapshot for the result viewer

If a run fails, `streamlit_ui/error_guidance.py` maps the exception to a
user-facing recovery message while preserving the original traceback in the GUI.

Relevant code path:

- `build_framework_config_from_editor(...)`
- `execute_workflow(...)`
- `CheckpointedWorkflowRunner(config=config).run_all(run_input)`
- `python -m quant_pd_framework.run_stage --manifest ... --stage-id ...`
- `QuantModelOrchestrator.run_context(...)`
- `build_run_snapshot(...)`

The `Run checkpointed step-by-step` option uses the same stage runner, but it
calls `run_next_checkpoint_stage(...)` once per button click instead of running
all pending stages. The checkpoint manifest lives at
`checkpoints/checkpoint_manifest.json` and is copied to `metadata/` after export.
The Step 2 `Diagnostics & Exports` toggle `Keep all checkpoints` maps to
`ArtifactConfig.keep_all_checkpoints`. It defaults to off; when off, stale
`.joblib` context checkpoints are deleted after newer safe checkpoints are
written, while the manifest is retained for audit status.

The Run Status panel renders a `Checkpoint Flow` chart from each progress
event's `stages` payload. That payload is built from the same checkpoint
manifest that is exported with the run, so the UI stage colors and the
auditable manifest status are aligned.

For the stage-by-stage user-facing definition, see
[Checkpoint Stage Guide](./CHECKPOINT_STAGE_GUIDE.md).

## 19. Result-Viewer Filters

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

## 20. Results Artifact Explorer, Reviewer Workspace, And Model Card

| GUI surface | Config or runtime target | Main implementation | Audit surface |
| --- | --- | --- | --- |
| `Artifact Explorer` | completed run artifact path map | `build_artifact_explorer_frame(...)`, `render_artifact_locations(...)` | artifact paths and file downloads |
| `Reviewer / Approval Workspace` | reviewer name, approval status, notes, exceptions | `ReviewerRecord`, `render_reviewer_workspace(...)` | optional `review_workspace.json` in run folder |
| `Download model card` | run snapshot plus reviewer record | `build_model_card_markdown(...)` | downloaded Markdown model card |

Notes:

- The reviewer workspace is development-stage governance, not production
  monitoring.
- The model card is generated from the completed run snapshot and current
  reviewer inputs so it can be attached to model documentation or validation
  review material.

## 21. Step 5: Decision Summary

| GUI surface | Config or runtime target | Main implementation | Audit surface |
| --- | --- | --- | --- |
| `Decision Summary` | completed run snapshot, metrics, diagnostics, warnings, feature importance, and artifact paths | `decision_summary.build_decision_summary(...)`, `render_decision_summary(...)` | `reports/decision_summary.md` |
| `Download decision summary` | completed run snapshot | `decision_summary.build_decision_summary_markdown(...)` | downloaded Markdown scorecard |
| `Validation Checklist` tab | `diagnostics_tables["validation_checklist"]` | `validation_evidence.build_validation_checklist(...)`, `render_decision_summary(...)` | `tables/governance/validation_checklist.*` |
| `Traceability Map` tab | `diagnostics_tables["evidence_traceability_map"]` | `validation_evidence.build_evidence_traceability_map(...)`, `render_decision_summary(...)` | `tables/governance/evidence_traceability_map.*` |

Notes:

- Step 5 does not run a model, change configuration, or replace validation
  judgment.
- It synthesizes completed-run evidence into a recommendation, decision level,
  primary metrics, issue table, top feature drivers, validation checklist,
  evidence index, and traceability map.

## 22. Authoritative Export Files for Traceability

For an audit review, the most important files are:

- `config/run_config.json`
- `metadata/step_manifest.json`
- `artifact_manifest.json`
  This now includes `core_artifacts`, `directories`, interactive figure paths,
  regulator-ready report paths, the artifact index, and the rerun bundle map.
- `metadata/metrics.json`
- `reports/run_report.md`
- `reports/decision_summary.md`
- `reports/model_documentation_pack.md`
- `reports/validation_pack.md`
- `metadata/reproducibility_manifest.json`
- `tables/governance/validation_checklist.*`
- `tables/governance/evidence_traceability_map.*`
- `tables/governance/report_payload_audit.*`
- `config/configuration_template.xlsx`
- `reports/interactive_report.html`
- `tables/feature_subset_search/subset_search_candidates.csv`,
  `tables/feature_subset_search/subset_search_frontier.csv`,
  `tables/feature_subset_search/subset_search_feature_frequency.csv`, and
  `tables/feature_subset_search/subset_search_significance_tests.csv` when
  the third execution mode is used

Together these create the formal record of what the GUI settings actually
became in code.
