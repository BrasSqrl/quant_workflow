"""Configuration assembly helpers for the Streamlit controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_pd_framework import (
    AdvancedImputationConfig,
    ArtifactConfig,
    CleaningConfig,
    ComparisonConfig,
    DataStructure,
    DocumentationConfig,
    ExecutionMode,
    ExplainabilityConfig,
    ExportProfile,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FeatureSubsetSearchConfig,
    ImputationSensitivityConfig,
    LargeDataExportPolicy,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    RegulatoryReportConfig,
    ReproducibilityConfig,
    ScorecardConfig,
    ScorecardMonotonicity,
    SuitabilityCheckConfig,
    TabularOutputFormat,
    TargetMode,
    VariableSelectionConfig,
    WorkflowGuardrailConfig,
)
from quant_pd_framework.gui_support import (
    FEATURE_REVIEW_COLUMNS,
    SCORECARD_OVERRIDE_COLUMNS,
    GUIBuildInputs,
    build_framework_config_from_editor,
    parse_expected_signs,
    parse_feature_dictionary_frame,
    parse_manual_review_frames,
    parse_scenario_rows,
    parse_transformation_frame,
)
from quant_pd_framework.workflow_guardrails import evaluate_workflow_guardrails


def build_preview_configuration(
    *,
    edited_schema: pd.DataFrame,
    feature_dictionary_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    feature_review_frame: pd.DataFrame,
    scorecard_override_frame: pd.DataFrame,
    preset_inputs,
    control_values: dict[str, Any],
) -> tuple[Any, list[Any], str | None]:
    preview_config = None
    preview_error: str | None = None
    preview_findings: list[Any] = []
    values = control_values

    try:
        scenario_testing_config = parse_scenario_rows(
            values["scenario_rows"].to_dict(orient="records")
        )
        scenario_testing_config.evaluation_split = values["scenario_split"]

        transformation_config = parse_transformation_frame(transformation_frame)
        transformation_config.auto_interactions_enabled = values["auto_interactions_enabled"]
        transformation_config.include_numeric_numeric_interactions = values[
            "include_numeric_numeric_interactions"
        ]
        transformation_config.include_categorical_numeric_interactions = values[
            "include_categorical_numeric_interactions"
        ]
        transformation_config.max_auto_interactions = int(values["max_auto_interactions"])
        transformation_config.max_categorical_levels = int(values["max_categorical_levels"])
        transformation_config.min_interaction_score = float(values["min_interaction_score"])

        inputs = GUIBuildInputs(
            preset_name=None
            if values["selected_preset_name_value"] == "custom"
            else values["selected_preset"],
            model=ModelConfig(
                model_type=ModelType(values["model_type"]),
                max_iter=int(values["max_iter"]),
                C=float(values["inverse_regularization"]),
                solver=values["solver"],
                l1_ratio=float(values["l1_ratio"]),
                class_weight=None if values["class_weight"] == "none" else values["class_weight"],
                threshold=float(values["threshold"]),
                scorecard_bins=int(values["scorecard_bins"]),
                quantile_alpha=float(values["quantile_alpha"]),
                xgboost_n_estimators=int(values["xgboost_n_estimators"]),
                xgboost_learning_rate=float(values["xgboost_learning_rate"]),
                xgboost_max_depth=int(values["xgboost_max_depth"]),
                xgboost_subsample=float(values["xgboost_subsample"]),
                xgboost_colsample_bytree=float(values["xgboost_colsample_bytree"]),
                tobit_left_censoring=float(values["tobit_left_censoring"]),
                tobit_right_censoring=values["tobit_right_censoring"],
            ),
            cleaning=CleaningConfig(
                trim_string_columns=values["trim_string_columns"],
                blank_strings_as_null=values["blank_strings_as_null"],
                drop_duplicate_rows=values["drop_duplicate_rows"],
                drop_rows_with_missing_target=values["drop_rows_with_missing_target"],
                drop_all_null_feature_columns=values["drop_all_null_feature_columns"],
            ),
            feature_engineering=FeatureEngineeringConfig(
                derive_date_parts=values["derive_date_parts"],
                drop_raw_date_columns=values["drop_raw_date_columns"],
                date_parts=values["date_parts"],
            ),
            comparison=ComparisonConfig(
                enabled=(
                    values["execution_mode"] != ExecutionMode.SEARCH_FEATURE_SUBSETS.value
                    and values["comparison_enabled"]
                    and bool(values["challenger_model_types"])
                ),
                challenger_model_types=[
                    ModelType(challenger_type)
                    for challenger_type in values["challenger_model_types"]
                ],
                ranking_metric=None
                if values["ranking_metric"] == "auto"
                else values["ranking_metric"],
            ),
            subset_search=FeatureSubsetSearchConfig(
                enabled=values["execution_mode"] == ExecutionMode.SEARCH_FEATURE_SUBSETS.value,
                candidate_feature_names=values["subset_search_candidate_features"],
                locked_include_features=values["subset_search_locked_include"],
                locked_exclude_features=values["subset_search_locked_exclude"],
                min_subset_size=int(values["subset_search_min_subset_size"]),
                max_subset_size=int(values["subset_search_max_subset_size"]),
                max_candidate_features=int(values["subset_search_max_candidate_features"]),
                ranking_split=values["subset_search_ranking_split"],
                ranking_metric=values["subset_search_ranking_metric"],
                top_candidate_count=int(values["subset_search_top_candidate_count"]),
                top_curve_count=int(values["subset_search_top_curve_count"]),
                include_significance_tests=values["subset_search_include_significance_tests"],
            ),
            feature_policy=FeaturePolicyConfig(
                enabled=values["feature_policy_enabled"],
                required_features=[
                    value.strip()
                    for value in values["policy_required_features"].split(",")
                    if value.strip()
                ],
                excluded_features=[
                    value.strip()
                    for value in values["policy_excluded_features"].split(",")
                    if value.strip()
                ],
                max_missing_pct=float(values["policy_max_missing_pct"])
                if values["feature_policy_enabled"]
                else None,
                max_vif=float(values["policy_max_vif"])
                if values["feature_policy_enabled"]
                else None,
                minimum_information_value=float(values["policy_min_iv"])
                if values["feature_policy_enabled"]
                and TargetMode(values["target_mode"]) == TargetMode.BINARY
                else None,
                expected_signs=parse_expected_signs(values["policy_expected_signs"]),
                monotonic_features=parse_expected_signs(values["policy_monotonic_features"]),
                error_on_violation=values["policy_error_on_violation"],
            ),
            feature_dictionary=parse_feature_dictionary_frame(feature_dictionary_frame),
            advanced_imputation=AdvancedImputationConfig(
                enabled=True,
                knn_neighbors=int(preset_inputs.advanced_imputation.knn_neighbors),
                iterative_max_iter=int(preset_inputs.advanced_imputation.iterative_max_iter),
                iterative_random_state=int(values["random_state"]),
                iterative_sample_posterior=(
                    bool(preset_inputs.advanced_imputation.iterative_sample_posterior)
                    or values["multiple_imputation_enabled"]
                ),
                max_auxiliary_numeric_features=int(
                    preset_inputs.advanced_imputation.max_auxiliary_numeric_features
                ),
                minimum_complete_rows=int(preset_inputs.advanced_imputation.minimum_complete_rows),
                multiple_imputation_enabled=values["multiple_imputation_enabled"],
                multiple_imputation_datasets=int(values["multiple_imputation_datasets"]),
                multiple_imputation_evaluation_split=values["multiple_imputation_split"],
                multiple_imputation_top_features=int(values["multiple_imputation_top_features"]),
            ),
            transformations=transformation_config,
            manual_review=(
                parse_manual_review_frames(
                    feature_review_frame,
                    scorecard_override_frame,
                    reviewer_name=values["manual_reviewer_name"],
                    require_review_complete=values["manual_review_required"],
                )
                if values["manual_review_enabled"]
                else parse_manual_review_frames(
                    pd.DataFrame(columns=FEATURE_REVIEW_COLUMNS),
                    pd.DataFrame(columns=SCORECARD_OVERRIDE_COLUMNS),
                )
            ),
            suitability_checks=SuitabilityCheckConfig(
                enabled=values["suitability_checks_enabled"],
                min_events_per_feature=float(values["suitability_min_events_per_feature"])
                if values["suitability_checks_enabled"]
                and TargetMode(values["target_mode"]) == TargetMode.BINARY
                else None,
                min_class_rate=float(values["suitability_min_class_rate"])
                if values["suitability_checks_enabled"]
                and TargetMode(values["target_mode"]) == TargetMode.BINARY
                else None,
                max_class_rate=float(values["suitability_max_class_rate"])
                if values["suitability_checks_enabled"]
                and TargetMode(values["target_mode"]) == TargetMode.BINARY
                else None,
                max_dominant_category_share=float(values["suitability_max_dominant_category_share"])
                if values["suitability_checks_enabled"]
                else None,
                error_on_failure=values["suitability_error_on_failure"],
            ),
            workflow_guardrails=WorkflowGuardrailConfig(
                enabled=values["workflow_guardrails_enabled"],
                fail_on_error=values["workflow_guardrails_fail_on_error"],
                enforce_documentation_requirements=values["workflow_guardrails_require_docs"],
            ),
            explainability=ExplainabilityConfig(
                enabled=values["explainability_enabled"],
                permutation_importance=values["permutation_importance_enabled"],
                feature_effect_curves=values["feature_effect_curves_enabled"],
                partial_dependence=values["partial_dependence_enabled"],
                ice_curves=values["ice_curves_enabled"],
                centered_ice_curves=values["centered_ice_curves_enabled"],
                accumulated_local_effects=values["ale_enabled"],
                two_way_effects=values["two_way_effects_enabled"],
                effect_confidence_bands=values["effect_confidence_bands_enabled"],
                monotonicity_diagnostics=values["effect_monotonicity_enabled"],
                segmented_effects=values["segmented_effects_enabled"],
                effect_stability=values["effect_stability_enabled"],
                marginal_effects=values["marginal_effects_enabled"],
                interaction_strength=values["interaction_strength_enabled"],
                effect_calibration=values["effect_calibration_enabled"],
                coefficient_breakdown=values["coefficient_breakdown_enabled"],
                top_n_features=int(values["explainability_top_n"]),
                grid_points=int(values["explainability_grid_points"]),
                sample_size=int(values["explainability_sample_size"]),
                ice_sample_size=int(values["explainability_ice_sample_size"]),
                effect_band_resamples=int(values["effect_band_resamples"]),
                two_way_grid_points=int(values["two_way_grid_points"]),
                max_effect_segments=int(values["max_effect_segments"]),
            ),
            calibration=values["calibration_config"],
            scorecard=ScorecardConfig(
                monotonicity=ScorecardMonotonicity(values["scorecard_monotonicity"]),
                min_bin_share=float(values["scorecard_min_bin_share"]),
                base_score=int(values["scorecard_base_score"]),
                points_to_double_odds=int(values["scorecard_pdo"]),
                odds_reference=float(values["scorecard_odds_reference"]),
                reason_code_count=int(values["scorecard_reason_code_count"]),
            ),
            scorecard_workbench=values["scorecard_workbench_config"],
            imputation_sensitivity=ImputationSensitivityConfig(
                enabled=values["imputation_sensitivity_enabled"],
                evaluation_split=values["imputation_sensitivity_split"],
                alternative_policies=[
                    MissingValuePolicy(policy)
                    for policy in values["imputation_sensitivity_policies"]
                ]
                if values["imputation_sensitivity_enabled"]
                else [],
                max_features=int(values["imputation_sensitivity_max_features"]),
                min_missing_count=int(values["imputation_sensitivity_min_missing_count"]),
            ),
            variable_selection=VariableSelectionConfig(
                enabled=values["variable_selection_enabled"],
                max_features=int(values["variable_selection_max_features"])
                if values["variable_selection_enabled"]
                else None,
                min_univariate_score=float(values["variable_selection_min_univariate_score"])
                if values["variable_selection_enabled"]
                else None,
                correlation_threshold=float(values["variable_selection_correlation_threshold"])
                if values["variable_selection_enabled"]
                else None,
                locked_include_features=[
                    value.strip()
                    for value in values["variable_selection_locked_include"].split(",")
                    if value.strip()
                ],
                locked_exclude_features=[
                    value.strip()
                    for value in values["variable_selection_locked_exclude"].split(",")
                    if value.strip()
                ],
            ),
            documentation=DocumentationConfig(
                enabled=values["documentation_enabled"],
                model_name=values["documentation_model_name"].strip() or "Quant Studio Model",
                model_owner=values["documentation_model_owner"].strip(),
                business_purpose=values["documentation_business_purpose"].strip(),
                portfolio_name=values["documentation_portfolio_name"].strip(),
                segment_name=values["documentation_segment_name"].strip(),
                horizon_definition=values["documentation_horizon_definition"].strip(),
                target_definition=values["documentation_target_definition"].strip(),
                loss_definition=values["documentation_loss_definition"].strip(),
                assumptions=_parse_multiline_list(values["documentation_assumptions"]),
                exclusions=_parse_multiline_list(values["documentation_exclusions"]),
                limitations=_parse_multiline_list(values["documentation_limitations"]),
                reviewer_notes=values["documentation_reviewer_notes"].strip(),
            ),
            regulatory_reporting=RegulatoryReportConfig(
                enabled=values["regulatory_reporting_enabled"],
                export_docx=values["regulatory_export_docx"],
                export_pdf=values["regulatory_export_pdf"],
                committee_template_name=values["regulatory_committee_template"].strip()
                or "committee_standard",
                validation_template_name=values["regulatory_validation_template"].strip()
                or "validation_standard",
                include_assumptions_section=values["regulatory_include_assumptions"],
                include_challenger_section=values["regulatory_include_challengers"],
                include_scenario_section=values["regulatory_include_scenarios"],
                include_appendix_section=values["regulatory_include_appendix"],
            ),
            scenario_testing=scenario_testing_config,
            diagnostics=values["diagnostic_config"],
            credit_risk=values["credit_risk_config"],
            robustness=values["robustness_config"],
            cross_validation=values["cross_validation_config"],
            reproducibility=ReproducibilityConfig(
                enabled=values["reproducibility_enabled"],
                capture_git_metadata=values["reproducibility_capture_git"],
                package_names=[
                    value.strip()
                    for value in values["reproducibility_packages_text"].split(",")
                    if value.strip()
                ],
            ),
            performance=values.get("performance_config", preset_inputs.performance),
            artifacts=ArtifactConfig(
                output_root=Path(values["output_root"].strip() or "artifacts"),
                include_enhanced_report_visuals=values.get(
                    "include_enhanced_report_visuals",
                    preset_inputs.artifacts.include_enhanced_report_visuals,
                ),
                include_advanced_visual_analytics=values.get(
                    "include_advanced_visual_analytics",
                    preset_inputs.artifacts.include_advanced_visual_analytics,
                ),
                export_individual_figure_files=values["export_individual_figure_files"],
                export_input_snapshot=values["export_input_snapshot"],
                export_code_snapshot=values["export_code_snapshot"],
                export_profile=ExportProfile(values["export_profile"]),
                tabular_output_format=TabularOutputFormat(values["tabular_output_format"]),
                large_data_export_policy=LargeDataExportPolicy(values["large_data_export_policy"]),
                large_data_sample_rows=int(values["large_data_sample_rows"]),
                parquet_compression=values["parquet_compression"],
            ),
            data_structure=DataStructure(values["data_structure"]),
            train_size=float(values["train_size"]),
            validation_size=float(values["validation_size"]),
            test_size=float(values["test_size"]),
            random_state=int(values["random_state"]),
            stratify=values["stratify"],
            execution_mode=ExecutionMode(values["execution_mode"]),
            existing_model_path=Path(values["existing_model_path_text"].strip())
            if values["existing_model_path_text"].strip()
            else None,
            existing_config_path=Path(values["existing_config_path_text"].strip())
            if values["existing_config_path_text"].strip()
            else None,
            target_mode=TargetMode(values["target_mode"]),
            target_output_column=values["target_output_column"].strip() or "default_flag",
            positive_values_text=values["positive_values_text"],
            drop_target_source_column=values["drop_target_source_column"],
            pass_through_unconfigured_columns=values["pass_through_unconfigured_columns"],
            output_root=Path(values["output_root"].strip() or "artifacts"),
        )
        preview_config = build_framework_config_from_editor(
            edited_schema,
            inputs,
            validate=False,
        )
        if preview_config.workflow_guardrails.enabled:
            preview_findings = evaluate_workflow_guardrails(preview_config)
        preview_config.validate()
    except Exception as exc:
        preview_error = str(exc)

    return preview_config, preview_findings, preview_error


def _parse_multiline_list(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]
