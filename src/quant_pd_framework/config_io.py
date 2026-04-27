"""Helpers for reading saved framework configs back into dataclasses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import (
    AdvancedImputationConfig,
    ArtifactConfig,
    CalibrationConfig,
    CalibrationRankingMetric,
    CalibrationStrategy,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    ComparisonConfig,
    CreditRiskDiagnosticConfig,
    CrossValidationConfig,
    CrossValidationStrategy,
    DataStructure,
    DependencyDiagnosticConfig,
    DiagnosticConfig,
    DistributionDiagnosticConfig,
    DocumentationConfig,
    ExecutionConfig,
    ExecutionMode,
    ExplainabilityConfig,
    ExportProfile,
    FeatureDictionaryConfig,
    FeatureDictionaryEntry,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FeatureReviewDecision,
    FeatureReviewDecisionType,
    FeatureSubsetSearchConfig,
    FeatureWorkbenchConfig,
    FrameworkConfig,
    ImputationSensitivityConfig,
    LargeDataExportPolicy,
    ManualReviewConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    OutlierDiagnosticConfig,
    PerformanceConfig,
    PresetName,
    PresetRecommendationConfig,
    RegulatoryReportConfig,
    ReproducibilityConfig,
    ResidualDiagnosticConfig,
    RobustnessConfig,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    ScorecardBinOverride,
    ScorecardConfig,
    ScorecardMonotonicity,
    ScorecardWorkbenchConfig,
    SplitConfig,
    StructuralBreakConfig,
    SuitabilityCheckConfig,
    TabularOutputFormat,
    TargetConfig,
    TargetMode,
    TimeSeriesDiagnosticConfig,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
    VariableSelectionConfig,
    WorkflowGuardrailConfig,
)


def load_framework_config(source: str | Path | dict[str, Any]) -> FrameworkConfig:
    """Loads a saved config dictionary or JSON file into FrameworkConfig."""

    payload, base_path = _load_payload(source)
    config = FrameworkConfig(
        schema=_build_schema_config(payload.get("schema", {})),
        cleaning=CleaningConfig(**payload.get("cleaning", {})),
        feature_engineering=FeatureEngineeringConfig(**payload.get("feature_engineering", {})),
        target=_build_target_config(payload.get("target", {})),
        split=_build_split_config(payload.get("split", {})),
        preset_name=_build_preset_name(payload.get("preset_name")),
        execution=_build_execution_config(payload.get("execution", {}), base_path),
        model=_build_model_config(payload.get("model", {})),
        comparison=_build_comparison_config(payload.get("comparison", {})),
        subset_search=_build_feature_subset_search_config(payload.get("subset_search", {})),
        feature_policy=_build_feature_policy_config(payload.get("feature_policy", {})),
        feature_dictionary=_build_feature_dictionary_config(payload.get("feature_dictionary", {})),
        advanced_imputation=_build_advanced_imputation_config(
            payload.get("advanced_imputation", {})
        ),
        transformations=_build_transformation_config(payload.get("transformations", {})),
        manual_review=_build_manual_review_config(payload.get("manual_review", {})),
        suitability_checks=_build_suitability_check_config(payload.get("suitability_checks", {})),
        workflow_guardrails=_build_workflow_guardrail_config(
            payload.get("workflow_guardrails", {})
        ),
        explainability=_build_explainability_config(payload.get("explainability", {})),
        calibration=_build_calibration_config(payload.get("calibration", {})),
        scorecard=_build_scorecard_config(payload.get("scorecard", {})),
        scorecard_workbench=_build_scorecard_workbench_config(
            payload.get("scorecard_workbench", {})
        ),
        imputation_sensitivity=_build_imputation_sensitivity_config(
            payload.get("imputation_sensitivity", {})
        ),
        variable_selection=_build_variable_selection_config(payload.get("variable_selection", {})),
        documentation=_build_documentation_config(payload.get("documentation", {})),
        regulatory_reporting=_build_regulatory_report_config(
            payload.get("regulatory_reporting", {})
        ),
        scenario_testing=_build_scenario_test_config(payload.get("scenario_testing", {})),
        diagnostics=_build_diagnostic_config(payload.get("diagnostics", {})),
        distribution_diagnostics=_build_distribution_diagnostic_config(
            payload.get("distribution_diagnostics", {})
        ),
        residual_diagnostics=_build_residual_diagnostic_config(
            payload.get("residual_diagnostics", {})
        ),
        outlier_diagnostics=_build_outlier_diagnostic_config(
            payload.get("outlier_diagnostics", {})
        ),
        dependency_diagnostics=_build_dependency_diagnostic_config(
            payload.get("dependency_diagnostics", {})
        ),
        time_series_diagnostics=_build_time_series_diagnostic_config(
            payload.get("time_series_diagnostics", {})
        ),
        structural_breaks=_build_structural_break_config(payload.get("structural_breaks", {})),
        feature_workbench=_build_feature_workbench_config(payload.get("feature_workbench", {})),
        preset_recommendations=_build_preset_recommendation_config(
            payload.get("preset_recommendations", {})
        ),
        credit_risk=_build_credit_risk_diagnostic_config(payload.get("credit_risk", {})),
        robustness=_build_robustness_config(payload.get("robustness", {})),
        cross_validation=_build_cross_validation_config(payload.get("cross_validation", {})),
        reproducibility=_build_reproducibility_config(payload.get("reproducibility", {})),
        performance=_build_performance_config(payload.get("performance", {})),
        artifacts=_build_artifact_config(payload.get("artifacts", {})),
    )
    config.validate()
    return config


def _load_payload(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, dict):
        return source, None

    source_path = Path(source)
    with source_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), source_path.resolve().parent


def _build_schema_config(payload: dict[str, Any]) -> SchemaConfig:
    column_specs = [
        ColumnSpec(
            name=spec["name"],
            source_name=spec.get("source_name"),
            enabled=spec.get("enabled", True),
            dtype=spec.get("dtype"),
            role=ColumnRole(spec.get("role", ColumnRole.FEATURE.value)),
            missing_value_policy=MissingValuePolicy(
                spec.get("missing_value_policy", MissingValuePolicy.INHERIT_DEFAULT.value)
            ),
            missing_value_fill_value=spec.get("missing_value_fill_value"),
            missing_value_group_columns=spec.get("missing_value_group_columns", []),
            create_missing_indicator=spec.get("create_missing_indicator", False),
            create_if_missing=spec.get("create_if_missing", False),
            default_value=spec.get("default_value"),
            keep_source=spec.get("keep_source", False),
        )
        for spec in payload.get("column_specs", [])
    ]
    return SchemaConfig(
        column_specs=column_specs,
        pass_through_unconfigured_columns=payload.get("pass_through_unconfigured_columns", True),
    )


def _build_target_config(payload: dict[str, Any]) -> TargetConfig:
    return TargetConfig(
        source_column=payload["source_column"],
        mode=TargetMode(payload.get("mode", TargetMode.BINARY.value)),
        output_column=payload.get("output_column", "default_flag"),
        positive_values=payload.get("positive_values"),
        drop_source_column=payload.get("drop_source_column", False),
    )


def _build_preset_name(value: str | None) -> PresetName | None:
    if not value:
        return None
    return PresetName(value)


def _build_split_config(payload: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        data_structure=DataStructure(
            payload.get("data_structure", DataStructure.CROSS_SECTIONAL.value)
        ),
        train_size=payload.get("train_size", 0.6),
        validation_size=payload.get("validation_size", 0.2),
        test_size=payload.get("test_size", 0.2),
        random_state=payload.get("random_state", 42),
        stratify=payload.get("stratify", True),
        date_column=payload.get("date_column"),
        entity_column=payload.get("entity_column"),
    )


def _build_execution_config(payload: dict[str, Any], base_path: Path | None) -> ExecutionConfig:
    return ExecutionConfig(
        mode=ExecutionMode(payload.get("mode", ExecutionMode.FIT_NEW_MODEL.value)),
        existing_model_path=_resolve_optional_path(payload.get("existing_model_path"), base_path),
        existing_config_path=_resolve_optional_path(payload.get("existing_config_path"), base_path),
    )


def _build_model_config(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        model_type=ModelType(payload.get("model_type", ModelType.LOGISTIC_REGRESSION.value)),
        max_iter=payload.get("max_iter", 1000),
        C=payload.get("C", 1.0),
        solver=payload.get("solver", "lbfgs"),
        l1_ratio=payload.get("l1_ratio", 0.5),
        class_weight=payload.get("class_weight", "balanced"),
        threshold=payload.get("threshold", 0.5),
        scorecard_bins=payload.get("scorecard_bins", 5),
        beta_clip_epsilon=payload.get("beta_clip_epsilon", 1e-4),
        lgd_positive_threshold=payload.get("lgd_positive_threshold", 1e-6),
        quantile_alpha=payload.get("quantile_alpha", 0.5),
        xgboost_n_estimators=payload.get("xgboost_n_estimators", 300),
        xgboost_learning_rate=payload.get("xgboost_learning_rate", 0.05),
        xgboost_max_depth=payload.get("xgboost_max_depth", 4),
        xgboost_subsample=payload.get("xgboost_subsample", 0.9),
        xgboost_colsample_bytree=payload.get("xgboost_colsample_bytree", 0.9),
        tobit_left_censoring=payload.get("tobit_left_censoring", 0.0),
        tobit_right_censoring=payload.get("tobit_right_censoring"),
    )


def _build_comparison_config(payload: dict[str, Any]) -> ComparisonConfig:
    return ComparisonConfig(
        enabled=payload.get("enabled", False),
        challenger_model_types=[
            ModelType(value) for value in payload.get("challenger_model_types", [])
        ],
        ranking_metric=payload.get("ranking_metric"),
        ranking_split=payload.get("ranking_split", "validation"),
    )


def _build_feature_subset_search_config(
    payload: dict[str, Any],
) -> FeatureSubsetSearchConfig:
    return FeatureSubsetSearchConfig(
        enabled=payload.get("enabled", False),
        candidate_feature_names=payload.get("candidate_feature_names", []),
        locked_include_features=payload.get("locked_include_features", []),
        locked_exclude_features=payload.get("locked_exclude_features", []),
        min_subset_size=payload.get("min_subset_size", 1),
        max_subset_size=payload.get("max_subset_size", 4),
        max_candidate_features=payload.get("max_candidate_features", 12),
        ranking_split=payload.get("ranking_split", "validation"),
        ranking_metric=payload.get("ranking_metric", "roc_auc"),
        top_candidate_count=payload.get("top_candidate_count", 25),
        top_curve_count=payload.get("top_curve_count", 5),
        include_significance_tests=payload.get("include_significance_tests", True),
    )


def _build_feature_policy_config(payload: dict[str, Any]) -> FeaturePolicyConfig:
    return FeaturePolicyConfig(
        enabled=payload.get("enabled", False),
        required_features=payload.get("required_features", []),
        excluded_features=payload.get("excluded_features", []),
        max_missing_pct=payload.get("max_missing_pct"),
        max_vif=payload.get("max_vif"),
        minimum_information_value=payload.get("minimum_information_value"),
        expected_signs=payload.get("expected_signs", {}),
        monotonic_features=payload.get("monotonic_features", {}),
        error_on_violation=payload.get("error_on_violation", False),
    )


def _build_feature_dictionary_config(payload: dict[str, Any]) -> FeatureDictionaryConfig:
    return FeatureDictionaryConfig(
        enabled=payload.get("enabled", False),
        require_documentation_for_selected_features=payload.get(
            "require_documentation_for_selected_features", False
        ),
        entries=[
            FeatureDictionaryEntry(
                feature_name=entry.get("feature_name", ""),
                business_name=entry.get("business_name", ""),
                definition=entry.get("definition", ""),
                source_system=entry.get("source_system", ""),
                unit=entry.get("unit", ""),
                allowed_range=entry.get("allowed_range", ""),
                missingness_meaning=entry.get("missingness_meaning", ""),
                expected_sign=entry.get("expected_sign", ""),
                inclusion_rationale=entry.get("inclusion_rationale", ""),
                notes=entry.get("notes", ""),
                enabled=entry.get("enabled", True),
            )
            for entry in payload.get("entries", [])
        ],
    )


def _build_advanced_imputation_config(payload: dict[str, Any]) -> AdvancedImputationConfig:
    return AdvancedImputationConfig(
        enabled=payload.get("enabled", True),
        knn_neighbors=payload.get("knn_neighbors", 5),
        iterative_max_iter=payload.get("iterative_max_iter", 10),
        iterative_random_state=payload.get("iterative_random_state", 42),
        iterative_sample_posterior=payload.get("iterative_sample_posterior", False),
        max_auxiliary_numeric_features=payload.get("max_auxiliary_numeric_features", 25),
        minimum_complete_rows=payload.get("minimum_complete_rows", 20),
        multiple_imputation_enabled=payload.get("multiple_imputation_enabled", False),
        multiple_imputation_datasets=payload.get("multiple_imputation_datasets", 5),
        multiple_imputation_evaluation_split=payload.get(
            "multiple_imputation_evaluation_split",
            "test",
        ),
        multiple_imputation_top_features=payload.get("multiple_imputation_top_features", 20),
    )


def _build_transformation_config(payload: dict[str, Any]) -> TransformationConfig:
    return TransformationConfig(
        enabled=payload.get("enabled", False),
        error_on_failure=payload.get("error_on_failure", True),
        auto_interactions_enabled=payload.get("auto_interactions_enabled", False),
        include_numeric_numeric_interactions=payload.get(
            "include_numeric_numeric_interactions",
            True,
        ),
        include_categorical_numeric_interactions=payload.get(
            "include_categorical_numeric_interactions",
            False,
        ),
        max_auto_interactions=payload.get("max_auto_interactions", 5),
        max_categorical_levels=payload.get("max_categorical_levels", 3),
        min_interaction_score=payload.get("min_interaction_score", 0.0),
        transformations=[
            TransformationSpec(
                transform_type=TransformationType(
                    entry.get("transform_type", TransformationType.WINSORIZE.value)
                ),
                source_feature=entry.get("source_feature", ""),
                output_feature=entry.get("output_feature"),
                secondary_feature=entry.get("secondary_feature"),
                categorical_value=entry.get("categorical_value"),
                lower_quantile=entry.get("lower_quantile"),
                upper_quantile=entry.get("upper_quantile"),
                parameter_value=entry.get("parameter_value"),
                window_size=entry.get("window_size"),
                lag_periods=entry.get("lag_periods"),
                bin_edges=[float(value) for value in entry.get("bin_edges", [])],
                enabled=entry.get("enabled", True),
                generated_automatically=entry.get("generated_automatically", False),
                notes=entry.get("notes", ""),
            )
            for entry in payload.get("transformations", [])
        ],
    )


def _build_manual_review_config(payload: dict[str, Any]) -> ManualReviewConfig:
    return ManualReviewConfig(
        enabled=payload.get("enabled", False),
        reviewer_name=payload.get("reviewer_name", ""),
        require_review_complete=payload.get("require_review_complete", False),
        feature_decisions=[
            FeatureReviewDecision(
                feature_name=entry.get("feature_name", ""),
                decision=FeatureReviewDecisionType(
                    entry.get("decision", FeatureReviewDecisionType.APPROVE.value)
                ),
                rationale=entry.get("rationale", ""),
            )
            for entry in payload.get("feature_decisions", [])
        ],
        scorecard_bin_overrides=[
            ScorecardBinOverride(
                feature_name=entry.get("feature_name", ""),
                bin_edges=[float(value) for value in entry.get("bin_edges", [])],
                rationale=entry.get("rationale", ""),
            )
            for entry in payload.get("scorecard_bin_overrides", [])
        ],
    )


def _build_suitability_check_config(payload: dict[str, Any]) -> SuitabilityCheckConfig:
    return SuitabilityCheckConfig(
        enabled=payload.get("enabled", True),
        min_events_per_feature=payload.get("min_events_per_feature", 10.0),
        min_class_rate=payload.get("min_class_rate", 0.01),
        max_class_rate=payload.get("max_class_rate", 0.99),
        max_dominant_category_share=payload.get("max_dominant_category_share", 0.98),
        min_non_null_target_rows=payload.get("min_non_null_target_rows", 30),
        error_on_failure=payload.get("error_on_failure", False),
    )


def _build_workflow_guardrail_config(payload: dict[str, Any]) -> WorkflowGuardrailConfig:
    return WorkflowGuardrailConfig(
        enabled=payload.get("enabled", True),
        fail_on_error=payload.get("fail_on_error", True),
        enforce_documentation_requirements=payload.get(
            "enforce_documentation_requirements",
            True,
        ),
    )


def _build_explainability_config(payload: dict[str, Any]) -> ExplainabilityConfig:
    return ExplainabilityConfig(
        enabled=payload.get("enabled", True),
        permutation_importance=payload.get("permutation_importance", True),
        feature_effect_curves=payload.get("feature_effect_curves", True),
        partial_dependence=payload.get("partial_dependence", True),
        ice_curves=payload.get("ice_curves", True),
        centered_ice_curves=payload.get("centered_ice_curves", True),
        accumulated_local_effects=payload.get("accumulated_local_effects", True),
        two_way_effects=payload.get("two_way_effects", True),
        effect_confidence_bands=payload.get("effect_confidence_bands", True),
        monotonicity_diagnostics=payload.get("monotonicity_diagnostics", True),
        segmented_effects=payload.get("segmented_effects", True),
        effect_stability=payload.get("effect_stability", True),
        marginal_effects=payload.get("marginal_effects", True),
        interaction_strength=payload.get("interaction_strength", True),
        effect_calibration=payload.get("effect_calibration", True),
        coefficient_breakdown=payload.get("coefficient_breakdown", True),
        top_n_features=payload.get("top_n_features", 5),
        grid_points=payload.get("grid_points", 12),
        sample_size=payload.get("sample_size", 2000),
        ice_sample_size=payload.get("ice_sample_size", 250),
        effect_band_resamples=payload.get("effect_band_resamples", 20),
        two_way_grid_points=payload.get("two_way_grid_points", 6),
        max_effect_segments=payload.get("max_effect_segments", 4),
    )


def _build_calibration_config(payload: dict[str, Any]) -> CalibrationConfig:
    return CalibrationConfig(
        bin_count=payload.get("bin_count", 10),
        strategy=CalibrationStrategy(payload.get("strategy", CalibrationStrategy.QUANTILE.value)),
        platt_scaling=payload.get("platt_scaling", True),
        isotonic_calibration=payload.get("isotonic_calibration", True),
        ranking_metric=CalibrationRankingMetric(
            payload.get(
                "ranking_metric",
                CalibrationRankingMetric.BRIER_SCORE.value,
            )
        ),
    )


def _build_scenario_test_config(payload: dict[str, Any]) -> ScenarioTestConfig:
    return ScenarioTestConfig(
        enabled=payload.get("enabled", False),
        evaluation_split=payload.get("evaluation_split", "test"),
        scenarios=[
            ScenarioConfig(
                name=scenario.get("name", ""),
                description=scenario.get("description", ""),
                enabled=scenario.get("enabled", True),
                feature_shocks=[
                    ScenarioFeatureShock(
                        feature_name=shock.get("feature_name", ""),
                        operation=ScenarioShockOperation(
                            shock.get("operation", ScenarioShockOperation.SET.value)
                        ),
                        value=shock.get("value"),
                    )
                    for shock in scenario.get("feature_shocks", [])
                ],
            )
            for scenario in payload.get("scenarios", [])
        ],
    )


def _build_scorecard_config(payload: dict[str, Any]) -> ScorecardConfig:
    return ScorecardConfig(
        monotonicity=ScorecardMonotonicity(
            payload.get("monotonicity", ScorecardMonotonicity.AUTO.value)
        ),
        min_bin_share=payload.get("min_bin_share", 0.05),
        base_score=payload.get("base_score", 600),
        points_to_double_odds=payload.get("points_to_double_odds", 50),
        odds_reference=payload.get("odds_reference", 20.0),
        reason_code_count=payload.get("reason_code_count", 3),
    )


def _build_scorecard_workbench_config(payload: dict[str, Any]) -> ScorecardWorkbenchConfig:
    return ScorecardWorkbenchConfig(
        enabled=payload.get("enabled", True),
        max_features=payload.get("max_features", 6),
        include_score_distribution=payload.get("include_score_distribution", True),
        include_reason_code_analysis=payload.get("include_reason_code_analysis", True),
    )


def _build_imputation_sensitivity_config(payload: dict[str, Any]) -> ImputationSensitivityConfig:
    return ImputationSensitivityConfig(
        enabled=payload.get("enabled", False),
        evaluation_split=payload.get("evaluation_split", "test"),
        alternative_policies=[
            MissingValuePolicy(policy)
            for policy in payload.get(
                "alternative_policies",
                [policy.value for policy in ImputationSensitivityConfig().alternative_policies],
            )
        ],
        selected_features=payload.get("selected_features", []),
        max_features=payload.get("max_features", 5),
        min_missing_count=payload.get("min_missing_count", 1),
        max_features_with_detail=payload.get("max_features_with_detail", 3),
    )


def _build_variable_selection_config(payload: dict[str, Any]) -> VariableSelectionConfig:
    return VariableSelectionConfig(
        enabled=payload.get("enabled", False),
        max_features=payload.get("max_features"),
        min_univariate_score=payload.get("min_univariate_score"),
        correlation_threshold=payload.get("correlation_threshold", 0.8),
        locked_include_features=payload.get("locked_include_features", []),
        locked_exclude_features=payload.get("locked_exclude_features", []),
    )


def _build_documentation_config(payload: dict[str, Any]) -> DocumentationConfig:
    return DocumentationConfig(
        enabled=payload.get("enabled", True),
        model_name=payload.get("model_name", "Quant Studio Model"),
        model_owner=payload.get("model_owner", ""),
        business_purpose=payload.get("business_purpose", ""),
        portfolio_name=payload.get("portfolio_name", ""),
        segment_name=payload.get("segment_name", ""),
        horizon_definition=payload.get("horizon_definition", ""),
        target_definition=payload.get("target_definition", ""),
        loss_definition=payload.get("loss_definition", ""),
        assumptions=payload.get("assumptions", []),
        exclusions=payload.get("exclusions", []),
        limitations=payload.get("limitations", []),
        reviewer_notes=payload.get("reviewer_notes", ""),
    )


def _build_regulatory_report_config(payload: dict[str, Any]) -> RegulatoryReportConfig:
    return RegulatoryReportConfig(
        enabled=payload.get("enabled", True),
        export_docx=payload.get("export_docx", True),
        export_pdf=payload.get("export_pdf", True),
        committee_template_name=payload.get("committee_template_name", "committee_standard"),
        validation_template_name=payload.get(
            "validation_template_name",
            "validation_standard",
        ),
        include_assumptions_section=payload.get("include_assumptions_section", True),
        include_challenger_section=payload.get("include_challenger_section", True),
        include_scenario_section=payload.get("include_scenario_section", True),
        include_appendix_section=payload.get("include_appendix_section", True),
    )


def _build_diagnostic_config(payload: dict[str, Any]) -> DiagnosticConfig:
    return DiagnosticConfig(
        data_quality=payload.get("data_quality", True),
        descriptive_statistics=payload.get("descriptive_statistics", True),
        missingness_analysis=payload.get("missingness_analysis", True),
        correlation_analysis=payload.get("correlation_analysis", True),
        vif_analysis=payload.get("vif_analysis", True),
        woe_iv_analysis=payload.get("woe_iv_analysis", True),
        psi_analysis=payload.get("psi_analysis", True),
        adf_analysis=payload.get("adf_analysis", True),
        model_specification_tests=payload.get("model_specification_tests", True),
        forecasting_statistical_tests=payload.get("forecasting_statistical_tests", True),
        calibration_analysis=payload.get("calibration_analysis", True),
        threshold_analysis=payload.get("threshold_analysis", True),
        lift_gain_analysis=payload.get("lift_gain_analysis", True),
        segment_analysis=payload.get("segment_analysis", True),
        residual_analysis=payload.get("residual_analysis", True),
        quantile_analysis=payload.get("quantile_analysis", True),
        qq_analysis=payload.get("qq_analysis", True),
        interactive_visualizations=payload.get("interactive_visualizations", True),
        static_image_exports=payload.get("static_image_exports", True),
        export_excel_workbook=payload.get("export_excel_workbook", True),
        top_n_features=payload.get("top_n_features", 15),
        top_n_categories=payload.get("top_n_categories", 10),
        max_plot_rows=payload.get("max_plot_rows", 20000),
        quantile_bucket_count=payload.get("quantile_bucket_count", 10),
        default_segment_column=payload.get("default_segment_column"),
    )


def _build_distribution_diagnostic_config(
    payload: dict[str, Any],
) -> DistributionDiagnosticConfig:
    return DistributionDiagnosticConfig(
        enabled=payload.get("enabled", True),
        include_normality_tests=payload.get("include_normality_tests", True),
        include_shift_tests=payload.get("include_shift_tests", True),
        top_features=payload.get("top_features", 8),
        minimum_rows=payload.get("minimum_rows", 30),
    )


def _build_residual_diagnostic_config(payload: dict[str, Any]) -> ResidualDiagnosticConfig:
    return ResidualDiagnosticConfig(
        enabled=payload.get("enabled", True),
        heteroskedasticity_tests=payload.get("heteroskedasticity_tests", True),
        segment_bias_analysis=payload.get("segment_bias_analysis", True),
        autocorrelation_tests=payload.get("autocorrelation_tests", True),
        minimum_rows=payload.get("minimum_rows", 30),
    )


def _build_outlier_diagnostic_config(payload: dict[str, Any]) -> OutlierDiagnosticConfig:
    return OutlierDiagnosticConfig(
        enabled=payload.get("enabled", True),
        zscore_threshold=payload.get("zscore_threshold", 3.0),
        leverage_multiplier=payload.get("leverage_multiplier", 2.0),
        cooks_distance_multiplier=payload.get("cooks_distance_multiplier", 4.0),
        max_rows=payload.get("max_rows", 50),
    )


def _build_dependency_diagnostic_config(payload: dict[str, Any]) -> DependencyDiagnosticConfig:
    return DependencyDiagnosticConfig(
        enabled=payload.get("enabled", True),
        clustering_correlation_threshold=payload.get("clustering_correlation_threshold", 0.7),
        maximum_features=payload.get("maximum_features", 12),
        condition_index_warning=payload.get("condition_index_warning", 30.0),
    )


def _build_time_series_diagnostic_config(payload: dict[str, Any]) -> TimeSeriesDiagnosticConfig:
    return TimeSeriesDiagnosticConfig(
        enabled=payload.get("enabled", True),
        maximum_lag=payload.get("maximum_lag", 5),
        seasonal_period=payload.get("seasonal_period", 4),
        minimum_series_length=payload.get("minimum_series_length", 12),
    )


def _build_structural_break_config(payload: dict[str, Any]) -> StructuralBreakConfig:
    return StructuralBreakConfig(
        enabled=payload.get("enabled", True),
        candidate_break_count=payload.get("candidate_break_count", 3),
        minimum_segment_size=payload.get("minimum_segment_size", 12),
        rolling_window_fraction=payload.get("rolling_window_fraction", 0.25),
    )


def _build_feature_workbench_config(payload: dict[str, Any]) -> FeatureWorkbenchConfig:
    return FeatureWorkbenchConfig(
        enabled=payload.get("enabled", True),
        max_features=payload.get("max_features", 12),
        include_preview_statistics=payload.get("include_preview_statistics", True),
        include_target_association=payload.get("include_target_association", True),
    )


def _build_preset_recommendation_config(
    payload: dict[str, Any],
) -> PresetRecommendationConfig:
    return PresetRecommendationConfig(
        enabled=payload.get("enabled", True),
        include_imputation_recommendations=payload.get("include_imputation_recommendations", True),
        include_transformation_recommendations=payload.get(
            "include_transformation_recommendations", True
        ),
        include_test_recommendations=payload.get("include_test_recommendations", True),
    )


def _build_credit_risk_diagnostic_config(payload: dict[str, Any]) -> CreditRiskDiagnosticConfig:
    migration_state_column = payload.get("migration_state_column")
    if migration_state_column is not None:
        migration_state_column = str(migration_state_column).strip() or None
    return CreditRiskDiagnosticConfig(
        enabled=payload.get("enabled", True),
        vintage_analysis=payload.get("vintage_analysis", True),
        migration_analysis=payload.get("migration_analysis", True),
        delinquency_transition_analysis=payload.get(
            "delinquency_transition_analysis",
            True,
        ),
        migration_state_column=migration_state_column,
        cohort_pd_analysis=payload.get("cohort_pd_analysis", True),
        lgd_segment_analysis=payload.get("lgd_segment_analysis", True),
        recovery_analysis=payload.get("recovery_analysis", True),
        macro_sensitivity_analysis=payload.get("macro_sensitivity_analysis", True),
        top_macro_features=payload.get("top_macro_features", 5),
        top_segments=payload.get("top_segments", 8),
        shock_std_multiplier=payload.get("shock_std_multiplier", 1.0),
    )


def _build_robustness_config(payload: dict[str, Any]) -> RobustnessConfig:
    return RobustnessConfig(
        enabled=payload.get("enabled", False),
        resample_count=payload.get("resample_count", 12),
        sample_fraction=payload.get("sample_fraction", 0.8),
        sample_with_replacement=payload.get("sample_with_replacement", True),
        evaluation_split=payload.get("evaluation_split", "test"),
        metric_stability=payload.get("metric_stability", True),
        coefficient_stability=payload.get("coefficient_stability", True),
        random_state=payload.get("random_state", 42),
    )


def _build_cross_validation_config(payload: dict[str, Any]) -> CrossValidationConfig:
    return CrossValidationConfig(
        enabled=payload.get("enabled", False),
        fold_count=payload.get("fold_count", 5),
        strategy=CrossValidationStrategy(payload.get("strategy", CrossValidationStrategy.AUTO)),
        shuffle=payload.get("shuffle", True),
        metric_stability=payload.get("metric_stability", True),
        coefficient_stability=payload.get("coefficient_stability", True),
        random_state=payload.get("random_state", 42),
    )


def _build_reproducibility_config(payload: dict[str, Any]) -> ReproducibilityConfig:
    return ReproducibilityConfig(
        enabled=payload.get("enabled", True),
        capture_git_metadata=payload.get("capture_git_metadata", True),
        package_names=payload.get(
            "package_names",
            ReproducibilityConfig().package_names,
        ),
    )


def _build_performance_config(payload: dict[str, Any]) -> PerformanceConfig:
    return PerformanceConfig(
        enabled=payload.get("enabled", True),
        large_data_mode=payload.get("large_data_mode", False),
        upload_warning_mb=payload.get("upload_warning_mb", 5120),
        ui_preview_rows=payload.get("ui_preview_rows", 50),
        html_table_preview_rows=payload.get("html_table_preview_rows", 12),
        html_max_figures_per_section=payload.get("html_max_figures_per_section", 6),
        html_max_tables_per_section=payload.get("html_max_tables_per_section", 6),
        diagnostic_sample_rows=payload.get("diagnostic_sample_rows", 20000),
        multiple_imputation_row_cap=payload.get("multiple_imputation_row_cap", 25000),
        lazy_html_figures=payload.get("lazy_html_figures", True),
        lazy_streamlit_results=payload.get("lazy_streamlit_results", True),
        optimize_dtypes=payload.get("optimize_dtypes", True),
        capture_memory_profile=payload.get("capture_memory_profile", True),
        deep_memory_profile=payload.get("deep_memory_profile", False),
        retain_full_working_data=payload.get("retain_full_working_data", False),
        downcast_numeric=payload.get("downcast_numeric", True),
        convert_low_cardinality_strings=payload.get("convert_low_cardinality_strings", True),
        category_max_unique_values=payload.get("category_max_unique_values", 500),
        category_max_unique_ratio=payload.get("category_max_unique_ratio", 0.5),
        max_categorical_cardinality=payload.get("max_categorical_cardinality", 500),
        max_categorical_cardinality_ratio=payload.get(
            "max_categorical_cardinality_ratio", 0.2
        ),
        allow_high_cardinality_categoricals=payload.get(
            "allow_high_cardinality_categoricals", False
        ),
        convert_csv_to_parquet=payload.get("convert_csv_to_parquet", False),
        csv_conversion_chunk_rows=payload.get("csv_conversion_chunk_rows", 100000),
        large_data_training_sample_rows=payload.get("large_data_training_sample_rows", 250000),
        large_data_score_chunk_rows=payload.get("large_data_score_chunk_rows", 100000),
        large_data_project_columns=payload.get("large_data_project_columns", True),
        large_data_auto_stage_parquet=payload.get("large_data_auto_stage_parquet", True),
        memory_limit_gb=payload.get("memory_limit_gb"),
        memory_estimate_file_multiplier=payload.get("memory_estimate_file_multiplier", 6.0),
        memory_estimate_dataframe_multiplier=payload.get(
            "memory_estimate_dataframe_multiplier", 4.0
        ),
    )


def _build_artifact_config(payload: dict[str, Any]) -> ArtifactConfig:
    return ArtifactConfig(
        output_root=Path(payload.get("output_root", "artifacts")),
        model_file_name=payload.get("model_file_name", "quant_model.joblib"),
        metrics_file_name=payload.get("metrics_file_name", "metrics.json"),
        input_snapshot_file_name=payload.get("input_snapshot_file_name", "input_snapshot.csv"),
        input_snapshot_parquet_file_name=payload.get(
            "input_snapshot_parquet_file_name",
            "input_snapshot.parquet",
        ),
        predictions_file_name=payload.get("predictions_file_name", "predictions.csv"),
        predictions_parquet_file_name=payload.get(
            "predictions_parquet_file_name",
            "predictions.parquet",
        ),
        feature_importance_file_name=payload.get(
            "feature_importance_file_name", "feature_importance.csv"
        ),
        backtest_file_name=payload.get("backtest_file_name", "backtest_summary.csv"),
        report_file_name=payload.get("report_file_name", "run_report.md"),
        interactive_report_file_name=payload.get(
            "interactive_report_file_name", "interactive_report.html"
        ),
        config_file_name=payload.get("config_file_name", "run_config.json"),
        statistical_tests_file_name=payload.get(
            "statistical_tests_file_name", "statistical_tests.json"
        ),
        workbook_file_name=payload.get("workbook_file_name", "analysis_workbook.xlsx"),
        model_summary_file_name=payload.get("model_summary_file_name", "model_summary.txt"),
        manifest_file_name=payload.get("manifest_file_name", "artifact_manifest.json"),
        step_manifest_file_name=payload.get("step_manifest_file_name", "step_manifest.json"),
        documentation_pack_file_name=payload.get(
            "documentation_pack_file_name",
            "model_documentation_pack.md",
        ),
        validation_pack_file_name=payload.get("validation_pack_file_name", "validation_pack.md"),
        committee_report_docx_file_name=payload.get(
            "committee_report_docx_file_name",
            "committee_report.docx",
        ),
        validation_report_docx_file_name=payload.get(
            "validation_report_docx_file_name",
            "validation_report.docx",
        ),
        committee_report_pdf_file_name=payload.get(
            "committee_report_pdf_file_name",
            "committee_report.pdf",
        ),
        validation_report_pdf_file_name=payload.get(
            "validation_report_pdf_file_name",
            "validation_report.pdf",
        ),
        reproducibility_manifest_file_name=payload.get(
            "reproducibility_manifest_file_name",
            "reproducibility_manifest.json",
        ),
        template_workbook_file_name=payload.get(
            "template_workbook_file_name",
            "configuration_template.xlsx",
        ),
        runner_script_file_name=payload.get("runner_script_file_name", "generated_run.py"),
        rerun_readme_file_name=payload.get("rerun_readme_file_name", "HOW_TO_RERUN.md"),
        tables_directory_name=payload.get("tables_directory_name", "tables"),
        figures_directory_name=payload.get("figures_directory_name", "figures"),
        html_directory_name=payload.get("html_directory_name", "html"),
        png_directory_name=payload.get("png_directory_name", "png"),
        code_snapshot_directory_name=payload.get("code_snapshot_directory_name", "code_snapshot"),
        decision_summary_file_name=payload.get(
            "decision_summary_file_name",
            "decision_summary.md",
        ),
        include_enhanced_report_visuals=payload.get("include_enhanced_report_visuals", True),
        include_advanced_visual_analytics=payload.get(
            "include_advanced_visual_analytics",
            False,
        ),
        export_individual_figure_files=payload.get("export_individual_figure_files", False),
        compact_prediction_exports=payload.get("compact_prediction_exports", True),
        export_input_snapshot=payload.get("export_input_snapshot", True),
        export_code_snapshot=payload.get("export_code_snapshot", True),
        export_profile=ExportProfile(payload.get("export_profile", ExportProfile.STANDARD.value)),
        tabular_output_format=TabularOutputFormat(
            payload.get("tabular_output_format", TabularOutputFormat.PARQUET.value)
        ),
        large_data_export_policy=LargeDataExportPolicy(
            payload.get("large_data_export_policy", LargeDataExportPolicy.FULL.value)
        ),
        large_data_sample_rows=payload.get("large_data_sample_rows", 50000),
        parquet_compression=payload.get("parquet_compression", "snappy"),
        run_debug_trace_file_name=payload.get(
            "run_debug_trace_file_name",
            "run_debug_trace.json",
        ),
    )


def _resolve_optional_path(value: str | None, base_path: Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or base_path is None:
        return path
    return (base_path / path).resolve()
