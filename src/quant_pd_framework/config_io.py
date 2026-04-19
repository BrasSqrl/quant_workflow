"""Helpers for reading saved framework configs back into dataclasses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    ComparisonConfig,
    DataStructure,
    DiagnosticConfig,
    ExecutionConfig,
    ExecutionMode,
    ExplainabilityConfig,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FrameworkConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    PresetName,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
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
        feature_policy=_build_feature_policy_config(payload.get("feature_policy", {})),
        explainability=_build_explainability_config(payload.get("explainability", {})),
        scenario_testing=_build_scenario_test_config(payload.get("scenario_testing", {})),
        diagnostics=_build_diagnostic_config(payload.get("diagnostics", {})),
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


def _build_explainability_config(payload: dict[str, Any]) -> ExplainabilityConfig:
    return ExplainabilityConfig(
        enabled=payload.get("enabled", True),
        permutation_importance=payload.get("permutation_importance", True),
        feature_effect_curves=payload.get("feature_effect_curves", True),
        coefficient_breakdown=payload.get("coefficient_breakdown", True),
        top_n_features=payload.get("top_n_features", 5),
        grid_points=payload.get("grid_points", 12),
        sample_size=payload.get("sample_size", 2000),
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


def _build_artifact_config(payload: dict[str, Any]) -> ArtifactConfig:
    return ArtifactConfig(
        output_root=Path(payload.get("output_root", "artifacts")),
        model_file_name=payload.get("model_file_name", "quant_model.joblib"),
        metrics_file_name=payload.get("metrics_file_name", "metrics.json"),
        input_snapshot_file_name=payload.get("input_snapshot_file_name", "input_snapshot.csv"),
        predictions_file_name=payload.get("predictions_file_name", "predictions.csv"),
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
        runner_script_file_name=payload.get("runner_script_file_name", "generated_run.py"),
        rerun_readme_file_name=payload.get("rerun_readme_file_name", "HOW_TO_RERUN.md"),
        tables_directory_name=payload.get("tables_directory_name", "tables"),
        figures_directory_name=payload.get("figures_directory_name", "figures"),
        html_directory_name=payload.get("html_directory_name", "html"),
        png_directory_name=payload.get("png_directory_name", "png"),
        json_directory_name=payload.get("json_directory_name", "json"),
        code_snapshot_directory_name=payload.get("code_snapshot_directory_name", "code_snapshot"),
        export_input_snapshot=payload.get("export_input_snapshot", True),
        export_code_snapshot=payload.get("export_code_snapshot", True),
    )


def _resolve_optional_path(value: str | None, base_path: Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or base_path is None:
        return path
    return (base_path / path).resolve()
