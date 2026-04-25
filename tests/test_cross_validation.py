"""Tests for optional cross-validation diagnostics."""

from __future__ import annotations

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    CrossValidationConfig,
    CrossValidationStrategy,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.config_io import load_framework_config
from tests.support import (
    build_binary_dataframe,
    build_common_schema,
    build_panel_forecast_dataframe,
    temporary_artifact_root,
)


def test_binary_cross_validation_exports_fold_metric_and_feature_outputs() -> None:
    dataframe = build_binary_dataframe(row_count=140)

    with temporary_artifact_root("cross_validation_binary") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="default_status",
                mode=TargetMode.BINARY,
                positive_values=[1],
            ),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                random_state=42,
                stratify=True,
            ),
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION, max_iter=300),
            diagnostics=DiagnosticConfig(
                interactive_visualizations=False,
                static_image_exports=False,
            ),
            cross_validation=CrossValidationConfig(enabled=True, fold_count=3),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metadata["cross_validation_summary"]["validation_method"] == (
            "stratified_kfold"
        )
        assert context.metadata["cross_validation_summary"]["final_model_refit"] is False
        assert "cross_validation_fold_metrics" in context.diagnostics_tables
        assert "cross_validation_metric_summary" in context.diagnostics_tables
        assert "cross_validation_feature_stability" in context.diagnostics_tables
        assert "cross_validation_metric_boxplot" in context.visualizations


def test_time_aware_cross_validation_uses_expanding_windows() -> None:
    dataframe = build_panel_forecast_dataframe(entity_count=6, periods_per_entity=8)

    with temporary_artifact_root("cross_validation_time_aware") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("segment_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="forecast_value",
                mode=TargetMode.CONTINUOUS,
                output_column="forecast_target",
            ),
            split=SplitConfig(
                data_structure=DataStructure.PANEL,
                date_column="as_of_date",
                entity_column="segment_id",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                random_state=42,
                stratify=False,
            ),
            model=ModelConfig(model_type=ModelType.LINEAR_REGRESSION),
            diagnostics=DiagnosticConfig(
                interactive_visualizations=False,
                static_image_exports=False,
            ),
            cross_validation=CrossValidationConfig(
                enabled=True,
                fold_count=3,
                strategy=CrossValidationStrategy.AUTO,
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metadata["cross_validation_summary"]["validation_method"] == (
            "time_series_expanding_window"
        )
        assert "cross_validation_fold_metrics" in context.diagnostics_tables


def test_cross_validation_config_round_trips_from_saved_config_payload() -> None:
    config = FrameworkConfig(
        schema=build_common_schema("account_id"),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="default_status",
            mode=TargetMode.BINARY,
            positive_values=[1],
        ),
        split=SplitConfig(data_structure=DataStructure.CROSS_SECTIONAL),
        cross_validation=CrossValidationConfig(
            enabled=True,
            fold_count=4,
            strategy=CrossValidationStrategy.STRATIFIED_KFOLD,
            shuffle=False,
        ),
    )

    loaded = load_framework_config(config.to_dict())

    assert loaded.cross_validation.enabled is True
    assert loaded.cross_validation.fold_count == 4
    assert loaded.cross_validation.strategy == CrossValidationStrategy.STRATIFIED_KFOLD
    assert loaded.cross_validation.shuffle is False
