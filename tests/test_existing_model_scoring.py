"""Regression tests for scoring new data on an existing fitted model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    ExecutionMode,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    build_sample_pd_dataframe,
    load_framework_config,
)
from tests.support import build_common_schema, temporary_artifact_root


def build_training_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=build_common_schema("loan_id", include_legacy_drop=True),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="default_status",
            output_column="default_flag",
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        diagnostics=DiagnosticConfig(
            interactive_visualizations=False,
            static_image_exports=False,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
    )


def build_scoring_config(
    trained_context,
    output_root: Path,
) -> FrameworkConfig:
    config = deepcopy(load_framework_config(trained_context.artifacts["config"]))
    config.execution.mode = ExecutionMode.SCORE_EXISTING_MODEL
    config.execution.existing_model_path = trained_context.artifacts["model"]
    config.execution.existing_config_path = trained_context.artifacts["config"]
    config.diagnostics.interactive_visualizations = False
    config.diagnostics.static_image_exports = False
    config.artifacts.output_root = output_root
    return config


def test_existing_model_scoring_with_labels_runs_full_validation() -> None:
    with temporary_artifact_root("pytest_existing_model_train") as training_root:
        training_context = QuantModelOrchestrator(config=build_training_config(training_root)).run(
            build_sample_pd_dataframe(row_count=220, random_state=13)
        )

        with temporary_artifact_root("pytest_existing_model_score_labeled") as scoring_root:
            scoring_config = build_scoring_config(training_context, scoring_root)
            new_dataframe = build_sample_pd_dataframe(row_count=160, random_state=31)
            scored_context = QuantModelOrchestrator(config=scoring_config).run(new_dataframe)

            assert scored_context.metadata["model_reused"] is True
            assert scored_context.metadata["labels_available"] is True
            assert scored_context.metrics["test"]["roc_auc"] is not None
            assert "calibration" in scored_context.diagnostics_tables
            assert "segment_performance" in scored_context.diagnostics_tables
            assert scored_context.artifacts["report"].exists()
            assert scored_context.artifacts["model"].exists()


def test_existing_model_scoring_without_labels_runs_score_only_outputs() -> None:
    with temporary_artifact_root("pytest_existing_model_train_unlabeled") as training_root:
        training_context = QuantModelOrchestrator(config=build_training_config(training_root)).run(
            build_sample_pd_dataframe(row_count=220, random_state=21)
        )

        with temporary_artifact_root("pytest_existing_model_score_unlabeled") as scoring_root:
            scoring_config = build_scoring_config(training_context, scoring_root)
            new_dataframe = build_sample_pd_dataframe(row_count=160, random_state=44).drop(
                columns=["default_status"]
            )
            scored_context = QuantModelOrchestrator(config=scoring_config).run(new_dataframe)

            assert scored_context.metadata["model_reused"] is True
            assert scored_context.metadata["labels_available"] is False
            assert scored_context.metrics["test"]["roc_auc"] is None
            assert "calibration" not in scored_context.diagnostics_tables
            assert "average_predicted_pd" in scored_context.backtest_summary.columns
            assert "observed_default_rate" not in scored_context.backtest_summary.columns
            assert "predicted_probability" in scored_context.predictions["test"].columns
