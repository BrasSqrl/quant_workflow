"""Regression tests for scoring new data on an existing fitted model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    DiagnosticConfig,
    ExecutionMode,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
    build_sample_pd_dataframe,
    load_framework_config,
)
from tests.support import build_common_schema, build_continuous_dataframe, temporary_artifact_root


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


def build_continuous_training_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ColumnSpec(
                    name="censored_target",
                    dtype="float",
                    role=ColumnRole.TARGET_SOURCE,
                ),
            ]
        ),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="censored_target",
            output_column="lgd_target",
            mode=TargetMode.CONTINUOUS,
        ),
        split=SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        model=ModelConfig(model_type=ModelType.LINEAR_REGRESSION),
        diagnostics=DiagnosticConfig(
            interactive_visualizations=False,
            static_image_exports=False,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
    )


def build_multiclass_dataframe(row_count: int):
    dataframe = build_sample_pd_dataframe(row_count=row_count, random_state=71)
    dataframe["risk_grade"] = dataframe["region"].map(
        {"north": "grade_a", "south": "grade_b", "east": "grade_c", "west": "grade_c"}
    )
    return dataframe


def build_multiclass_training_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ColumnSpec(name="default_status", enabled=False),
                ColumnSpec(
                    name="risk_grade",
                    dtype="string",
                    role=ColumnRole.TARGET_SOURCE,
                ),
            ]
        ),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="risk_grade",
            output_column="risk_grade_target",
            mode=TargetMode.MULTICLASS,
        ),
        split=SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        model=ModelConfig(model_type=ModelType.DECISION_TREE),
        diagnostics=DiagnosticConfig(
            interactive_visualizations=False,
            static_image_exports=False,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
    )


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


def test_existing_continuous_model_scoring_runs_validation() -> None:
    with temporary_artifact_root("pytest_existing_continuous_train") as training_root:
        training_context = QuantModelOrchestrator(
            config=build_continuous_training_config(training_root)
        ).run(build_continuous_dataframe(row_count=180))

        with temporary_artifact_root("pytest_existing_continuous_score") as scoring_root:
            scoring_config = build_scoring_config(training_context, scoring_root)
            scored_context = QuantModelOrchestrator(config=scoring_config).run(
                build_continuous_dataframe(row_count=120)
            )

            assert scored_context.metadata["model_reused"] is True
            assert scored_context.metrics["test"]["rmse"] is not None
            assert "predicted_value" in scored_context.predictions["test"].columns


def test_existing_multiclass_model_scoring_runs_validation() -> None:
    with temporary_artifact_root("pytest_existing_multiclass_train") as training_root:
        training_context = QuantModelOrchestrator(
            config=build_multiclass_training_config(training_root)
        ).run(build_multiclass_dataframe(row_count=180))

        with temporary_artifact_root("pytest_existing_multiclass_score") as scoring_root:
            scoring_config = build_scoring_config(training_context, scoring_root)
            scored_context = QuantModelOrchestrator(config=scoring_config).run(
                build_multiclass_dataframe(row_count=120)
            )

            assert scored_context.metadata["model_reused"] is True
            assert scored_context.metrics["test"]["accuracy"] is not None
            assert "predicted_class" in scored_context.predictions["test"].columns
