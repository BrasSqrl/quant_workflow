"""Regression tests for the post-calibration roadmap features."""

from __future__ import annotations

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ComparisonConfig,
    DataStructure,
    DocumentationConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    RobustnessConfig,
    ScorecardConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
    VariableSelectionConfig,
)
from tests.support import (
    build_binary_dataframe,
    build_common_schema,
    build_lifetime_pd_dataframe,
    temporary_artifact_root,
)


def test_scorecard_mode_exports_points_reason_codes_and_documentation_pack() -> None:
    dataframe = build_binary_dataframe(row_count=260)
    with temporary_artifact_root("pytest_scorecard_docs") as artifact_root:
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
            ),
            model=ModelConfig(model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION),
            scorecard=ScorecardConfig(reason_code_count=2),
            variable_selection=VariableSelectionConfig(enabled=True, max_features=4),
            documentation=DocumentationConfig(
                enabled=True,
                model_name="Retail Scorecard",
                business_purpose="Scorecard development regression test.",
                assumptions=["Assumption A"],
            ),
            comparison=ComparisonConfig(
                enabled=True,
                challenger_model_types=[ModelType.LOGISTIC_REGRESSION],
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "scorecard_points_table" in context.diagnostics_tables
        assert "scorecard_scaling_summary" in context.diagnostics_tables
        assert "scorecard_feature_summary" in context.diagnostics_tables
        assert "scorecard_reason_code_frequency" in context.diagnostics_tables
        assert "scorecard_points" in context.predictions["test"].columns
        assert "reason_code_1" in context.predictions["test"].columns
        assert "variable_selection" in context.diagnostics_tables
        assert "scorecard_feature_iv" in context.visualizations
        assert context.artifacts["documentation_pack"].exists()


def test_discrete_time_hazard_model_exports_lifetime_curve() -> None:
    dataframe = build_lifetime_pd_dataframe(entity_count=18, periods_per_entity=8)
    with temporary_artifact_root("pytest_lifetime_pd") as artifact_root:
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
                data_structure=DataStructure.PANEL,
                date_column="as_of_date",
                entity_column="account_id",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=ModelConfig(model_type=ModelType.DISCRETE_TIME_HAZARD_MODEL),
            variable_selection=VariableSelectionConfig(enabled=True, max_features=5),
            documentation=DocumentationConfig(
                enabled=True,
                model_name="Lifetime PD",
                horizon_definition="Monthly lifetime horizon",
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "lifetime_pd_curve" in context.diagnostics_tables
        assert "lifetime_pd_curve" in context.visualizations
        assert "hazard_period_index" in context.predictions["test"].columns


def test_robustness_testing_exports_metric_and_feature_stability_outputs() -> None:
    dataframe = build_binary_dataframe(row_count=240)
    with temporary_artifact_root("pytest_robustness") as artifact_root:
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
            ),
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
            robustness=RobustnessConfig(
                enabled=True,
                resample_count=4,
                sample_fraction=0.75,
                evaluation_split="test",
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "robustness_metric_distribution" in context.diagnostics_tables
        assert "robustness_metric_summary" in context.diagnostics_tables
        assert "robustness_feature_stability" in context.diagnostics_tables
        assert "robustness_metric_boxplot" in context.visualizations
        assert "robustness_feature_stability" in context.visualizations
