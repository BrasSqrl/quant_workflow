"""Tests for comparison, explainability, policy, scenario, and preset support."""

from __future__ import annotations

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ComparisonConfig,
    DataStructure,
    ExplainabilityConfig,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    PresetName,
    QuantModelOrchestrator,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
    get_preset_definition,
)
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def test_development_features_produce_expected_outputs() -> None:
    dataframe = build_binary_dataframe(row_count=220)
    with temporary_artifact_root("pytest_development_features") as artifact_root:
        config = FrameworkConfig(
            preset_name=PresetName.PD_DEVELOPMENT,
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
            comparison=ComparisonConfig(
                enabled=True,
                challenger_model_types=[
                    ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                    ModelType.SCORECARD_LOGISTIC_REGRESSION,
                ],
            ),
            feature_policy=FeaturePolicyConfig(
                enabled=True,
                required_features=["balance", "utilization"],
                max_missing_pct=100.0,
                expected_signs={"balance": "positive"},
                error_on_violation=False,
            ),
            explainability=ExplainabilityConfig(
                enabled=True,
                permutation_importance=True,
                feature_effect_curves=True,
                coefficient_breakdown=True,
                top_n_features=3,
                grid_points=5,
                sample_size=120,
            ),
            scenario_testing=ScenarioTestConfig(
                enabled=True,
                evaluation_split="test",
                scenarios=[
                    ScenarioConfig(
                        name="Utilization Up",
                        feature_shocks=[
                            ScenarioFeatureShock(
                                feature_name="utilization",
                                operation=ScenarioShockOperation.ADD,
                                value=0.10,
                            )
                        ],
                    )
                ],
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.comparison_results is not None
        assert "model_comparison" in context.diagnostics_tables
        assert "feature_policy_checks" in context.diagnostics_tables
        assert "feature_effect_curves" in context.diagnostics_tables
        assert "permutation_importance" in context.diagnostics_tables
        assert "scenario_summary" in context.diagnostics_tables
        assert context.metadata.get("comparison_recommended_model") is not None


def test_preset_definitions_cover_expected_workflows() -> None:
    pd_preset = get_preset_definition(PresetName.PD_DEVELOPMENT)
    ccar_preset = get_preset_definition(PresetName.CCAR_FORECASTING)

    assert pd_preset.model.model_type == ModelType.LOGISTIC_REGRESSION
    assert pd_preset.target_mode == TargetMode.BINARY
    assert ccar_preset.model.model_type == ModelType.PANEL_REGRESSION
    assert ccar_preset.data_structure == DataStructure.PANEL
