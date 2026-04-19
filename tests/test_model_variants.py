"""Smoke tests for additional supported model families."""

from __future__ import annotations

import pytest

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from tests.support import (
    build_binary_dataframe,
    build_common_schema,
    build_continuous_dataframe,
    build_panel_forecast_dataframe,
    temporary_artifact_root,
)


@pytest.mark.parametrize(
    ("model_type", "model_config"),
    [
        (ModelType.LOGISTIC_REGRESSION, ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)),
        (
            ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
            ModelConfig(
                model_type=ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                l1_ratio=0.35,
                max_iter=400,
            ),
        ),
        (
            ModelType.SCORECARD_LOGISTIC_REGRESSION,
            ModelConfig(
                model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION,
                scorecard_bins=4,
                max_iter=400,
            ),
        ),
        (
            ModelType.PROBIT_REGRESSION,
            ModelConfig(model_type=ModelType.PROBIT_REGRESSION, max_iter=300),
        ),
        (
            ModelType.XGBOOST,
            ModelConfig(
                model_type=ModelType.XGBOOST,
                xgboost_n_estimators=20,
                xgboost_max_depth=3,
            ),
        ),
    ],
)
def test_binary_model_variants_run(model_type: ModelType, model_config: ModelConfig) -> None:
    dataframe = build_binary_dataframe()
    with temporary_artifact_root(f"pytest_{model_type.value}") as artifact_root:
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
            model=model_config,
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["roc_auc"] is not None
        assert context.feature_importance is not None
        assert "quantile_summary" in context.diagnostics_tables
        assert context.artifacts["interactive_report"] is not None
        assert "package_zip" not in context.artifacts


@pytest.mark.parametrize(
    ("model_type", "model_config"),
    [
        (ModelType.LINEAR_REGRESSION, ModelConfig(model_type=ModelType.LINEAR_REGRESSION)),
        (
            ModelType.BETA_REGRESSION,
            ModelConfig(model_type=ModelType.BETA_REGRESSION, max_iter=300),
        ),
        (
            ModelType.TWO_STAGE_LGD_MODEL,
            ModelConfig(model_type=ModelType.TWO_STAGE_LGD_MODEL, max_iter=300),
        ),
        (
            ModelType.QUANTILE_REGRESSION,
            ModelConfig(model_type=ModelType.QUANTILE_REGRESSION, quantile_alpha=0.65),
        ),
        (
            ModelType.TOBIT_REGRESSION,
            ModelConfig(
                model_type=ModelType.TOBIT_REGRESSION,
                max_iter=250,
                tobit_left_censoring=0.0,
                tobit_right_censoring=1.0,
            ),
        ),
    ],
)
def test_continuous_model_variants_run(model_type: ModelType, model_config: ModelConfig) -> None:
    dataframe = build_continuous_dataframe()
    with temporary_artifact_root(f"pytest_{model_type.value}") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("loan_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="censored_target",
                mode=TargetMode.CONTINUOUS,
                output_column="target_value",
            ),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=model_config,
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["rmse"] is not None
        assert context.feature_importance is not None
        assert (
            "residual_summary" in context.diagnostics_tables
            or model_type == ModelType.LINEAR_REGRESSION
        )
        assert context.artifacts["workbook"] is not None


def test_panel_regression_variant_runs_on_panel_data() -> None:
    dataframe = build_panel_forecast_dataframe()
    with temporary_artifact_root("pytest_panel_regression") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("segment_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(drop_raw_date_columns=False),
            target=TargetConfig(
                source_column="forecast_value",
                mode=TargetMode.CONTINUOUS,
                output_column="target_value",
            ),
            split=SplitConfig(
                data_structure=DataStructure.PANEL,
                date_column="as_of_date",
                entity_column="segment_id",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=ModelConfig(model_type=ModelType.PANEL_REGRESSION),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["rmse"] is not None
        assert context.feature_importance is not None
        assert "segment_id" in context.feature_columns
