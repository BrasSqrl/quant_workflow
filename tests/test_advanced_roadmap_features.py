"""Regression tests for the remaining advanced roadmap items."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ImputationSensitivityConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
)
from tests.support import (
    build_panel_forecast_dataframe,
    temporary_artifact_root,
)


def build_time_series_binary_dataframe(row_count: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(seed=17)
    as_of_date = pd.date_range("2023-01-01", periods=row_count, freq="D")
    balance = rng.normal(15000, 3200, size=row_count).clip(1500, None)
    utilization = rng.uniform(0.08, 0.95, size=row_count)
    inquiries = rng.poisson(1.1, size=row_count)
    fico_bucket = rng.choice(["A", "B", "C"], size=row_count, p=[0.25, 0.45, 0.30])
    macro_index = np.linspace(-0.6, 0.7, row_count) + rng.normal(0, 0.08, size=row_count)
    default_probability = 1.0 / (
        1.0
        + np.exp(
            -(
                -5.2
                + 0.00009 * balance
                + 2.8 * utilization
                + 0.25 * inquiries
                + 0.75 * macro_index
                + np.where(fico_bucket == "C", 0.8, 0.0)
            )
        )
    )
    default_status = (rng.uniform(size=row_count) < default_probability).astype(int)
    return pd.DataFrame(
        {
            "as_of_date": as_of_date,
            "account_id": [f"T{index:04d}" for index in range(row_count)],
            "balance": balance,
            "utilization": utilization,
            "inquiries": inquiries,
            "fico_bucket": fico_bucket,
            "macro_index": macro_index,
            "default_status": default_status,
        }
    )


def build_missingness_binary_dataframe(row_count: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(seed=29)
    balance_true = rng.normal(12000, 2500, size=row_count).clip(1000, None)
    utilization = rng.uniform(0.1, 0.98, size=row_count)
    inquiries = rng.poisson(1.0, size=row_count)
    segment_true = rng.choice(["retail", "small_business", "consumer"], size=row_count)
    income_true = rng.normal(70000, 12000, size=row_count).clip(20000, None)
    probability = 1.0 / (
        1.0
        + np.exp(
            -(
                -4.6
                + 0.0001 * balance_true
                + 2.6 * utilization
                + 0.25 * inquiries
                + np.where(segment_true == "small_business", 0.55, 0.0)
            )
        )
    )
    default_status = (rng.uniform(size=row_count) < probability).astype(int)

    balance = balance_true.copy()
    income = income_true.copy()
    segment = segment_true.astype(object)
    balance[rng.choice(row_count, size=18, replace=False)] = np.nan
    income[rng.choice(row_count, size=14, replace=False)] = np.nan
    for index in rng.choice(row_count, size=12, replace=False):
        segment[index] = None

    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=row_count, freq="D"),
            "account_id": [f"M{index:04d}" for index in range(row_count)],
            "balance": balance,
            "income": income,
            "utilization": utilization,
            "inquiries": inquiries,
            "segment": segment,
            "default_status": default_status,
        }
    )


def test_transformation_library_and_interaction_engine() -> None:
    dataframe = build_time_series_binary_dataframe()
    with temporary_artifact_root("pytest_transform_interactions") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.TIME_SERIES,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            transformations=TransformationConfig(
                enabled=True,
                auto_interactions_enabled=True,
                include_numeric_numeric_interactions=True,
                include_categorical_numeric_interactions=True,
                max_auto_interactions=3,
                transformations=[
                    TransformationSpec(
                        transform_type=TransformationType.YEO_JOHNSON,
                        source_feature="balance",
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.CAPPED_ZSCORE,
                        source_feature="utilization",
                        parameter_value=2.5,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.LAG,
                        source_feature="utilization",
                        lag_periods=1,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.ROLLING_MEAN,
                        source_feature="macro_index",
                        window_size=4,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.PCT_CHANGE,
                        source_feature="balance",
                        lag_periods=1,
                    ),
                ],
            ),
            diagnostics=DiagnosticConfig(
                quantile_bucket_count=4,
                forecasting_statistical_tests=False,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "balance_yeo_johnson" in context.feature_columns
        assert "utilization_zscore" in context.feature_columns
        assert "utilization_lag_1" in context.feature_columns
        assert "macro_index_rollmean_4" in context.feature_columns
        assert "balance_pct_change_1" in context.feature_columns
        assert "interaction_candidates" in context.diagnostics_tables
        generated_interactions = context.metadata.get("generated_interaction_features", [])
        assert generated_interactions
        assert all(
            feature_name in context.feature_columns for feature_name in generated_interactions
        )


def test_imputation_sensitivity_missingness_and_specification_outputs() -> None:
    dataframe = build_missingness_binary_dataframe()
    with temporary_artifact_root("pytest_imputation_sensitivity") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                    ColumnSpec(
                        name="balance",
                        missing_value_policy=MissingValuePolicy.MEAN,
                    ),
                    ColumnSpec(
                        name="income",
                        missing_value_policy=MissingValuePolicy.MEDIAN,
                    ),
                    ColumnSpec(
                        name="segment",
                        missing_value_policy=MissingValuePolicy.MODE,
                    ),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                random_state=13,
            ),
            imputation_sensitivity=ImputationSensitivityConfig(
                enabled=True,
                evaluation_split="test",
                alternative_policies=[
                    MissingValuePolicy.MEAN,
                    MissingValuePolicy.MEDIAN,
                    MissingValuePolicy.MODE,
                ],
                max_features=3,
                min_missing_count=3,
            ),
            diagnostics=DiagnosticConfig(
                quantile_bucket_count=4,
                model_specification_tests=True,
                forecasting_statistical_tests=False,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "imputation_sensitivity_summary" in context.diagnostics_tables
        assert "imputation_sensitivity_detail" in context.diagnostics_tables
        assert "missingness_by_split" in context.diagnostics_tables
        assert "missingness_target_association" in context.diagnostics_tables
        assert "missingness_indicator_correlation" in context.diagnostics_tables
        assert "model_specification_tests" in context.diagnostics_tables
        assert "model_influence_summary" in context.diagnostics_tables


def test_forecasting_statistical_test_outputs() -> None:
    dataframe = build_panel_forecast_dataframe(entity_count=10, periods_per_entity=18)
    with temporary_artifact_root("pytest_forecasting_tests") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="segment_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="forecast_value",
                mode=TargetMode.CONTINUOUS,
                output_column="forecast_value",
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
            diagnostics=DiagnosticConfig(
                quantile_bucket_count=5,
                forecasting_statistical_tests=True,
                model_specification_tests=True,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "forecasting_statistical_tests" in context.diagnostics_tables
        assert "adf_tests" in context.diagnostics_tables
