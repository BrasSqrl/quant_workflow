"""Regression tests for model feature construction."""

from __future__ import annotations

import pandas as pd

from quant_pd_framework import (
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.steps.feature_engineering import FeatureEngineeringStep
from tests.support import build_binary_dataframe


def _scorecard_panel_context(feature_engineering: FeatureEngineeringConfig) -> PipelineContext:
    dataframe = build_binary_dataframe(row_count=80).assign(
        application_date=lambda frame: pd.to_datetime(frame["as_of_date"])
        - pd.to_timedelta(30, unit="D")
    )
    return PipelineContext(
        config=FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="application_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=feature_engineering,
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.PANEL,
                date_column="as_of_date",
                entity_column="loan_id",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=ModelConfig(model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION),
        ),
        run_id="pytest_feature_engineering",
        raw_input=dataframe,
        raw_data=dataframe,
        working_data=dataframe,
        target_column="default_status",
        metadata={"target_source_column": "default_status"},
    )


def test_date_part_features_default_to_off() -> None:
    assert FeatureEngineeringConfig().derive_date_parts is False


def test_date_part_features_are_controlled_by_date_part_toggle() -> None:
    context = _scorecard_panel_context(
        FeatureEngineeringConfig(
            derive_date_parts=False,
            drop_raw_date_columns=True,
        )
    )

    result = FeatureEngineeringStep().run(context)

    assert not any(column.startswith("as_of_date_") for column in result.feature_columns)
    assert not any(column.startswith("application_date_") for column in result.feature_columns)
    assert "as_of_date" not in result.feature_columns
    assert "application_date" not in result.feature_columns


def test_generated_date_parts_remain_features_when_raw_dates_are_dropped() -> None:
    context = _scorecard_panel_context(
        FeatureEngineeringConfig(
            derive_date_parts=True,
            drop_raw_date_columns=True,
            date_parts=["year", "month"],
        )
    )

    result = FeatureEngineeringStep().run(context)

    assert "as_of_date_year" in result.feature_columns
    assert "as_of_date_month" in result.feature_columns
    assert "application_date_year" in result.feature_columns
    assert "application_date_month" in result.feature_columns
    assert "as_of_date" not in result.feature_columns
    assert "application_date" not in result.feature_columns
    assert "application_date" not in result.working_data.columns
    assert result.metadata["date_feature_engineering"]["removed_raw_date_columns"] == [
        "application_date"
    ]


def test_raw_date_retention_does_not_make_raw_dates_model_features() -> None:
    context = _scorecard_panel_context(
        FeatureEngineeringConfig(
            derive_date_parts=True,
            drop_raw_date_columns=False,
            date_parts=["quarter"],
        )
    )

    result = FeatureEngineeringStep().run(context)

    assert "as_of_date" in result.working_data.columns
    assert "application_date" in result.working_data.columns
    assert "as_of_date" not in result.feature_columns
    assert "application_date" not in result.feature_columns
    assert "as_of_date_quarter" in result.feature_columns
    assert "application_date_quarter" in result.feature_columns
    assert set(result.metadata["date_feature_engineering"]["retained_raw_date_columns"]) == {
        "application_date",
        "as_of_date",
    }
