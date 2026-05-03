"""Regression tests for optional validation and test splits."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    SplitStrategy,
    TargetConfig,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.steps.splitting import SplitStep
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def _split_context(split_config: SplitConfig) -> PipelineContext:
    dataframe = build_binary_dataframe(row_count=80)
    return _split_context_with_dataframe(dataframe, split_config)


def _split_context_with_dataframe(
    dataframe,
    split_config: SplitConfig,
) -> PipelineContext:
    return PipelineContext(
        config=FrameworkConfig(
            schema=SchemaConfig(),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status"),
            split=split_config,
        ),
        run_id="pytest_zero_split",
        raw_input=dataframe,
        raw_data=dataframe,
        working_data=dataframe,
        target_column="default_status",
        metadata={"labels_available": True},
    )


def test_split_config_allows_zero_validation_and_test_sizes() -> None:
    SplitConfig(train_size=1.0, validation_size=0.0, test_size=0.0).validate()

    with pytest.raises(ValueError, match="validation_size must be non-negative"):
        SplitConfig(train_size=1.1, validation_size=-0.1, test_size=0.0).validate()


def test_cross_sectional_split_can_omit_test_split() -> None:
    context = _split_context(
        SplitConfig(
            data_structure=DataStructure.CROSS_SECTIONAL,
            train_size=0.8,
            validation_size=0.2,
            test_size=0.0,
            stratify=False,
        )
    )

    result = SplitStep().run(context)

    assert set(result.split_frames) == {"train", "validation"}
    assert len(result.split_frames["train"]) == 64
    assert len(result.split_frames["validation"]) == 16
    assert "test" not in result.split_frames


def test_time_aware_split_can_omit_validation_split() -> None:
    context = _split_context(
        SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            train_size=0.75,
            validation_size=0.0,
            test_size=0.25,
            date_column="as_of_date",
            stratify=False,
        )
    )

    result = SplitStep().run(context)

    assert set(result.split_frames) == {"train", "test"}
    assert len(result.split_frames["train"]) == 60
    assert len(result.split_frames["test"]) == 20
    assert "validation" not in result.split_frames


def test_date_cutoff_split_uses_explicit_out_of_time_boundaries() -> None:
    dataframe = build_binary_dataframe(row_count=12).copy()
    dataframe["as_of_date"] = pd.date_range("2024-01-01", periods=12, freq="MS")
    context = _split_context_with_dataframe(
        dataframe,
        SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            split_strategy=SplitStrategy.DATE_CUTOFF,
            date_column="as_of_date",
            validation_start_date="2024-07-01",
            test_start_date="2024-10-01",
        ),
    )

    result = SplitStep().run(context)

    assert {split_name: len(frame) for split_name, frame in result.split_frames.items()} == {
        "train": 6,
        "validation": 3,
        "test": 3,
    }
    assert result.metadata["split_assignment"]["strategy"] == "date_cutoff"
    assert result.metadata["split_summary"]["test"]["min_date"] == "2024-10-01"


def test_explicit_date_window_split_requires_complete_non_overlapping_windows() -> None:
    dataframe = build_binary_dataframe(row_count=9).copy()
    dataframe["as_of_date"] = pd.date_range("2024-01-01", periods=9, freq="MS")
    context = _split_context_with_dataframe(
        dataframe,
        SplitConfig(
            data_structure=DataStructure.TIME_SERIES,
            split_strategy=SplitStrategy.EXPLICIT_DATE_WINDOWS,
            date_column="as_of_date",
            train_start_date="2024-01-01",
            train_end_date="2024-03-31",
            validation_start_date="2024-04-01",
            validation_end_date="2024-06-30",
            test_start_date="2024-07-01",
            test_end_date="2024-09-30",
        ),
    )

    result = SplitStep().run(context)

    assert {split_name: len(frame) for split_name, frame in result.split_frames.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert result.split_frames["validation"]["as_of_date"].min() == pd.Timestamp("2024-04-01")


def test_custom_column_split_accepts_validation_and_oot_aliases() -> None:
    dataframe = build_binary_dataframe(row_count=12).copy()
    dataframe["split_flag"] = ["train"] * 6 + ["val"] * 3 + ["oot"] * 3
    context = _split_context_with_dataframe(
        dataframe,
        SplitConfig(
            split_strategy=SplitStrategy.CUSTOM_COLUMN,
            custom_split_column="split_flag",
        ),
    )

    result = SplitStep().run(context)

    assert {split_name: len(frame) for split_name, frame in result.split_frames.items()} == {
        "train": 6,
        "validation": 3,
        "test": 3,
    }
    assert result.metadata["split_assignment"]["custom_split_column"] == "split_flag"


def test_full_workflow_uses_validation_when_test_split_is_zero() -> None:
    dataframe = build_binary_dataframe(row_count=140)

    with temporary_artifact_root("pytest_zero_test_split") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                train_size=0.8,
                validation_size=0.2,
                test_size=0.0,
                stratify=False,
            ),
            diagnostics=DiagnosticConfig(
                interactive_visualizations=False,
                static_image_exports=False,
            ),
            artifacts=ArtifactConfig(
                output_root=artifact_root,
                export_input_snapshot=False,
                export_code_snapshot=False,
            ),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert set(context.split_frames) == {"train", "validation"}
        assert set(context.metrics) == {"train", "validation"}
        assert context.metadata["backtest_split"] == "validation"
