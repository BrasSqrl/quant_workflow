"""Regression tests for optional validation and test splits."""

from __future__ import annotations

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
    TargetConfig,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.steps.splitting import SplitStep
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def _split_context(split_config: SplitConfig) -> PipelineContext:
    dataframe = build_binary_dataframe(row_count=80)
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
