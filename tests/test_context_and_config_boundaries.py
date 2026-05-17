"""Regression checks for context/config encapsulation boundaries."""

from __future__ import annotations

import inspect

import pytest

from quant_pd_framework import (
    CleaningConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    SchemaConfig,
    TargetConfig,
)
from quant_pd_framework.config import (
    DataStructure,
    PerformanceConfig,
    SplitConfig,
    SplitStrategy,
)
from quant_pd_framework.context import PipelineContext, PipelineMetadataKey
from quant_pd_framework.orchestrator import build_run_id
from quant_pd_framework.stage_runner import CheckpointedWorkflowRunner


def test_checkpoint_runner_uses_public_run_id_factory() -> None:
    source = inspect.getsource(CheckpointedWorkflowRunner.start)

    assert "_build_run_id" not in source
    assert build_run_id().startswith("run_")


def test_config_enum_coercion_happens_at_construction() -> None:
    split = SplitConfig(data_structure="panel", split_strategy="random")
    performance = PerformanceConfig(
        large_data_backend="disk_backed",
        large_data_model_policy="certified_only",
        large_data_partition_strategy="split",
        large_data_worker_mode="worker_service",
    )

    assert split.data_structure == DataStructure.PANEL
    assert split.split_strategy == SplitStrategy.RANDOM
    assert performance.large_data_backend.value == "disk_backed"
    assert performance.large_data_model_policy.value == "certified_only"
    assert performance.large_data_partition_strategy.value == "split"
    assert performance.large_data_worker_mode.value == "worker_service"

    split_values = (split.data_structure, split.split_strategy)
    performance_values = (
        performance.large_data_backend,
        performance.large_data_model_policy,
        performance.large_data_partition_strategy,
        performance.large_data_worker_mode,
    )
    split.validate()
    performance.validate()
    assert (split.data_structure, split.split_strategy) == split_values
    assert (
        performance.large_data_backend,
        performance.large_data_model_policy,
        performance.large_data_partition_strategy,
        performance.large_data_worker_mode,
    ) == performance_values


def test_pipeline_metadata_schema_validates_known_keys() -> None:
    context = PipelineContext(
        config=FrameworkConfig(
            schema=SchemaConfig(),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status"),
            split=SplitConfig(),
        ),
        run_id="pytest_metadata_schema",
        raw_input=None,
    )

    context.set_metadata(
        PipelineMetadataKey.LARGE_DATA_SAMPLE,
        {"sample_rows_requested": 100, "sample_rows_loaded": 10},
    )

    assert context.metadata["large_data_sample"]["sample_rows_loaded"] == 10
    assert context.get_metadata_dict(PipelineMetadataKey.LARGE_DATA_SAMPLE)[
        "sample_rows_requested"
    ] == 100

    with pytest.raises(TypeError, match="large_data_sample"):
        context.set_metadata(PipelineMetadataKey.LARGE_DATA_SAMPLE, ["not", "a", "mapping"])
