"""Tests for governed segmented model builds and router scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SchemaConfig,
    SegmentedModelConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.segmented_model import build_segment_key_series
from quant_pd_framework.steps.validation import ValidationStep
from tests.support import (
    build_binary_dataframe,
    build_continuous_dataframe,
    temporary_artifact_root,
)


def test_segment_key_creation_handles_multiple_columns_and_missing_values() -> None:
    dataframe = build_binary_dataframe(row_count=5)
    dataframe.loc[0, "channel"] = None
    dataframe["dpd_bucket"] = ["0", "1-30", "0", "31-60", "1-30"]

    keys = build_segment_key_series(dataframe, ["channel", "dpd_bucket"]).tolist()

    assert keys[0] == "channel=<missing> | dpd_bucket=0"
    assert keys[1].startswith("channel=")
    assert "dpd_bucket=1-30" in keys[1]


def test_binary_segmented_fit_exports_router_and_segment_evidence() -> None:
    dataframe = build_binary_dataframe(row_count=240)
    with temporary_artifact_root("pytest_segmented_binary") as artifact_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                    ColumnSpec(name="channel", dtype="string", role=ColumnRole.SEGMENT),
                ]
            ),
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
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION, max_iter=500),
            segmented_model=SegmentedModelConfig(
                enabled=True,
                segment_columns=["channel"],
                min_segment_rows=10,
                min_segment_events=1,
                max_segments=10,
            ),
            diagnostics=DiagnosticConfig(static_image_exports=False),
            artifacts=ArtifactConfig(output_root=artifact_root, export_code_snapshot=False),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metadata["segmented_model"]["enabled"] is True
        assert "segment_model_inventory" in context.diagnostics_tables
        assert "segment_metrics" in context.diagnostics_tables
        assert "segment_key" in context.predictions["test"].columns
        assert "used_global_fallback" in context.predictions["test"].columns
        assert "channel" not in context.feature_columns
        assert Path(context.artifacts["segmented_model_manifest"]).exists()
        assert (
            Path(context.artifacts["tables_dir"])
            / "segmented_model"
            / "segment_model_inventory.csv"
        ).exists()

        manifest = json.loads(Path(context.artifacts["manifest"]).read_text(encoding="utf-8"))
        assert manifest["segmented_model"]["manifest"] == str(
            context.artifacts["segmented_model_manifest"]
        )


def test_segmented_config_blocks_unsupported_model_family() -> None:
    config = SegmentedModelConfig(enabled=True, segment_columns=["region"])

    with pytest.raises(ValueError, match="not supported"):
        config.validate(ModelType.TOBIT_REGRESSION, TargetMode.CONTINUOUS)


def test_segmented_validation_blocks_segment_count_above_cap() -> None:
    dataframe = build_binary_dataframe(row_count=30)
    dataframe["segment"] = [f"segment_{index}" for index in range(len(dataframe))]
    config = FrameworkConfig(
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ColumnSpec(name="segment", dtype="string", role=ColumnRole.SEGMENT),
            ]
        ),
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
        ),
        model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
        segmented_model=SegmentedModelConfig(
            enabled=True,
            segment_columns=["segment"],
            min_segment_rows=5,
            min_segment_events=1,
            max_segments=3,
        ),
    )
    context = PipelineContext(
        config=config,
        run_id="segmented_validation_test",
        raw_input=dataframe,
        raw_data=dataframe,
        working_data=dataframe,
        target_column="default_status",
    )
    context.metadata["labels_available"] = True

    with pytest.raises(ValueError, match="above the configured maximum"):
        ValidationStep().run(context)


def test_continuous_segmented_fit_routes_small_segments_to_global_fallback() -> None:
    dataframe = build_continuous_dataframe(row_count=160)
    dataframe["segment"] = "core"
    dataframe.loc[:4, "segment"] = "tiny"
    with temporary_artifact_root("pytest_segmented_continuous") as artifact_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
                    ColumnSpec(name="segment", dtype="string", role=ColumnRole.SEGMENT),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="censored_target", mode=TargetMode.CONTINUOUS),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.7,
                validation_size=0.1,
                test_size=0.2,
            ),
            model=ModelConfig(model_type=ModelType.LINEAR_REGRESSION),
            segmented_model=SegmentedModelConfig(
                enabled=True,
                segment_columns=["segment"],
                min_segment_rows=20,
                min_segment_events=0,
                max_segments=5,
            ),
            diagnostics=DiagnosticConfig(static_image_exports=False),
            artifacts=ArtifactConfig(output_root=artifact_root, export_code_snapshot=False),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        inventory = context.diagnostics_tables["segment_model_inventory"]
        assert "fallback_global" in set(inventory["status"])
        assert context.predictions["test"]["segment_model_id"].notna().all()
