"""Tests for the Step 3 Large Data Mode bypass control."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pd_framework.config import (
    CleaningConfig,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.large_data import build_dataset_handle
from quant_pd_framework.streamlit_ui.data import SelectedInputDataset
from quant_pd_framework.streamlit_ui.run_execution import (
    build_config_for_large_data_execution_override,
    build_execution_plan_cards,
    build_run_input,
    resolve_large_data_execution_override,
)
from tests.support import build_common_schema


def _selected_file(path: Path, source_kind: str = "data_load") -> SelectedInputDataset:
    path.write_text("placeholder", encoding="utf-8")
    handle = build_dataset_handle(
        path,
        {"source_kind": source_kind, "relative_path": str(path), "size_bytes": 1},
    )
    return SelectedInputDataset(
        pd.DataFrame({"x": [1], "target": [0]}),
        str(path),
        {"source_kind": source_kind, "relative_path": str(path), "size_bytes": 1},
        dataset_handle=handle,
        large_data_mode=True,
    )


def _test_config() -> FrameworkConfig:
    return FrameworkConfig(
        schema=build_common_schema("account_id"),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="default_status",
            mode=TargetMode.BINARY,
            positive_values=[1],
        ),
        split=SplitConfig(data_structure=DataStructure.CROSS_SECTIONAL),
    )


def test_large_data_override_keeps_detected_mode_by_default(tmp_path: Path) -> None:
    selected_input = _selected_file(tmp_path / "input.parquet")

    override = resolve_large_data_execution_override(
        selected_input=selected_input,
        detected_large_data_mode=True,
        force_standard_requested=False,
        force_standard_confirmed=False,
        reason="",
    )

    assert override.effective_large_data_mode is True
    assert override.user_override_disabled is False
    assert override.blocked_reason == ""


def test_large_data_override_requires_confirmation(tmp_path: Path) -> None:
    selected_input = _selected_file(tmp_path / "input.parquet")

    override = resolve_large_data_execution_override(
        selected_input=selected_input,
        detected_large_data_mode=True,
        force_standard_requested=True,
        force_standard_confirmed=False,
        reason="",
    )

    assert override.effective_large_data_mode is True
    assert "Confirm" in override.blocked_reason


def test_confirmed_large_data_override_forces_standard_execution(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    input_path.write_text("placeholder", encoding="utf-8")
    selected_input = _selected_file(input_path)

    override = resolve_large_data_execution_override(
        selected_input=selected_input,
        detected_large_data_mode=True,
        force_standard_requested=True,
        force_standard_confirmed=True,
        reason="Sized instance validation",
    )
    config = build_config_for_large_data_execution_override(_test_config(), override)

    assert override.effective_large_data_mode is False
    assert override.user_override_disabled is True
    assert config.performance.large_data_mode is False
    assert config.performance.large_data_auto_detected is True
    assert config.performance.large_data_user_override_disabled is True
    assert config.performance.large_data_override_reason == "Sized instance validation"
    assert config.performance.large_data_standard_execution_override_reason == (
        "Sized instance validation"
    )
    assert config.performance.large_data_effective_mode == "standard_in_memory_forced"


def test_large_data_override_blocks_s3_inputs() -> None:
    selected_input = SelectedInputDataset(
        pd.DataFrame({"x": [1], "target": [0]}),
        "s3://bucket/input.parquet",
        {"source_kind": "s3", "relative_path": "s3://bucket/input.parquet"},
        dataset_handle=object(),
        large_data_mode=True,
    )

    override = resolve_large_data_execution_override(
        selected_input=selected_input,
        detected_large_data_mode=True,
        force_standard_requested=True,
        force_standard_confirmed=True,
        reason="Try full memory",
    )

    assert override.effective_large_data_mode is True
    assert "S3 inputs require Large Data Mode" in override.blocked_reason


def test_standard_override_uses_full_file_path_not_preview(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    input_path.write_text("placeholder", encoding="utf-8")
    selected_input = _selected_file(input_path, source_kind="local_path")

    run_input = build_run_input(
        dataframe=selected_input.dataframe,
        selected_input=selected_input,
        large_data_mode=False,
    )

    assert run_input == input_path


def test_execution_plan_labels_forced_standard_data_mode(tmp_path: Path) -> None:
    selected_input = _selected_file(tmp_path / "input.parquet")
    override = resolve_large_data_execution_override(
        selected_input=selected_input,
        detected_large_data_mode=True,
        force_standard_requested=True,
        force_standard_confirmed=True,
        reason="",
    )
    config = build_config_for_large_data_execution_override(_test_config(), override)

    cards = build_execution_plan_cards(
        preview_config=config,
        data_source_label="large file",
        large_data_mode=override.effective_large_data_mode,
        large_data_user_override_disabled=override.user_override_disabled,
    )

    assert {
        "label": "Data Mode",
        "value": "Standard in-memory execution forced by user override",
    } in cards
