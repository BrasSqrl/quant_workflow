"""Tests for the GUI helper functions that build framework configs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_pd_framework import (
    ColumnRole,
    DataStructure,
    ExecutionMode,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    PresetName,
    ScenarioShockOperation,
    TargetMode,
)
from quant_pd_framework.gui_support import (
    GUIBuildInputs,
    build_column_editor_frame,
    build_framework_config_from_editor,
    build_gui_inputs_from_preset,
    default_challengers_for_target_mode,
    parse_positive_values,
    parse_scenario_rows,
)


def test_build_framework_config_from_editor_maps_roles_and_split_fields() -> None:
    dataframe = pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "loan_id": ["L1", "L2", "L3", "L4", "L5"],
            "balance": [10, 20, 30, 40, 50],
            "default_status": [0, 1, 0, 1, 0],
        }
    )
    editor = build_column_editor_frame(dataframe)
    editor.loc[editor["name"] == "as_of_date", "role"] = ColumnRole.DATE.value
    editor.loc[editor["name"] == "loan_id", "role"] = ColumnRole.IDENTIFIER.value
    editor.loc[editor["name"] == "default_status", "role"] = ColumnRole.TARGET_SOURCE.value
    editor.loc[editor["name"] == "balance", "missing_value_policy"] = (
        MissingValuePolicy.MEDIAN.value
    )

    new_row = {
        "enabled": True,
        "source_name": "",
        "name": "portfolio_segment",
        "role": ColumnRole.FEATURE.value,
        "dtype": "string",
        "missing_value_policy": MissingValuePolicy.CONSTANT.value,
        "missing_value_fill_value": "retail",
        "create_if_missing": True,
        "default_value": "retail",
        "keep_source": False,
    }
    editor = pd.concat([editor, pd.DataFrame([new_row])], ignore_index=True)

    config = build_framework_config_from_editor(
        editor,
        GUIBuildInputs(
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
            data_structure=DataStructure.TIME_SERIES,
            positive_values_text="1, default",
            output_root=Path("artifacts/ui_test"),
        ),
    )

    assert config.target.source_column == "default_status"
    assert config.target.positive_values == ["1", "default"]
    assert config.split.date_column == "as_of_date"
    assert config.split.entity_column == "loan_id"
    assert config.split.stratify is False
    assert any(
        spec.name == "portfolio_segment" and spec.create_if_missing
        for spec in config.schema.column_specs
    )
    assert any(
        spec.name == "balance"
        and spec.missing_value_policy == MissingValuePolicy.MEDIAN
        for spec in config.schema.column_specs
    )
    assert any(
        spec.name == "portfolio_segment"
        and spec.missing_value_policy == MissingValuePolicy.CONSTANT
        and spec.missing_value_fill_value == "retail"
        for spec in config.schema.column_specs
    )


def test_build_framework_config_from_editor_requires_exactly_one_target() -> None:
    dataframe = pd.DataFrame({"balance": [1, 2], "default_status": [0, 1]})
    editor = build_column_editor_frame(dataframe)

    with pytest.raises(ValueError, match="exactly one enabled row as the target source"):
        build_framework_config_from_editor(editor, GUIBuildInputs())


def test_parse_positive_values_handles_blanks() -> None:
    assert parse_positive_values(" 1, default , charged_off ") == ["1", "default", "charged_off"]
    assert parse_positive_values("   ") is None


def test_build_framework_config_allows_prior_run_config_in_existing_model_mode() -> None:
    dataframe = pd.DataFrame({"balance": [1, 2], "utilization": [0.1, 0.2]})
    editor = build_column_editor_frame(dataframe)

    config = build_framework_config_from_editor(
        editor,
        GUIBuildInputs(
            execution_mode=ExecutionMode.SCORE_EXISTING_MODEL,
            existing_model_path=Path("artifacts/prior_run/quant_model.joblib"),
            existing_config_path=Path("artifacts/prior_run/run_config.json"),
        ),
    )

    assert config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL
    assert config.execution.existing_config_path == Path("artifacts/prior_run/run_config.json")


def test_build_gui_inputs_from_preset_returns_expected_defaults() -> None:
    inputs = build_gui_inputs_from_preset(PresetName.LGD_SEVERITY)

    assert inputs.preset_name == PresetName.LGD_SEVERITY
    assert inputs.target_mode == TargetMode.CONTINUOUS
    assert inputs.model.model_type == ModelType.TWO_STAGE_LGD_MODEL
    assert inputs.scorecard_workbench.enabled is True
    assert inputs.robustness.enabled is False


def test_parse_scenario_rows_groups_rows_into_typed_scenarios() -> None:
    scenario_config = parse_scenario_rows(
        [
            {
                "scenario_name": "Base Stress",
                "description": "Utilization shock",
                "feature_name": "utilization",
                "operation": ScenarioShockOperation.ADD.value,
                "value": 0.1,
                "enabled": True,
            }
        ]
    )

    assert scenario_config.enabled is True
    assert scenario_config.scenarios[0].feature_shocks[0].operation == ScenarioShockOperation.ADD


def test_default_challengers_for_target_mode_returns_matching_families() -> None:
    assert ModelType.SCORECARD_LOGISTIC_REGRESSION in default_challengers_for_target_mode(
        TargetMode.BINARY
    )
    assert ModelType.TWO_STAGE_LGD_MODEL in default_challengers_for_target_mode(
        TargetMode.CONTINUOUS
    )
