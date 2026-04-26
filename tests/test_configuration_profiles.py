"""Tests for Streamlit configuration profile persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_pd_framework import ColumnRole, DataStructure, ModelConfig, ModelType
from quant_pd_framework.gui_support import (
    GUIBuildInputs,
    build_column_editor_frame,
    build_framework_config_from_editor,
)
from quant_pd_framework.streamlit_ui.config_profiles import (
    build_configuration_profile,
    compare_profile_to_dataset,
    framework_config_from_profile,
    gui_inputs_from_framework_config,
    load_configuration_profile,
    profile_table_to_frame,
    profile_to_download_bytes,
    save_configuration_profile,
)

TEST_OUTPUT_ROOT = Path("artifacts") / "test_configuration_profiles"


def test_configuration_profile_round_trips_config_and_workspace_tables() -> None:
    dataframe = pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "balance": [100.0, 150.0, 120.0, 90.0],
            "default_status": [0, 1, 0, 0],
        }
    )
    schema_frame = build_column_editor_frame(dataframe)
    schema_frame.loc[schema_frame["name"] == "as_of_date", "role"] = ColumnRole.DATE.value
    schema_frame.loc[schema_frame["name"] == "default_status", "role"] = (
        ColumnRole.TARGET_SOURCE.value
    )
    config = build_framework_config_from_editor(
        schema_frame,
        GUIBuildInputs(
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
            data_structure=DataStructure.TIME_SERIES,
            output_root=TEST_OUTPUT_ROOT / "round_trip_artifacts",
        ),
    )
    feature_dictionary_frame = pd.DataFrame(
        [
            {
                "enabled": True,
                "feature_name": "balance",
                "business_name": "Current Balance",
                "definition": "Outstanding exposure",
                "source_system": "core",
                "unit": "USD",
                "allowed_range": ">=0",
                "missingness_meaning": "",
                "expected_sign": "positive",
                "inclusion_rationale": "Risk exposure",
                "notes": "",
            }
        ]
    )
    transformation_frame = pd.DataFrame(
        [{"enabled": True, "transform_type": "log1p", "source_feature": "balance"}]
    )
    profile = build_configuration_profile(
        profile_name="PD baseline",
        notes="Quarterly development setup",
        dataframe=dataframe,
        data_source_label="Unit Test Data",
        source_metadata={"kind": "unit"},
        framework_config=config,
        schema_frame=schema_frame,
        feature_dictionary_frame=feature_dictionary_frame,
        transformation_frame=transformation_frame,
        feature_review_frame=pd.DataFrame(columns=["feature_name", "decision", "rationale"]),
        scorecard_override_frame=pd.DataFrame(columns=["feature_name", "bin_edges", "rationale"]),
    )

    output_path = save_configuration_profile(profile, directory=TEST_OUTPUT_ROOT / "profiles")
    loaded = load_configuration_profile(output_path)

    loaded_config = framework_config_from_profile(loaded)
    assert loaded_config.model.model_type == ModelType.LOGISTIC_REGRESSION
    assert loaded_config.split.data_structure == DataStructure.TIME_SERIES
    assert loaded["dataset_fingerprint"]["row_count"] == 4
    assert "records" not in loaded["dataset_fingerprint"]
    assert profile_table_to_frame(loaded, "schema").shape == schema_frame.shape
    assert profile_table_to_frame(loaded, "feature_dictionary").iloc[0]["feature_name"] == "balance"


def test_configuration_profile_download_json_is_portable() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "default_status": [0, 1, 0]})
    schema_frame = build_column_editor_frame(dataframe)
    schema_frame.loc[schema_frame["name"] == "default_status", "role"] = (
        ColumnRole.TARGET_SOURCE.value
    )
    config = build_framework_config_from_editor(
        schema_frame,
        GUIBuildInputs(output_root=TEST_OUTPUT_ROOT / "portable_artifacts"),
    )
    profile = build_configuration_profile(
        profile_name="portable",
        notes="",
        dataframe=dataframe,
        data_source_label="Unit Test Data",
        source_metadata=None,
        framework_config=config,
        schema_frame=schema_frame,
        feature_dictionary_frame=pd.DataFrame(),
        transformation_frame=pd.DataFrame(),
        feature_review_frame=pd.DataFrame(),
        scorecard_override_frame=pd.DataFrame(),
    )

    profile_bytes = profile_to_download_bytes(profile)
    loaded_payload = json.loads(profile_bytes.decode("utf-8"))

    assert loaded_payload["metadata"]["profile_name"] == "portable"
    assert loaded_payload["framework_config"]["target"]["source_column"] == "default_status"
    assert loaded_payload["dataset_fingerprint"]["columns"] == ["feature", "default_status"]


def test_gui_inputs_from_framework_config_restores_key_defaults() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "default_status": [0, 1, 0]})
    schema_frame = build_column_editor_frame(dataframe)
    schema_frame.loc[schema_frame["name"] == "default_status", "role"] = (
        ColumnRole.TARGET_SOURCE.value
    )
    config = build_framework_config_from_editor(
        schema_frame,
        GUIBuildInputs(
            model=ModelConfig(model_type=ModelType.XGBOOST, threshold=0.42),
            train_size=0.7,
            validation_size=0.15,
            test_size=0.15,
            positive_values_text="1,charged_off",
            output_root=TEST_OUTPUT_ROOT / "restore_defaults_artifacts",
        ),
    )

    inputs = gui_inputs_from_framework_config(config)

    assert inputs.model.model_type == ModelType.XGBOOST
    assert inputs.model.threshold == 0.42
    assert inputs.train_size == 0.7
    assert inputs.validation_size == 0.15
    assert inputs.test_size == 0.15
    assert inputs.positive_values_text == "1,charged_off"
    assert inputs.output_root == TEST_OUTPUT_ROOT / "restore_defaults_artifacts"


def test_compare_profile_to_dataset_reports_non_blocking_mismatch() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "default_status": [0, 1, 0]})
    schema_frame = build_column_editor_frame(dataframe)
    schema_frame.loc[schema_frame["name"] == "default_status", "role"] = (
        ColumnRole.TARGET_SOURCE.value
    )
    config = build_framework_config_from_editor(
        schema_frame,
        GUIBuildInputs(output_root=TEST_OUTPUT_ROOT / "mismatch_artifacts"),
    )
    profile = build_configuration_profile(
        profile_name="mismatch",
        notes="",
        dataframe=dataframe,
        data_source_label="Unit Test Data",
        source_metadata=None,
        framework_config=config,
        schema_frame=schema_frame,
        feature_dictionary_frame=pd.DataFrame(),
        transformation_frame=pd.DataFrame(),
        feature_review_frame=pd.DataFrame(),
        scorecard_override_frame=pd.DataFrame(),
    )
    new_dataframe = pd.DataFrame({"other_feature": [1, 2], "default_status": [0, 1]})

    warnings = compare_profile_to_dataset(profile, new_dataframe)

    assert any("missing profile columns" in warning for warning in warnings)
    assert any("not present in the profile" in warning for warning in warnings)
    assert any("rows" in warning for warning in warnings)
