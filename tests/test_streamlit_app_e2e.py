"""End-to-end Streamlit regression tests for the main GUI flows."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
    build_sample_pd_dataframe,
)
from tests.support import build_common_schema, temporary_artifact_root

_ORIGINAL_TEMP_DIR_CLEANUP = tempfile.TemporaryDirectory._cleanup


def _safe_tempdir_cleanup(*args, **kwargs):
    try:
        return _ORIGINAL_TEMP_DIR_CLEANUP(*args, **kwargs)
    except PermissionError:
        return None


tempfile.TemporaryDirectory._cleanup = staticmethod(_safe_tempdir_cleanup)


def _build_app_test():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    return streamlit_testing.AppTest.from_file("app/streamlit_app.py")


def _find_by_label(elements, label: str):
    for element in elements:
        if getattr(element, "label", None) == label:
            return element
    raise AssertionError(f"Unable to find widget with label '{label}'.")


def _configure_sample_schema(at) -> None:
    from app.streamlit_app import build_editor_key

    dataframe = build_sample_pd_dataframe()
    editor_key = build_editor_key(dataframe, "bundled_sample")
    schema_key = f"{editor_key}_schema_frame"
    schema = at.session_state[schema_key].copy(deep=True)
    schema.loc[schema["name"] == "as_of_date", "role"] = ColumnRole.DATE.value
    schema.loc[schema["name"] == "loan_id", "role"] = ColumnRole.IDENTIFIER.value
    schema.loc[schema["name"] == "default_status", "role"] = ColumnRole.TARGET_SOURCE.value
    at.session_state[schema_key] = schema


def _build_existing_model_bundle() -> tuple[Path, Path]:
    dataframe = build_sample_pd_dataframe()
    with temporary_artifact_root("pytest_gui_existing_model") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("loan_id"),
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
            model=ModelConfig(),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)
        model_path = Path(context.artifacts["model"])
        config_path = Path(context.artifacts["config"])
        preserved_root = output_root.parent / f"{output_root.name}_preserved"
        preserved_root.mkdir(parents=True, exist_ok=True)
        preserved_model = preserved_root / model_path.name
        preserved_config = preserved_root / config_path.name
        preserved_model.write_bytes(model_path.read_bytes())
        preserved_config.write_bytes(config_path.read_bytes())
    return preserved_model, preserved_config


def test_streamlit_app_runs_new_model_and_renders_result_tabs() -> None:
    at = _build_app_test()
    at.run(timeout=120)

    _configure_sample_schema(at)
    at.run(timeout=120)
    _find_by_label(at.button, "Run Quant Model Workflow").click().run(timeout=240)

    assert not [element.value for element in at.error]
    success_messages = [element.value for element in at.success]
    assert any("Completed run" in message for message in success_messages)
    result_section = _find_by_label(at.radio, "Result section")
    option_labels = list(result_section.options)
    assert "Overview" in option_labels
    assert "Model Performance" in option_labels
    assert "Governance" in option_labels


def test_streamlit_guided_mode_locks_advanced_controls_until_enabled() -> None:
    at = _build_app_test()
    at.run(timeout=120)

    assert _find_by_label(at.radio, "Workspace mode").value == "guided"
    assert _find_by_label(at.checkbox, "Enable variable selection").disabled is True
    assert _find_by_label(at.checkbox, "Enable explainability outputs").disabled is True

    _find_by_label(at.radio, "Workspace mode").set_value("advanced").run(timeout=120)

    assert _find_by_label(at.checkbox, "Enable variable selection").disabled is False
    assert _find_by_label(at.checkbox, "Enable explainability outputs").disabled is False


def test_streamlit_subset_size_controls_clamp_when_features_are_deselected() -> None:
    from app.streamlit_app import build_editor_key

    at = _build_app_test()
    at.run(timeout=120)

    dataframe = build_sample_pd_dataframe()
    schema_key = f"{build_editor_key(dataframe, 'bundled_sample')}_schema_frame"
    schema = at.session_state[schema_key].copy(deep=True)
    schema.loc[schema["name"] == "as_of_date", "role"] = ColumnRole.DATE.value
    schema.loc[schema["name"] == "loan_id", "role"] = ColumnRole.IDENTIFIER.value
    schema.loc[schema["name"] == "default_status", "role"] = ColumnRole.TARGET_SOURCE.value
    allowed_features = {"annual_income", "debt_to_income", "utilization"}
    feature_mask = schema["role"].eq(ColumnRole.FEATURE.value)
    schema.loc[feature_mask & ~schema["name"].isin(allowed_features), "enabled"] = False
    at.session_state[schema_key] = schema

    at.run(timeout=120)

    assert not [element.value for element in at.error]
    assert _find_by_label(at.number_input, "Maximum subset size").value <= 3


def test_streamlit_app_scores_existing_model_bundle() -> None:
    model_path, config_path = _build_existing_model_bundle()

    at = _build_app_test()
    at.run(timeout=120)
    _find_by_label(at.selectbox, "Execution mode").select("score_existing_model").run(timeout=120)
    _find_by_label(at.text_input, "Existing model artifact path").input(str(model_path)).run(
        timeout=120
    )
    _find_by_label(at.text_input, "Existing run config path").input(str(config_path)).run(
        timeout=120
    )
    _find_by_label(at.button, "Run Quant Model Workflow").click().run(timeout=240)

    assert not [element.value for element in at.error]
    success_messages = [element.value for element in at.success]
    assert any("Completed run" in message for message in success_messages)
