"""Tests for enterprise Streamlit workflow helper logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pd_framework import ColumnRole, ModelConfig, ModelType
from quant_pd_framework.gui_support import (
    GUIBuildInputs,
    build_column_editor_frame,
    build_framework_config_from_editor,
)
from quant_pd_framework.streamlit_ui.enterprise_workflow import (
    ReviewerRecord,
    WorkflowStatus,
    build_artifact_explorer_frame,
    build_config_diff_frame,
    build_model_card_markdown,
    build_preflight_summary,
    build_workflow_step_states,
    collect_readiness_issues,
)
from quant_pd_framework.workflow_guardrails import GuardrailFinding


def test_workflow_step_states_flag_ready_and_stale_results() -> None:
    current_config = {"model": {"model_type": "logistic_regression"}}
    states = build_workflow_step_states(
        dataframe_loaded=True,
        preview_config=object(),
        preview_error=None,
        preview_findings=[],
        last_run_snapshot={"config": {"model": {"model_type": "xgboost"}}},
        current_config=current_config,
    )

    assert states[0].status == WorkflowStatus.COMPLETE
    assert states[2].status == WorkflowStatus.READY
    assert states[3].status == WorkflowStatus.NEEDS_ATTENTION
    assert states[4].status == WorkflowStatus.NEEDS_ATTENTION
    assert states[4].label == "Decision Summary"


def test_collect_readiness_issues_maps_errors_and_guardrails() -> None:
    issues = collect_readiness_issues(
        preview_error="Mark exactly one enabled row as the target source.",
        preview_findings=[
            GuardrailFinding(
                severity="warning",
                code="model_family",
                message="Review model family.",
                field_path="model.model_type",
            )
        ],
        profile_warnings=["Current dataset is missing profile columns: x"],
    )

    assert [issue.severity for issue in issues] == ["error", "warning", "warning"]
    assert issues[0].recommended_action.startswith("Open Column Designer")
    assert issues[1].field_path == "model.model_type"


def test_build_config_diff_frame_groups_changes() -> None:
    diff = build_config_diff_frame(
        current_config={"model": {"model_type": "logistic_regression"}, "split": {"train": 0.7}},
        baseline_config={"model": {"model_type": "xgboost"}, "split": {"train": 0.7}},
        baseline_label="profile",
    )

    assert diff.shape[0] == 1
    assert diff.iloc[0]["section"] == "model"
    assert diff.iloc[0]["baseline_value"] == "xgboost"


def test_build_preflight_summary_uses_config_and_editor_tables() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "default_status": [0, 1, 0]})
    editor = build_column_editor_frame(dataframe)
    editor.loc[editor["name"] == "default_status", "role"] = ColumnRole.TARGET_SOURCE.value
    config = build_framework_config_from_editor(
        editor,
        GUIBuildInputs(
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
            output_root=Path("artifacts") / "test_enterprise_workflow",
        ),
    )
    transformations = pd.DataFrame([{"enabled": True, "transform_type": "log1p"}])

    cards, details = build_preflight_summary(
        dataframe=dataframe,
        data_source_label="Unit Test",
        preview_config=config,
        edited_schema=editor,
        transformation_frame=transformations,
    )

    assert any(card["label"] == "Rows" and card["value"] == "3" for card in cards)
    assert any(card["label"] == "Transformations" and card["value"] == "1" for card in cards)
    assert "Output root" in details["area"].tolist()


def test_model_card_includes_reviewer_decision() -> None:
    snapshot = {
        "run_id": "run_1",
        "execution_mode": "fit_new_model",
        "model_type": "logistic_regression",
        "target_mode": "binary",
        "target_column": "default_flag",
        "feature_columns": ["balance", "utilization"],
        "labels_available": True,
        "metrics": {"test": {"roc_auc": 0.8}},
        "warnings": ["review calibration"],
    }
    card = build_model_card_markdown(
        snapshot=snapshot,
        reviewer_record=ReviewerRecord(
            reviewer_name="Reviewer",
            approval_status="Approved with exceptions",
            review_notes="Acceptable.",
            exception_notes="Monitor calibration.",
        ),
    )

    assert "Approved with exceptions" in card
    assert "roc_auc" in card
    assert "Monitor calibration." in card


def test_artifact_explorer_frame_adds_purpose_and_download_hint() -> None:
    frame = build_artifact_explorer_frame({"config": "artifacts/missing_run/run_config.json"})

    config_row = frame.loc[frame["key"] == "config"].iloc[0]
    assert "Resolved run configuration" in config_row["purpose"]
    assert config_row["download_hint"] == "Not available"
