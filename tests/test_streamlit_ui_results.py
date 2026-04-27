"""Regression tests for Streamlit result rendering helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant_pd_framework import TargetMode
from quant_pd_framework.streamlit_ui.artifact_summary import build_artifact_summary_frame
from quant_pd_framework.streamlit_ui.error_guidance import classify_workflow_exception
from quant_pd_framework.streamlit_ui.results import (
    build_suitability_display_table,
    format_memory_bytes,
    resolve_score_column_for_display,
    with_report_enhancement_visualizations,
)
from quant_pd_framework.streamlit_ui.state import (
    build_run_diagnostics,
    infer_snapshot_score_column,
)
from tests.support import temporary_artifact_root


def test_resolve_score_column_for_display_falls_back_to_probability_column() -> None:
    snapshot = {
        "score_column": "prediction_score",
        "target_mode": TargetMode.BINARY.value,
    }
    prediction_frame = pd.DataFrame(
        {
            "predicted_probability": [0.12, 0.34, 0.56],
            "predicted_class": [0, 0, 1],
            "balance": [100.0, 150.0, 200.0],
        }
    )

    resolved = resolve_score_column_for_display(snapshot, prediction_frame)

    assert resolved == "predicted_probability"


def test_report_enhancement_visualizations_respect_snapshot_toggle() -> None:
    snapshot = {
        "include_enhanced_report_visuals": False,
        "metrics": {"test": {"roc_auc": 0.8}},
        "diagnostics_tables": {"roc_curve": pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})},
        "visualizations": {},
        "target_mode": TargetMode.BINARY.value,
        "labels_available": True,
        "predictions": {},
    }

    updated_snapshot = with_report_enhancement_visualizations(snapshot)

    assert updated_snapshot["visualizations"] == {}


def test_format_memory_bytes_uses_compact_units() -> None:
    assert format_memory_bytes(None) == "Not captured"
    assert format_memory_bytes("not-a-number") == "Not captured"
    assert format_memory_bytes(0) == "0 B"
    assert format_memory_bytes(1536) == "1.5 KB"
    assert format_memory_bytes(1024**3) == "1.0 GB"


def test_build_run_diagnostics_uses_peak_tracked_dataframe_memory() -> None:
    context = SimpleNamespace(
        metadata={"run_elapsed_seconds": 12.3},
        debug_trace=[
            {
                "before": {"tracked_dataframe_memory_bytes": 1024},
                "after": {"tracked_dataframe_memory_bytes": 4096},
            },
            {"before": {"tracked_dataframe_memory_bytes": 2048}, "after": {}},
            "unexpected-record",
        ],
    )

    diagnostics = build_run_diagnostics(context)

    assert diagnostics["elapsed_seconds"] == 12.3
    assert diagnostics["peak_tracked_dataframe_memory_bytes"] == 4096
    assert diagnostics["memory_profile_available"] is True


def test_advanced_visual_analytics_respects_snapshot_toggle() -> None:
    snapshot = {
        "include_enhanced_report_visuals": False,
        "include_advanced_visual_analytics": True,
        "metrics": {"test": {"roc_auc": 0.8}},
        "diagnostics_tables": {
            "feature_importance": pd.DataFrame({"feature_name": ["balance"], "coefficient": [0.5]})
        },
        "visualizations": {},
        "target_mode": TargetMode.BINARY.value,
        "labels_available": True,
        "predictions": {
            "test": pd.DataFrame(
                {
                    "predicted_probability": [0.1, 0.2, 0.8, 0.9],
                    "default_status": [0, 0, 1, 1],
                    "balance": [100, 200, 300, 400],
                }
            )
        },
    }

    updated_snapshot = with_report_enhancement_visualizations(snapshot)

    assert "advanced_contribution_beeswarm" in updated_snapshot["visualizations"]


def test_suitability_display_table_puts_failures_first_with_plain_language() -> None:
    table = pd.DataFrame(
        [
            {
                "check_name": "positive_class_rate",
                "check_label": "Positive class rate",
                "subject": "train_target",
                "status": "pass",
                "status_label": "Pass",
                "observed_value": 0.25,
                "threshold": "[0.01, 0.99]",
                "interpretation": "The positive class rate is within range.",
                "why_it_matters": "Class balance affects validation stability.",
                "recommended_action": "No action required.",
            },
            {
                "check_name": "events_per_feature",
                "check_label": "Events per feature",
                "subject": "train_target",
                "status": "fail",
                "status_label": "Fail",
                "observed_value": 8.3,
                "threshold": 10.0,
                "interpretation": "The training split has fewer target events per feature.",
                "why_it_matters": "Low events per feature increases overfitting risk.",
                "recommended_action": "Reduce selected features or add more event observations.",
            },
        ]
    )

    display = build_suitability_display_table(table)

    assert display.iloc[0]["status_label"] == "Fail"
    assert display.iloc[0]["check_label"] == "Events per feature"
    assert "recommended_action" in display.columns
    assert "status" not in display.columns


def test_infer_snapshot_score_column_uses_binary_prediction_outputs_when_metadata_missing() -> None:
    context = SimpleNamespace(
        metadata={},
        predictions={
            "test": pd.DataFrame(
                {
                    "predicted_probability": [0.1, 0.2],
                    "predicted_class": [0, 0],
                }
            )
        },
        config=SimpleNamespace(target=SimpleNamespace(mode=TargetMode.BINARY)),
    )

    resolved = infer_snapshot_score_column(context)

    assert resolved == "predicted_probability"


def test_artifact_summary_frame_prioritizes_primary_locations() -> None:
    with temporary_artifact_root("pytest_artifact_summary") as artifact_root:
        report_path = artifact_root / "interactive_report.html"
        report_path.write_text("<html></html>", encoding="utf-8")

        summary = build_artifact_summary_frame(
            {
                "interactive_report": report_path,
                "model": artifact_root / "missing_model.joblib",
                "output_root": artifact_root,
            }
        )

    assert summary.iloc[0]["key"] == "output_root"
    assert summary.loc[summary["key"].eq("interactive_report"), "status"].iloc[0] == "Available"
    assert summary.loc[summary["key"].eq("model"), "status"].iloc[0] == "Recorded, not found"


def test_workflow_error_guidance_classifies_memory_failures() -> None:
    guidance = classify_workflow_exception(MemoryError("Unable to allocate array"))

    assert "memory" in guidance.title.lower()
    assert "Large Data Mode" in guidance.recommended_action
