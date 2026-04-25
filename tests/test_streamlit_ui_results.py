"""Regression tests for Streamlit result rendering helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant_pd_framework import TargetMode
from quant_pd_framework.streamlit_ui.artifact_summary import build_artifact_summary_frame
from quant_pd_framework.streamlit_ui.error_guidance import classify_workflow_exception
from quant_pd_framework.streamlit_ui.results import resolve_score_column_for_display
from quant_pd_framework.streamlit_ui.state import infer_snapshot_score_column
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
    assert (
        summary.loc[summary["key"].eq("model"), "status"].iloc[0]
        == "Recorded, not found"
    )


def test_workflow_error_guidance_classifies_memory_failures() -> None:
    guidance = classify_workflow_exception(MemoryError("Unable to allocate array"))

    assert "memory" in guidance.title.lower()
    assert "Large Data Mode" in guidance.recommended_action
