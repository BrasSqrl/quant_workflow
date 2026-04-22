"""Regression tests for Streamlit result rendering helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant_pd_framework import TargetMode
from quant_pd_framework.streamlit_ui.results import resolve_score_column_for_display
from quant_pd_framework.streamlit_ui.state import infer_snapshot_score_column


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
