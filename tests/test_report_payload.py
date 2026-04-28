"""Regression tests for interactive report payload controls."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from quant_pd_framework.report_payload import (
    count_figure_points,
    optimize_report_visualizations,
)


def test_report_payload_optimizer_downsamples_large_figures() -> None:
    frame = pd.DataFrame({"x": range(1000), "y": range(1000)})
    figure = px.scatter(frame, x="x", y="y")

    optimized, audit = optimize_report_visualizations(
        {"large_scatter": figure},
        max_points_per_figure=100,
        max_figure_payload_mb=5.0,
        max_total_figure_payload_mb=10.0,
    )

    assert count_figure_points(optimized["large_scatter"]) <= 100
    assert audit.loc[audit["figure_name"] == "large_scatter", "action"].iloc[0] == "downsampled"
    assert audit.loc[audit["figure_name"] == "__total__", "report_points"].iloc[0] <= 100


def test_report_payload_optimizer_skips_figures_over_payload_cap() -> None:
    frame = pd.DataFrame(
        {
            "x": range(500),
            "y": range(500),
            "label": ["long-label-value-" * 20 for _ in range(500)],
        }
    )
    figure = px.scatter(frame, x="x", y="y", hover_data=["label"])

    optimized, audit = optimize_report_visualizations(
        {"oversized": figure},
        max_points_per_figure=500,
        max_figure_payload_mb=0.001,
        max_total_figure_payload_mb=10.0,
    )

    assert "oversized" not in optimized
    assert audit.loc[audit["figure_name"] == "oversized", "action"].iloc[0] == "skipped"
