"""Tests for the shared GUI/report presentation helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.io as pio

from quant_pd_framework.presentation import (
    apply_fintech_figure_theme,
    build_asset_catalog,
    build_interactive_report_html,
    friendly_asset_title,
    prepare_display_table,
)


def test_build_asset_catalog_groups_assets_into_expected_sections() -> None:
    tables = {
        "missingness": pd.DataFrame({"column_name": ["balance"], "missing_pct": [0.0]}),
        "split_metrics": pd.DataFrame({"split": ["test"], "roc_auc": [0.81]}),
        "psi": pd.DataFrame({"feature_name": ["balance"], "psi": [0.04]}),
        "adf_tests": pd.DataFrame({"split": ["test"], "series_name": ["prediction_mean"]}),
        "scorecard_feature_summary": pd.DataFrame(
            {"feature_name": ["balance"], "information_value": [0.12]}
        ),
        "robustness_metric_summary": pd.DataFrame(
            {"metric_name": ["roc_auc"], "mean_value": [0.81]}
        ),
    }
    figures = {
        "correlation_heatmap": px.imshow(pd.DataFrame([[1.0, 0.2], [0.2, 1.0]])),
        "segment_performance_chart": px.bar(
            pd.DataFrame({"segment": ["A"], "value": [0.1]}),
            x="segment",
            y="value",
        ),
        "quantile_backtest": px.line(
            pd.DataFrame({"risk_band": ["Q1"], "average_score": [0.1]}),
            x="risk_band",
            y="average_score",
        ),
        "scorecard_feature_iv": px.bar(
            pd.DataFrame({"feature_name": ["balance"], "information_value": [0.12]}),
            x="feature_name",
            y="information_value",
        ),
    }

    catalog = build_asset_catalog(tables, figures)

    assert catalog["data_quality"]["tables"][0].key == "missingness"
    assert catalog["model_performance"]["tables"][0].key == "split_metrics"
    assert catalog["stability_drift"]["tables"][0].key == "psi"
    assert catalog["backtesting_time"]["tables"][0].key == "adf_tests"
    assert catalog["scorecard_workbench"]["tables"][0].key == "scorecard_feature_summary"
    assert [descriptor.key for descriptor in catalog["stability_drift"]["tables"]] == [
        "psi",
        "robustness_metric_summary",
    ]
    assert catalog["data_quality"]["figures"][0].key == "correlation_heatmap"
    assert catalog["sample_segmentation"]["figures"][0].key == "segment_performance_chart"
    assert catalog["backtesting_time"]["figures"][0].key == "quantile_backtest"
    assert catalog["scorecard_workbench"]["figures"][0].key == "scorecard_feature_iv"


def test_friendly_asset_title_handles_time_backtest_names() -> None:
    assert (
        friendly_asset_title("time_backtest_validation", kind="figure")
        == "Observed vs Predicted Over Time (Validation)"
    )


def test_prepare_display_table_stringifies_interval_values() -> None:
    table = pd.DataFrame(
        {
            "bucket": pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0]),
            "value": [0.125678, 0.5],
        }
    )

    prepared = prepare_display_table(table)

    assert prepared["bucket"].tolist() == ["(0.0, 1.0]", "(1.0, 2.0]"]
    assert prepared["value"].tolist() == [0.1257, 0.5]


def test_apply_fintech_figure_theme_normalizes_interval_axes_for_plotly_json() -> None:
    frame = pd.DataFrame(
        {
            "bucket": pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0]),
            "score": [0.2, 0.4],
        }
    )
    figure = px.line(frame, x="bucket", y="score", markers=True)

    themed = apply_fintech_figure_theme(figure, title="Interval Safe Figure")
    json_payload = pio.to_json(themed)

    assert "(0.0, 1.0]" in json_payload
    assert "(1.0, 2.0]" in json_payload


def test_interactive_report_html_wraps_plot_and_table_content() -> None:
    table = pd.DataFrame(
        {
            "bucket": pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0]),
            "psi": [0.1, 0.2],
        }
    )
    figure = apply_fintech_figure_theme(
        px.bar(pd.DataFrame({"segment": ["A"], "value": [0.2]}), x="segment", y="value"),
        title="Segment Performance",
    )

    html = build_interactive_report_html(
        run_id="test_run",
        model_type="logistic_regression",
        execution_mode="fit_new_model",
        target_mode="binary",
        labels_available=True,
        warning_count=0,
        metrics={"test": {"roc_auc": 0.81, "ks_statistic": 0.32, "brier_score": 0.18}},
        input_rows=100,
        feature_count=5,
        split_summary={"test": {"rows": 20}},
        warnings=[],
        events=[],
        diagnostics_tables={"psi": table},
        visualizations={"segment_performance_chart": figure},
    )

    assert 'class="plot-shell"' in html
    assert 'class="table-shell"' in html
    assert "Quant Studio Report test_run" in html
    assert '<script src="https://cdn.plot.ly/' not in html
    assert "sendout package" not in html
