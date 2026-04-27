"""Tests for the shared GUI/report presentation helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.io as pio

from quant_pd_framework.presentation import (
    apply_advanced_visual_analytics,
    apply_fintech_figure_theme,
    build_asset_catalog,
    build_interactive_report_html,
    enhance_report_visualizations,
    friendly_asset_title,
    prepare_display_table,
    report_chart_guidance,
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
    assert catalog["statistical_tests"]["tables"][0].key == "adf_tests"
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


def test_enhance_report_visualizations_adds_companion_charts() -> None:
    diagnostics_tables = {
        "roc_curve": pd.DataFrame({"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0]}),
        "precision_recall_curve": pd.DataFrame(
            {"recall": [0.0, 0.6, 1.0], "precision": [1.0, 0.75, 0.2]}
        ),
        "calibration": pd.DataFrame(
            {
                "bin": ["Q1", "Q2"],
                "mean_predicted_probability": [0.05, 0.35],
                "observed_default_rate": [0.04, 0.42],
            }
        ),
        "psi": pd.DataFrame({"feature_name": ["balance"], "psi": [0.12]}),
        "vif": pd.DataFrame({"feature_name": ["income"], "vif": [6.2]}),
        "missingness_by_split": pd.DataFrame(
            {
                "split": ["train", "test"],
                "column_name": ["balance", "balance"],
                "missing_pct": [0.01, 0.05],
            }
        ),
        "feature_importance": pd.DataFrame(
            {"feature_name": ["balance", "income"], "coefficient": [0.8, -0.3]}
        ),
        "segment_performance": pd.DataFrame(
            {
                "split": ["test", "test"],
                "region": ["north", "south"],
                "observation_count": [50, 40],
                "average_score": [0.21, 0.35],
                "average_actual": [0.20, 0.44],
            }
        ),
        "scenario_summary": pd.DataFrame(
            {"scenario_name": ["Downside", "Upside"], "mean_delta": [0.08, -0.03]}
        ),
        "cross_validation_metric_distribution": pd.DataFrame(
            {"metric_name": ["roc_auc", "roc_auc"], "metric_value": [0.78, 0.82]}
        ),
        "feature_effect_stability": pd.DataFrame(
            {
                "feature_name": ["balance", "balance", "balance"],
                "split": ["train", "validation", "test"],
                "feature_value": [1, 1, 1],
                "average_prediction": [0.20, 0.22, 0.21],
            }
        ),
    }
    predictions = {
        "test": pd.DataFrame(
            {
                "predicted_probability": [0.1, 0.2, 0.8, 0.9],
                "default_status": [0, 0, 1, 1],
            }
        )
    }

    enhanced = enhance_report_visualizations(
        metrics={
            "train": {"roc_auc": 0.86, "ks_statistic": 0.42},
            "test": {
                "roc_auc": 0.81,
                "average_precision": 0.58,
                "ks_statistic": 0.37,
                "brier_score": 0.16,
            },
        },
        diagnostics_tables=diagnostics_tables,
        visualizations={},
        target_mode="binary",
        labels_available=True,
        predictions=predictions,
    )

    assert {
        "split_metric_slope_chart",
        "roc_curve_annotated",
        "precision_recall_curve_annotated",
        "ks_curve_annotated",
        "score_distribution_violin",
        "calibration_residual_bars",
        "psi_threshold_bars",
        "vif_threshold_bars",
        "missingness_split_heatmap",
        "feature_importance_waterfall",
        "segment_performance_dumbbell",
        "scenario_tornado",
        "cross_validation_metric_violin",
        "feature_effect_stability_small_multiples",
    }.issubset(enhanced)


def test_advanced_visual_analytics_adds_exploratory_charts() -> None:
    diagnostics_tables = {
        "feature_importance": pd.DataFrame(
            {
                "feature_name": ["balance", "income", "utilization"],
                "coefficient": [0.7, -0.4, 0.3],
            }
        ),
        "interaction_strength": pd.DataFrame(
            {
                "feature_x": ["balance", "balance", "income"],
                "feature_y": ["income", "utilization", "utilization"],
                "mean_absolute_interaction": [0.08, 0.05, 0.04],
            }
        ),
        "partial_dependence": pd.DataFrame(
            {
                "feature_name": ["balance", "balance", "income", "income"],
                "feature_value": [1, 2, 1, 2],
                "average_prediction": [0.2, 0.3, 0.4, 0.35],
                "sort_order": [1, 2, 1, 2],
            }
        ),
        "ice_curves": pd.DataFrame(
            {
                "feature_name": ["balance", "balance", "balance", "balance"],
                "observation_id": [1, 1, 2, 2],
                "feature_value": [1, 2, 1, 2],
                "prediction": [0.21, 0.29, 0.18, 0.31],
                "sort_order": [1, 2, 1, 2],
            }
        ),
        "correlation_matrix": pd.DataFrame(
            {
                "feature_name": ["balance", "income", "utilization"],
                "balance": [1.0, 0.62, 0.25],
                "income": [0.62, 1.0, -0.41],
                "utilization": [0.25, -0.41, 1.0],
            }
        ),
        "lift_gain": pd.DataFrame(
            {
                "bucket": [1, 2, 3],
                "lift": [2.5, 1.2, 0.5],
                "capture_rate": [0.55, 0.30, 0.15],
                "cumulative_capture_rate": [0.55, 0.85, 1.0],
            }
        ),
        "model_comparison": pd.DataFrame(
            {
                "model": ["Champion", "Challenger"],
                "roc_auc": [0.82, 0.79],
                "ks_statistic": [0.41, 0.37],
                "brier_score": [0.15, 0.17],
            }
        ),
        "scenario_summary": pd.DataFrame(
            {"scenario_name": ["Downside", "Upside"], "mean_delta": [0.08, -0.04]}
        ),
    }
    predictions = {
        "test": pd.DataFrame(
            {
                "as_of_date": pd.date_range("2024-01-01", periods=80, freq="D"),
                "predicted_probability": [0.05 + (index % 20) / 25 for index in range(80)],
                "default_status": [0, 1] * 40,
                "region": ["north", "south", "east", "west"] * 20,
                "balance": [100 + index for index in range(80)],
                "income": [500 - index for index in range(80)],
                "utilization": [(index % 10) / 10 for index in range(80)],
            }
        )
    }

    advanced = apply_advanced_visual_analytics(
        metrics={"test": {"roc_auc": 0.82, "ks_statistic": 0.41, "brier_score": 0.15}},
        diagnostics_tables=diagnostics_tables,
        visualizations={},
        target_mode="binary",
        labels_available=True,
        predictions=predictions,
    )

    assert {
        "advanced_contribution_beeswarm",
        "advanced_interaction_heatmap",
        "advanced_feature_effect_matrix",
        "advanced_segment_calibration_small_multiples",
        "advanced_score_ridgeline",
        "advanced_temporal_score_stream",
        "advanced_correlation_network",
        "advanced_lift_gain_heatmap",
        "advanced_score_decile_treemap",
        "advanced_model_comparison_radar",
        "advanced_scenario_waterfall",
        "advanced_feature_importance_lollipop",
    }.issubset(advanced)


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

    assert 'class="plot-shell plotly-lazy"' in html
    assert 'class="plot-fallback"' in html
    assert 'class="static-plot-fallback"' not in html
    assert 'class="table-shell"' in html
    assert "Quant Studio Report test_run" in html
    assert "Quant Studio Regulatory Model Development Report" in html
    assert '<script src="https://cdn.plot.ly/' not in html
    assert "sendout package" not in html
    assert "Use this chart with the companion tables" not in html


def test_interactive_report_omits_non_specific_chart_guidance() -> None:
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
        metrics={},
        input_rows=100,
        feature_count=5,
        split_summary={"test": {"rows": 20}},
        warnings=[],
        events=[],
        diagnostics_tables={},
        visualizations={"segment_performance_chart": figure},
        include_enhanced_report_visuals=False,
    )

    assert report_chart_guidance("segment_performance_chart") == ""
    assert "Use this chart with the companion tables" not in html
    assert 'class="chart-guidance"' not in html


def test_interactive_report_keeps_specific_chart_guidance() -> None:
    figure = apply_fintech_figure_theme(
        px.line(
            pd.DataFrame(
                {
                    "mean_predicted_probability": [0.1, 0.5, 0.9],
                    "observed_default_rate": [0.08, 0.52, 0.93],
                }
            ),
            x="mean_predicted_probability",
            y="observed_default_rate",
        ),
        title="Calibration Curve",
    )

    html = build_interactive_report_html(
        run_id="test_run",
        model_type="logistic_regression",
        execution_mode="fit_new_model",
        target_mode="binary",
        labels_available=True,
        warning_count=0,
        metrics={},
        input_rows=100,
        feature_count=5,
        split_summary={"test": {"rows": 20}},
        warnings=[],
        events=[],
        diagnostics_tables={},
        visualizations={"calibration_curve": figure},
        include_enhanced_report_visuals=False,
    )

    assert 'class="chart-guidance"' in html
    assert "Closer to the diagonal is better" in html


def test_interactive_report_can_skip_enhanced_companion_visuals() -> None:
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
        diagnostics_tables={"roc_curve": pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})},
        visualizations={},
        include_enhanced_report_visuals=False,
    )

    assert "ROC Curve With Review Bands" not in html
