"""Diagnostic registry used by runtime, documentation, and audit tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_pd_framework.config import TargetMode


@dataclass(frozen=True)
class DiagnosticDescriptor:
    key: str
    label: str
    family: str
    config_path: str
    tables: tuple[str, ...]
    figures: tuple[str, ...] = ()
    target_modes: tuple[TargetMode, ...] | None = None
    requires_labels: bool = False
    large_data_behavior: str = "sampled"


DIAGNOSTIC_REGISTRY: tuple[DiagnosticDescriptor, ...] = (
    DiagnosticDescriptor(
        key="data_quality",
        label="Data quality",
        family="Data",
        config_path="diagnostics.data_quality",
        tables=("data_quality_summary",),
        large_data_behavior="sampled metadata",
    ),
    DiagnosticDescriptor(
        key="descriptive_statistics",
        label="Descriptive statistics",
        family="Data",
        config_path="diagnostics.descriptive_statistics",
        tables=("descriptive_statistics",),
    ),
    DiagnosticDescriptor(
        key="missingness_analysis",
        label="Missingness analysis",
        family="Data",
        config_path="diagnostics.missingness_analysis",
        tables=("missingness", "missingness_by_split"),
        figures=("missingness", "missingness_by_split"),
    ),
    DiagnosticDescriptor(
        key="correlation_analysis",
        label="Correlation analysis",
        family="Feature Review",
        config_path="diagnostics.correlation_analysis",
        tables=("correlation_matrix",),
        figures=("correlation_heatmap",),
    ),
    DiagnosticDescriptor(
        key="vif_analysis",
        label="VIF analysis",
        family="Feature Review",
        config_path="diagnostics.vif_analysis",
        tables=("vif",),
        figures=("vif_profile",),
    ),
    DiagnosticDescriptor(
        key="woe_iv_analysis",
        label="WoE / IV analysis",
        family="Binary Model",
        config_path="diagnostics.woe_iv_analysis",
        tables=("woe_iv_summary", "woe_iv_detail"),
        target_modes=(TargetMode.BINARY,),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="psi_analysis",
        label="Population Stability Index",
        family="Stability",
        config_path="diagnostics.psi_analysis",
        tables=("psi",),
        figures=("psi_profile",),
    ),
    DiagnosticDescriptor(
        key="adf_analysis",
        label="ADF stationarity tests",
        family="Time Series",
        config_path="diagnostics.adf_analysis",
        tables=("adf_tests",),
    ),
    DiagnosticDescriptor(
        key="model_specification_tests",
        label="Model specification tests",
        family="Statistical Tests",
        config_path="diagnostics.model_specification_tests",
        tables=("model_specification_tests", "model_influence_summary"),
        figures=("model_influence_plot",),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="forecasting_statistical_tests",
        label="Forecasting statistical tests",
        family="Time Series",
        config_path="diagnostics.forecasting_statistical_tests",
        tables=(
            "forecasting_statistical_tests",
            "cointegration_tests",
            "granger_causality_tests",
        ),
    ),
    DiagnosticDescriptor(
        key="calibration_analysis",
        label="Calibration analysis",
        family="Binary Model",
        config_path="diagnostics.calibration_analysis",
        tables=("calibration", "calibration_summary"),
        figures=("calibration_curve", "calibration_method_comparison"),
        target_modes=(TargetMode.BINARY,),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="threshold_analysis",
        label="Threshold analysis",
        family="Binary Model",
        config_path="diagnostics.threshold_analysis",
        tables=("threshold_analysis",),
        figures=("threshold_analysis",),
        target_modes=(TargetMode.BINARY,),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="lift_gain_analysis",
        label="Lift and gain analysis",
        family="Binary Model",
        config_path="diagnostics.lift_gain_analysis",
        tables=("lift_gain",),
        figures=("gain_chart", "lift_chart"),
        target_modes=(TargetMode.BINARY,),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="segment_analysis",
        label="Segment analysis",
        family="Segmentation",
        config_path="diagnostics.segment_analysis",
        tables=("segment_performance",),
        figures=("segment_performance_chart", "segment_volume"),
    ),
    DiagnosticDescriptor(
        key="residual_analysis",
        label="Residual analysis",
        family="Continuous Model",
        config_path="diagnostics.residual_analysis",
        tables=("residual_summary",),
        figures=("residuals_vs_predicted", "actual_vs_predicted"),
        target_modes=(TargetMode.CONTINUOUS,),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="quantile_analysis",
        label="Quantile analysis",
        family="Backtesting",
        config_path="diagnostics.quantile_analysis",
        tables=("quantile_summary",),
        figures=("quantile_backtest",),
    ),
    DiagnosticDescriptor(
        key="qq_analysis",
        label="QQ analysis",
        family="Continuous Model",
        config_path="diagnostics.qq_analysis",
        tables=("qq_plot_data",),
        figures=("qq_plot",),
        target_modes=(TargetMode.CONTINUOUS,),
        requires_labels=True,
    ),
    DiagnosticDescriptor(
        key="explainability",
        label="Feature-effect explainability",
        family="Explainability",
        config_path="explainability.enabled",
        tables=(
            "partial_dependence",
            "ice_curves",
            "accumulated_local_effects",
            "two_way_feature_effects",
        ),
        figures=("partial_dependence",),
        large_data_behavior="sampled and batched",
    ),
    DiagnosticDescriptor(
        key="robustness",
        label="Robustness diagnostics",
        family="Robustness",
        config_path="robustness.enabled",
        tables=("robustness_metric_distribution", "robustness_feature_distribution"),
        figures=("robustness_metric_boxplot", "robustness_feature_stability"),
        large_data_behavior="off by default for large data",
    ),
    DiagnosticDescriptor(
        key="credit_risk",
        label="Credit risk development diagnostics",
        family="Credit Risk",
        config_path="credit_risk.enabled",
        tables=("vintage_summary", "migration_matrix", "lgd_segment_summary"),
        figures=("vintage_curve", "migration_heatmap", "lgd_segment_chart"),
    ),
)


def build_diagnostic_registry_table(context: Any) -> pd.DataFrame:
    """Builds a runtime registry showing enabled, skipped, and emitted diagnostics."""

    rows: list[dict[str, Any]] = []
    labels_available = bool(context.metadata.get("labels_available", False))
    target_mode = context.config.target.mode
    emitted_tables = set(context.diagnostics_tables)
    emitted_figures = set(context.visualizations)

    for descriptor in DIAGNOSTIC_REGISTRY:
        configured = bool(_resolve_config_path(context.config, descriptor.config_path))
        mode_allowed = descriptor.target_modes is None or target_mode in descriptor.target_modes
        label_allowed = not descriptor.requires_labels or labels_available
        emitted = bool(
            emitted_tables.intersection(descriptor.tables)
            or emitted_figures.intersection(descriptor.figures)
        )
        if not configured:
            status = "disabled"
        elif not mode_allowed:
            status = "skipped_target_mode"
        elif not label_allowed:
            status = "skipped_labels_unavailable"
        elif emitted:
            status = "emitted"
        else:
            status = "configured_no_output"

        rows.append(
            {
                "diagnostic_key": descriptor.key,
                "label": descriptor.label,
                "family": descriptor.family,
                "config_path": descriptor.config_path,
                "configured": configured,
                "status": status,
                "requires_labels": descriptor.requires_labels,
                "target_modes": ",".join(mode.value for mode in descriptor.target_modes)
                if descriptor.target_modes
                else "all",
                "large_data_behavior": descriptor.large_data_behavior,
                "expected_tables": ", ".join(descriptor.tables),
                "expected_figures": ", ".join(descriptor.figures),
            }
        )
    return pd.DataFrame(rows)


def _resolve_config_path(config: Any, dotted_path: str) -> Any:
    current = config
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current
