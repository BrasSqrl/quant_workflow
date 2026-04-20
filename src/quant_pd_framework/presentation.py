"""Shared presentation rules for the GUI and exported diagnostic reports."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline.offline import get_plotlyjs

FINTECH_COLORWAY = [
    "#16324F",
    "#2A6F97",
    "#2C8C7B",
    "#C28A2C",
    "#7C5CFC",
    "#D46A6A",
    "#607089",
    "#0E7490",
]
FINTECH_NEUTRAL = {
    "bg": "#F6F4EF",
    "surface": "#FFFDFC",
    "surface_alt": "#F3EEE5",
    "line": "#D8D1C4",
    "text": "#112033",
    "muted": "#5F6B7A",
    "accent": "#C28A2C",
}

SECTION_SPECS: OrderedDict[str, dict[str, str]] = OrderedDict(
    [
        (
            "data_quality",
            {
                "title": "Data Quality",
                "description": (
                    "Schema integrity, completeness, summary statistics, and feature health."
                ),
            },
        ),
        (
            "sample_segmentation",
            {
                "title": "Sample / Segmentation",
                "description": (
                    "Population mix, split composition, and segment-level score behavior."
                ),
            },
        ),
        (
            "credit_risk_development",
            {
                "title": "Credit-Risk Development",
                "description": (
                    "Vintage curves, migration views, recovery segmentation, and "
                    "macro-sensitivity diagnostics for credit development workflows."
                ),
            },
        ),
        (
            "model_performance",
            {
                "title": "Model Performance",
                "description": (
                    "Ranking power, predictive strength, residual behavior, and feature importance."
                ),
            },
        ),
        (
            "scorecard_workbench",
            {
                "title": "Scorecard / Binning Workbench",
                "description": (
                    "WoE bins, partial points, reason codes, and feature-level scorecard views."
                ),
            },
        ),
        (
            "calibration_thresholds",
            {
                "title": "Calibration / Thresholds",
                "description": (
                    "Probability alignment, decision thresholds, lift, gain, "
                    "and credit-risk diagnostics."
                ),
            },
        ),
        (
            "stability_drift",
            {
                "title": "Stability / Drift",
                "description": (
                    "Population stability and feature drift between development and scored samples."
                ),
            },
        ),
        (
            "backtesting_time",
            {
                "title": "Backtesting / Time Diagnostics",
                "description": (
                    "Quantile backtests, out-of-time validation, and time-aware statistical tests."
                ),
            },
        ),
        (
            "governance_export",
            {
                "title": "Governance / Export Bundle",
                "description": (
                    "Warnings, execution trace, rerun assets, and distribution-ready outputs."
                ),
            },
        ),
    ]
)

TABLE_LABELS = {
    "split_metrics": "Split Metric Summary",
    "feature_importance": "Feature Importance Table",
    "backtest_summary": "Backtest Summary Table",
    "model_comparison": "Model Comparison Summary",
    "data_quality_summary": "Run Quality Summary",
    "imputation_rules": "Imputation Rules",
    "imputation_group_rules": "Imputation Group Rules",
    "imputation_sensitivity_summary": "Imputation Sensitivity Summary",
    "imputation_sensitivity_detail": "Imputation Sensitivity Detail",
    "assumption_checks": "Suitability And Assumption Checks",
    "feature_dictionary": "Feature Dictionary",
    "governed_transformations": "Governed Transformations",
    "interaction_candidates": "Interaction Candidates",
    "variable_selection": "Variable Selection",
    "manual_review_feature_decisions": "Manual Feature Review",
    "scorecard_bin_overrides": "Scorecard Bin Overrides",
    "documentation_metadata": "Documentation Metadata",
    "reproducibility_manifest": "Reproducibility Manifest",
    "model_numerical_diagnostics": "Model Numerical Diagnostics",
    "numerical_warning_summary": "Numerical Warning Summary",
    "descriptive_statistics": "Descriptive Statistics",
    "missingness": "Missingness Profile",
    "missingness_by_split": "Missingness By Split",
    "missingness_target_association": "Missingness Target Association",
    "missingness_indicator_correlation": "Missingness Indicator Correlation",
    "correlation_matrix": "Correlation Matrix",
    "vif": "Variance Inflation Factors",
    "quantile_summary": "Quantile Performance Table",
    "threshold_analysis": "Threshold Sweep Table",
    "calibration": "Calibration Table",
    "calibration_summary": "Calibration Summary",
    "roc_curve": "ROC Curve Data",
    "precision_recall_curve": "Precision-Recall Data",
    "lift_gain": "Lift and Gain Table",
    "woe_iv_summary": "WoE / IV Summary",
    "woe_iv_detail": "WoE / IV Detail",
    "psi": "Population Stability Index",
    "segment_performance": "Segment Performance",
    "workflow_guardrails": "Workflow Guardrails",
    "adf_tests": "Augmented Dickey-Fuller Tests",
    "model_specification_tests": "Model Specification Tests",
    "model_influence_summary": "Model Influence Summary",
    "forecasting_statistical_tests": "Forecasting Statistical Tests",
    "cointegration_tests": "Cointegration Tests",
    "granger_causality_tests": "Granger-Causality Tests",
    "residual_summary": "Residual Summary",
    "qq_plot_data": "QQ Plot Data",
    "coefficient_breakdown": "Coefficient Breakdown",
    "feature_effect_curves": "Feature Effect Curves",
    "permutation_importance": "Permutation Importance",
    "feature_policy_checks": "Feature Policy Checks",
    "scenario_summary": "Scenario Summary",
    "scenario_definitions": "Scenario Definitions",
    "scenario_segment_impacts": "Scenario Segment Impacts",
    "scorecard_woe_table": "Scorecard WoE Table",
    "scorecard_points_table": "Scorecard Points Table",
    "scorecard_scaling_summary": "Scorecard Scaling Summary",
    "scorecard_feature_summary": "Scorecard Feature Summary",
    "scorecard_reason_code_frequency": "Scorecard Reason Code Frequency",
    "lgd_stage_one_coefficients": "LGD Stage-One Coefficients",
    "lgd_stage_two_coefficients": "LGD Stage-Two Coefficients",
    "lifetime_pd_curve": "Lifetime PD Curve Table",
    "vintage_summary": "Vintage Summary",
    "cohort_pd_summary": "Cohort PD Summary",
    "migration_matrix": "Migration Matrix",
    "roll_rate_summary": "Roll-Rate Summary",
    "lgd_segment_summary": "LGD Segment Summary",
    "recovery_segmentation": "Recovery Segmentation",
    "macro_sensitivity": "Macro Sensitivity Summary",
    "robustness_metric_distribution": "Robustness Metric Distribution",
    "robustness_metric_summary": "Robustness Metric Summary",
    "robustness_feature_distribution": "Robustness Feature Distribution",
    "robustness_feature_stability": "Robustness Feature Stability",
}

FIGURE_LABELS = {
    "missingness": "Missingness by Column",
    "missingness_by_split": "Missingness by Split",
    "missingness_indicator_heatmap": "Missingness Indicator Correlation",
    "correlation_heatmap": "Correlation Heatmap",
    "vif_profile": "VIF Feature Profile",
    "quantile_backtest": "Quantile Backtest",
    "threshold_analysis": "Threshold Performance Sweep",
    "calibration_curve": "Calibration Curve",
    "calibration_method_comparison": "Calibration Method Comparison",
    "roc_curve": "ROC Curve",
    "precision_recall_curve": "Precision-Recall Curve",
    "gain_chart": "Cumulative Gain",
    "lift_chart": "Lift by Quantile",
    "feature_importance_overview": "Feature Importance Overview",
    "split_metric_overview": "Metric Comparison by Split",
    "score_distribution_overview": "Score Distribution by Split",
    "segment_performance_chart": "Segment Performance",
    "segment_volume": "Segment Observation Mix",
    "vintage_curve": "Vintage Curve",
    "cohort_pd_curve": "Cohort PD Curve",
    "migration_heatmap": "Migration Heatmap",
    "lgd_segment_chart": "LGD Segment Performance",
    "recovery_segment_chart": "Recovery Segmentation",
    "macro_sensitivity_chart": "Macro Sensitivity",
    "psi_profile": "Population Stability Profile",
    "imputation_sensitivity_impact": "Imputation Sensitivity Impact",
    "residuals_vs_predicted": "Residuals vs Predicted",
    "actual_vs_predicted": "Actual vs Predicted",
    "qq_plot": "Residual QQ Plot",
    "model_influence_plot": "Influence Summary",
    "model_comparison_chart": "Model Comparison Chart",
    "permutation_importance": "Permutation Importance",
    "scenario_summary_chart": "Scenario Summary",
    "scenario_segment_impact": "Scenario Impact by Segment",
    "lifetime_pd_curve": "Lifetime PD Curve",
    "robustness_metric_boxplot": "Robustness Metric Distribution",
    "robustness_metric_summary_chart": "Robustness Metric Summary",
    "robustness_feature_stability": "Feature Stability Profile",
    "scorecard_feature_iv": "Scorecard Feature Information Value",
    "scorecard_score_distribution": "Scorecard Points Distribution",
    "scorecard_reason_code_frequency_chart": "Reason Code Frequency",
}

ASSET_DESCRIPTIONS = {
    "data_quality_summary": "High-level run metadata and feature coverage checks.",
    "imputation_rules": "Training-fit missing-value rules applied to each feature column.",
    "imputation_group_rules": (
        "Segment-aware train-fit fill values learned for grouped imputation rules."
    ),
    "imputation_sensitivity_summary": (
        "Feature-level summary of how alternative fill rules changed scores and key metrics."
    ),
    "imputation_sensitivity_detail": (
        "Policy-by-policy imputation sensitivity results on the chosen evaluation split."
    ),
    "assumption_checks": "Pre-fit suitability checks tied to the chosen model family.",
    "feature_dictionary": "Business definitions and rationale for modeled features.",
    "governed_transformations": "Explicit transformations fit on train and replayed downstream.",
    "interaction_candidates": "Screened interaction candidates ranked by train-split association.",
    "variable_selection": "Train-split feature screening results and selection rationale.",
    "manual_review_feature_decisions": (
        "Human review decisions that overrode or confirmed feature selection."
    ),
    "scorecard_bin_overrides": "Manual numeric bin edges applied to scorecard development.",
    "documentation_metadata": "Captured model-purpose and governance metadata.",
    "reproducibility_manifest": (
        "Run fingerprint metadata for reruns, audits, and package-version traceability."
    ),
    "model_numerical_diagnostics": (
        "Structured estimation-health diagnostics such as convergence, iteration counts, "
        "and optimizer status."
    ),
    "numerical_warning_summary": (
        "Normalized warning records captured during model fitting instead of surfacing as "
        "raw library warnings."
    ),
    "missingness": "Null-rate profile for the current modeled dataset.",
    "missingness_by_split": "Null-rate profile broken out across train, validation, and test.",
    "missingness_target_association": (
        "Association between a feature being missing and the observed target outcome."
    ),
    "missingness_indicator_correlation": (
        "Pairwise correlation across raw missingness indicators for the modeled features."
    ),
    "correlation_heatmap": (
        "Pairwise feature correlation across the most material numeric drivers."
    ),
    "vif": "Collinearity pressure among top numeric features.",
    "segment_performance": (
        "Observed and predicted behavior across the current default segment cut."
    ),
    "split_metrics": "Primary metric pack broken out by train, validation, and test.",
    "feature_importance": "Ranked feature influence from the active model family.",
    "roc_curve": "Binary discrimination curve on held-out data.",
    "precision_recall_curve": "Precision-recall tradeoff under class imbalance.",
    "calibration_curve": "Observed versus predicted risk alignment.",
    "calibration_summary": (
        "Method-level calibration metrics including Brier, log loss, error, and "
        "Hosmer-Lemeshow statistics."
    ),
    "calibration_method_comparison": (
        "Comparison of base and challenger recalibration methods on the held-out test split."
    ),
    "threshold_analysis": ("Decision threshold tradeoffs across key classification metrics."),
    "lift_gain": "Captured defaults by ranked score bucket.",
    "psi": "Population shift between development and current scored populations.",
    "adf_tests": "Stationarity checks for predictions and key time-varying series.",
    "model_specification_tests": (
        "Estimator-form diagnostics such as Box-Tidwell, link tests, and condition indices."
    ),
    "model_influence_summary": (
        "Observation-level leverage and Cook's distance for the fitted specification."
    ),
    "forecasting_statistical_tests": (
        "Residual autocorrelation and heteroskedasticity diagnostics for forecasting runs."
    ),
    "cointegration_tests": "Cointegration checks between the target series and top drivers.",
    "granger_causality_tests": "Macro-driver Granger-causality results on the aggregated series.",
    "quantile_backtest": "Observed and predicted performance by ordered risk bucket.",
    "residual_summary": "Regression error distribution summary.",
    "model_comparison": (
        "Primary-versus-challenger comparison across held-out development splits."
    ),
    "coefficient_breakdown": "Signed coefficient summary for interpretable model review.",
    "feature_effect_curves": (
        "Average predicted response when each key feature is stressed in isolation."
    ),
    "permutation_importance": "Held-out metric degradation when each top feature is shuffled.",
    "feature_policy_checks": (
        "Governance checks for required, excluded, monotonic, and sign-constrained features."
    ),
    "workflow_guardrails": (
        "Preset-aware readiness findings that check model-family, data-structure, and "
        "documentation expectations before execution."
    ),
    "scenario_summary": "Average score impact for each configured scenario shock.",
    "scenario_segment_impacts": "Scenario deltas broken out by the active review segment.",
    "scorecard_woe_table": "Weight-of-evidence bins used by the scorecard challenger.",
    "scorecard_points_table": "Score contribution by feature bucket for the scorecard model.",
    "scorecard_scaling_summary": "Base-score and points-to-double-odds scaling settings.",
    "vintage_summary": ("Observed and predicted behavior grouped by booking or reporting vintage."),
    "cohort_pd_summary": (
        "Observed and predicted default behavior grouped by first-observed cohort."
    ),
    "migration_matrix": "Observed transition counts and rates between ordered delinquency states.",
    "roll_rate_summary": "Improving, stable, and worsening transition shares for detected states.",
    "lgd_segment_summary": "Actual versus predicted LGD by the active segment cut.",
    "recovery_segmentation": "Recovery-rate view derived from LGD actuals or predictions.",
    "macro_sensitivity": (
        "Average score change after shocking macro-linked drivers by one standard deviation."
    ),
    "scorecard_feature_summary": (
        "Feature-level summary of information value, points spread, and monotonicity."
    ),
    "scorecard_reason_code_frequency": (
        "Frequency with which each feature appears in the exported reason-code slots."
    ),
    "lgd_stage_one_coefficients": (
        "Probability-of-loss stage coefficients for the LGD two-stage model."
    ),
    "lgd_stage_two_coefficients": "Severity-stage coefficients for the LGD two-stage model.",
    "lifetime_pd_curve": "Cumulative lifetime PD implied by the discrete-time hazard model.",
    "robustness_metric_distribution": ("Held-out metric values across repeated train resamples."),
    "robustness_metric_summary": (
        "Mean, spread, and percentile summary for repeated held-out metrics."
    ),
    "robustness_feature_distribution": (
        "Resample-level effect and importance values for each modeled feature."
    ),
    "robustness_feature_stability": (
        "Feature-level stability summary across repeated train resamples."
    ),
    "split_metric_overview": (
        "Comparison of the primary metrics across train, validation, and test."
    ),
    "feature_importance_overview": "Top drivers highlighted for executive scanning.",
    "score_distribution_overview": "Modeled score density across available splits.",
    "segment_performance_chart": ("Observed and predicted rates across the selected segment view."),
    "segment_volume": "Relative concentration of observations by segment.",
    "psi_profile": "Feature-level PSI detail for stability review.",
    "vif_profile": "Visual view of multicollinearity pressure across top features.",
    "actual_vs_predicted": "Regression fit of observed versus predicted values.",
    "qq_plot": "Residual distribution against a normal reference line.",
    "model_comparison_chart": (
        "Validation or test metric comparison across champion and challengers."
    ),
    "scenario_summary_chart": "Average predicted-score impact from each configured scenario.",
    "scenario_segment_impact": "Segment-level differences under each configured scenario.",
    "scorecard_feature_iv": "Top scorecard features ranked by information value.",
    "scorecard_score_distribution": "Distribution of total scorecard points on the scored split.",
    "scorecard_reason_code_frequency_chart": (
        "How often each feature appears as a leading reason code."
    ),
    "robustness_metric_boxplot": "Distribution of held-out metrics across repeated resamples.",
    "robustness_metric_summary_chart": (
        "Average and standard deviation of held-out metrics across repeated resamples."
    ),
}

FEATURED_ASSETS = {
    "data_quality_summary",
    "imputation_rules",
    "assumption_checks",
    "feature_dictionary",
    "governed_transformations",
    "variable_selection",
    "manual_review_feature_decisions",
    "documentation_metadata",
    "reproducibility_manifest",
    "missingness",
    "correlation_heatmap",
    "segment_performance",
    "split_metrics",
    "feature_importance",
    "feature_importance_overview",
    "roc_curve_chart",
    "calibration_curve",
    "calibration_summary",
    "calibration_method_comparison",
    "quantile_backtest",
    "psi_profile",
    "adf_tests",
    "robustness_metric_summary",
    "robustness_feature_stability",
    "scorecard_feature_summary",
    "scorecard_feature_iv",
}


@dataclass(frozen=True, slots=True)
class AssetDescriptor:
    """Metadata used to render charts and tables consistently across surfaces."""

    key: str
    title: str
    kind: str
    section: str
    description: str
    featured: bool


def build_asset_catalog(
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, go.Figure],
) -> OrderedDict[str, dict[str, list[AssetDescriptor]]]:
    """Groups diagnostic assets into the shared reporting taxonomy."""

    catalog: OrderedDict[str, dict[str, list[AssetDescriptor]]] = OrderedDict(
        (
            section_id,
            {
                "title": section_spec["title"],
                "description": section_spec["description"],
                "tables": [],
                "figures": [],
            },
        )
        for section_id, section_spec in SECTION_SPECS.items()
    )

    for key in tables:
        descriptor = AssetDescriptor(
            key=key,
            title=friendly_asset_title(key, kind="table"),
            kind="table",
            section=infer_asset_section(key, kind="table"),
            description=ASSET_DESCRIPTIONS.get(key, ""),
            featured=key in FEATURED_ASSETS,
        )
        catalog[descriptor.section]["tables"].append(descriptor)

    for key in figures:
        descriptor = AssetDescriptor(
            key=key,
            title=friendly_asset_title(key, kind="figure"),
            kind="figure",
            section=infer_asset_section(key, kind="figure"),
            description=ASSET_DESCRIPTIONS.get(key, ""),
            featured=key in FEATURED_ASSETS
            or key.startswith("time_backtest_")
            or key.startswith("feature_importance_overview"),
        )
        catalog[descriptor.section]["figures"].append(descriptor)

    return catalog


def infer_asset_section(asset_key: str, *, kind: str) -> str:
    """Maps an asset key to the section where it should be rendered."""

    if asset_key in {
        "data_quality_summary",
        "imputation_rules",
        "imputation_group_rules",
        "imputation_sensitivity_summary",
        "imputation_sensitivity_detail",
        "assumption_checks",
        "feature_dictionary",
        "governed_transformations",
        "interaction_candidates",
        "variable_selection",
        "documentation_metadata",
        "descriptive_statistics",
        "missingness",
        "missingness_by_split",
        "missingness_target_association",
        "missingness_indicator_correlation",
        "missingness_indicator_heatmap",
        "correlation_matrix",
        "correlation_heatmap",
        "vif",
        "vif_profile",
    }:
        return "data_quality"
    if asset_key in {
        "segment_performance",
        "segment_performance_chart",
        "segment_volume",
    }:
        return "sample_segmentation"
    if asset_key in {
        "vintage_summary",
        "vintage_curve",
        "cohort_pd_summary",
        "cohort_pd_curve",
        "migration_matrix",
        "migration_heatmap",
        "roll_rate_summary",
        "lgd_segment_summary",
        "lgd_segment_chart",
        "recovery_segmentation",
        "recovery_segment_chart",
        "macro_sensitivity",
        "macro_sensitivity_chart",
    }:
        return "credit_risk_development"
    if asset_key in {
        "split_metrics",
        "feature_importance",
        "feature_importance_overview",
        "model_comparison",
        "model_comparison_chart",
        "split_metric_overview",
        "score_distribution_overview",
        "roc_curve",
        "roc_curve_chart",
        "precision_recall_curve",
        "precision_recall_curve_chart",
        "residual_summary",
        "residuals_vs_predicted",
        "actual_vs_predicted",
        "qq_plot_data",
        "qq_plot",
        "model_specification_tests",
        "model_influence_summary",
        "model_influence_plot",
        "coefficient_breakdown",
        "feature_effect_curves",
        "permutation_importance",
        "lgd_stage_one_coefficients",
        "lgd_stage_two_coefficients",
        "scenario_summary",
        "scenario_definitions",
        "scenario_segment_impacts",
        "scenario_summary_chart",
        "scenario_segment_impact",
    }:
        return "model_performance"
    if asset_key in {
        "scorecard_woe_table",
        "scorecard_points_table",
        "scorecard_scaling_summary",
        "scorecard_feature_summary",
        "scorecard_reason_code_frequency",
        "scorecard_feature_iv",
        "scorecard_score_distribution",
        "scorecard_reason_code_frequency_chart",
    }:
        return "scorecard_workbench"
    if asset_key in {
        "threshold_analysis",
        "threshold_analysis_chart",
        "calibration",
        "calibration_summary",
        "calibration_curve",
        "calibration_method_comparison",
        "lift_gain",
        "gain_chart",
        "lift_chart",
        "woe_iv_summary",
        "woe_iv_detail",
    }:
        return "calibration_thresholds"
    if asset_key in {
        "psi",
        "psi_profile",
        "imputation_sensitivity_impact",
        "robustness_metric_distribution",
        "robustness_metric_summary",
        "robustness_metric_boxplot",
        "robustness_metric_summary_chart",
        "robustness_feature_distribution",
        "robustness_feature_stability",
    }:
        return "stability_drift"
    if asset_key in {
        "backtest_summary",
        "quantile_summary",
        "quantile_backtest",
        "adf_tests",
        "forecasting_statistical_tests",
        "cointegration_tests",
        "granger_causality_tests",
        "lifetime_pd_curve",
    }:
        return "backtesting_time"
    if asset_key in {
        "feature_policy_checks",
        "workflow_guardrails",
        "manual_review_feature_decisions",
        "reproducibility_manifest",
        "model_numerical_diagnostics",
        "numerical_warning_summary",
    }:
        return "governance_export"
    if asset_key in {"scorecard_bin_overrides"}:
        return "scorecard_workbench"
    if asset_key.startswith("scorecard_bad_rate_"):
        return "scorecard_workbench"
    if asset_key.startswith("scorecard_woe_"):
        return "scorecard_workbench"
    if asset_key.startswith("scorecard_points_"):
        return "scorecard_workbench"
    if asset_key.startswith("time_backtest_"):
        return "backtesting_time"
    if asset_key.startswith("feature_effect_"):
        return "model_performance"
    return "model_performance" if kind == "figure" else "data_quality"


def friendly_asset_title(asset_key: str, *, kind: str = "figure") -> str:
    """Converts internal asset keys into polished display labels."""

    if kind == "table" and asset_key in TABLE_LABELS:
        return TABLE_LABELS[asset_key]
    if kind == "figure" and asset_key in FIGURE_LABELS:
        return FIGURE_LABELS[asset_key]
    if asset_key.startswith("scorecard_bad_rate_"):
        feature_name = asset_key.removeprefix("scorecard_bad_rate_").replace("_", " ").title()
        return f"Scorecard Bad Rate by Bucket ({feature_name})"
    if asset_key.startswith("scorecard_woe_"):
        feature_name = asset_key.removeprefix("scorecard_woe_").replace("_", " ").title()
        return f"Scorecard WoE by Bucket ({feature_name})"
    if asset_key.startswith("scorecard_points_"):
        feature_name = asset_key.removeprefix("scorecard_points_").replace("_", " ").title()
        return f"Scorecard Points by Bucket ({feature_name})"
    if asset_key.startswith("time_backtest_"):
        split_name = asset_key.removeprefix("time_backtest_").replace("_", " ").title()
        return f"Observed vs Predicted Over Time ({split_name})"
    return asset_key.replace("_", " ").title()


def apply_fintech_figure_theme(
    figure: go.Figure,
    *,
    title: str | None = None,
    height: int = 410,
) -> go.Figure:
    """Applies the shared premium-fintech chart styling to a Plotly figure."""

    resolved_title = title or figure.layout.title.text or ""
    figure.update_layout(
        template="plotly_white",
        colorway=FINTECH_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=FINTECH_NEUTRAL["surface"],
        font={
            "family": '"Aptos", "Segoe UI", "Helvetica Neue", sans-serif',
            "size": 13,
            "color": FINTECH_NEUTRAL["text"],
        },
        title={
            "text": resolved_title,
            "x": 0.0,
            "xanchor": "left",
            "font": {
                "family": '"Aptos Display", "Aptos", "Segoe UI", sans-serif',
                "size": 20,
                "color": FINTECH_NEUTRAL["text"],
            },
        },
        height=max(int(figure.layout.height or 0), height),
        margin={"l": 22, "r": 22, "t": 78, "b": 28},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": "#102338",
            "font": {"color": "#F8FAFC", "family": '"Aptos", "Segoe UI", sans-serif'},
        },
    )

    figure.update_xaxes(
        showline=True,
        linecolor=FINTECH_NEUTRAL["line"],
        gridcolor="rgba(17, 32, 51, 0.06)",
        zeroline=False,
        tickfont={"color": FINTECH_NEUTRAL["muted"]},
        title_font={"color": FINTECH_NEUTRAL["muted"], "size": 12},
    )
    figure.update_yaxes(
        showline=True,
        linecolor=FINTECH_NEUTRAL["line"],
        gridcolor="rgba(17, 32, 51, 0.08)",
        zeroline=False,
        tickfont={"color": FINTECH_NEUTRAL["muted"]},
        title_font={"color": FINTECH_NEUTRAL["muted"], "size": 12},
    )

    for index, trace in enumerate(figure.data):
        if trace.type == "bar":
            marker_color = getattr(trace.marker, "color", None)
            if marker_color is None or (isinstance(marker_color, str) and marker_color == ""):
                trace.marker.color = FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)]
            trace.marker.line = {"width": 0}
            trace.opacity = trace.opacity or 0.95
        elif trace.type == "scatter":
            if getattr(trace, "line", None) is not None:
                trace.line.width = max(getattr(trace.line, "width", 0) or 0, 2.8)
            if getattr(trace, "marker", None) is not None:
                marker_size = getattr(trace.marker, "size", None)
                if marker_size is None or np.isscalar(marker_size):
                    trace.marker.size = max(float(marker_size or 0), 7)
                else:
                    trace.marker.sizemin = max(
                        float(getattr(trace.marker, "sizemin", 0) or 0),
                        7,
                    )
        elif trace.type == "heatmap" and not trace.colorscale:
            trace.colorscale = [
                [0.0, "#F2D7A6"],
                [0.5, "#FFFDFC"],
                [1.0, "#16324F"],
            ]

    _make_figure_display_safe(figure)
    return figure


def summarize_run_kpis(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    input_rows: int | None,
    feature_count: int,
    labels_available: bool,
    execution_mode: str,
    model_type: str,
    target_mode: str,
    warning_count: int,
) -> list[dict[str, str]]:
    """Builds the primary KPI strip shown in the GUI and HTML reports."""

    primary_split = metrics.get("test") or next(iter(metrics.values()), {})
    cards = [
        {"label": "Execution Mode", "value": execution_mode.replace("_", " ").title()},
        {"label": "Model Family", "value": model_type.replace("_", " ").title()},
        {"label": "Input Rows", "value": _format_number(input_rows)},
        {"label": "Feature Count", "value": _format_number(feature_count)},
        {"label": "Labels Available", "value": "Yes" if labels_available else "No"},
        {"label": "Warnings", "value": _format_number(warning_count)},
    ]

    if target_mode == "binary":
        cards.extend(
            [
                {
                    "label": "Test ROC AUC",
                    "value": format_metric_value(primary_split.get("roc_auc")),
                },
                {
                    "label": "Test KS",
                    "value": format_metric_value(primary_split.get("ks_statistic")),
                },
                {
                    "label": "Test Brier",
                    "value": format_metric_value(primary_split.get("brier_score")),
                },
            ]
        )
    else:
        cards.extend(
            [
                {
                    "label": "Test RMSE",
                    "value": format_metric_value(primary_split.get("rmse")),
                },
                {
                    "label": "Test MAE",
                    "value": format_metric_value(primary_split.get("mae")),
                },
                {
                    "label": "Test R^2",
                    "value": format_metric_value(primary_split.get("r2")),
                },
            ]
        )

    return cards


def format_metric_value(value: Any) -> str:
    """Formats metrics consistently for cards, tables, and HTML reports."""

    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        absolute = abs(value)
        if absolute >= 1000:
            return f"{value:,.0f}"
        if absolute >= 100:
            return f"{value:,.1f}"
        return f"{value:,.3f}"
    return str(value)


def plotly_display_config() -> dict[str, Any]:
    """Provides a cleaner mode bar and export behavior for interactive charts."""

    return {
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "toggleSpikelines",
        ],
    }


def prepare_display_table(table: pd.DataFrame) -> pd.DataFrame:
    """Converts tables into a display-safe frame for Streamlit and HTML rendering."""

    preview = table.copy(deep=True)
    for column in preview.columns:
        series = preview[column]
        if pd.api.types.is_numeric_dtype(series):
            preview[column] = series.map(
                lambda value: round(float(value), 4) if pd.notna(value) else value
            )
            continue
        if (
            isinstance(series.dtype, pd.IntervalDtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
        ):
            preview[column] = series.map(_normalize_display_scalar)
    return preview


def build_interactive_report_html(
    *,
    run_id: str,
    model_type: str,
    execution_mode: str,
    target_mode: str,
    labels_available: bool,
    warning_count: int,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    input_rows: int | None,
    feature_count: int,
    split_summary: Mapping[str, Mapping[str, Any]],
    warnings: list[str],
    events: list[str],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    visualizations: Mapping[str, go.Figure],
) -> str:
    """Builds the polished standalone HTML dashboard report for each run."""

    asset_catalog = build_asset_catalog(diagnostics_tables, visualizations)
    metric_cards = summarize_run_kpis(
        metrics=metrics,
        input_rows=input_rows,
        feature_count=feature_count,
        labels_available=labels_available,
        execution_mode=execution_mode,
        model_type=model_type,
        target_mode=target_mode,
        warning_count=warning_count,
    )
    metric_cards_html = "".join(
        f"""
        <article class="metric-card">
          <div class="metric-label">{escape(card["label"])}</div>
          <div class="metric-value">{escape(card["value"])}</div>
        </article>
        """
        for card in metric_cards
    )

    split_chips_html = "".join(
        f"""
        <div class="split-chip">
          <span>{escape(split_name.title())}</span>
          <strong>{escape(format_metric_value(summary.get("rows")))} rows</strong>
        </div>
        """
        for split_name, summary in split_summary.items()
    )

    diagnostic_sections_html = "".join(
        _build_section_html(
            section_id=section_id,
            section_title=section_payload["title"],
            section_description=section_payload["description"],
            figure_descriptors=section_payload["figures"],
            table_descriptors=section_payload["tables"],
            figures=visualizations,
            tables=diagnostics_tables,
        )
        for section_id, section_payload in asset_catalog.items()
        if section_payload["figures"] or section_payload["tables"]
    )
    sections_html = diagnostic_sections_html + _build_governance_section_html(
        warning_count=warning_count,
        warnings=warnings,
        events=events,
    )
    plotly_js_bundle = get_plotlyjs()

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Quant Studio Report {escape(run_id)}</title>
    <script>{plotly_js_bundle}</script>
    <style>
      :root {{
        --bg: {FINTECH_NEUTRAL["bg"]};
        --surface: {FINTECH_NEUTRAL["surface"]};
        --surface-alt: {FINTECH_NEUTRAL["surface_alt"]};
        --line: {FINTECH_NEUTRAL["line"]};
        --text: {FINTECH_NEUTRAL["text"]};
        --muted: {FINTECH_NEUTRAL["muted"]};
        --accent: {FINTECH_NEUTRAL["accent"]};
        --shadow: 0 20px 55px rgba(17, 32, 51, 0.08);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Aptos", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(42, 111, 151, 0.09), transparent 24%),
          radial-gradient(circle at top right, rgba(194, 138, 44, 0.11), transparent 22%),
          linear-gradient(180deg, #fcfaf6 0%, var(--bg) 100%);
        color: var(--text);
      }}
      .page {{
        width: min(1400px, calc(100% - 48px));
        margin: 0 auto;
        padding: 36px 0 64px;
      }}
      .hero {{
        display: grid;
        gap: 24px;
        padding: 28px 30px;
        border-radius: 28px;
        background:
          linear-gradient(135deg, rgba(255, 253, 252, 0.98), rgba(246, 238, 225, 0.96));
        border: 1px solid rgba(22, 50, 79, 0.10);
        box-shadow: var(--shadow);
      }}
      .hero-kicker {{
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 12px;
        color: var(--accent);
        margin-bottom: 10px;
      }}
      .hero h1 {{
        margin: 0;
        font-family: "Aptos Display", "Aptos", "Segoe UI", sans-serif;
        font-size: clamp(34px, 4vw, 54px);
        line-height: 1.02;
      }}
      .hero p {{
        margin: 10px 0 0;
        color: var(--muted);
        max-width: 780px;
        font-size: 15px;
      }}
      .hero-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .meta-chip, .split-chip {{
        border-radius: 999px;
        padding: 10px 14px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.82);
        color: var(--text);
        font-size: 13px;
      }}
      .split-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 26px;
      }}
      .metric-card {{
        padding: 18px;
        border-radius: 22px;
        background: var(--surface);
        border: 1px solid rgba(17, 32, 51, 0.08);
        box-shadow: 0 16px 32px rgba(17, 32, 51, 0.05);
      }}
      .metric-label {{
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 10px;
      }}
      .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: var(--text);
      }}
      .section-shell {{
        margin-top: 28px;
        padding: 26px 28px 30px;
        border-radius: 26px;
        background: rgba(255, 253, 250, 0.96);
        border: 1px solid rgba(17, 32, 51, 0.08);
        box-shadow: var(--shadow);
      }}
      .section-header {{
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 18px;
        margin-bottom: 18px;
      }}
      .section-header h2 {{
        margin: 0;
        font-size: 28px;
      }}
      .section-header p {{
        margin: 6px 0 0;
        color: var(--muted);
        max-width: 760px;
      }}
      .figure-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
        gap: 18px;
        align-items: start;
      }}
      .table-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
        gap: 18px;
        margin-top: 18px;
        align-items: start;
      }}
      .asset-card {{
        min-width: 0;
        padding: 18px;
        border-radius: 22px;
        background: var(--surface);
        border: 1px solid rgba(17, 32, 51, 0.08);
        overflow: hidden;
      }}
      .asset-card h3 {{
        margin: 0 0 8px;
        font-size: 18px;
      }}
      .asset-card p {{
        margin: 0 0 14px;
        color: var(--muted);
        font-size: 14px;
      }}
      .asset-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        overflow: hidden;
        border-radius: 16px;
      }}
      .table-shell {{
        width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
      }}
      .plot-shell {{
        width: 100%;
        min-width: 0;
        overflow: hidden;
      }}
      .plot-shell .js-plotly-plot,
      .plot-shell .plot-container,
      .plot-shell .svg-container {{
        width: 100% !important;
        max-width: 100% !important;
      }}
      .plot-shell .modebar {{
        right: 8px !important;
      }}
      .asset-table th {{
        text-align: left;
        background: var(--surface-alt);
        color: var(--text);
        padding: 10px 12px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      .asset-table td {{
        padding: 10px 12px;
        border-top: 1px solid rgba(17, 32, 51, 0.06);
      }}
      .asset-note {{
        margin-top: 10px;
        color: var(--muted);
        font-size: 12px;
      }}
      .asset-card ul, .asset-card ol {{
        margin: 10px 0 0 18px;
        color: var(--muted);
      }}
      @media (max-width: 900px) {{
        .page {{
          width: min(100%, calc(100% - 24px));
          padding-top: 18px;
        }}
        .hero {{
          padding: 22px;
        }}
        .section-shell {{
          padding: 22px;
        }}
        .figure-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <div>
          <div class="hero-kicker">Quantitative Validation Dashboard</div>
          <h1>Run {escape(run_id)}</h1>
          <p>
            Premium validation package for {escape(model_type.replace("_", " ").title())}
            operating in {escape(execution_mode.replace("_", " ").title())}.
          </p>
        </div>
        <div class="hero-meta">
          <div class="meta-chip">Target: <strong>{escape(target_mode.title())}</strong></div>
          <div class="meta-chip">
            Labels Available: <strong>{"Yes" if labels_available else "No"}</strong>
          </div>
          <div class="meta-chip">
            Warnings: <strong>{escape(format_metric_value(warning_count))}</strong>
          </div>
        </div>
        <div class="split-strip">{split_chips_html}</div>
        <div class="metrics-grid">{metric_cards_html}</div>
      </section>
      {sections_html}
    </main>
  </body>
</html>
"""


def _build_section_html(
    *,
    section_id: str,
    section_title: str,
    section_description: str,
    figure_descriptors: list[AssetDescriptor],
    table_descriptors: list[AssetDescriptor],
    figures: Mapping[str, go.Figure],
    tables: Mapping[str, pd.DataFrame],
) -> str:
    figure_cards = "".join(
        _build_figure_card_html(descriptor, figures[descriptor.key])
        for descriptor in figure_descriptors
        if descriptor.key in figures
    )
    table_cards = "".join(
        _build_table_card_html(descriptor, tables[descriptor.key])
        for descriptor in table_descriptors
        if descriptor.key in tables
    )

    if not figure_cards and not table_cards:
        return ""

    return f"""
    <section class="section-shell" id="{escape(section_id)}">
      <div class="section-header">
        <div>
          <h2>{escape(section_title)}</h2>
          <p>{escape(section_description)}</p>
        </div>
      </div>
      {"<div class='figure-grid'>" + figure_cards + "</div>" if figure_cards else ""}
      {"<div class='table-grid'>" + table_cards + "</div>" if table_cards else ""}
    </section>
    """


def _build_figure_card_html(descriptor: AssetDescriptor, figure: go.Figure) -> str:
    safe_figure = go.Figure(figure)
    _make_figure_display_safe(safe_figure)
    figure_html = pio.to_html(
        safe_figure,
        include_plotlyjs=False,
        full_html=False,
        default_width="100%",
        default_height=f"{int(safe_figure.layout.height or 440)}px",
        config=plotly_display_config(),
    )
    description = escape(descriptor.description or "")
    return f"""
    <article class="asset-card">
      <h3>{escape(descriptor.title)}</h3>
      {f"<p>{description}</p>" if description else ""}
      <div class="plot-shell">{figure_html}</div>
    </article>
    """


def _build_table_card_html(descriptor: AssetDescriptor, table: pd.DataFrame) -> str:
    preview = _prepare_table_preview(table).head(12)
    description = escape(descriptor.description or "")
    return f"""
    <article class="asset-card">
      <h3>{escape(descriptor.title)}</h3>
      {f"<p>{description}</p>" if description else ""}
      <div class="table-shell">
        {preview.to_html(index=False, classes="asset-table", border=0)}
      </div>
      <div class="asset-note">
        Showing {min(len(table), 12)} of {len(table):,} rows. Full export is available
        in the tables directory.
      </div>
    </article>
    """


def _prepare_table_preview(table: pd.DataFrame) -> pd.DataFrame:
    return prepare_display_table(table)


def _build_governance_section_html(
    *,
    warning_count: int,
    warnings: list[str],
    events: list[str],
) -> str:
    warning_items = (
        "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
        if warnings
        else "<li>No warnings were recorded for this run.</li>"
    )
    event_items = (
        "".join(f"<li>{escape(event)}</li>" for event in events)
        if events
        else "<li>No pipeline events were recorded.</li>"
    )
    export_items = "".join(
        [
            "<li><strong>run_config.json</strong> stores the fully resolved configuration.</li>",
            "<li><strong>generated_run.py</strong> reruns the bundle without the GUI.</li>",
            "<li><strong>input_snapshot.csv</strong> preserves the scored input when enabled.</li>",
            (
                "<li><strong>committee_report.docx/.pdf</strong> and "
                "<strong>validation_report.docx/.pdf</strong> package the run for "
                "committee and validation review.</li>"
            ),
            (
                "<li><strong>tables/</strong> and <strong>figures/</strong> "
                "hold distribution-ready outputs.</li>"
            ),
        ]
    )
    return f"""
    <section class="section-shell" id="governance_export">
      <div class="section-header">
        <div>
          <h2>{escape(SECTION_SPECS["governance_export"]["title"])}</h2>
          <p>{escape(SECTION_SPECS["governance_export"]["description"])}</p>
        </div>
      </div>
      <div class="table-grid">
        <article class="asset-card">
          <h3>Run Warnings</h3>
          <p>Alerts, caveats, and validation notes that need reviewer attention.</p>
          <div class="asset-note">Warning count: {escape(format_metric_value(warning_count))}</div>
          <ul>{warning_items}</ul>
        </article>
        <article class="asset-card">
          <h3>Pipeline Events</h3>
          <p>Ordered execution events captured during orchestration.</p>
          <ol>{event_items}</ol>
        </article>
        <article class="asset-card">
          <h3>Bundle Contents</h3>
          <p>Primary artifacts included so the run can be reviewed or rerun outside the GUI.</p>
          <ul>{export_items}</ul>
        </article>
      </div>
    </section>
    """


def _format_number(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def _make_figure_display_safe(figure: go.Figure) -> None:
    for trace in figure.data:
        for attribute_name in (
            "x",
            "y",
            "z",
            "text",
            "hovertext",
            "customdata",
            "ids",
        ):
            if not hasattr(trace, attribute_name):
                continue
            attribute_value = getattr(trace, attribute_name)
            normalized = _normalize_plotly_value(attribute_value)
            if normalized is not attribute_value:
                setattr(trace, attribute_name, normalized)


def _normalize_plotly_value(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"b", "i", "u", "f", "M"}:
            return value
        return [_normalize_plotly_value(item) for item in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        if getattr(value, "dtype", None) is not None and pd.api.types.is_numeric_dtype(value.dtype):
            return value.tolist()
        return [_normalize_plotly_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_plotly_value(item) for item in value]
    return _normalize_display_scalar(value)


def _normalize_display_scalar(value: Any) -> Any:
    if pd.isna(value):
        return value
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
