"""Shared presentation rules for the GUI and exported diagnostic reports."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots

FINTECH_COLORWAY = [
    "#16324F",
    "#2A6F97",
    "#2C8C7B",
    "#C28A2C",
    "#4F6FB7",
    "#D46A6A",
    "#0F7B6C",
    "#E09F3E",
    "#3C5A73",
    "#9A6B2F",
    "#607089",
    "#0E7490",
]
FINTECH_NEUTRAL = {
    "bg": "#F5F1EA",
    "surface": "#FFFDFC",
    "surface_alt": "#F3EEE5",
    "line": "#D8D1C4",
    "text": "#112033",
    "muted": "#5F6B7A",
    "accent": "#C28A2C",
}
REPORT_SEVERITY_COLORS = {
    "great": "#0F8B5F",
    "good": "#2A6F97",
    "watch": "#D99A2B",
    "bad": "#C44536",
    "info": "#607089",
}

SECTION_SPECS: OrderedDict[str, dict[str, str]] = OrderedDict(
    [
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
            "sample_segmentation",
            {
                "title": "Sample / Segmentation",
                "description": (
                    "Population mix, split composition, and segment-level score behavior."
                ),
            },
        ),
        (
            "feature_effects",
            {
                "title": "Feature Effects / Explainability",
                "description": (
                    "Partial dependence, ICE, ALE, marginal effects, interaction strength, "
                    "and effect stability diagnostics."
                ),
            },
        ),
        (
            "advanced_visual_analytics",
            {
                "title": "Advanced Visual Analytics",
                "description": (
                    "Optional exploratory visuals for deeper pattern discovery, executive "
                    "storytelling, and model-insight review."
                ),
            },
        ),
        (
            "statistical_tests",
            {
                "title": "Statistical Tests",
                "description": (
                    "Hypothesis-test and test-like evidence for stationarity, specification, "
                    "distribution shifts, dependency, and challenger significance."
                ),
            },
        ),
        (
            "feature_subset_search",
            {
                "title": "Feature Subset Search",
                "description": (
                    "Candidate feature-set ranking, ROC and KS comparison, significance "
                    "tests, and parsimony tradeoffs."
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
            "data_quality",
            {
                "title": "Data Quality",
                "description": (
                    "Schema integrity, completeness, summary statistics, and feature health."
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
    "advanced_imputation_summary": "Advanced Imputation Summary",
    "multiple_imputation_metric_paths": "Multiple-Imputation Metric Paths",
    "multiple_imputation_pooled_coefficients": "Multiple-Imputation Pooled Coefficients",
    "multiple_imputation_pooling_summary": "Multiple-Imputation Pooling Summary",
    "imputation_sensitivity_summary": "Imputation Sensitivity Summary",
    "imputation_sensitivity_detail": "Imputation Sensitivity Detail",
    "assumption_checks": "Suitability And Assumption Checks",
    "feature_dictionary": "Feature Dictionary",
    "governed_transformations": "Governed Transformations",
    "interaction_candidates": "Interaction Candidates",
    "subset_search_scope": "Subset Search Scope",
    "subset_search_candidates": "Subset Search Candidate Ranking",
    "subset_search_frontier": "Subset Search Frontier",
    "subset_search_feature_frequency": "Subset Search Feature Frequency",
    "subset_search_significance_tests": "Subset Search Significance Tests",
    "subset_search_selected_candidate": "Selected Candidate Snapshot",
    "subset_search_selected_coefficients": "Selected Candidate Coefficients",
    "subset_search_nonwinning_candidates": "Non-Winning Candidate Ranking",
    "variable_selection": "Variable Selection",
    "manual_review_feature_decisions": "Manual Feature Review",
    "scorecard_bin_overrides": "Scorecard Bin Overrides",
    "documentation_metadata": "Documentation Metadata",
    "reproducibility_manifest": "Reproducibility Manifest",
    "validation_checklist": "Validation Checklist",
    "evidence_traceability_map": "Evidence Traceability Map",
    "report_payload_audit": "Report Payload Audit",
    "performance_hardening_actions": "Performance Hardening Actions",
    "model_numerical_diagnostics": "Model Numerical Diagnostics",
    "numerical_warning_summary": "Numerical Warning Summary",
    "descriptive_statistics": "Descriptive Statistics",
    "missingness": "Missingness Profile",
    "missingness_by_split": "Missingness By Split",
    "missingness_target_association": "Missingness Target Association",
    "missingness_indicator_correlation": "Missingness Indicator Correlation",
    "missingness_predictive_power": "Missingness Predictive Power",
    "littles_mcar_test": "Little's MCAR Test",
    "distribution_tests": "Distribution Tests",
    "distribution_shift_tests": "Distribution Shift Tests",
    "correlation_matrix": "Correlation Matrix",
    "vif": "Variance Inflation Factors",
    "dependency_cluster_summary": "Dependency Cluster Summary",
    "condition_index_detail": "Condition Index Detail",
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
    "model_dfbetas_summary": "DFBETAs Summary",
    "model_dffits_summary": "DFFITS Summary",
    "model_comparison_significance_tests": "Model Comparison Significance Tests",
    "forecasting_statistical_tests": "Forecasting Statistical Tests",
    "cointegration_tests": "Cointegration Tests",
    "granger_causality_tests": "Granger-Causality Tests",
    "time_series_extension_tests": "Time-Series Extension Tests",
    "structural_break_tests": "Structural Break Tests",
    "residual_summary": "Residual Summary",
    "residual_diagnostics": "Residual Diagnostics",
    "residual_segment_bias": "Residual Segment Bias",
    "outlier_flags": "Outlier Flags",
    "qq_plot_data": "QQ Plot Data",
    "coefficient_breakdown": "Coefficient Breakdown",
    "feature_effect_curves": "Feature Effect Curves",
    "partial_dependence": "Partial Dependence",
    "ice_curves": "ICE Curves",
    "centered_ice_curves": "Centered ICE Curves",
    "accumulated_local_effects": "Accumulated Local Effects",
    "two_way_feature_effects": "2D Feature Effects",
    "feature_effect_confidence_bands": "Feature Effect Confidence Bands",
    "feature_effect_monotonicity": "Feature Effect Monotonicity",
    "segmented_feature_effects": "Segmented Feature Effects",
    "feature_effect_stability": "Feature Effect Stability",
    "average_marginal_effects": "Average Marginal Effects",
    "interaction_strength": "Interaction Strength",
    "feature_effect_calibration": "Feature Effect Calibration",
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
    "binning_framework_summary": "Binning Framework Summary",
    "manual_binning_profile": "Manual Binning Profile",
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
    "robustness_framework_summary": "Robustness Framework Summary",
    "cross_validation_fold_metrics": "Cross-Validation Fold Metrics",
    "cross_validation_metric_distribution": "Cross-Validation Metric Distribution",
    "cross_validation_metric_summary": "Cross-Validation Metric Summary",
    "cross_validation_feature_distribution": "Cross-Validation Feature Distribution",
    "cross_validation_feature_stability": "Cross-Validation Feature Stability",
    "feature_construction_workbench": "Feature Construction Workbench",
    "preset_imputation_recommendations": "Preset Imputation Recommendations",
    "preset_transformation_recommendations": "Preset Transformation Recommendations",
    "preset_test_recommendations": "Preset Test Recommendations",
}

FIGURE_LABELS = {
    "missingness": "Missingness by Column",
    "missingness_by_split": "Missingness by Split",
    "missingness_indicator_heatmap": "Missingness Indicator Correlation",
    "distribution_shift_overview": "Distribution Shift Overview",
    "correlation_heatmap": "Correlation Heatmap",
    "vif_profile": "VIF Feature Profile",
    "dependency_cluster_heatmap": "Dependency Cluster Heatmap",
    "quantile_backtest": "Quantile Backtest",
    "threshold_analysis": "Threshold Performance Sweep",
    "calibration_curve": "Calibration Curve",
    "calibration_method_comparison": "Calibration Method Comparison",
    "calibration_residual_bars": "Calibration Residual Review",
    "roc_curve": "ROC Curve",
    "roc_curve_annotated": "ROC Curve With Review Bands",
    "precision_recall_curve": "Precision-Recall Curve",
    "precision_recall_curve_annotated": "Precision-Recall Curve With Average Precision",
    "ks_curve_annotated": "KS Separation Curve",
    "gain_chart": "Cumulative Gain",
    "lift_chart": "Lift by Quantile",
    "feature_importance_overview": "Feature Importance Overview",
    "feature_importance_waterfall": "Feature Importance Waterfall",
    "subset_search_auc_frontier": "Subset Search ROC AUC Frontier",
    "subset_search_ks_frontier": "Subset Search KS Frontier",
    "subset_search_metric_frontier": "Subset Search Performance Frontier",
    "subset_search_selected_roc_curve": "Selected Candidate ROC Curve",
    "subset_search_selected_ks_curve": "Selected Candidate KS Curve",
    "subset_search_feature_frequency_chart": "Subset Search Feature Frequency",
    "split_metric_overview": "Metric Comparison by Split",
    "split_metric_slope_chart": "Split Metric Slope Chart",
    "score_distribution_overview": "Score Distribution by Split",
    "score_distribution_violin": "Score Distribution Violin",
    "segment_performance_chart": "Segment Performance",
    "segment_performance_dumbbell": "Segment Performance Dumbbell",
    "segment_volume": "Segment Observation Mix",
    "vintage_curve": "Vintage Curve",
    "cohort_pd_curve": "Cohort PD Curve",
    "migration_heatmap": "Migration Heatmap",
    "lgd_segment_chart": "LGD Segment Performance",
    "recovery_segment_chart": "Recovery Segmentation",
    "macro_sensitivity_chart": "Macro Sensitivity",
    "psi_profile": "Population Stability Profile",
    "psi_threshold_bars": "PSI Threshold Review",
    "vif_threshold_bars": "VIF Threshold Review",
    "missingness_split_heatmap": "Missingness Heatmap by Split",
    "imputation_sensitivity_impact": "Imputation Sensitivity Impact",
    "residuals_vs_predicted": "Residuals vs Predicted",
    "actual_vs_predicted": "Actual vs Predicted",
    "residual_segment_bias": "Residual Bias by Segment",
    "qq_plot": "Residual QQ Plot",
    "model_influence_plot": "Influence Summary",
    "outlier_influence_map": "Outlier / Influence Map",
    "model_comparison_chart": "Model Comparison Chart",
    "permutation_importance": "Permutation Importance",
    "average_marginal_effects": "Average Marginal Effects",
    "interaction_strength": "Interaction Strength",
    "scenario_summary_chart": "Scenario Summary",
    "scenario_tornado": "Scenario Impact Tornado",
    "scenario_segment_impact": "Scenario Impact by Segment",
    "lifetime_pd_curve": "Lifetime PD Curve",
    "robustness_metric_boxplot": "Robustness Metric Distribution",
    "robustness_metric_summary_chart": "Robustness Metric Summary",
    "robustness_feature_stability": "Feature Stability Profile",
    "cross_validation_metric_boxplot": "Cross-Validation Metric Distribution",
    "cross_validation_metric_violin": "Cross-Validation Metric Violin",
    "cross_validation_metric_summary_chart": "Cross-Validation Metric Summary",
    "cross_validation_feature_stability": "Cross-Validation Feature Stability",
    "feature_construction_association": "Constructed Feature Association",
    "manual_binning_distribution": "Manual Binning Distribution",
    "seasonality_profile": "Seasonality Profile",
    "structural_break_profile": "Structural Break / Regime Profile",
    "feature_effect_stability_small_multiples": "Feature Effect Stability Small Multiples",
    "advanced_contribution_beeswarm": "Contribution Beeswarm (Coefficient Proxy)",
    "advanced_interaction_heatmap": "Feature Interaction Heatmap",
    "advanced_feature_effect_matrix": "PDP / ICE Effect Matrix",
    "advanced_segment_calibration_small_multiples": "Segment Calibration Small Multiples",
    "advanced_score_ridgeline": "Score Distribution Ridgeline",
    "advanced_temporal_score_stream": "Temporal Score Stream",
    "advanced_correlation_network": "Feature Correlation Network",
    "advanced_lift_gain_heatmap": "Lift / Gain Decile Heatmap",
    "advanced_score_decile_treemap": "Score Decile Risk Treemap",
    "advanced_model_comparison_radar": "Model Comparison Radar",
    "advanced_scenario_waterfall": "Scenario Sensitivity Waterfall",
    "advanced_feature_importance_lollipop": "Feature Importance Lollipop",
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
    "advanced_imputation_summary": (
        "Model-based imputation settings, auxiliary features, and fallback behavior "
        "for KNN and iterative fills."
    ),
    "multiple_imputation_metric_paths": (
        "Metric values across repeated stochastic imputations used in the pooling framework."
    ),
    "multiple_imputation_pooled_coefficients": (
        "Rubin-style pooled surrogate coefficients and uncertainty summary across "
        "multiply imputed datasets."
    ),
    "multiple_imputation_pooling_summary": (
        "Pooled metric summary across the multiply imputed surrogate-model runs."
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
    "subset_search_scope": (
        "Search-scope metadata including the candidate feature pool, subset limits, "
        "ranking metric, and success/failure counts."
    ),
    "subset_search_candidates": (
        "Candidate subset leaderboard ranked on the chosen held-out split."
    ),
    "subset_search_frontier": (
        "Best-performing subset by feature-count bucket to show parsimony versus performance."
    ),
    "subset_search_feature_frequency": (
        "How often each feature appears across the top-ranked candidate subsets."
    ),
    "subset_search_significance_tests": (
        "Paired comparison tests between the leading candidate subsets."
    ),
    "subset_search_selected_candidate": (
        "Winning candidate snapshot chosen to move forward into the main development workflow."
    ),
    "subset_search_selected_coefficients": (
        "Feature-level coefficients or fallback importance values for the selected "
        "candidate subset."
    ),
    "subset_search_nonwinning_candidates": (
        "Ranked non-winning candidate subsets for side-by-side ROC AUC, KS, and parsimony review."
    ),
    "variable_selection": "Train-split feature screening results and selection rationale.",
    "manual_review_feature_decisions": (
        "Human review decisions that overrode or confirmed feature selection."
    ),
    "scorecard_bin_overrides": "Manual numeric bin edges applied to scorecard development.",
    "documentation_metadata": "Captured model-purpose and governance metadata.",
    "reproducibility_manifest": (
        "Run fingerprint metadata for reruns, audits, and package-version traceability."
    ),
    "validation_checklist": (
        "Reviewer checklist showing complete, attention-needed, and not-applicable "
        "evidence areas for the completed run."
    ),
    "evidence_traceability_map": (
        "Question-to-artifact map that tells reviewers which exported file answers "
        "each common model-development review question."
    ),
    "report_payload_audit": (
        "Interactive-report chart size decisions, including figures kept, downsampled, "
        "or skipped under the configured payload caps."
    ),
    "performance_hardening_actions": (
        "Large-run safeguards applied automatically to keep diagnostics and exports usable."
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
    "missingness_predictive_power": (
        "Ranked summary of where missingness itself is materially associated with the target."
    ),
    "littles_mcar_test": (
        "Approximate Little's MCAR test for whether the observed missingness pattern is "
        "consistent with missing completely at random."
    ),
    "distribution_tests": (
        "Split-level distribution shape diagnostics including skewness, kurtosis, and "
        "normality tests for the top numeric features."
    ),
    "distribution_shift_tests": (
        "Kolmogorov-Smirnov shift checks comparing train to validation and test feature "
        "distributions."
    ),
    "distribution_shift_overview": (
        "Boxplot overview of train-versus-test distribution movement for leading features."
    ),
    "correlation_heatmap": (
        "Pairwise feature correlation across the most material numeric drivers."
    ),
    "vif": "Collinearity pressure among top numeric features.",
    "dependency_cluster_summary": (
        "Correlation-based dependency clusters among the most material numeric features."
    ),
    "condition_index_detail": (
        "Singular-value breakdown behind the condition-index multicollinearity review."
    ),
    "dependency_cluster_heatmap": (
        "Heatmap view of the correlation structure used for dependency clustering."
    ),
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
    "calibration_residual_bars": (
        "Observed-minus-predicted calibration residuals colored by practical review bands."
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
    "model_dfbetas_summary": (
        "Largest per-observation coefficient perturbations from the surrogate influence review."
    ),
    "model_dffits_summary": (
        "Observation-level fitted-value influence summary from the surrogate specification review."
    ),
    "forecasting_statistical_tests": (
        "Residual autocorrelation and heteroskedasticity diagnostics for forecasting runs."
    ),
    "cointegration_tests": "Cointegration checks between the target series and top drivers.",
    "granger_causality_tests": "Macro-driver Granger-causality results on the aggregated series.",
    "time_series_extension_tests": (
        "Extended econometric checks such as Breusch-Godfrey, KPSS, and "
        "Phillips-Perron diagnostics."
    ),
    "seasonality_profile": (
        "Average residual profile by repeating seasonal bucket on the aggregated time series."
    ),
    "structural_break_tests": (
        "Candidate breakpoint table for Chow-style, CUSUM, and CUSUM-squares stability review."
    ),
    "structural_break_profile": (
        "Rolling regime signal used to visualize potential structural breaks over time."
    ),
    "quantile_backtest": "Observed and predicted performance by ordered risk bucket.",
    "residual_summary": "Regression error distribution summary.",
    "residual_diagnostics": (
        "Residual-bias, heteroskedasticity, and autocorrelation checks on the scored split."
    ),
    "residual_segment_bias": (
        "Average residual bias broken out by the current default segment cut."
    ),
    "outlier_flags": (
        "Flagged observations from leverage, Cook's distance, and residual z-score screening."
    ),
    "outlier_influence_map": (
        "Influence-map view for spotting high-leverage and high-residual observations."
    ),
    "model_comparison": (
        "Primary-versus-challenger comparison across held-out development splits."
    ),
    "model_comparison_significance_tests": (
        "Paired significance tests for champion-versus-challenger performance differences."
    ),
    "coefficient_breakdown": "Signed coefficient summary for interpretable model review.",
    "feature_effect_curves": (
        "Average predicted response when each key feature is stressed in isolation."
    ),
    "partial_dependence": (
        "Formal PDP table showing average prediction response over feature grids."
    ),
    "ice_curves": "Individual conditional expectation values for sampled observations.",
    "centered_ice_curves": "ICE values centered to each observation's starting response.",
    "accumulated_local_effects": (
        "ALE curves for correlated numeric predictors using local interval effects."
    ),
    "two_way_feature_effects": "Two-feature response surface for candidate interactions.",
    "feature_effect_confidence_bands": (
        "Bootstrap uncertainty bands around top numeric partial-dependence curves."
    ),
    "feature_effect_monotonicity": (
        "Monotonicity review of feature-effect curves against expected directions."
    ),
    "segmented_feature_effects": (
        "Feature-effect curves broken out by the configured default segment."
    ),
    "feature_effect_stability": (
        "Train, validation, and test feature-effect curves compared on a common grid."
    ),
    "advanced_contribution_beeswarm": (
        "Coefficient-based proxy contribution view for top numeric drivers. This is not a "
        "formal SHAP calculation, but it helps show direction, spread, and outlier influence."
    ),
    "advanced_interaction_heatmap": (
        "Pairwise interaction-strength matrix for the strongest detected feature interactions."
    ),
    "advanced_feature_effect_matrix": (
        "Compact matrix of partial-dependence and ICE-style feature-effect curves."
    ),
    "advanced_segment_calibration_small_multiples": (
        "Segment-level observed-versus-predicted calibration panels for the largest segments."
    ),
    "advanced_score_ridgeline": (
        "Layered score-distribution view across splits or outcomes for separation review."
    ),
    "advanced_temporal_score_stream": (
        "Time-bucketed score stream to show whether average model scores move by split over time."
    ),
    "advanced_correlation_network": (
        "Network view of high-correlation feature relationships and possible redundancy clusters."
    ),
    "advanced_lift_gain_heatmap": (
        "Decile-level lift, capture, and cumulative gain shown as a compact heatmap."
    ),
    "advanced_score_decile_treemap": (
        "Score-decile portfolio map sized by observation count and colored by observed risk."
    ),
    "advanced_model_comparison_radar": (
        "Radar chart comparing available model candidates or splits across common "
        "performance metrics."
    ),
    "advanced_scenario_waterfall": (
        "Cumulative scenario impact view showing how sensitivity cases move the average score."
    ),
    "advanced_feature_importance_lollipop": (
        "Colorful top-driver lollipop view for coefficient or feature-importance magnitude."
    ),
    "average_marginal_effects": (
        "Finite-difference marginal effects on the model prediction scale."
    ),
    "interaction_strength": (
        "Strength of non-additive two-feature effects across response-surface grids."
    ),
    "feature_effect_calibration": ("Actual-versus-predicted calibration by feature bucket."),
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
    "binning_framework_summary": (
        "Feature-level summary of scorecard binning quality and monotonicity review."
    ),
    "manual_binning_profile": (
        "Bucket counts and target averages for explicitly configured manual-bin features."
    ),
    "manual_binning_distribution": (
        "Observation mix across manually configured binning features and buckets."
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
    "robustness_framework_summary": (
        "Coefficient-of-variation summary for the robustness framework's metric outputs."
    ),
    "cross_validation_fold_metrics": (
        "Fold-level validation metrics from temporary cross-validation models."
    ),
    "cross_validation_metric_distribution": (
        "Long-form fold metric values used to assess validation stability."
    ),
    "cross_validation_metric_summary": (
        "Mean, spread, and range of metrics across cross-validation folds."
    ),
    "cross_validation_feature_distribution": (
        "Fold-level feature effects or importances from temporary cross-validation models."
    ),
    "cross_validation_feature_stability": (
        "Feature-level stability summary across cross-validation folds."
    ),
    "feature_construction_workbench": (
        "Constructed-feature preview covering engineered feature types, coverage, and "
        "target association."
    ),
    "feature_construction_association": (
        "Absolute target association across constructed features in the workbench."
    ),
    "preset_imputation_recommendations": (
        "Preset-aligned imputation framework recommendations compared with the current run."
    ),
    "preset_transformation_recommendations": (
        "Preset-aligned transformation framework recommendations compared with the current run."
    ),
    "preset_test_recommendations": (
        "Preset-aligned testing framework recommendations compared with the current run."
    ),
    "split_metric_overview": (
        "Comparison of the primary metrics across train, validation, and test."
    ),
    "split_metric_slope_chart": (
        "Train, validation, and test metric movement shown as a slope chart to highlight drift."
    ),
    "feature_importance_overview": "Top drivers highlighted for executive scanning.",
    "feature_importance_waterfall": (
        "Signed feature contributions or importances displayed as a waterfall-style ranking."
    ),
    "subset_search_auc_frontier": (
        "Relationship between held-out ROC AUC and subset size for the top-ranked candidates."
    ),
    "subset_search_ks_frontier": (
        "Relationship between held-out KS and subset size for the top-ranked candidates."
    ),
    "subset_search_metric_frontier": (
        "Performance-versus-parsimony scatter used to choose a candidate subset to carry forward."
    ),
    "subset_search_selected_roc_curve": ("Held-out ROC curve for the winning candidate subset."),
    "subset_search_selected_ks_curve": ("Held-out KS curve for the winning candidate subset."),
    "subset_search_feature_frequency_chart": (
        "Frequency with which each feature appears across the top-ranked candidates."
    ),
    "score_distribution_overview": "Modeled score density across available splits.",
    "score_distribution_violin": (
        "Score distribution shape by split, including median and tail behavior."
    ),
    "segment_performance_chart": ("Observed and predicted rates across the selected segment view."),
    "segment_performance_dumbbell": (
        "Observed-versus-predicted segment gaps shown as a dumbbell comparison."
    ),
    "segment_volume": "Relative concentration of observations by segment.",
    "psi_profile": "Feature-level PSI detail for stability review.",
    "psi_threshold_bars": "PSI values colored by common stability review bands.",
    "vif_profile": "Visual view of multicollinearity pressure across top features.",
    "vif_threshold_bars": "VIF values colored by common multicollinearity review bands.",
    "missingness_split_heatmap": "Missing-value rates across split and feature.",
    "roc_curve_annotated": "ROC curve with random-model reference and AUC annotation.",
    "precision_recall_curve_annotated": (
        "Precision-recall curve with average precision annotation."
    ),
    "ks_curve_annotated": "KS separation curve with maximum gap marker.",
    "actual_vs_predicted": "Regression fit of observed versus predicted values.",
    "qq_plot": "Residual distribution against a normal reference line.",
    "model_comparison_chart": (
        "Validation or test metric comparison across champion and challengers."
    ),
    "scenario_summary_chart": "Average predicted-score impact from each configured scenario.",
    "scenario_tornado": "Scenario impacts ranked as a tornado chart.",
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
    "cross_validation_metric_boxplot": (
        "Distribution of fold-level validation metrics from cross-validation."
    ),
    "cross_validation_metric_violin": (
        "Fold-level validation metric distribution shown as a violin plot."
    ),
    "cross_validation_metric_summary_chart": (
        "Average and standard deviation of validation metrics across folds."
    ),
    "feature_effect_stability_small_multiples": (
        "Feature-effect stability curves faceted by feature for split-by-split review."
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
    "feature_importance_waterfall",
    "subset_search_candidates",
    "subset_search_frontier",
    "subset_search_auc_frontier",
    "subset_search_ks_frontier",
    "subset_search_selected_candidate",
    "subset_search_selected_coefficients",
    "subset_search_selected_roc_curve",
    "roc_curve_chart",
    "roc_curve_annotated",
    "precision_recall_curve_annotated",
    "ks_curve_annotated",
    "calibration_curve",
    "calibration_summary",
    "calibration_method_comparison",
    "calibration_residual_bars",
    "score_distribution_violin",
    "split_metric_slope_chart",
    "segment_performance_dumbbell",
    "psi_threshold_bars",
    "vif_threshold_bars",
    "missingness_split_heatmap",
    "scenario_tornado",
    "cross_validation_metric_violin",
    "feature_effect_stability_small_multiples",
    "quantile_backtest",
    "psi_profile",
    "adf_tests",
    "robustness_metric_summary",
    "robustness_feature_stability",
    "cross_validation_metric_summary",
    "cross_validation_feature_stability",
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


CHART_GUIDANCE = {
    "threshold_analysis": (
        "Choose thresholds against the operating objective. Watch for cliff points where "
        "small threshold moves sharply change recall, precision, or approval volume."
    ),
    "calibration_curve": (
        "Closer to the diagonal is better. Curves above the line indicate observed risk "
        "is higher than predicted in those bins."
    ),
    "calibration_method_comparison": (
        "Lower Brier and log-loss values are better. Prefer recalibration only when it "
        "improves probability alignment without weakening discrimination."
    ),
    "gain_chart": (
        "Steeper early capture is better because more observed events are found in the "
        "highest-risk ranked groups."
    ),
    "lift_chart": (
        "Higher lift in top quantiles is better. Rapid decay suggests the score is most "
        "useful for prioritizing the riskiest accounts."
    ),
    "roc_curve_annotated": (
        "Higher lift above the diagonal is better. Use AUC with KS and calibration rather "
        "than treating one discrimination metric as sufficient."
    ),
    "precision_recall_curve_annotated": (
        "Most useful when events are rare. Better models hold higher precision as recall increases."
    ),
    "ks_curve_annotated": (
        "The highlighted maximum gap shows separation between event and non-event score "
        "distributions."
    ),
    "calibration_residual_bars": (
        "Bars near zero are best. Watch and bad colors indicate bins where observed outcomes "
        "depart materially from predicted risk."
    ),
    "psi_threshold_bars": (
        "Great is generally below 0.05, good below 0.10, watch begins near 0.10, and bad "
        "typically starts near 0.25."
    ),
    "vif_threshold_bars": (
        "Great is generally below 2.5, good below 5, watch begins near 5, and bad generally "
        "starts near 10."
    ),
    "segment_performance_dumbbell": (
        "Shorter observed-versus-predicted gaps are better. Large gaps identify segments "
        "needing calibration or policy review."
    ),
    "score_distribution_violin": (
        "Review whether train, validation, and test score distributions have similar shape, "
        "center, and tail behavior."
    ),
    "split_metric_slope_chart": (
        "Stable movement from train to validation to test supports generalization. Sharp "
        "drops are a watch item."
    ),
    "feature_importance_waterfall": (
        "Use this as a ranked driver view. Signed bars show direction where the model family "
        "supports signed coefficients."
    ),
    "scenario_tornado": (
        "Larger absolute bars indicate stronger sensitivity to scenario assumptions and may "
        "need narrative support."
    ),
    "cross_validation_metric_violin": (
        "Narrower fold distributions are better. Wide distributions indicate validation "
        "instability."
    ),
    "feature_effect_stability_small_multiples": (
        "Curves with similar shape across splits are more defensible than effects that reverse "
        "or move sharply."
    ),
    "advanced_contribution_beeswarm": (
        "This is a coefficient proxy, not formal SHAP. Wider spread indicates features whose "
        "observed values create more dispersion in modeled contribution."
    ),
    "advanced_interaction_heatmap": (
        "Darker cells indicate stronger non-additive interaction signal and should be reviewed "
        "for policy, stability, and business plausibility."
    ),
    "advanced_feature_effect_matrix": (
        "Use the matrix to scan whether top feature responses are monotonic, smooth, and similar "
        "across PDP/ICE-style views."
    ),
    "advanced_segment_calibration_small_multiples": (
        "Panels far from the diagonal indicate segments where the score may be miscalibrated."
    ),
    "advanced_score_ridgeline": (
        "Better separation appears as visibly different score distributions across outcomes "
        "or splits."
    ),
    "advanced_temporal_score_stream": (
        "Sudden level shifts can indicate population drift, data changes, or time-varying risk."
    ),
    "advanced_correlation_network": (
        "Dense connected clusters can indicate redundancy and may support feature reduction."
    ),
    "advanced_lift_gain_heatmap": (
        "Strong models usually show higher lift and capture in the highest-risk deciles."
    ),
    "advanced_score_decile_treemap": (
        "Large, high-risk tiles show where portfolio exposure and observed risk concentrate."
    ),
    "advanced_model_comparison_radar": (
        "Balanced shapes are easier to defend than models that win one metric but materially "
        "lag others."
    ),
    "advanced_scenario_waterfall": (
        "Large cumulative movements need clear narrative support and scenario-design justification."
    ),
    "advanced_feature_importance_lollipop": (
        "Use this as a quick visual ranking of top drivers before reading coefficient tables."
    ),
}


def report_asset_badge(asset_key: str, *, featured: bool = False) -> tuple[str, str]:
    """Returns the short interpretation badge used in reports and the live UI."""

    if asset_key.startswith("advanced_"):
        return "Advanced Analytics", "info"
    if asset_key in {"psi_threshold_bars", "vif_threshold_bars", "calibration_residual_bars"}:
        return "Review Bands", "watch"
    if asset_key in {
        "roc_curve_annotated",
        "precision_recall_curve_annotated",
        "ks_curve_annotated",
    }:
        return "Decision Metric", "good"
    if asset_key in {
        "feature_importance_waterfall",
        "segment_performance_dumbbell",
        "scenario_tornado",
        "cross_validation_metric_violin",
        "feature_effect_stability_small_multiples",
        "score_distribution_violin",
        "split_metric_slope_chart",
    }:
        return "Companion View", "info"
    if featured:
        return "Featured", "great"
    return "Supporting Evidence", "info"


def report_chart_guidance(asset_key: str) -> str:
    """Returns concise reviewer guidance for a chart, when available."""

    return CHART_GUIDANCE.get(asset_key, "")


SUBSET_SEARCH_HIGHLIGHT_TABLE_KEYS = {
    "subset_search_selected_candidate",
    "subset_search_selected_coefficients",
    "subset_search_nonwinning_candidates",
}
SUBSET_SEARCH_HIGHLIGHT_FIGURE_KEYS = {
    "subset_search_selected_roc_curve",
    "subset_search_selected_ks_curve",
}


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


def prune_subset_search_highlight_assets(
    catalog: OrderedDict[str, dict[str, list[AssetDescriptor]]],
) -> OrderedDict[str, dict[str, list[AssetDescriptor]]]:
    """Removes the dedicated selected-candidate assets from generic section rendering."""

    filtered: OrderedDict[str, dict[str, list[AssetDescriptor]]] = OrderedDict()
    for section_id, payload in catalog.items():
        filtered[section_id] = {
            **payload,
            "tables": [
                descriptor
                for descriptor in payload["tables"]
                if descriptor.key not in SUBSET_SEARCH_HIGHLIGHT_TABLE_KEYS
            ],
            "figures": [
                descriptor
                for descriptor in payload["figures"]
                if descriptor.key not in SUBSET_SEARCH_HIGHLIGHT_FIGURE_KEYS
            ],
        }
    return filtered


def infer_asset_section(asset_key: str, *, kind: str) -> str:
    """Maps an asset key to the section where it should be rendered."""

    if asset_key.startswith("advanced_"):
        return "advanced_visual_analytics"
    if asset_key in {
        "data_quality_summary",
        "imputation_rules",
        "imputation_group_rules",
        "advanced_imputation_summary",
        "multiple_imputation_metric_paths",
        "multiple_imputation_pooled_coefficients",
        "multiple_imputation_pooling_summary",
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
        "missingness_split_heatmap",
        "missingness_target_association",
        "missingness_indicator_correlation",
        "missingness_predictive_power",
        "missingness_indicator_heatmap",
        "feature_construction_workbench",
        "feature_construction_association",
        "correlation_matrix",
        "correlation_heatmap",
        "vif",
        "vif_profile",
        "vif_threshold_bars",
    }:
        return "data_quality"
    if asset_key in {
        "segment_performance",
        "segment_performance_chart",
        "segment_performance_dumbbell",
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
        "feature_importance_waterfall",
        "model_comparison",
        "model_comparison_chart",
        "split_metric_overview",
        "split_metric_slope_chart",
        "score_distribution_overview",
        "score_distribution_violin",
        "roc_curve",
        "roc_curve_chart",
        "roc_curve_annotated",
        "precision_recall_curve",
        "precision_recall_curve_chart",
        "precision_recall_curve_annotated",
        "ks_curve_annotated",
        "residual_summary",
        "residuals_vs_predicted",
        "actual_vs_predicted",
        "qq_plot_data",
        "qq_plot",
        "residual_segment_bias",
        "outlier_influence_map",
        "model_influence_plot",
        "lgd_stage_one_coefficients",
        "lgd_stage_two_coefficients",
    }:
        return "model_performance"
    if asset_key in {
        "coefficient_breakdown",
        "feature_effect_curves",
        "partial_dependence",
        "ice_curves",
        "centered_ice_curves",
        "accumulated_local_effects",
        "two_way_feature_effects",
        "feature_effect_confidence_bands",
        "feature_effect_monotonicity",
        "segmented_feature_effects",
        "feature_effect_stability",
        "feature_effect_stability_small_multiples",
        "average_marginal_effects",
        "interaction_strength",
        "feature_effect_calibration",
        "permutation_importance",
    }:
        return "feature_effects"
    if asset_key in {
        "scenario_summary",
        "scenario_definitions",
        "scenario_segment_impacts",
        "scenario_summary_chart",
        "scenario_tornado",
        "scenario_segment_impact",
    }:
        return "model_performance"
    if asset_key in {
        "subset_search_scope",
        "subset_search_candidates",
        "subset_search_frontier",
        "subset_search_feature_frequency",
        "subset_search_significance_tests",
        "subset_search_selected_candidate",
        "subset_search_selected_coefficients",
        "subset_search_nonwinning_candidates",
        "subset_search_auc_frontier",
        "subset_search_ks_frontier",
        "subset_search_metric_frontier",
        "subset_search_selected_roc_curve",
        "subset_search_selected_ks_curve",
        "subset_search_feature_frequency_chart",
    }:
        return "feature_subset_search"
    if asset_key in {
        "scorecard_woe_table",
        "scorecard_points_table",
        "scorecard_scaling_summary",
        "scorecard_feature_summary",
        "scorecard_reason_code_frequency",
        "scorecard_feature_iv",
        "scorecard_score_distribution",
        "scorecard_reason_code_frequency_chart",
        "binning_framework_summary",
        "manual_binning_profile",
        "manual_binning_distribution",
    }:
        return "scorecard_workbench"
    if asset_key in {
        "threshold_analysis",
        "threshold_analysis_chart",
        "calibration",
        "calibration_summary",
        "calibration_curve",
        "calibration_method_comparison",
        "calibration_residual_bars",
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
        "distribution_shift_overview",
        "dependency_cluster_heatmap",
        "imputation_sensitivity_impact",
        "robustness_metric_distribution",
        "robustness_metric_summary",
        "robustness_metric_boxplot",
        "robustness_metric_summary_chart",
        "robustness_feature_distribution",
        "robustness_feature_stability",
        "robustness_framework_summary",
        "cross_validation_fold_metrics",
        "cross_validation_metric_distribution",
        "cross_validation_metric_summary",
        "cross_validation_metric_boxplot",
        "cross_validation_metric_violin",
        "cross_validation_metric_summary_chart",
        "cross_validation_feature_distribution",
        "cross_validation_feature_stability",
    }:
        return "stability_drift"
    if asset_key in {
        "backtest_summary",
        "quantile_summary",
        "quantile_backtest",
        "lifetime_pd_curve",
    }:
        return "backtesting_time"
    if asset_key in {
        "adf_tests",
        "distribution_tests",
        "distribution_shift_tests",
        "littles_mcar_test",
        "dependency_cluster_summary",
        "condition_index_detail",
        "residual_diagnostics",
        "outlier_flags",
        "model_specification_tests",
        "model_influence_summary",
        "model_dfbetas_summary",
        "model_dffits_summary",
        "model_comparison_significance_tests",
        "forecasting_statistical_tests",
        "cointegration_tests",
        "granger_causality_tests",
        "time_series_extension_tests",
        "seasonality_profile",
        "structural_break_tests",
        "structural_break_profile",
    }:
        return "statistical_tests"
    if asset_key in {
        "preset_imputation_recommendations",
        "preset_transformation_recommendations",
        "preset_test_recommendations",
        "feature_policy_checks",
        "workflow_guardrails",
        "manual_review_feature_decisions",
        "reproducibility_manifest",
        "validation_checklist",
        "evidence_traceability_map",
        "report_payload_audit",
        "performance_hardening_actions",
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
    if asset_key.startswith(
        (
            "feature_effect_",
            "partial_dependence_",
            "ice_",
            "centered_ice_",
            "accumulated_local_effect_",
            "two_way_effect_",
            "segmented_feature_effect_",
        )
    ):
        return "feature_effects"
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
    existing_meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    theme_token = "quant_studio_premium_fintech_v1"
    if (
        existing_meta.get("quant_studio_theme") == theme_token
        and existing_meta.get("quant_studio_theme_title") == resolved_title
        and int(figure.layout.height or 0) >= height
    ):
        return figure

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
    figure.update_layout(
        meta={
            **existing_meta,
            "quant_studio_theme": theme_token,
            "quant_studio_theme_title": resolved_title,
        }
    )
    return figure


def _prepare_report_card_figure(figure: go.Figure) -> None:
    """Optimizes a themed figure for card rendering inside the HTML report."""

    legend_trace_count = sum(
        1
        for trace in figure.data
        if getattr(trace, "showlegend", None) is not False and getattr(trace, "name", None)
    )
    top_margin = 34
    if legend_trace_count:
        top_margin = 72
    if legend_trace_count > 4:
        top_margin = 94

    figure.update_layout(
        autosize=True,
        height=max(int(figure.layout.height or 0), 470),
        margin={"l": 56, "r": 24, "t": top_margin, "b": 58},
        title={"text": ""},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": FINTECH_NEUTRAL["text"]},
            "itemsizing": "constant",
        },
    )


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

    if execution_mode == "search_feature_subsets":
        search_summary = metrics.get("subset_search", {})
        return [
            {"label": "Execution Mode", "value": execution_mode.replace("_", " ").title()},
            {"label": "Model Family", "value": model_type.replace("_", " ").title()},
            {"label": "Input Rows", "value": _format_number(input_rows)},
            {"label": "Candidate Features", "value": _format_number(feature_count)},
            {
                "label": "Enumerated Subsets",
                "value": format_metric_value(search_summary.get("enumerated_subsets")),
            },
            {
                "label": "Successful Subsets",
                "value": format_metric_value(search_summary.get("successful_subsets")),
            },
            {
                "label": "Best Validation AUC",
                "value": format_metric_value(search_summary.get("best_validation_roc_auc")),
            },
            {
                "label": "Best Validation KS",
                "value": format_metric_value(search_summary.get("best_validation_ks_statistic")),
            },
            {
                "label": "Best Feature Count",
                "value": format_metric_value(search_summary.get("best_feature_count")),
            },
            {"label": "Warnings", "value": _format_number(warning_count)},
        ]

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


def enhance_report_visualizations(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    visualizations: Mapping[str, go.Figure],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, go.Figure]:
    """Adds report-only companion charts without changing pipeline calculations."""

    enhanced = dict(visualizations)
    builders = [
        _build_split_metric_slope_chart,
        _build_roc_curve_annotated,
        _build_precision_recall_curve_annotated,
        _build_ks_curve_annotated,
        _build_score_distribution_violin,
        _build_calibration_residual_bars,
        _build_psi_threshold_bars,
        _build_vif_threshold_bars,
        _build_missingness_split_heatmap,
        _build_feature_importance_waterfall,
        _build_segment_performance_dumbbell,
        _build_scenario_tornado,
        _build_cross_validation_metric_violin,
        _build_feature_effect_stability_small_multiples,
    ]
    for builder in builders:
        try:
            result = builder(
                metrics=metrics,
                diagnostics_tables=diagnostics_tables,
                target_mode=target_mode,
                labels_available=labels_available,
                predictions=predictions,
            )
        except Exception:
            continue
        if result is None:
            continue
        figure_key, figure = result
        enhanced.setdefault(figure_key, figure)
    return enhanced


def apply_advanced_visual_analytics(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    visualizations: Mapping[str, go.Figure],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, go.Figure]:
    """Adds optional exploratory visuals for richer model-insight review."""

    advanced = dict(visualizations)
    builders = [
        _build_advanced_contribution_beeswarm,
        _build_advanced_interaction_heatmap,
        _build_advanced_feature_effect_matrix,
        _build_advanced_segment_calibration_small_multiples,
        _build_advanced_score_ridgeline,
        _build_advanced_temporal_score_stream,
        _build_advanced_correlation_network,
        _build_advanced_lift_gain_heatmap,
        _build_advanced_score_decile_treemap,
        _build_advanced_model_comparison_radar,
        _build_advanced_scenario_waterfall,
        _build_advanced_feature_importance_lollipop,
    ]
    for builder in builders:
        try:
            result = builder(
                metrics=metrics,
                diagnostics_tables=diagnostics_tables,
                target_mode=target_mode,
                labels_available=labels_available,
                predictions=predictions,
            )
        except Exception:
            continue
        if result is None:
            continue
        figure_key, figure = result
        advanced.setdefault(figure_key, figure)
    return advanced


def _build_advanced_contribution_beeswarm(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available
    if predictions is None:
        return None
    importance = diagnostics_tables.get("feature_importance")
    if importance is None or importance.empty or "feature_name" not in importance.columns:
        return None
    value_column = "coefficient" if "coefficient" in importance.columns else "importance_value"
    if value_column not in importance.columns:
        return None
    frame = _combine_prediction_frames(predictions, max_rows=8000)
    if frame.empty:
        return None
    ranked = importance[["feature_name", value_column]].copy(deep=True)
    ranked[value_column] = pd.to_numeric(ranked[value_column], errors="coerce")
    ranked = ranked.dropna(subset=[value_column])
    ranked["abs_value"] = ranked[value_column].abs()
    features = [
        str(feature)
        for feature in ranked.sort_values("abs_value", ascending=False)["feature_name"].head(8)
        if str(feature) in frame.columns and pd.api.types.is_numeric_dtype(frame[str(feature)])
    ]
    if not features:
        return None
    coefficient_map = ranked.set_index("feature_name")[value_column].to_dict()
    figure = go.Figure()
    rng = np.random.default_rng(42)
    for index, feature_name in enumerate(features):
        values = pd.to_numeric(frame[feature_name], errors="coerce").dropna()
        if values.empty:
            continue
        if len(values) > 1500:
            values = values.sample(1500, random_state=42)
        std = float(values.std()) or 1.0
        centered = (values - float(values.mean())) / std
        contribution = centered * float(coefficient_map.get(feature_name, 0.0))
        jitter = rng.normal(0, 0.08, len(contribution))
        marker = {
            "size": 7,
            "opacity": 0.42,
            "color": centered,
            "colorscale": "Tealrose",
            "showscale": index == 0,
        }
        if index == 0:
            marker["colorbar"] = {"title": "Std value"}
        figure.add_trace(
            go.Scatter(
                x=contribution,
                y=np.full(len(contribution), len(features) - index) + jitter,
                mode="markers",
                name=feature_name,
                marker=marker,
                hovertemplate=(
                    f"{feature_name}<br>Proxy contribution=%{{x:.4f}}"
                    "<br>Standardized value=%{marker.color:.3f}<extra></extra>"
                ),
            )
        )
    if not figure.data:
        return None
    figure.add_vline(x=0.0, line_dash="dash", line_color=REPORT_SEVERITY_COLORS["info"])
    figure.update_layout(
        title="Contribution Beeswarm (Coefficient Proxy)",
        xaxis_title="Standardized Value x Coefficient",
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(1, len(features) + 1)),
            "ticktext": list(reversed(features)),
            "title": "Feature",
        },
        showlegend=False,
    )
    return "advanced_contribution_beeswarm", apply_fintech_figure_theme(
        figure,
        title="Contribution Beeswarm (Coefficient Proxy)",
        height=520,
    )


def _build_advanced_interaction_heatmap(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("interaction_strength")
    required = {"feature_x", "feature_y", "mean_absolute_interaction"}
    if table is None or table.empty or not required.issubset(table.columns):
        return None
    working = table.copy(deep=True)
    working["mean_absolute_interaction"] = pd.to_numeric(
        working["mean_absolute_interaction"], errors="coerce"
    )
    working = working.dropna(subset=["mean_absolute_interaction"]).head(30)
    features = sorted(set(working["feature_x"].astype(str)) | set(working["feature_y"].astype(str)))
    if len(features) < 2:
        return None
    matrix = pd.DataFrame(0.0, index=features, columns=features)
    for _, row in working.iterrows():
        left = str(row["feature_x"])
        right = str(row["feature_y"])
        value = float(row["mean_absolute_interaction"])
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns,
            y=matrix.index,
            colorscale="YlOrRd",
            colorbar={"title": "Interaction"},
        )
    )
    figure.update_layout(
        title="Feature Interaction Heatmap",
        xaxis_title="Feature",
        yaxis_title="Feature",
    )
    return "advanced_interaction_heatmap", apply_fintech_figure_theme(
        figure,
        title="Feature Interaction Heatmap",
        height=520,
    )


def _build_advanced_feature_effect_matrix(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    pdp = diagnostics_tables.get("partial_dependence")
    if pdp is None or pdp.empty:
        pdp = diagnostics_tables.get("feature_effect_curves")
    required = {"feature_name", "feature_value", "average_prediction"}
    if pdp is None or pdp.empty or not required.issubset(pdp.columns):
        return None
    ice = diagnostics_tables.get("ice_curves")
    features = pdp["feature_name"].dropna().astype(str).drop_duplicates().head(6).tolist()
    if not features:
        return None
    cols = 2 if len(features) > 1 else 1
    rows = int(np.ceil(len(features) / cols))
    figure = make_subplots(rows=rows, cols=cols, subplot_titles=features)
    for index, feature_name in enumerate(features):
        row_index = index // cols + 1
        col_index = index % cols + 1
        feature_frame = (
            pdp.loc[pdp["feature_name"].astype(str) == feature_name]
            .copy(deep=True)
            .sort_values("sort_order" if "sort_order" in pdp.columns else "feature_value")
        )
        if (
            ice is not None
            and not ice.empty
            and {"feature_name", "observation_id"}.issubset(ice.columns)
        ):
            ice_frame = ice.loc[ice["feature_name"].astype(str) == feature_name].copy(deep=True)
            for observation_id in ice_frame["observation_id"].drop_duplicates().head(12):
                obs = ice_frame.loc[ice_frame["observation_id"] == observation_id].sort_values(
                    "sort_order" if "sort_order" in ice_frame.columns else "feature_value"
                )
                figure.add_trace(
                    go.Scatter(
                        x=obs["feature_value"].astype(str),
                        y=obs["prediction"],
                        mode="lines",
                        line={"color": "rgba(42,111,151,0.18)", "width": 1},
                        showlegend=False,
                        name=f"ICE {observation_id}",
                    ),
                    row=row_index,
                    col=col_index,
                )
        figure.add_trace(
            go.Scatter(
                x=feature_frame["feature_value"].astype(str),
                y=feature_frame["average_prediction"],
                mode="lines+markers",
                line={"color": REPORT_SEVERITY_COLORS["good"], "width": 3},
                name="PDP",
                showlegend=index == 0,
            ),
            row=row_index,
            col=col_index,
        )
    figure.update_layout(title="PDP / ICE Effect Matrix", height=max(480, rows * 300))
    return "advanced_feature_effect_matrix", apply_fintech_figure_theme(
        figure,
        title="PDP / ICE Effect Matrix",
        height=max(480, rows * 300),
    )


def _build_advanced_segment_calibration_small_multiples(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, diagnostics_tables
    if target_mode != "binary" or not labels_available or predictions is None:
        return None
    frame = _select_prediction_frame(predictions)
    score_column = _resolve_prediction_score_column(frame)
    target_column = _resolve_prediction_target_column(frame)
    if score_column is None or target_column is None:
        return None
    segment_column = _select_segment_column(frame, exclude={score_column, target_column})
    if segment_column is None:
        return None
    working = frame[[score_column, target_column, segment_column]].dropna().copy(deep=True)
    if working.empty or working[target_column].nunique(dropna=True) < 2:
        return None
    top_segments = working[segment_column].astype(str).value_counts().head(6).index.tolist()
    cols = 2 if len(top_segments) > 1 else 1
    rows = int(np.ceil(len(top_segments) / cols))
    figure = make_subplots(rows=rows, cols=cols, subplot_titles=top_segments)
    for index, segment in enumerate(top_segments):
        row_index = index // cols + 1
        col_index = index % cols + 1
        segment_frame = working.loc[working[segment_column].astype(str) == segment].copy(deep=True)
        if len(segment_frame) < 20:
            continue
        segment_frame["bucket"] = _safe_quantile_bucket(segment_frame[score_column], bins=6)
        grouped = (
            segment_frame.dropna(subset=["bucket"])
            .groupby("bucket", observed=False)
            .agg(
                mean_score=(score_column, "mean"),
                observed_rate=(target_column, "mean"),
            )
            .reset_index()
        )
        if grouped.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=grouped["mean_score"],
                y=grouped["observed_rate"],
                mode="lines+markers",
                name=str(segment),
                line={"color": FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)]},
                showlegend=False,
            ),
            row=row_index,
            col=col_index,
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"color": "rgba(96,112,137,0.35)", "dash": "dash"},
                showlegend=False,
            ),
            row=row_index,
            col=col_index,
        )
    if not figure.data:
        return None
    figure.update_layout(
        title=f"Segment Calibration Small Multiples by {segment_column}",
        height=max(500, rows * 290),
    )
    return "advanced_segment_calibration_small_multiples", apply_fintech_figure_theme(
        figure,
        title=f"Segment Calibration Small Multiples by {segment_column}",
        height=max(500, rows * 290),
    )


def _build_advanced_score_ridgeline(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, diagnostics_tables, target_mode
    if predictions is None:
        return None
    frame = _combine_prediction_frames(predictions, max_rows=30000)
    score_column = _resolve_prediction_score_column(frame)
    if frame.empty or score_column is None:
        return None
    group_column = None
    if labels_available:
        target_column = _resolve_prediction_target_column(frame)
        if target_column is not None:
            frame = frame.assign(
                _outcome_group=frame[target_column].map({0: "Non-Event", 1: "Event"})
            )
            group_column = "_outcome_group"
    if group_column is None and "split" in frame.columns:
        group_column = "split"
    if group_column is None:
        return None
    figure = go.Figure()
    for index, group_name in enumerate(_ordered_unique(frame[group_column])):
        values = pd.to_numeric(
            frame.loc[frame[group_column].astype(str) == str(group_name), score_column],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        figure.add_trace(
            go.Violin(
                x=values,
                y=[str(group_name).title()] * len(values),
                orientation="h",
                name=str(group_name).title(),
                side="positive",
                width=1.6,
                points=False,
                meanline_visible=True,
                line_color=FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)],
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        title="Score Distribution Ridgeline",
        xaxis_title="Predicted Score",
        yaxis_title="Group",
        violinmode="overlay",
    )
    return "advanced_score_ridgeline", apply_fintech_figure_theme(
        figure,
        title="Score Distribution Ridgeline",
    )


def _build_advanced_temporal_score_stream(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, diagnostics_tables, target_mode, labels_available
    if predictions is None:
        return None
    frame = _combine_prediction_frames(predictions, max_rows=50000)
    score_column = _resolve_prediction_score_column(frame)
    date_column = _select_date_column(frame)
    if frame.empty or score_column is None or date_column is None:
        return None
    working = frame[[date_column, score_column, "split"]].copy(deep=True)
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[date_column, score_column])
    if working.empty:
        return None
    working["period"] = working[date_column].dt.to_period("M").dt.to_timestamp()
    grouped = (
        working.groupby(["period", "split"], dropna=False)[score_column]
        .mean()
        .reset_index()
        .sort_values("period")
    )
    if grouped["period"].nunique() < 2:
        return None
    figure = go.Figure()
    for index, split_name in enumerate(_ordered_unique(grouped["split"])):
        split_frame = grouped.loc[grouped["split"].astype(str) == str(split_name)]
        figure.add_trace(
            go.Scatter(
                x=split_frame["period"],
                y=split_frame[score_column],
                mode="lines",
                stackgroup="one",
                name=str(split_name).title(),
                line={"color": FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)]},
            )
        )
    figure.update_layout(
        title="Temporal Score Stream",
        xaxis_title="Period",
        yaxis_title="Average Predicted Score",
    )
    return "advanced_temporal_score_stream", apply_fintech_figure_theme(
        figure,
        title="Temporal Score Stream",
    )


def _build_advanced_correlation_network(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("correlation_matrix")
    if table is None or table.empty or "feature_name" not in table.columns:
        return None
    matrix = table.set_index("feature_name")
    numeric_matrix = matrix.apply(pd.to_numeric, errors="coerce")
    features = [
        str(feature) for feature in numeric_matrix.index if feature in numeric_matrix.columns
    ]
    if len(features) < 2:
        return None
    edges: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = numeric_matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.35:
                edges.append((left, right, float(value)))
    edges = sorted(edges, key=lambda item: abs(item[2]), reverse=True)[:30]
    if not edges:
        return None
    nodes = sorted(set([left for left, _, _ in edges] + [right for _, right, _ in edges]))
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    positions = {
        node: (float(np.cos(angle)), float(np.sin(angle)))
        for node, angle in zip(nodes, angles, strict=True)
    }
    figure = go.Figure()
    for left, right, value in edges:
        figure.add_trace(
            go.Scatter(
                x=[positions[left][0], positions[right][0]],
                y=[positions[left][1], positions[right][1]],
                mode="lines",
                line={
                    "width": max(1.0, min(6.0, abs(value) * 6.0)),
                    "color": REPORT_SEVERITY_COLORS["bad"]
                    if value < 0
                    else REPORT_SEVERITY_COLORS["good"],
                },
                hoverinfo="text",
                text=f"{left} - {right}: {value:.3f}",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker={"size": 18, "color": REPORT_SEVERITY_COLORS["good"]},
            showlegend=False,
        )
    )
    figure.update_layout(
        title="Feature Correlation Network",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return "advanced_correlation_network", apply_fintech_figure_theme(
        figure,
        title="Feature Correlation Network",
        height=540,
    )


def _build_advanced_lift_gain_heatmap(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("lift_gain")
    required = {"bucket", "lift", "capture_rate", "cumulative_capture_rate"}
    if table is None or table.empty or not required.issubset(table.columns):
        return None
    working = table.copy(deep=True).sort_values("bucket")
    rows = ["Lift", "Capture Rate", "Cumulative Capture"]
    values = np.vstack(
        [
            pd.to_numeric(working["lift"], errors="coerce").fillna(0.0).to_numpy(),
            pd.to_numeric(working["capture_rate"], errors="coerce").fillna(0.0).to_numpy(),
            pd.to_numeric(working["cumulative_capture_rate"], errors="coerce")
            .fillna(0.0)
            .to_numpy(),
        ]
    )
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=working["bucket"].astype(str),
            y=rows,
            colorscale="Tealgrn",
            colorbar={"title": "Value"},
        )
    )
    figure.update_layout(
        title="Lift / Gain Decile Heatmap",
        xaxis_title="Risk Bucket",
        yaxis_title="Metric",
    )
    return "advanced_lift_gain_heatmap", apply_fintech_figure_theme(
        figure,
        title="Lift / Gain Decile Heatmap",
    )


def _build_advanced_score_decile_treemap(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, diagnostics_tables
    if target_mode != "binary" or not labels_available or predictions is None:
        return None
    frame = _select_prediction_frame(predictions)
    score_column = _resolve_prediction_score_column(frame)
    target_column = _resolve_prediction_target_column(frame)
    if frame.empty or score_column is None or target_column is None:
        return None
    working = frame[[score_column, target_column]].dropna().copy(deep=True)
    if working.empty:
        return None
    working["score_decile"] = _safe_quantile_bucket(working[score_column], bins=10)
    grouped = (
        working.dropna(subset=["score_decile"])
        .groupby("score_decile", observed=False)
        .agg(
            observation_count=(score_column, "size"),
            observed_rate=(target_column, "mean"),
            average_score=(score_column, "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return None
    labels = grouped["score_decile"].astype(str)
    figure = go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(grouped),
            values=grouped["observation_count"],
            marker={
                "colors": grouped["observed_rate"],
                "colorscale": "YlOrRd",
                "colorbar": {"title": "Observed Rate"},
            },
            customdata=np.stack(
                [grouped["average_score"], grouped["observed_rate"]],
                axis=-1,
            ),
            hovertemplate=(
                "Decile=%{label}<br>Rows=%{value:,}<br>Average score=%{customdata[0]:.3f}"
                "<br>Observed rate=%{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(title="Score Decile Risk Treemap")
    return "advanced_score_decile_treemap", apply_fintech_figure_theme(
        figure,
        title="Score Decile Risk Treemap",
    )


def _build_advanced_model_comparison_radar(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del target_mode, labels_available, predictions
    table = diagnostics_tables.get("model_comparison")
    comparison_frame: pd.DataFrame
    name_column = "model"
    if table is not None and not table.empty:
        comparison_frame = table.copy(deep=True)
        for candidate in ("model", "model_name", "candidate_model", "challenger"):
            if candidate in comparison_frame.columns:
                name_column = candidate
                break
        if name_column not in comparison_frame.columns:
            comparison_frame[name_column] = [f"Model {index + 1}" for index in range(len(table))]
    else:
        rows = []
        for split_name, split_metrics in metrics.items():
            row = {"model": str(split_name).title()}
            row.update(split_metrics)
            rows.append(row)
        comparison_frame = pd.DataFrame(rows)
    if comparison_frame.empty or name_column not in comparison_frame.columns:
        return None
    numeric_columns = [
        column
        for column in comparison_frame.columns
        if column != name_column and pd.api.types.is_numeric_dtype(comparison_frame[column])
    ][:6]
    if len(numeric_columns) < 2:
        return None
    normalized = comparison_frame[[name_column, *numeric_columns]].copy(deep=True)
    for column in numeric_columns:
        normalized[column] = _normalize_metric_series(normalized[column], metric_name=column)
    theta = [column.replace("_", " ").title() for column in numeric_columns]
    figure = go.Figure()
    for index, (_, row) in enumerate(normalized.head(6).iterrows()):
        values = [float(row[column]) for column in numeric_columns]
        figure.add_trace(
            go.Scatterpolar(
                r=[*values, values[0]],
                theta=[*theta, theta[0]],
                fill="toself",
                name=str(row[name_column]),
                line={"color": FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)]},
            )
        )
    figure.update_layout(
        title="Model Comparison Radar",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
    )
    return "advanced_model_comparison_radar", apply_fintech_figure_theme(
        figure,
        title="Model Comparison Radar",
    )


def _build_advanced_scenario_waterfall(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("scenario_summary")
    if table is None or table.empty or not {"scenario_name", "mean_delta"}.issubset(table.columns):
        return None
    working = table[["scenario_name", "mean_delta"]].copy(deep=True)
    working["mean_delta"] = pd.to_numeric(working["mean_delta"], errors="coerce")
    working = working.dropna(subset=["mean_delta"])
    if working.empty:
        return None
    working["abs_delta"] = working["mean_delta"].abs()
    working = working.sort_values("abs_delta", ascending=False).head(10)
    figure = go.Figure(
        go.Waterfall(
            x=working["scenario_name"].astype(str),
            y=working["mean_delta"],
            measure=["relative"] * len(working),
            connector={"line": {"color": "#D8D1C4"}},
            increasing={"marker": {"color": REPORT_SEVERITY_COLORS["bad"]}},
            decreasing={"marker": {"color": REPORT_SEVERITY_COLORS["great"]}},
            name="Scenario Delta",
        )
    )
    figure.update_layout(
        title="Scenario Sensitivity Waterfall",
        xaxis_title="Scenario",
        yaxis_title="Average Score Delta",
    )
    return "advanced_scenario_waterfall", apply_fintech_figure_theme(
        figure,
        title="Scenario Sensitivity Waterfall",
        height=470,
    )


def _build_advanced_feature_importance_lollipop(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("feature_importance")
    if table is None or table.empty or "feature_name" not in table.columns:
        return None
    value_column = "coefficient" if "coefficient" in table.columns else "importance_value"
    if value_column not in table.columns:
        return None
    working = table[["feature_name", value_column]].copy(deep=True)
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    if working.empty:
        return None
    working["abs_value"] = working[value_column].abs()
    working = working.sort_values("abs_value", ascending=True).tail(18)
    figure = go.Figure()
    for _, row in working.iterrows():
        color = (
            REPORT_SEVERITY_COLORS["great"]
            if float(row[value_column]) >= 0
            else REPORT_SEVERITY_COLORS["bad"]
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0, row[value_column]],
                y=[str(row["feature_name"]), str(row["feature_name"])],
                mode="lines",
                line={"color": "rgba(96,112,137,0.35)", "width": 2},
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[row[value_column]],
                y=[str(row["feature_name"])],
                mode="markers",
                marker={"size": 13, "color": color},
                showlegend=False,
            )
        )
    figure.add_vline(x=0.0, line_dash="dash", line_color=REPORT_SEVERITY_COLORS["info"])
    figure.update_layout(
        title="Feature Importance Lollipop",
        xaxis_title=value_column.replace("_", " ").title(),
        yaxis_title="Feature",
    )
    return "advanced_feature_importance_lollipop", apply_fintech_figure_theme(
        figure,
        title="Feature Importance Lollipop",
        height=520,
    )


def _build_split_metric_slope_chart(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del diagnostics_tables, labels_available, predictions
    metric_names = (
        ["roc_auc", "average_precision", "ks_statistic", "brier_score"]
        if target_mode == "binary"
        else ["rmse", "mae", "r2", "explained_variance"]
    )
    split_order = ["train", "validation", "test"]
    available_splits = [split for split in split_order if split in metrics]
    if len(available_splits) < 2:
        available_splits = list(metrics.keys())
    if len(available_splits) < 2:
        return None
    figure = go.Figure()
    for index, metric_name in enumerate(metric_names):
        values = [
            metrics.get(split, {}).get(metric_name)
            for split in available_splits
            if metrics.get(split, {}).get(metric_name) is not None
        ]
        if len(values) < 2:
            continue
        x_values = [
            split.title()
            for split in available_splits
            if metrics.get(split, {}).get(metric_name) is not None
        ]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=[float(value) for value in values],
                mode="lines+markers",
                name=metric_name.replace("_", " ").title(),
                line={"color": FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)]},
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        title="Split Metric Slope Chart",
        xaxis_title="Split",
        yaxis_title="Metric Value",
    )
    return "split_metric_slope_chart", apply_fintech_figure_theme(
        figure,
        title="Split Metric Slope Chart",
    )


def _build_roc_curve_annotated(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del predictions
    table = diagnostics_tables.get("roc_curve")
    if target_mode != "binary" or not labels_available or table is None or table.empty:
        return None
    if not {"fpr", "tpr"}.issubset(table.columns):
        return None
    auc_value = (metrics.get("test") or {}).get("roc_auc")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=table["fpr"],
            y=table["tpr"],
            mode="lines",
            name="Model ROC",
            fill="tozeroy",
            line={"color": REPORT_SEVERITY_COLORS["good"], "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Model",
            line={"color": REPORT_SEVERITY_COLORS["info"], "dash": "dash"},
        )
    )
    if auc_value is not None:
        figure.add_annotation(
            x=0.62,
            y=0.18,
            text=f"AUC {format_metric_value(auc_value)}",
            showarrow=False,
            bgcolor="#FFFFFF",
            bordercolor="#D8D1C4",
            borderwidth=1,
        )
    figure.update_layout(
        title="ROC Curve With Review Bands",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return "roc_curve_annotated", apply_fintech_figure_theme(
        figure,
        title="ROC Curve With Review Bands",
    )


def _build_precision_recall_curve_annotated(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del predictions
    table = diagnostics_tables.get("precision_recall_curve")
    if target_mode != "binary" or not labels_available or table is None or table.empty:
        return None
    if not {"precision", "recall"}.issubset(table.columns):
        return None
    average_precision = (metrics.get("test") or {}).get("average_precision")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=table["recall"],
            y=table["precision"],
            mode="lines",
            name="Precision-Recall",
            line={"color": REPORT_SEVERITY_COLORS["great"], "width": 3},
        )
    )
    if average_precision is not None:
        figure.add_annotation(
            x=0.62,
            y=0.15,
            text=f"Average Precision {format_metric_value(average_precision)}",
            showarrow=False,
            bgcolor="#FFFFFF",
            bordercolor="#D8D1C4",
            borderwidth=1,
        )
    figure.update_layout(
        title="Precision-Recall Curve With Average Precision",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return "precision_recall_curve_annotated", apply_fintech_figure_theme(
        figure,
        title="Precision-Recall Curve With Average Precision",
    )


def _build_ks_curve_annotated(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, diagnostics_tables
    if target_mode != "binary" or not labels_available or predictions is None:
        return None
    frame = _select_prediction_frame(predictions)
    score_column = _resolve_prediction_score_column(frame)
    target_column = _resolve_prediction_target_column(frame)
    if score_column is None or target_column is None:
        return None
    scored = frame[[score_column, target_column]].dropna()
    if scored[target_column].nunique(dropna=True) < 2:
        return None
    positives = np.sort(scored.loc[scored[target_column].astype(int) == 1, score_column])
    negatives = np.sort(scored.loc[scored[target_column].astype(int) == 0, score_column])
    if len(positives) == 0 or len(negatives) == 0:
        return None
    all_scores = np.sort(scored[score_column].dropna().unique())
    positive_cdf = np.searchsorted(positives, all_scores, side="right") / len(positives)
    negative_cdf = np.searchsorted(negatives, all_scores, side="right") / len(negatives)
    gaps = np.abs(positive_cdf - negative_cdf)
    max_index = int(np.argmax(gaps))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=all_scores, y=positive_cdf, mode="lines", name="Event CDF"))
    figure.add_trace(go.Scatter(x=all_scores, y=negative_cdf, mode="lines", name="Non-Event CDF"))
    figure.add_trace(
        go.Scatter(
            x=[all_scores[max_index], all_scores[max_index]],
            y=[positive_cdf[max_index], negative_cdf[max_index]],
            mode="lines",
            name=f"Max KS {gaps[max_index]:.3f}",
            line={"color": REPORT_SEVERITY_COLORS["watch"], "dash": "dash", "width": 4},
        )
    )
    figure.update_layout(
        title="KS Separation Curve", xaxis_title="Predicted Score", yaxis_title="Cumulative Share"
    )
    return "ks_curve_annotated", apply_fintech_figure_theme(figure, title="KS Separation Curve")


def _build_score_distribution_violin(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, diagnostics_tables, target_mode, labels_available
    if predictions is None:
        return None
    frame = _combine_prediction_frames(predictions, max_rows=50000)
    score_column = _resolve_prediction_score_column(frame)
    if frame.empty or score_column is None or "split" not in frame.columns:
        return None
    figure = go.Figure()
    for index, split_name in enumerate(_ordered_unique(frame["split"])):
        split_values = pd.to_numeric(
            frame.loc[frame["split"] == split_name, score_column],
            errors="coerce",
        ).dropna()
        if split_values.empty:
            continue
        figure.add_trace(
            go.Violin(
                x=[str(split_name).title()] * len(split_values),
                y=split_values,
                name=str(split_name).title(),
                box_visible=True,
                meanline_visible=True,
                points=False,
                line_color=FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)],
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        title="Score Distribution Violin", xaxis_title="Split", yaxis_title="Predicted Score"
    )
    return "score_distribution_violin", apply_fintech_figure_theme(
        figure,
        title="Score Distribution Violin",
    )


def _build_calibration_residual_bars(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, predictions
    table = diagnostics_tables.get("calibration")
    if target_mode != "binary" or not labels_available or table is None or table.empty:
        return None
    required = {"mean_predicted_probability", "observed_default_rate"}
    if not required.issubset(table.columns):
        return None
    working = table.copy(deep=True)
    working["calibration_gap"] = pd.to_numeric(
        working["observed_default_rate"], errors="coerce"
    ) - pd.to_numeric(working["mean_predicted_probability"], errors="coerce")
    label_column = "method_label" if "method_label" in working.columns else "method_name"
    if label_column not in working.columns:
        working[label_column] = "Model"
    x_values = (
        working["bin"].astype(str)
        if "bin" in working.columns
        else working.groupby(label_column).cumcount().add(1).astype(str)
    )
    figure = go.Figure()
    for method_name in _ordered_unique(working[label_column]):
        method_frame = working.loc[working[label_column] == method_name]
        colors = [
            REPORT_SEVERITY_COLORS["bad"]
            if abs(value) > 0.10
            else REPORT_SEVERITY_COLORS["watch"]
            if abs(value) > 0.05
            else REPORT_SEVERITY_COLORS["great"]
            for value in method_frame["calibration_gap"].fillna(0.0)
        ]
        figure.add_trace(
            go.Bar(
                x=x_values.loc[method_frame.index],
                y=method_frame["calibration_gap"],
                name=str(method_name),
                marker={"color": colors},
            )
        )
    figure.add_hline(y=0.0, line_dash="dash", line_color=REPORT_SEVERITY_COLORS["info"])
    figure.update_layout(
        title="Calibration Residual Bars",
        xaxis_title="Calibration Bin",
        yaxis_title="Observed - Predicted",
    )
    return "calibration_residual_bars", apply_fintech_figure_theme(
        figure,
        title="Calibration Residual Bars",
    )


def _build_psi_threshold_bars(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("psi")
    if table is None or table.empty or not {"feature_name", "psi"}.issubset(table.columns):
        return None
    working = table.copy(deep=True).sort_values("psi", ascending=True).tail(20)
    colors = [_psi_color(value) for value in pd.to_numeric(working["psi"], errors="coerce")]
    figure = go.Figure(
        go.Bar(
            x=working["psi"],
            y=working["feature_name"].astype(str),
            orientation="h",
            marker={"color": colors},
            name="PSI",
        )
    )
    figure.add_vline(
        x=0.10,
        line_dash="dash",
        line_color=REPORT_SEVERITY_COLORS["watch"],
        annotation_text="Watch",
    )
    figure.add_vline(
        x=0.25,
        line_dash="dash",
        line_color=REPORT_SEVERITY_COLORS["bad"],
        annotation_text="Concern",
    )
    figure.update_layout(title="PSI Threshold Review", xaxis_title="PSI", yaxis_title="Feature")
    return "psi_threshold_bars", apply_fintech_figure_theme(figure, title="PSI Threshold Review")


def _build_vif_threshold_bars(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("vif")
    if table is None or table.empty or not {"feature_name", "vif"}.issubset(table.columns):
        return None
    working = table.copy(deep=True).replace([np.inf, -np.inf], np.nan).dropna(subset=["vif"])
    if working.empty:
        return None
    working = working.sort_values("vif", ascending=True).tail(20)
    colors = [_vif_color(value) for value in pd.to_numeric(working["vif"], errors="coerce")]
    figure = go.Figure(
        go.Bar(
            x=working["vif"],
            y=working["feature_name"].astype(str),
            orientation="h",
            marker={"color": colors},
            name="VIF",
        )
    )
    figure.add_vline(
        x=5, line_dash="dash", line_color=REPORT_SEVERITY_COLORS["watch"], annotation_text="Watch"
    )
    figure.add_vline(
        x=10, line_dash="dash", line_color=REPORT_SEVERITY_COLORS["bad"], annotation_text="Concern"
    )
    figure.update_layout(title="VIF Threshold Review", xaxis_title="VIF", yaxis_title="Feature")
    return "vif_threshold_bars", apply_fintech_figure_theme(figure, title="VIF Threshold Review")


def _build_missingness_split_heatmap(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("missingness_by_split")
    if (
        table is None
        or table.empty
        or not {"split", "column_name", "missing_pct"}.issubset(table.columns)
    ):
        return None
    working = table.copy(deep=True)
    top_columns = (
        working.groupby("column_name")["missing_pct"]
        .max()
        .sort_values(ascending=False)
        .head(25)
        .index
    )
    pivot = (
        working.loc[working["column_name"].isin(top_columns)]
        .pivot_table(index="column_name", columns="split", values="missing_pct", aggfunc="max")
        .fillna(0.0)
    )
    if pivot.empty:
        return None
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(column).title() for column in pivot.columns],
            y=pivot.index.astype(str),
            colorscale=[[0, "#F8FAFC"], [0.5, "#E09F3E"], [1, "#C44536"]],
            colorbar={"title": "Missing %"},
        )
    )
    figure.update_layout(
        title="Missingness Heatmap by Split", xaxis_title="Split", yaxis_title="Feature"
    )
    return "missingness_split_heatmap", apply_fintech_figure_theme(
        figure,
        title="Missingness Heatmap by Split",
    )


def _build_feature_importance_waterfall(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("feature_importance")
    if table is None or table.empty or "feature_name" not in table.columns:
        return None
    value_column = "coefficient" if "coefficient" in table.columns else "importance_value"
    if value_column not in table.columns:
        return None
    working = table[["feature_name", value_column]].copy(deep=True)
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    if working.empty:
        return None
    if value_column == "coefficient":
        working["rank_value"] = working[value_column].abs()
        measure = ["relative"] * len(working.head(12))
        y_values = working.sort_values("rank_value", ascending=False).head(12)[value_column]
    else:
        working["rank_value"] = working[value_column]
        measure = ["relative"] * len(working.head(12))
        y_values = working.sort_values("rank_value", ascending=False).head(12)[value_column]
    chart = working.sort_values("rank_value", ascending=False).head(12)
    figure = go.Figure(
        go.Waterfall(
            x=chart["feature_name"].astype(str),
            y=y_values,
            measure=measure,
            connector={"line": {"color": "#D8D1C4"}},
            increasing={"marker": {"color": REPORT_SEVERITY_COLORS["great"]}},
            decreasing={"marker": {"color": REPORT_SEVERITY_COLORS["bad"]}},
            totals={"marker": {"color": REPORT_SEVERITY_COLORS["info"]}},
            name="Feature Effect",
        )
    )
    figure.update_layout(
        title="Feature Importance Waterfall",
        xaxis_title="Feature",
        yaxis_title=value_column.replace("_", " ").title(),
    )
    return "feature_importance_waterfall", apply_fintech_figure_theme(
        figure,
        title="Feature Importance Waterfall",
        height=470,
    )


def _build_segment_performance_dumbbell(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("segment_performance")
    if (
        table is None
        or table.empty
        or not {"average_score", "average_actual"}.issubset(table.columns)
    ):
        return None
    segment_candidates = [
        column
        for column in table.columns
        if column not in {"split", "observation_count", "average_score", "average_actual"}
    ]
    if not segment_candidates:
        return None
    segment_column = segment_candidates[0]
    working = table.copy(deep=True)
    if "split" in working.columns and "test" in set(working["split"].astype(str)):
        working = working.loc[working["split"].astype(str) == "test"]
    working = working.sort_values("observation_count", ascending=False).head(15)
    if working.empty:
        return None
    y_values = working[segment_column].fillna("Missing").astype(str)
    figure = go.Figure()
    for _, row in working.iterrows():
        figure.add_trace(
            go.Scatter(
                x=[row["average_actual"], row["average_score"]],
                y=[str(row[segment_column]), str(row[segment_column])],
                mode="lines",
                showlegend=False,
                line={"color": "#D8D1C4", "width": 3},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=working["average_actual"],
            y=y_values,
            mode="markers",
            name="Observed",
            marker={"color": REPORT_SEVERITY_COLORS["great"], "size": 10},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=working["average_score"],
            y=y_values,
            mode="markers",
            name="Predicted",
            marker={"color": REPORT_SEVERITY_COLORS["good"], "size": 10},
        )
    )
    figure.update_layout(
        title="Segment Performance Dumbbell", xaxis_title="Rate", yaxis_title="Segment"
    )
    return "segment_performance_dumbbell", apply_fintech_figure_theme(
        figure,
        title="Segment Performance Dumbbell",
    )


def _build_scenario_tornado(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("scenario_summary")
    if table is None or table.empty or not {"scenario_name", "mean_delta"}.issubset(table.columns):
        return None
    working = table.copy(deep=True)
    working["mean_delta"] = pd.to_numeric(working["mean_delta"], errors="coerce")
    working = working.dropna(subset=["mean_delta"])
    if working.empty:
        return None
    working["abs_delta"] = working["mean_delta"].abs()
    working = working.sort_values("abs_delta", ascending=True).tail(20)
    colors = [
        REPORT_SEVERITY_COLORS["bad"] if value < 0 else REPORT_SEVERITY_COLORS["great"]
        for value in working["mean_delta"]
    ]
    figure = go.Figure(
        go.Bar(
            x=working["mean_delta"],
            y=working["scenario_name"].astype(str),
            orientation="h",
            marker={"color": colors},
            name="Score Delta",
        )
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color=REPORT_SEVERITY_COLORS["info"])
    figure.update_layout(
        title="Scenario Impact Tornado", xaxis_title="Average Score Delta", yaxis_title="Scenario"
    )
    return "scenario_tornado", apply_fintech_figure_theme(figure, title="Scenario Impact Tornado")


def _build_cross_validation_metric_violin(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("cross_validation_metric_distribution")
    if table is None or table.empty or not {"metric_name", "metric_value"}.issubset(table.columns):
        return None
    figure = go.Figure()
    for index, metric_name in enumerate(_ordered_unique(table["metric_name"])):
        values = pd.to_numeric(
            table.loc[table["metric_name"] == metric_name, "metric_value"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        figure.add_trace(
            go.Violin(
                x=[str(metric_name).replace("_", " ").title()] * len(values),
                y=values,
                name=str(metric_name).replace("_", " ").title(),
                box_visible=True,
                meanline_visible=True,
                points="all",
                line_color=FINTECH_COLORWAY[index % len(FINTECH_COLORWAY)],
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        title="Cross-Validation Metric Violin", xaxis_title="Metric", yaxis_title="Fold Value"
    )
    return "cross_validation_metric_violin", apply_fintech_figure_theme(
        figure,
        title="Cross-Validation Metric Violin",
    )


def _build_feature_effect_stability_small_multiples(
    *,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    diagnostics_tables: Mapping[str, pd.DataFrame],
    target_mode: str,
    labels_available: bool,
    predictions: Mapping[str, pd.DataFrame] | None,
) -> tuple[str, go.Figure] | None:
    del metrics, target_mode, labels_available, predictions
    table = diagnostics_tables.get("feature_effect_stability")
    required = {"feature_name", "split", "feature_value", "average_prediction"}
    if table is None or table.empty or not required.issubset(table.columns):
        return None
    features = table["feature_name"].dropna().astype(str).drop_duplicates().head(4).tolist()
    if not features:
        return None
    figure = make_subplots(
        rows=len(features),
        cols=1,
        shared_xaxes=False,
        subplot_titles=features,
        vertical_spacing=0.08,
    )
    for row_index, feature_name in enumerate(features, start=1):
        feature_frame = table.loc[table["feature_name"].astype(str) == feature_name]
        for split_index, split_name in enumerate(_ordered_unique(feature_frame["split"])):
            split_frame = feature_frame.loc[feature_frame["split"] == split_name].sort_values(
                "feature_value"
            )
            figure.add_trace(
                go.Scatter(
                    x=split_frame["feature_value"],
                    y=split_frame["average_prediction"],
                    mode="lines+markers",
                    name=str(split_name).title(),
                    legendgroup=str(split_name),
                    showlegend=row_index == 1,
                    line={"color": FINTECH_COLORWAY[split_index % len(FINTECH_COLORWAY)]},
                ),
                row=row_index,
                col=1,
            )
    figure.update_layout(
        title="Feature Effect Stability Small Multiples", height=max(420, 230 * len(features))
    )
    return "feature_effect_stability_small_multiples", apply_fintech_figure_theme(
        figure,
        title="Feature Effect Stability Small Multiples",
        height=max(420, 230 * len(features)),
    )


def _combine_prediction_frames(
    predictions: Mapping[str, pd.DataFrame],
    *,
    max_rows: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name, frame in predictions.items():
        if frame is None or frame.empty:
            continue
        sampled = frame.copy(deep=False)
        if "split" not in sampled.columns:
            sampled = sampled.assign(split=split_name)
        frames.append(sampled)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > max_rows:
        return combined.sample(max_rows, random_state=42)
    return combined


def _select_prediction_frame(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    for split_name in ("test", "validation", "train"):
        frame = predictions.get(split_name)
        if frame is not None and not frame.empty:
            if "split" not in frame.columns:
                return frame.assign(split=split_name)
            return frame
    return _combine_prediction_frames(predictions, max_rows=50000)


def _resolve_prediction_score_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "predicted_probability_recommended",
        "predicted_probability",
        "prediction_score",
        "predicted_score",
        "predicted_value",
    ):
        if column in frame.columns:
            return column
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    return numeric_columns[0] if numeric_columns else None


def _resolve_prediction_target_column(frame: pd.DataFrame) -> str | None:
    preferred = [
        "target",
        "default_flag",
        "default_status",
        "actual",
        "observed",
    ]
    for column in preferred:
        if column in frame.columns and frame[column].nunique(dropna=True) <= 2:
            return column
    binary_candidates = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column
        not in {
            "predicted_probability_recommended",
            "predicted_probability",
            "prediction_score",
            "predicted_score",
            "predicted_value",
            "predicted_class",
        }
        and frame[column].nunique(dropna=True) <= 2
    ]
    return binary_candidates[0] if binary_candidates else None


def _select_segment_column(frame: pd.DataFrame, *, exclude: set[str]) -> str | None:
    excluded = {
        *exclude,
        "split",
        "predicted_class",
        "predicted_probability",
        "predicted_probability_recommended",
        "prediction_score",
        "predicted_score",
        "predicted_value",
    }
    candidates = []
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
            continue
        unique_count = int(series.astype(str).nunique(dropna=True))
        if 2 <= unique_count <= 12:
            candidates.append((column, unique_count))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[1], str(item[0])))[0][0]


def _select_date_column(frame: pd.DataFrame) -> str | None:
    preferred = [
        column
        for column in frame.columns
        if "date" in str(column).lower() or "time" in str(column).lower()
    ]
    candidates = [*preferred, *[column for column in frame.columns if column not in preferred]]
    for column in candidates:
        if column in {
            "split",
            "predicted_probability",
            "predicted_probability_recommended",
            "prediction_score",
            "predicted_score",
            "predicted_value",
            "predicted_class",
        }:
            continue
        if column not in preferred and not (
            pd.api.types.is_datetime64_any_dtype(frame[column])
            or pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        ):
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().sum() >= max(5, int(len(frame) * 0.20)) and parsed.nunique() >= 2:
            return str(column)
    return None


def _safe_quantile_bucket(series: pd.Series, *, bins: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(numeric, q=min(bins, max(1, numeric.nunique())), duplicates="drop")
    except ValueError:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="object")


def _normalize_metric_series(series: pd.Series, *, metric_name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lower_is_better = any(
        token in metric_name.lower()
        for token in ("rmse", "mae", "mse", "brier", "log_loss", "error")
    )
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(max_value, min_value):
        return pd.Series([1.0] * len(values), index=values.index)
    normalized = (values - min_value) / (max_value - min_value)
    if lower_is_better:
        normalized = 1.0 - normalized
    return normalized.clip(0.0, 1.0)


def _ordered_unique(series: pd.Series) -> list[Any]:
    preferred = ["train", "validation", "test"]
    values = [value for value in preferred if value in set(series.astype(str))]
    values.extend(
        value
        for value in series.dropna().astype(str).drop_duplicates().tolist()
        if value not in values
    )
    return values


def _psi_color(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return REPORT_SEVERITY_COLORS["info"]
    if numeric >= 0.25:
        return REPORT_SEVERITY_COLORS["bad"]
    if numeric >= 0.10:
        return REPORT_SEVERITY_COLORS["watch"]
    if numeric >= 0.05:
        return REPORT_SEVERITY_COLORS["good"]
    return REPORT_SEVERITY_COLORS["great"]


def _vif_color(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return REPORT_SEVERITY_COLORS["info"]
    if numeric >= 10:
        return REPORT_SEVERITY_COLORS["bad"]
    if numeric >= 5:
        return REPORT_SEVERITY_COLORS["watch"]
    if numeric >= 2.5:
        return REPORT_SEVERITY_COLORS["good"]
    return REPORT_SEVERITY_COLORS["great"]


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
    table_preview_rows: int = 12,
    max_figures_per_section: int = 6,
    max_tables_per_section: int = 6,
    include_enhanced_report_visuals: bool = True,
    include_advanced_visual_analytics: bool = False,
    predictions: Mapping[str, pd.DataFrame] | None = None,
) -> str:
    """Builds the polished standalone HTML dashboard report for each run."""

    enhanced_visualizations = (
        enhance_report_visualizations(
            metrics=metrics,
            diagnostics_tables=diagnostics_tables,
            visualizations=visualizations,
            target_mode=target_mode,
            labels_available=labels_available,
            predictions=predictions,
        )
        if include_enhanced_report_visuals
        else dict(visualizations)
    )
    report_visualizations = (
        apply_advanced_visual_analytics(
            metrics=metrics,
            diagnostics_tables=diagnostics_tables,
            visualizations=enhanced_visualizations,
            target_mode=target_mode,
            labels_available=labels_available,
            predictions=predictions,
        )
        if include_advanced_visual_analytics
        else enhanced_visualizations
    )
    asset_catalog = build_asset_catalog(diagnostics_tables, report_visualizations)
    subset_search_highlight_html = ""
    if execution_mode == "search_feature_subsets":
        subset_search_highlight_html = _build_subset_search_highlight_html(
            tables=diagnostics_tables,
            figures=report_visualizations,
        )
        asset_catalog = prune_subset_search_highlight_assets(asset_catalog)
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
            figures=report_visualizations,
            tables=diagnostics_tables,
            table_preview_rows=table_preview_rows,
            max_figures_per_section=max_figures_per_section,
            max_tables_per_section=max_tables_per_section,
        )
        for section_id, section_payload in asset_catalog.items()
        if section_payload["figures"] or section_payload["tables"]
    )
    sections_html = diagnostic_sections_html + _build_governance_section_html(
        execution_mode=execution_mode,
        warning_count=warning_count,
        warnings=warnings,
        events=events,
    )
    plotly_render_script = _build_plotly_render_script()
    plotly_js_bundle = get_plotlyjs()

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Quant Studio Report {escape(run_id)}</title>
    <style>
      :root {{
        --bg: {FINTECH_NEUTRAL["bg"]};
        --surface: {FINTECH_NEUTRAL["surface"]};
        --surface-alt: {FINTECH_NEUTRAL["surface_alt"]};
        --line: {FINTECH_NEUTRAL["line"]};
        --text: {FINTECH_NEUTRAL["text"]};
        --muted: {FINTECH_NEUTRAL["muted"]};
        --accent: {FINTECH_NEUTRAL["accent"]};
        --great: {REPORT_SEVERITY_COLORS["great"]};
        --good: {REPORT_SEVERITY_COLORS["good"]};
        --watch: {REPORT_SEVERITY_COLORS["watch"]};
        --bad: {REPORT_SEVERITY_COLORS["bad"]};
        --info: {REPORT_SEVERITY_COLORS["info"]};
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
      .hero-head {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(280px, 0.42fr);
        gap: 28px;
        align-items: start;
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
      .report-status-grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
      }}
      .report-status-card {{
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(17, 32, 51, 0.08);
      }}
      .report-status-card span {{
        display: block;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 5px;
      }}
      .report-status-card strong {{
        font-size: 15px;
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
      details.section-shell {{
        display: block;
      }}
      details.section-shell > summary {{
        list-style: none;
      }}
      details.section-shell > summary::-webkit-details-marker {{
        display: none;
      }}
      .section-summary {{
        cursor: pointer;
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
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 18px;
        border-radius: 22px;
        background: var(--surface);
        border: 1px solid rgba(17, 32, 51, 0.08);
        overflow: hidden;
        isolation: isolate;
      }}
      .asset-card__topline {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-bottom: -2px;
      }}
      .asset-kicker {{
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
      }}
      .asset-badge {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.03em;
        border: 1px solid rgba(17, 32, 51, 0.08);
        background: rgba(96, 112, 137, 0.10);
        color: var(--info);
        white-space: nowrap;
      }}
      .asset-badge--great {{
        background: rgba(15, 139, 95, 0.11);
        color: var(--great);
      }}
      .asset-badge--good {{
        background: rgba(42, 111, 151, 0.12);
        color: var(--good);
      }}
      .asset-badge--watch {{
        background: rgba(217, 154, 43, 0.14);
        color: #8A5A09;
      }}
      .asset-badge--bad {{
        background: rgba(196, 69, 54, 0.12);
        color: var(--bad);
      }}
      .asset-card h3 {{
        margin: 0 0 4px;
        font-size: 18px;
        line-height: 1.18;
      }}
      .asset-card p {{
        margin: 0 0 8px;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.35;
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
        position: relative;
        width: 100%;
        min-width: 0;
        overflow: hidden;
        min-height: 470px;
        isolation: isolate;
      }}
      .plot-shell > div {{
        width: 100% !important;
        max-width: 100% !important;
      }}
      .plot-shell .js-plotly-plot,
      .plot-shell .plot-container,
      .plot-shell .svg-container {{
        width: 100% !important;
        max-width: 100% !important;
        position: relative !important;
      }}
      .plot-shell .modebar {{
        right: 8px !important;
      }}
      .plot-fallback {{
        display: grid;
        min-height: 280px;
        place-items: center;
        border-radius: 18px;
        background: linear-gradient(135deg, #F8FAFC, #F3EEE5);
        color: var(--muted);
        text-align: center;
        padding: 24px;
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
      .chart-guidance {{
        margin: -2px 0 2px;
        padding: 8px 10px;
        border-radius: 14px;
        background: rgba(243, 238, 229, 0.72);
        color: var(--muted);
        font-size: 12px;
        line-height: 1.35;
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
        .hero-head {{
          grid-template-columns: 1fr;
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
        <div class="hero-head">
          <div>
            <div class="hero-kicker">Quant Studio Regulatory Model Development Report</div>
            <h1>Run {escape(run_id)}</h1>
            <p>
              Formal model-development evidence package for
              {escape(model_type.replace("_", " ").title())}, operating in
              {escape(execution_mode.replace("_", " ").title())}. Review the
              executive metrics first, then expand each diagnostic section for
              supporting charts, tables, and export evidence.
            </p>
          </div>
          <aside class="report-status-grid" aria-label="Report status">
            <div class="report-status-card">
              <span>Organization</span>
              <strong>Model Development &amp; Validation</strong>
            </div>
            <div class="report-status-card">
              <span>Report Style</span>
              <strong>Formal Regulatory Review</strong>
            </div>
            <div class="report-status-card">
              <span>Generated By</span>
              <strong>Quant Studio</strong>
            </div>
          </aside>
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
      {subset_search_highlight_html}
      {sections_html}
    </main>
    <script>{plotly_js_bundle}</script>
    <script>{plotly_render_script}</script>
  </body>
</html>
"""


def _build_subset_search_highlight_html(
    *,
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, go.Figure],
) -> str:
    selected_candidate = tables.get("subset_search_selected_candidate", pd.DataFrame())
    selected_coefficients = tables.get("subset_search_selected_coefficients", pd.DataFrame())
    nonwinning_candidates = tables.get("subset_search_nonwinning_candidates", pd.DataFrame())

    summary_cards: list[str] = []
    if not selected_candidate.empty:
        summary_cards.append(
            _build_table_card_html(
                AssetDescriptor(
                    key="subset_search_selected_candidate",
                    title=friendly_asset_title(
                        "subset_search_selected_candidate",
                        kind="table",
                    ),
                    kind="table",
                    section="feature_subset_search",
                    description=ASSET_DESCRIPTIONS.get(
                        "subset_search_selected_candidate",
                        "",
                    ),
                    featured=True,
                ),
                selected_candidate,
                preview_rows=1,
            )
        )
    if not selected_coefficients.empty:
        summary_cards.append(
            _build_table_card_html(
                AssetDescriptor(
                    key="subset_search_selected_coefficients",
                    title=friendly_asset_title(
                        "subset_search_selected_coefficients",
                        kind="table",
                    ),
                    kind="table",
                    section="feature_subset_search",
                    description=ASSET_DESCRIPTIONS.get(
                        "subset_search_selected_coefficients",
                        "",
                    ),
                    featured=True,
                ),
                selected_coefficients,
                preview_rows=25,
            )
        )

    figure_cards = "".join(
        _build_figure_card_html(
            AssetDescriptor(
                key=figure_key,
                title=friendly_asset_title(figure_key, kind="figure"),
                kind="figure",
                section="feature_subset_search",
                description=ASSET_DESCRIPTIONS.get(figure_key, ""),
                featured=True,
            ),
            figures[figure_key],
        )
        for figure_key in [
            "subset_search_selected_roc_curve",
            "subset_search_selected_ks_curve",
        ]
        if figure_key in figures
    )

    ranked_table_html = ""
    if not nonwinning_candidates.empty:
        ranked_table_html = _build_table_card_html(
            AssetDescriptor(
                key="subset_search_nonwinning_candidates",
                title=friendly_asset_title(
                    "subset_search_nonwinning_candidates",
                    kind="table",
                ),
                kind="table",
                section="feature_subset_search",
                description=ASSET_DESCRIPTIONS.get(
                    "subset_search_nonwinning_candidates",
                    "",
                ),
                featured=True,
            ),
            nonwinning_candidates,
            preview_rows=25,
        )

    if not summary_cards and not figure_cards and not ranked_table_html:
        return ""

    return f"""
    <section class="section-shell" id="subset_search_selection_summary">
      <div class="section-header">
        <div>
          <h2>Selected Candidate Summary</h2>
          <p>
            Selected-candidate evidence sits first in the subset-search report. The winning
            subset is shown with its coefficients or feature-importance values, while the
            alternative subsets are kept in a ranked comparison table for cleaner review.
          </p>
        </div>
      </div>
      {"<div class='table-grid'>" + "".join(summary_cards) + "</div>" if summary_cards else ""}
      {"<div class='figure-grid'>" + figure_cards + "</div>" if figure_cards else ""}
      {"<div class='table-grid'>" + ranked_table_html + "</div>" if ranked_table_html else ""}
    </section>
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
    table_preview_rows: int,
    max_figures_per_section: int,
    max_tables_per_section: int,
) -> str:
    visible_figure_descriptors = figure_descriptors[:max_figures_per_section]
    visible_table_descriptors = table_descriptors[:max_tables_per_section]
    figure_cards = "".join(
        _build_figure_card_html(descriptor, figures[descriptor.key])
        for descriptor in visible_figure_descriptors
        if descriptor.key in figures
    )
    table_cards = "".join(
        _build_table_card_html(
            descriptor,
            tables[descriptor.key],
            preview_rows=table_preview_rows,
        )
        for descriptor in visible_table_descriptors
        if descriptor.key in tables
    )

    if not figure_cards and not table_cards:
        return ""

    return f"""
    <details class="section-shell" id="{escape(section_id)}" open>
      <summary class="section-summary">
        <div class="section-header">
          <div>
            <h2>{escape(section_title)}</h2>
            <p>{escape(section_description)}</p>
            {
        _build_section_limit_note(
            figure_descriptors,
            table_descriptors,
            max_figures_per_section,
            max_tables_per_section,
        )
    }
          </div>
        </div>
      </summary>
      {"<div class='figure-grid'>" + figure_cards + "</div>" if figure_cards else ""}
      {"<div class='table-grid'>" + table_cards + "</div>" if table_cards else ""}
    </details>
    """


def _build_asset_badge_html(descriptor: AssetDescriptor) -> str:
    label, badge_class = report_asset_badge(
        descriptor.key,
        featured=descriptor.featured,
    )
    return f'<span class="asset-badge asset-badge--{escape(badge_class)}">{escape(label)}</span>'


def _build_chart_guidance_html(descriptor: AssetDescriptor) -> str:
    guidance = report_chart_guidance(descriptor.key)
    if not guidance:
        return ""
    return f'<div class="chart-guidance">{escape(guidance)}</div>'


def _build_figure_card_html(descriptor: AssetDescriptor, figure: go.Figure) -> str:
    safe_figure = go.Figure(figure)
    _make_figure_display_safe(safe_figure)
    _prepare_report_card_figure(safe_figure)
    figure_payload = json.dumps(
        safe_figure.to_plotly_json(),
        default=_json_default,
        separators=(",", ":"),
    )
    figure_id = f"plot_{descriptor.key}_{abs(hash(descriptor.key))}".replace("-", "_")
    figure_height = int(safe_figure.layout.height or 440)
    config_payload = json.dumps(plotly_display_config(), separators=(",", ":"))
    config_payload_html = escape(config_payload)
    payload_id = f"{figure_id}_payload"
    fallback = (
        "Chart loading. If this remains visible, open the report in a modern "
        "browser or serve it from a local HTTP server."
    )
    description = escape(descriptor.description or "")
    return f"""
    <article class="asset-card">
      <div class="asset-card__topline">
        <span class="asset-kicker">Chart</span>
        {_build_asset_badge_html(descriptor)}
      </div>
      <h3>{escape(descriptor.title)}</h3>
      {f"<p>{description}</p>" if description else ""}
      {_build_chart_guidance_html(descriptor)}
      <div
        class="plot-shell plotly-lazy"
        id="{escape(figure_id)}"
        data-figure-payload="{escape(payload_id)}"
        data-config="{config_payload_html}"
        style="min-height: {figure_height}px;"
      >
        <div class="plot-fallback">{escape(fallback)}</div>
      </div>
      <script type="application/json" id="{escape(payload_id)}">
        {_safe_json_script(figure_payload)}
      </script>
    </article>
    """


def _build_table_card_html(
    descriptor: AssetDescriptor,
    table: pd.DataFrame,
    *,
    preview_rows: int,
) -> str:
    preview = _prepare_table_preview(table).head(preview_rows)
    description = escape(descriptor.description or "")
    return f"""
    <article class="asset-card">
      <div class="asset-card__topline">
        <span class="asset-kicker">Table</span>
        {_build_asset_badge_html(descriptor)}
      </div>
      <h3>{escape(descriptor.title)}</h3>
      {f"<p>{description}</p>" if description else ""}
      <div class="table-shell">
        {preview.to_html(index=False, classes="asset-table", border=0)}
      </div>
      <div class="asset-note">
        Showing {min(len(table), preview_rows)} of {len(table):,} rows. Full export is available
        in the tables directory.
      </div>
    </article>
    """


def _prepare_table_preview(table: pd.DataFrame) -> pd.DataFrame:
    return prepare_display_table(table)


def _build_section_limit_note(
    figure_descriptors: list[AssetDescriptor],
    table_descriptors: list[AssetDescriptor],
    max_figures_per_section: int,
    max_tables_per_section: int,
) -> str:
    notes: list[str] = []
    if len(figure_descriptors) > max_figures_per_section:
        notes.append(
            f"Showing the first {max_figures_per_section} figures in this section in the HTML view."
        )
    if len(table_descriptors) > max_tables_per_section:
        notes.append(
            f"Showing the first {max_tables_per_section} tables in this section in the HTML view."
        )
    if not notes:
        return ""
    return f"<p>{escape(' '.join(notes))}</p>"


def _build_governance_section_html(
    *,
    execution_mode: str,
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
    if execution_mode == "search_feature_subsets":
        export_items = "".join(
            [
                (
                    "<li><strong>config/run_config.json</strong> stores the resolved "
                    "subset-search configuration.</li>"
                ),
                (
                    "<li><strong>subset_search_candidates.csv</strong> and related tables in "
                    "<strong>tables/</strong> preserve the candidate ranking.</li>"
                ),
                (
                    "<li><strong>reports/interactive_report.html</strong> packages the "
                    "comparison-only visuals for review outside the GUI.</li>"
                ),
                (
                    "<li><strong>figures/</strong> stores ROC, KS, frontier, and feature-"
                    "frequency visuals for the leading candidates.</li>"
                ),
            ]
        )
    else:
        export_items = "".join(
            [
                (
                    "<li><strong>config/run_config.json</strong> stores the fully resolved "
                    "configuration.</li>"
                ),
                (
                    "<li><strong>code/generated_run.py</strong> reruns the bundle "
                    "without the GUI.</li>"
                ),
                (
                    "<li><strong>data/input/input_snapshot.csv</strong> preserves the "
                    "scored input when enabled.</li>"
                ),
                (
                    "<li><strong>reports/committee_report.docx/.pdf</strong> and "
                    "<strong>reports/validation_report.docx/.pdf</strong> package the "
                    "run for committee and validation review.</li>"
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


def _build_plotly_render_script() -> str:
    return r"""
      (function () {
        function setFallback(node, message) {
          node.innerHTML = "";
          var fallback = document.createElement("div");
          fallback.className = "plot-fallback";
          fallback.textContent = message;
          node.appendChild(fallback);
        }

        function renderFigures() {
          var nodes = document.querySelectorAll(".plotly-lazy");
          if (!nodes.length) {
            return;
          }
          if (!window.Plotly) {
            nodes.forEach(function (node) {
              setFallback(
                node,
                "Plotly could not load. Serve this report from a local HTTP server " +
                  "or open it in a modern browser."
              );
            });
            return;
          }
          nodes.forEach(function (node) {
            try {
              var payloadNode = document.getElementById(node.dataset.figurePayload);
              if (!payloadNode) {
                setFallback(node, "Chart payload was not found in this report.");
                return;
              }
              var figure = JSON.parse(payloadNode.textContent || "{}");
              var config = JSON.parse(node.dataset.config || "{}");
              node.innerHTML = "";
              window.Plotly.newPlot(
                node,
                figure.data || [],
                figure.layout || {},
                config
              ).then(function () {
                window.Plotly.Plots.resize(node);
              });
            } catch (error) {
              setFallback(node, "Chart could not render: " + error.message);
            }
          });
        }

        if (document.readyState === "loading") {
          document.addEventListener("DOMContentLoaded", renderFigures);
        } else {
          renderFigures();
        }
        window.addEventListener("resize", function () {
          document.querySelectorAll(".plotly-lazy").forEach(function (node) {
            if (window.Plotly && node.data) {
              window.Plotly.Plots.resize(node);
            }
          });
        });
      })();
    """


def _safe_json_script(payload: str) -> str:
    return payload.replace("</", "<\\/")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_default(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_default(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, pd.Period):
        return str(value)
    return str(value)


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
