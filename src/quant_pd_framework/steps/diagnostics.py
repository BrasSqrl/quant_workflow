"""Builds statistical tests, tables, and interactive visualizations for a run."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from ..base import BasePipelineStep
from ..config import CalibrationStrategy, DataStructure, ExecutionMode, ModelType, TargetMode
from ..context import PipelineContext
from ..models import build_model_adapter
from ..presentation import apply_fintech_figure_theme, friendly_asset_title
from .evaluation import EvaluationStep


class DiagnosticsStep(BasePipelineStep):
    """
    Generates validation tables and interactive visuals for the completed model run.

    The diagnostics layer is intentionally broad so the same run folder can
    serve model builders, validators, and business users without rebuilding the
    analysis in separate notebooks.
    """

    name = "diagnostics"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.feature_importance is None or context.backtest_summary is None:
            raise ValueError("Diagnostics require evaluation and backtesting to finish first.")

        diagnostic_config = context.config.diagnostics
        labels_available = self._labels_available(context)
        has_target_variation = int(context.metadata.get("target_unique_values", 0)) >= 2
        preserved_tables = {
            table_name: table.copy(deep=True)
            for table_name, table in context.diagnostics_tables.items()
        }
        context.diagnostics_tables = {
            **preserved_tables,
            "split_metrics": pd.DataFrame(context.metrics).T.rename_axis("split").reset_index(),
            "feature_importance": context.feature_importance.copy(deep=True),
            "backtest_summary": context.backtest_summary.copy(deep=True),
        }
        context.visualizations = {}
        context.statistical_tests = {}

        top_features = self._select_top_features(context)
        target_mode = context.config.target.mode
        self._add_metric_overview_outputs(context)
        self._add_feature_importance_overview(context)
        self._add_model_artifact_outputs(context)
        if context.config.documentation.enabled:
            self._add_documentation_metadata(context)
        self._add_feature_dictionary_outputs(context)
        self._add_manual_review_outputs(context)

        if diagnostic_config.data_quality:
            self._build_data_quality_outputs(context)
        if diagnostic_config.descriptive_statistics:
            self._add_descriptive_statistics(context)
        if diagnostic_config.missingness_analysis:
            self._add_missingness_outputs(context)
        if diagnostic_config.correlation_analysis:
            self._add_correlation_outputs(context, top_features)
        if diagnostic_config.vif_analysis:
            self._add_vif_outputs(context, top_features)

        if target_mode == TargetMode.BINARY:
            if diagnostic_config.quantile_analysis:
                self._add_quantile_outputs(context, labels_available)
            if diagnostic_config.threshold_analysis and labels_available and has_target_variation:
                self._add_threshold_outputs(context)
            elif diagnostic_config.threshold_analysis and not labels_available:
                context.warn(
                    "Skipped threshold analysis because labels are unavailable "
                    "for this scored dataset."
                )
            if diagnostic_config.calibration_analysis and labels_available and has_target_variation:
                self._add_calibration_outputs(context)
            elif diagnostic_config.calibration_analysis and not labels_available:
                context.warn(
                    "Skipped calibration analysis because labels are "
                    "unavailable for this scored dataset."
                )
            if diagnostic_config.lift_gain_analysis and labels_available and has_target_variation:
                self._add_lift_gain_outputs(context)
            elif diagnostic_config.lift_gain_analysis and not labels_available:
                context.warn(
                    "Skipped lift and gain analysis because labels are "
                    "unavailable for this scored dataset."
                )
            if diagnostic_config.woe_iv_analysis and labels_available and has_target_variation:
                self._add_woe_iv_outputs(context, top_features)
            elif diagnostic_config.woe_iv_analysis and not labels_available:
                context.warn(
                    "Skipped WoE/IV analysis because labels are unavailable "
                    "for this scored dataset."
                )
            if context.config.model.model_type == ModelType.DISCRETE_TIME_HAZARD_MODEL:
                self._add_lifetime_pd_outputs(context, labels_available)
        else:
            if diagnostic_config.quantile_analysis:
                self._add_regression_quantile_outputs(context, labels_available)
            if diagnostic_config.residual_analysis and labels_available:
                self._add_residual_outputs(context)
            elif diagnostic_config.residual_analysis and not labels_available:
                context.warn(
                    "Skipped residual analysis because labels are unavailable "
                    "for this scored dataset."
                )
            if diagnostic_config.qq_analysis and labels_available:
                self._add_qq_outputs(context)
            elif diagnostic_config.qq_analysis and not labels_available:
                context.warn(
                    "Skipped QQ analysis because labels are unavailable for this scored dataset."
                )

        if diagnostic_config.psi_analysis:
            self._add_psi_outputs(context, top_features)
        if diagnostic_config.segment_analysis:
            self._add_segment_outputs(context, labels_available)
        if diagnostic_config.adf_analysis:
            self._add_adf_outputs(context, top_features, labels_available)
        if context.comparison_results is not None:
            self._add_model_comparison_outputs(context)
        if context.config.robustness.enabled:
            self._add_robustness_outputs(context)
        if context.config.explainability.enabled:
            self._add_explainability_outputs(context, top_features, labels_available)
        if context.config.scorecard_workbench.enabled:
            self._add_scorecard_workbench_outputs(context)
        if context.config.scenario_testing.enabled:
            self._add_scenario_outputs(context)
        if context.config.feature_policy.enabled:
            self._add_feature_policy_outputs(context)

        self._apply_visual_theme(context)
        return context

    def _add_metric_overview_outputs(self, context: PipelineContext) -> None:
        metrics_table = pd.DataFrame(context.metrics).T.rename_axis("split").reset_index()
        numeric_columns = metrics_table.select_dtypes(include="number").columns.tolist()
        comparison_columns = [
            column
            for column in (
                ["roc_auc", "average_precision", "ks_statistic", "brier_score"]
                if context.config.target.mode == TargetMode.BINARY
                else ["rmse", "mae", "r2", "explained_variance"]
            )
            if column in numeric_columns
        ]
        if not comparison_columns:
            return
        metric_overview = metrics_table.melt(
            id_vars=["split"],
            value_vars=comparison_columns,
            var_name="metric",
            value_name="value",
        )
        context.visualizations["split_metric_overview"] = px.bar(
            metric_overview,
            x="metric",
            y="value",
            color="split",
            barmode="group",
            title="Metric Comparison by Split",
            labels={"metric": "Metric", "value": "Value", "split": "Split"},
        )

    def _add_feature_importance_overview(self, context: PipelineContext) -> None:
        importance = context.feature_importance.copy(deep=True).head(
            context.config.diagnostics.top_n_features
        )
        if importance.empty or "feature_name" not in importance.columns:
            return
        value_column = (
            "importance_value"
            if "importance_value" in importance.columns
            else "abs_coefficient"
            if "abs_coefficient" in importance.columns
            else None
        )
        if value_column is None:
            return
        importance = importance.sort_values(value_column, ascending=True)
        context.visualizations["feature_importance_overview"] = px.bar(
            importance,
            x=value_column,
            y="feature_name",
            orientation="h",
            title="Feature Importance Overview",
            labels={value_column: "Importance", "feature_name": "Feature"},
        )

    def _add_model_artifact_outputs(self, context: PipelineContext) -> None:
        for artifact_name, artifact in context.model_artifacts.items():
            if isinstance(artifact, pd.DataFrame):
                context.diagnostics_tables[artifact_name] = artifact.copy(deep=True)

    def _build_data_quality_outputs(self, context: PipelineContext) -> None:
        data_quality = pd.DataFrame(
            [
                {"metric": "run_id", "value": context.run_id},
                {"metric": "execution_mode", "value": context.config.execution.mode.value},
                {
                    "metric": "labels_available",
                    "value": bool(context.metadata.get("labels_available", False)),
                },
                {"metric": "model_type", "value": context.config.model.model_type.value},
                {"metric": "target_mode", "value": context.config.target.mode.value},
                {
                    "metric": "input_rows",
                    "value": context.metadata.get("input_shape", {}).get("rows"),
                },
                {
                    "metric": "input_columns",
                    "value": context.metadata.get("input_shape", {}).get("columns"),
                },
                {"metric": "feature_count", "value": len(context.feature_columns)},
                {"metric": "numeric_feature_count", "value": len(context.numeric_features)},
                {"metric": "categorical_feature_count", "value": len(context.categorical_features)},
                {"metric": "dropped_column_count", "value": len(set(context.dropped_columns))},
            ]
        )
        context.diagnostics_tables["data_quality_summary"] = data_quality

    def _add_documentation_metadata(self, context: PipelineContext) -> None:
        documentation = context.config.documentation
        rows = [
            {"field": "model_name", "value": documentation.model_name},
            {"field": "model_owner", "value": documentation.model_owner},
            {"field": "business_purpose", "value": documentation.business_purpose},
            {"field": "portfolio_name", "value": documentation.portfolio_name},
            {"field": "segment_name", "value": documentation.segment_name},
            {"field": "horizon_definition", "value": documentation.horizon_definition},
            {"field": "target_definition", "value": documentation.target_definition},
            {"field": "loss_definition", "value": documentation.loss_definition},
            {"field": "assumptions", "value": "; ".join(documentation.assumptions)},
            {"field": "exclusions", "value": "; ".join(documentation.exclusions)},
            {"field": "limitations", "value": "; ".join(documentation.limitations)},
            {"field": "reviewer_notes", "value": documentation.reviewer_notes},
        ]
        context.diagnostics_tables["documentation_metadata"] = pd.DataFrame(rows)

    def _add_feature_dictionary_outputs(self, context: PipelineContext) -> None:
        feature_dictionary = context.config.feature_dictionary
        entry_map = {
            entry.feature_name: entry
            for entry in feature_dictionary.entries
        }
        rows: list[dict[str, Any]] = []
        for feature_name in context.feature_columns:
            entry = entry_map.get(feature_name)
            rows.append(
                {
                    "feature_name": feature_name,
                    "present_in_model": True,
                    "documented": bool(entry and entry.definition.strip()),
                    "business_name": "" if entry is None else entry.business_name,
                    "definition": "" if entry is None else entry.definition,
                    "source_system": "" if entry is None else entry.source_system,
                    "unit": "" if entry is None else entry.unit,
                    "allowed_range": "" if entry is None else entry.allowed_range,
                    "missingness_meaning": "" if entry is None else entry.missingness_meaning,
                    "expected_sign": "" if entry is None else entry.expected_sign,
                    "inclusion_rationale": "" if entry is None else entry.inclusion_rationale,
                    "notes": "" if entry is None else entry.notes,
                }
            )

        for feature_name, entry in entry_map.items():
            if feature_name in context.feature_columns:
                continue
            rows.append(
                {
                    "feature_name": feature_name,
                    "present_in_model": False,
                    "documented": bool(entry.definition.strip()),
                    "business_name": entry.business_name,
                    "definition": entry.definition,
                    "source_system": entry.source_system,
                    "unit": entry.unit,
                    "allowed_range": entry.allowed_range,
                    "missingness_meaning": entry.missingness_meaning,
                    "expected_sign": entry.expected_sign,
                    "inclusion_rationale": entry.inclusion_rationale,
                    "notes": entry.notes,
                }
            )

        if rows:
            feature_dictionary_table = pd.DataFrame(rows)
            context.diagnostics_tables["feature_dictionary"] = feature_dictionary_table
            if feature_dictionary.require_documentation_for_selected_features:
                undocumented = feature_dictionary_table.loc[
                    feature_dictionary_table["present_in_model"]
                    & ~feature_dictionary_table["documented"]
                ]
                if not undocumented.empty:
                    preview = ", ".join(undocumented["feature_name"].head(10))
                    raise ValueError(
                        "Feature dictionary coverage is required for selected features, but "
                        f"these modeled features are undocumented: {preview}."
                    )

    def _add_manual_review_outputs(self, context: PipelineContext) -> None:
        manual_review = context.config.manual_review
        if manual_review.scorecard_bin_overrides:
            context.diagnostics_tables["scorecard_bin_overrides"] = pd.DataFrame(
                [
                    {
                        "feature_name": override.feature_name,
                        "bin_edges": ", ".join(str(value) for value in override.bin_edges),
                        "rationale": override.rationale,
                        "reviewer_name": manual_review.reviewer_name,
                    }
                    for override in manual_review.scorecard_bin_overrides
                ]
            )

    def _add_descriptive_statistics(self, context: PipelineContext) -> None:
        dataframe = context.working_data
        if dataframe is None:
            return
        columns = list(context.feature_columns)
        if self._labels_available(context) and context.target_column in dataframe.columns:
            columns.append(context.target_column)
        descriptive = (
            dataframe[columns].describe(include="all").T.rename_axis("column_name").reset_index()
        )
        context.diagnostics_tables["descriptive_statistics"] = descriptive

    def _add_missingness_outputs(self, context: PipelineContext) -> None:
        dataframe = context.working_data
        if dataframe is None:
            return
        missingness = (
            dataframe.isna()
            .mean()
            .mul(100)
            .rename("missing_pct")
            .rename_axis("column_name")
            .reset_index()
            .sort_values("missing_pct", ascending=False)
        )
        context.diagnostics_tables["missingness"] = missingness
        context.visualizations["missingness"] = px.bar(
            missingness.head(25),
            x="column_name",
            y="missing_pct",
            title="Missingness by Column",
            labels={"column_name": "Column", "missing_pct": "Missing %"},
        )

    def _add_correlation_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
    ) -> None:
        dataframe = context.working_data
        if dataframe is None:
            return
        numeric_columns = [column for column in top_features if column in context.numeric_features]
        if len(numeric_columns) < 2:
            return
        correlation_matrix = dataframe[numeric_columns].corr(numeric_only=True)
        correlation_table = (
            correlation_matrix.rename_axis("feature_name").reset_index().rename_axis(None, axis=1)
        )
        context.diagnostics_tables["correlation_matrix"] = correlation_table
        context.visualizations["correlation_heatmap"] = px.imshow(
            correlation_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
            origin="lower",
            aspect="auto",
        )

    def _add_vif_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
    ) -> None:
        dataframe = context.split_frames.get("train")
        numeric_columns = [column for column in top_features if column in context.numeric_features]
        if dataframe is None or len(numeric_columns) < 2:
            return

        design = dataframe[numeric_columns].copy()
        design = design.fillna(design.median(numeric_only=True))
        if design.shape[1] < 2:
            return

        vif_rows = []
        matrix = design.to_numpy(dtype=float)
        for index, column in enumerate(numeric_columns):
            try:
                vif_value = variance_inflation_factor(matrix, index)
            except Exception:
                vif_value = np.nan
            vif_rows.append({"feature_name": column, "vif": float(vif_value)})

        vif_table = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
        context.diagnostics_tables["vif"] = vif_table
        context.visualizations["vif_profile"] = px.bar(
            vif_table.head(12),
            x="feature_name",
            y="vif",
            title="Variance Inflation Factor Profile",
            labels={"feature_name": "Feature", "vif": "VIF"},
        )

    def _add_quantile_outputs(self, context: PipelineContext, labels_available: bool) -> None:
        backtest = context.backtest_summary.copy(deep=True)
        context.diagnostics_tables["quantile_summary"] = backtest

        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(
            go.Bar(
                x=backtest["risk_band"],
                y=backtest["observation_count"],
                name="Observation Count",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=backtest["risk_band"],
                y=backtest["average_predicted_pd"],
                name="Average Predicted PD",
                mode="lines+markers",
            ),
            secondary_y=True,
        )
        if labels_available and "observed_default_rate" in backtest.columns:
            figure.add_trace(
                go.Scatter(
                    x=backtest["risk_band"],
                    y=backtest["observed_default_rate"],
                    name="Observed Default Rate",
                    mode="lines+markers",
                ),
                secondary_y=True,
            )
        figure.update_layout(title="Quantile Backtest")
        figure.update_yaxes(title_text="Observations", secondary_y=False)
        figure.update_yaxes(title_text="Rate", secondary_y=True)
        context.visualizations["quantile_backtest"] = figure

    def _add_regression_quantile_outputs(
        self, context: PipelineContext, labels_available: bool
    ) -> None:
        summary = context.backtest_summary.copy(deep=True)
        context.diagnostics_tables["quantile_summary"] = summary
        value_columns = ["average_predicted_value"]
        if labels_available and "observed_average" in summary.columns:
            value_columns.append("observed_average")
        context.visualizations["quantile_backtest"] = px.line(
            summary,
            x=summary.columns[0],
            y=value_columns,
            title="Prediction Quantile Backtest",
            markers=True,
        )

    def _add_threshold_outputs(self, context: PipelineContext) -> None:
        scored_test = context.predictions["test"]
        y_true = scored_test[context.target_column].astype(int).to_numpy()
        probability = scored_test["predicted_probability"].to_numpy()
        threshold_rows = []
        for threshold in np.arange(0.05, 1.0, 0.05):
            predicted_class = (probability >= threshold).astype(int)
            tp = np.sum((predicted_class == 1) & (y_true == 1))
            tn = np.sum((predicted_class == 0) & (y_true == 0))
            fp = np.sum((predicted_class == 1) & (y_true == 0))
            fn = np.sum((predicted_class == 0) & (y_true == 1))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
            f1_score = (
                2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            )
            threshold_rows.append(
                {
                    "threshold": float(round(threshold, 2)),
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                }
            )

        threshold_table = pd.DataFrame(threshold_rows)
        context.diagnostics_tables["threshold_analysis"] = threshold_table
        context.visualizations["threshold_analysis"] = px.line(
            threshold_table,
            x="threshold",
            y=["precision", "recall", "accuracy", "f1_score"],
            title="Threshold Performance Sweep",
            markers=True,
        )

    def _add_calibration_outputs(self, context: PipelineContext) -> None:
        validation_frame = context.predictions.get("validation")
        scored_test = context.predictions["test"]
        y_true = scored_test[context.target_column].astype(int).to_numpy()
        probability = self._clip_probability(
            scored_test["predicted_probability"].to_numpy(dtype=float)
        )
        if len(np.unique(y_true)) < 2:
            context.warn(
                "Skipped calibration analysis because the test split does not contain both "
                "target classes."
            )
            return

        calibration_methods: dict[str, dict[str, Any]] = {
            "base": {
                "label": "Base Model",
                "method_name": "base",
                "fitted_on_split": "model_output",
            }
        }
        calibration_config = context.config.calibration

        if (
            validation_frame is not None
            and context.target_column in validation_frame.columns
            and "predicted_probability" in validation_frame.columns
        ):
            y_validation = validation_frame[context.target_column].astype(int).to_numpy()
            probability_validation = self._clip_probability(
                validation_frame["predicted_probability"].to_numpy(dtype=float)
            )
            if len(np.unique(y_validation)) >= 2:
                if calibration_config.platt_scaling:
                    platt_model = self._fit_platt_calibrator(probability_validation, y_validation)
                    if platt_model is not None:
                        calibration_methods["platt"] = {
                            "label": "Platt Scaling",
                            "method_name": "platt",
                            "fitted_on_split": "validation",
                            "transformer": platt_model,
                        }
                if calibration_config.isotonic_calibration:
                    isotonic_model = self._fit_isotonic_calibrator(
                        probability_validation,
                        y_validation,
                    )
                    if isotonic_model is not None:
                        calibration_methods["isotonic"] = {
                            "label": "Isotonic Calibration",
                            "method_name": "isotonic",
                            "fitted_on_split": "validation",
                            "transformer": isotonic_model,
                        }
            elif calibration_config.platt_scaling or calibration_config.isotonic_calibration:
                context.warn(
                    "Calibration challengers were skipped because the validation split "
                    "does not contain both target classes."
                )
        elif calibration_config.platt_scaling or calibration_config.isotonic_calibration:
            context.warn(
                "Calibration challengers were skipped because a labeled validation split "
                "was unavailable."
            )

        comparison_rows: list[dict[str, Any]] = []
        calibration_tables: list[pd.DataFrame] = []
        figure = go.Figure()
        ranking_metric = calibration_config.ranking_metric.value
        recommended_column = "predicted_probability"
        recommended_method = "base"

        for method_name, method_payload in calibration_methods.items():
            calibrated_probability = self._apply_calibration_method(
                method_payload,
                probability,
            )
            calibration_table = self._build_calibration_table(
                y_true=y_true,
                probability=calibrated_probability,
                bin_count=calibration_config.bin_count,
                strategy=calibration_config.strategy,
            )
            if calibration_table.empty:
                continue
            calibration_table.insert(0, "method_name", method_name)
            calibration_table.insert(1, "method_label", method_payload["label"])
            calibration_tables.append(calibration_table)
            comparison_rows.append(
                self._summarize_calibration_method(
                    method_payload=method_payload,
                    y_true=y_true,
                    probability=calibrated_probability,
                    calibration_table=calibration_table,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=calibration_table["mean_predicted_probability"],
                    y=calibration_table["observed_default_rate"],
                    mode="lines+markers",
                    name=method_payload["label"],
                )
            )

        if not comparison_rows or not calibration_tables:
            return

        calibration_summary = pd.DataFrame(comparison_rows).sort_values(
            ranking_metric,
            ascending=True,
            kind="stable",
        )
        recommended_method = str(calibration_summary.iloc[0]["method_name"])
        recommended_column = (
            "predicted_probability"
            if recommended_method == "base"
            else f"predicted_probability_{recommended_method}"
        )

        for split_name, split_frame in context.predictions.items():
            if "predicted_probability" not in split_frame.columns:
                continue
            split_probability = self._clip_probability(
                split_frame["predicted_probability"].to_numpy(dtype=float)
            )
            updated_frame = split_frame.copy(deep=True)
            for method_name, method_payload in calibration_methods.items():
                if method_name == "base":
                    continue
                updated_frame[f"predicted_probability_{method_name}"] = (
                    self._apply_calibration_method(method_payload, split_probability)
                )
            if recommended_column in updated_frame.columns:
                updated_frame["predicted_probability_recommended"] = updated_frame[
                    recommended_column
                ]
            else:
                updated_frame["predicted_probability_recommended"] = updated_frame[
                    "predicted_probability"
                ]
            context.predictions[split_name] = updated_frame

        context.diagnostics_tables["calibration"] = pd.concat(
            calibration_tables,
            ignore_index=True,
        )
        context.diagnostics_tables["calibration_summary"] = calibration_summary
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line={"dash": "dash"},
            )
        )
        figure.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed Default Rate",
        )
        context.visualizations["calibration_curve"] = figure
        context.visualizations["calibration_method_comparison"] = px.bar(
            calibration_summary,
            x="method_label",
            y=ranking_metric,
            color="method_label",
            title="Calibration Method Comparison",
            hover_data={
                "brier_score": ":.4f",
                "log_loss": ":.4f",
                "expected_calibration_error": ":.4f",
                "maximum_calibration_error": ":.4f",
                "calibration_intercept": ":.4f",
                "calibration_slope": ":.4f",
                "method_name": False,
                "method_label": False,
            },
            labels={
                "method_label": "Method",
                ranking_metric: ranking_metric.replace("_", " ").title(),
            },
        )
        context.metadata["calibration_ranking_metric"] = ranking_metric
        context.metadata["recommended_calibration_method"] = recommended_method
        context.metadata["recommended_calibration_score_column"] = recommended_column
        context.metadata["calibration_methods_evaluated"] = calibration_summary[
            "method_name"
        ].tolist()
        context.statistical_tests["calibration_methods"] = calibration_summary.to_dict(
            orient="records"
        )
        context.log(
            "Evaluated calibration methods and recommended "
            f"`{recommended_method}` using `{ranking_metric}`."
        )

        fpr, tpr, _ = roc_curve(y_true, probability)
        precision, recall, _ = precision_recall_curve(y_true, probability)
        context.diagnostics_tables["roc_curve"] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        context.diagnostics_tables["precision_recall_curve"] = pd.DataFrame(
            {"precision": precision, "recall": recall}
        )
        context.visualizations["roc_curve"] = px.area(
            pd.DataFrame({"fpr": fpr, "tpr": tpr}),
            x="fpr",
            y="tpr",
            title="ROC Curve",
            labels={"fpr": "False Positive Rate", "tpr": "True Positive Rate"},
        )
        context.visualizations["precision_recall_curve"] = px.line(
            pd.DataFrame({"precision": precision, "recall": recall}),
            x="recall",
            y="precision",
            title="Precision-Recall Curve",
        )

    def _fit_platt_calibrator(
        self,
        probability: np.ndarray,
        y_true: np.ndarray,
    ) -> LogisticRegression | None:
        try:
            design = self._safe_logit(probability).reshape(-1, 1)
            calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
            calibrator.fit(design, y_true.astype(int))
        except Exception:
            return None
        return calibrator

    def _fit_isotonic_calibrator(
        self,
        probability: np.ndarray,
        y_true: np.ndarray,
    ) -> IsotonicRegression | None:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(probability, y_true.astype(int))
        except Exception:
            return None
        return calibrator

    def _apply_calibration_method(
        self,
        method_payload: dict[str, Any],
        probability: np.ndarray,
    ) -> np.ndarray:
        method_name = str(method_payload["method_name"])
        clipped_probability = self._clip_probability(probability)
        if method_name == "base":
            return clipped_probability
        transformer = method_payload.get("transformer")
        if method_name == "platt" and transformer is not None:
            transformed = transformer.predict_proba(
                self._safe_logit(clipped_probability).reshape(-1, 1)
            )[:, 1]
            return self._clip_probability(transformed)
        if method_name == "isotonic" and transformer is not None:
            transformed = transformer.predict(clipped_probability)
            return self._clip_probability(np.asarray(transformed, dtype=float))
        return clipped_probability

    def _build_calibration_table(
        self,
        *,
        y_true: np.ndarray,
        probability: np.ndarray,
        bin_count: int,
        strategy: CalibrationStrategy,
    ) -> pd.DataFrame:
        if len(probability) == 0:
            return pd.DataFrame()

        clipped_probability = self._clip_probability(probability)
        effective_bin_count = max(2, min(int(bin_count), len(clipped_probability)))
        if strategy == CalibrationStrategy.QUANTILE:
            ordered_index = np.argsort(clipped_probability, kind="stable")
            ranked_positions = np.arange(len(clipped_probability))
            bucket_codes = np.zeros(len(clipped_probability), dtype=int)
            quantile_codes = pd.qcut(
                ranked_positions,
                q=effective_bin_count,
                labels=False,
                duplicates="drop",
            )
            bucket_codes[ordered_index] = np.asarray(quantile_codes, dtype=int)
        else:
            bucket_codes = np.minimum(
                (clipped_probability * effective_bin_count).astype(int),
                effective_bin_count - 1,
            )

        calibration_frame = pd.DataFrame(
            {
                "bucket_index": bucket_codes,
                "predicted_probability": clipped_probability,
                "target": y_true.astype(int),
            }
        )
        grouped = (
            calibration_frame.groupby("bucket_index", sort=True, dropna=False)
            .agg(
                observation_count=("target", "size"),
                observed_default_count=("target", "sum"),
                expected_default_count=("predicted_probability", "sum"),
                mean_predicted_probability=("predicted_probability", "mean"),
                observed_default_rate=("target", "mean"),
                probability_min=("predicted_probability", "min"),
                probability_max=("predicted_probability", "max"),
            )
            .reset_index()
        )
        grouped["bucket_label"] = grouped["bucket_index"].map(
            lambda value: f"Bin {int(value) + 1}"
        )
        grouped["non_default_count"] = (
            grouped["observation_count"] - grouped["observed_default_count"]
        )
        grouped["absolute_gap"] = (
            grouped["observed_default_rate"] - grouped["mean_predicted_probability"]
        ).abs()
        ordered_columns = [
            "bucket_index",
            "bucket_label",
            "observation_count",
            "observed_default_count",
            "non_default_count",
            "expected_default_count",
            "mean_predicted_probability",
            "observed_default_rate",
            "absolute_gap",
            "probability_min",
            "probability_max",
        ]
        return grouped.loc[:, ordered_columns]

    def _summarize_calibration_method(
        self,
        *,
        method_payload: dict[str, Any],
        y_true: np.ndarray,
        probability: np.ndarray,
        calibration_table: pd.DataFrame,
    ) -> dict[str, Any]:
        clipped_probability = self._clip_probability(probability)
        calibration_intercept, calibration_slope = self._calibration_slope_intercept(
            y_true,
            clipped_probability,
        )
        ece, mce = self._calibration_error_metrics(calibration_table)
        hosmer_lemeshow_statistic, hosmer_lemeshow_p_value = (
            self._hosmer_lemeshow_statistic(calibration_table)
        )
        return {
            "method_name": method_payload["method_name"],
            "method_label": method_payload["label"],
            "fitted_on_split": method_payload.get("fitted_on_split", "model_output"),
            "observation_count": int(len(y_true)),
            "mean_predicted_probability": float(np.mean(clipped_probability)),
            "observed_default_rate": float(np.mean(y_true)),
            "brier_score": float(brier_score_loss(y_true, clipped_probability)),
            "log_loss": float(log_loss(y_true, clipped_probability, labels=[0, 1])),
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "hosmer_lemeshow_statistic": hosmer_lemeshow_statistic,
            "hosmer_lemeshow_p_value": hosmer_lemeshow_p_value,
        }

    def _calibration_slope_intercept(
        self,
        y_true: np.ndarray,
        probability: np.ndarray,
    ) -> tuple[float, float]:
        try:
            design = sm.add_constant(self._safe_logit(probability), has_constant="add")
            fit = sm.GLM(y_true, design, family=sm.families.Binomial()).fit()
            intercept = float(fit.params[0])
            slope = float(fit.params[1])
        except Exception:
            intercept = float("nan")
            slope = float("nan")
        return intercept, slope

    def _calibration_error_metrics(self, calibration_table: pd.DataFrame) -> tuple[float, float]:
        if calibration_table.empty:
            return float("nan"), float("nan")
        total = float(calibration_table["observation_count"].sum()) or 1.0
        gaps = calibration_table["absolute_gap"].to_numpy(dtype=float)
        weights = calibration_table["observation_count"].to_numpy(dtype=float) / total
        expected_calibration_error = float(np.sum(weights * gaps))
        maximum_calibration_error = float(np.max(gaps))
        return expected_calibration_error, maximum_calibration_error

    def _hosmer_lemeshow_statistic(
        self,
        calibration_table: pd.DataFrame,
    ) -> tuple[float, float]:
        if calibration_table.empty:
            return float("nan"), float("nan")
        observed_defaults = calibration_table["observed_default_count"].to_numpy(dtype=float)
        expected_defaults = calibration_table["expected_default_count"].to_numpy(dtype=float)
        observation_count = calibration_table["observation_count"].to_numpy(dtype=float)
        observed_non_defaults = observation_count - observed_defaults
        expected_non_defaults = observation_count - expected_defaults
        default_term = (observed_defaults - expected_defaults) ** 2 / np.maximum(
            expected_defaults,
            1e-9,
        )
        non_default_term = (
            (observed_non_defaults - expected_non_defaults) ** 2
            / np.maximum(expected_non_defaults, 1e-9)
        )
        statistic = float(np.sum(default_term + non_default_term))
        degrees_of_freedom = max(int(len(calibration_table) - 2), 1)
        p_value = float(1.0 - chi2.cdf(statistic, df=degrees_of_freedom))
        return statistic, p_value

    def _clip_probability(self, probability: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)

    def _safe_logit(self, probability: np.ndarray) -> np.ndarray:
        clipped_probability = self._clip_probability(probability)
        return np.log(clipped_probability / (1.0 - clipped_probability))

    def _add_lift_gain_outputs(self, context: PipelineContext) -> None:
        scored_test = context.predictions["test"].copy(deep=True)
        scored_test = scored_test.sort_values("predicted_probability", ascending=False).reset_index(
            drop=True
        )
        scored_test["bucket"] = (
            pd.qcut(
                scored_test.index + 1,
                q=min(context.config.diagnostics.quantile_bucket_count, len(scored_test)),
                labels=False,
            )
            + 1
        )
        lift_gain = (
            scored_test.groupby("bucket", dropna=False)
            .agg(
                observation_count=("predicted_probability", "size"),
                default_count=(context.target_column, "sum"),
            )
            .reset_index()
            .sort_values("bucket")
        )
        total_defaults = float(lift_gain["default_count"].sum()) or 1.0
        lift_gain["capture_rate"] = lift_gain["default_count"] / total_defaults
        lift_gain["cumulative_capture_rate"] = lift_gain["capture_rate"].cumsum()
        lift_gain["lift"] = lift_gain["capture_rate"] / (1.0 / len(lift_gain))
        context.diagnostics_tables["lift_gain"] = lift_gain
        context.visualizations["gain_chart"] = px.line(
            lift_gain,
            x="bucket",
            y="cumulative_capture_rate",
            title="Cumulative Gain Chart",
            markers=True,
        )
        context.visualizations["lift_chart"] = px.bar(
            lift_gain,
            x="bucket",
            y="lift",
            title="Lift by Quantile Bucket",
        )

    def _add_lifetime_pd_outputs(self, context: PipelineContext, labels_available: bool) -> None:
        scored_test = context.predictions.get("test")
        if scored_test is None or "hazard_period_index" not in scored_test.columns:
            return
        grouped = (
            scored_test.groupby("hazard_period_index", dropna=False)
            .agg(
                observation_count=("predicted_probability", "size"),
                mean_predicted_hazard=("predicted_probability", "mean"),
            )
            .reset_index()
            .sort_values("hazard_period_index")
        )
        grouped["predicted_survival_rate"] = (1.0 - grouped["mean_predicted_hazard"]).cumprod()
        grouped["predicted_cumulative_pd"] = 1.0 - grouped["predicted_survival_rate"]
        if labels_available and context.target_column in scored_test.columns:
            observed = (
                scored_test.groupby("hazard_period_index", dropna=False)[context.target_column]
                .mean()
                .reset_index(name="observed_hazard_rate")
                .sort_values("hazard_period_index")
            )
            grouped = grouped.merge(observed, on="hazard_period_index", how="left")
            grouped["observed_survival_rate"] = (
                1.0 - grouped["observed_hazard_rate"].fillna(0.0)
            ).cumprod()
            grouped["observed_cumulative_pd"] = 1.0 - grouped["observed_survival_rate"]
        context.diagnostics_tables["lifetime_pd_curve"] = grouped
        plot_columns = ["predicted_cumulative_pd"]
        if "observed_cumulative_pd" in grouped.columns:
            plot_columns.append("observed_cumulative_pd")
        context.visualizations["lifetime_pd_curve"] = px.line(
            grouped,
            x="hazard_period_index",
            y=plot_columns,
            markers=True,
            title="Lifetime PD Curve",
            labels={
                "hazard_period_index": "Period",
                "value": "Cumulative PD",
                "variable": "Series",
            },
        )

    def _add_woe_iv_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
    ) -> None:
        dataframe = context.split_frames.get("train")
        if dataframe is None:
            return

        woe_rows = []
        detailed_rows = []
        for feature in top_features[: context.config.diagnostics.top_n_features]:
            feature_series = dataframe[feature]
            if feature in context.numeric_features:
                bucketed = self._bucket_numeric_series(
                    feature_series,
                    max(3, min(10, context.config.diagnostics.quantile_bucket_count)),
                )
            else:
                bucketed = feature_series.fillna("Missing").astype(str)

            summary = (
                pd.DataFrame(
                    {
                        "bucket": bucketed.astype(str),
                        "target": dataframe[context.target_column].astype(int),
                    }
                )
                .groupby("bucket", dropna=False)
                .agg(
                    bad=("target", "sum"),
                    total=("target", "size"),
                )
                .reset_index()
            )
            summary["good"] = summary["total"] - summary["bad"]
            total_bad = max(float(summary["bad"].sum()), 1.0)
            total_good = max(float(summary["good"].sum()), 1.0)
            summary["bad_pct"] = summary["bad"].clip(lower=0.5) / total_bad
            summary["good_pct"] = summary["good"].clip(lower=0.5) / total_good
            summary["woe"] = np.log(summary["good_pct"] / summary["bad_pct"])
            summary["iv_component"] = (summary["good_pct"] - summary["bad_pct"]) * summary["woe"]
            iv_value = float(summary["iv_component"].sum())
            woe_rows.append({"feature_name": feature, "information_value": iv_value})
            summary.insert(0, "feature_name", feature)
            detailed_rows.append(summary)

        if woe_rows:
            context.diagnostics_tables["woe_iv_summary"] = pd.DataFrame(woe_rows).sort_values(
                "information_value", ascending=False
            )
            context.diagnostics_tables["woe_iv_detail"] = pd.concat(
                detailed_rows,
                ignore_index=True,
            )

    def _add_psi_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
    ) -> None:
        train_frame = context.split_frames.get("train")
        test_frame = context.split_frames.get("test")
        if train_frame is None or test_frame is None:
            return

        psi_rows = []
        for feature in top_features[: context.config.diagnostics.top_n_features]:
            psi_value = self._compute_population_stability_index(
                train_frame[feature],
                test_frame[feature],
            )
            psi_rows.append({"feature_name": feature, "psi": psi_value})

        score_column = (
            "predicted_probability"
            if context.config.target.mode == TargetMode.BINARY
            else "predicted_value"
        )
        psi_rows.append(
            {
                "feature_name": score_column,
                "psi": self._compute_population_stability_index(
                    context.predictions["train"][score_column],
                    context.predictions["test"][score_column],
                ),
            }
        )
        psi_table = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
        context.diagnostics_tables["psi"] = psi_table
        context.visualizations["psi_profile"] = px.bar(
            psi_table.head(12),
            x="feature_name",
            y="psi",
            title="Population Stability Index",
            labels={"feature_name": "Feature", "psi": "PSI"},
        )

    def _add_segment_outputs(self, context: PipelineContext, labels_available: bool) -> None:
        segment_column = context.config.diagnostics.default_segment_column
        if not segment_column:
            segment_column = (
                context.categorical_features[0] if context.categorical_features else None
            )
        if not segment_column:
            return

        rows = []
        score_column = (
            "predicted_probability"
            if context.config.target.mode == TargetMode.BINARY
            else "predicted_value"
        )
        for split_name, scored_frame in context.predictions.items():
            if segment_column not in scored_frame.columns:
                continue
            aggregations: dict[str, tuple[str, str]] = {
                "observation_count": (score_column, "size"),
                "average_score": (score_column, "mean"),
            }
            if labels_available and context.target_column in scored_frame.columns:
                aggregations["average_actual"] = (context.target_column, "mean")
            grouped = (
                scored_frame.groupby(segment_column, dropna=False).agg(**aggregations).reset_index()
            )
            grouped.insert(0, "split", split_name)
            rows.append(grouped)

        if not rows:
            return

        segment_table = pd.concat(rows, ignore_index=True)
        context.diagnostics_tables["segment_performance"] = segment_table
        if "average_actual" in segment_table.columns:
            long_segment_table = segment_table.melt(
                id_vars=["split", segment_column, "observation_count"],
                value_vars=["average_actual", "average_score"],
                var_name="metric",
                value_name="value",
            )
            context.visualizations["segment_performance_chart"] = px.bar(
                long_segment_table,
                x=segment_column,
                y="value",
                color="split",
                barmode="group",
                facet_row="metric",
                title=f"Observed vs Predicted by {segment_column}",
            )
        else:
            context.visualizations["segment_performance_chart"] = px.bar(
                segment_table,
                x=segment_column,
                y="average_score",
                color="split",
                barmode="group",
                title=f"Average Score by {segment_column}",
            )
        volume_table = (
            segment_table.groupby(segment_column, dropna=False)["observation_count"]
            .sum()
            .reset_index()
            .sort_values("observation_count", ascending=False)
            .head(context.config.diagnostics.top_n_categories)
        )
        context.visualizations["segment_volume"] = px.bar(
            volume_table,
            x=segment_column,
            y="observation_count",
            title=f"Observation Mix by {segment_column}",
            labels={segment_column: "Segment", "observation_count": "Observations"},
        )

    def _add_adf_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
        labels_available: bool,
    ) -> None:
        if context.config.split.data_structure not in {
            DataStructure.TIME_SERIES,
            DataStructure.PANEL,
        }:
            return

        date_column = context.config.split.date_column
        if not date_column:
            return

        rows = []
        for split_name, scored_frame in context.predictions.items():
            if date_column not in scored_frame.columns:
                continue
            aggregations: dict[str, tuple[str, str]] = {
                "prediction_mean": (
                    "predicted_probability"
                    if context.config.target.mode == TargetMode.BINARY
                    else "predicted_value",
                    "mean",
                ),
            }
            if labels_available and context.target_column in scored_frame.columns:
                aggregations["target_mean"] = (context.target_column, "mean")
            aggregated = scored_frame.groupby(date_column, dropna=False).agg(**aggregations)
            for series_name in aggregated.columns.tolist():
                rows.append(
                    self._run_adf_test(
                        aggregated[series_name].dropna(),
                        split_name=split_name,
                        series_name=series_name,
                    )
                )

            for feature in top_features[: min(3, len(top_features))]:
                if feature not in scored_frame.columns:
                    continue
                try:
                    feature_series = (
                        scored_frame.groupby(date_column, dropna=False)[feature].mean().dropna()
                    )
                except TypeError:
                    continue
                rows.append(
                    self._run_adf_test(
                        feature_series,
                        split_name=split_name,
                        series_name=feature,
                    )
                )

            time_plot = aggregated.reset_index()
            context.visualizations[f"time_backtest_{split_name}"] = px.line(
                time_plot,
                x=date_column,
                y=time_plot.columns.drop(date_column).tolist(),
                title=f"Observed vs Predicted Over Time ({split_name.title()})"
                if "target_mean" in aggregated.columns
                else f"Predicted Score Over Time ({split_name.title()})",
            )

        rows = [row for row in rows if row]
        if rows:
            context.diagnostics_tables["adf_tests"] = pd.DataFrame(rows)
            context.statistical_tests["adf"] = context.diagnostics_tables["adf_tests"].to_dict(
                orient="records"
            )

    def _add_residual_outputs(self, context: PipelineContext) -> None:
        scored_test = context.predictions["test"].copy(deep=True)
        scored_test["residual"] = (
            scored_test[context.target_column] - scored_test["predicted_value"]
        )
        residual_summary = scored_test["residual"].describe().reset_index()
        residual_summary.columns = ["statistic", "value"]
        context.diagnostics_tables["residual_summary"] = residual_summary
        sampled = self._sample_frame(scored_test[["predicted_value", "residual"]], context)
        context.visualizations["residuals_vs_predicted"] = px.scatter(
            sampled,
            x="predicted_value",
            y="residual",
            title="Residuals vs Predicted",
            opacity=0.5,
        )
        context.visualizations["actual_vs_predicted"] = px.scatter(
            self._sample_frame(
                scored_test[[context.target_column, "predicted_value"]],
                context,
            ),
            x=context.target_column,
            y="predicted_value",
            title="Actual vs Predicted",
            opacity=0.5,
            trendline="ols",
        )

    def _add_qq_outputs(self, context: PipelineContext) -> None:
        scored_test = context.predictions["test"].copy(deep=True)
        qq = ProbPlot(scored_test["residual"].dropna().to_numpy())
        qq_table = pd.DataFrame(
            {
                "theoretical_quantile": qq.theoretical_quantiles,
                "sample_quantile": np.sort(scored_test["residual"].dropna().to_numpy()),
            }
        )
        context.diagnostics_tables["qq_plot_data"] = qq_table
        context.visualizations["qq_plot"] = px.scatter(
            qq_table,
            x="theoretical_quantile",
            y="sample_quantile",
            title="Residual QQ Plot",
            trendline="ols",
        )

    def _add_model_comparison_outputs(self, context: PipelineContext) -> None:
        comparison_table = context.comparison_results
        if comparison_table is None or comparison_table.empty:
            return

        context.diagnostics_tables["model_comparison"] = comparison_table.copy(deep=True)
        ranking_split = context.config.comparison.ranking_split
        ranking_frame = comparison_table.loc[
            comparison_table["split"] == ranking_split
        ].dropna(subset=["ranking_value"])
        if ranking_frame.empty:
            return

        ranking_frame = ranking_frame.copy(deep=True)
        ranking_frame["model_label"] = (
            ranking_frame["model_type"].astype(str).str.replace("_", " ").str.title()
        )
        context.visualizations["model_comparison_chart"] = px.bar(
            ranking_frame,
            x="model_label",
            y="ranking_value",
            color="is_primary",
            title=(
                f"Model Comparison on {ranking_split.title()} "
                f"({ranking_frame['ranking_metric'].iloc[0]})"
            ),
            labels={
                "model_label": "Model",
                "ranking_value": "Metric Value",
                "is_primary": "Primary Model",
            },
        )

    def _add_robustness_outputs(self, context: PipelineContext) -> None:
        robustness = context.config.robustness
        if context.config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
            context.warn(
                "Skipped robustness testing because existing-model scoring does not refit "
                "the model on repeated resamples."
            )
            return
        if not self._labels_available(context) or context.target_column is None:
            context.warn(
                "Skipped robustness testing because labels are unavailable for repeated "
                "held-out evaluation."
            )
            return

        train_frame = context.split_frames.get("train")
        evaluation_frame = context.split_frames.get(robustness.evaluation_split)
        if train_frame is None or evaluation_frame is None:
            context.warn(
                "Skipped robustness testing because the required train/evaluation "
                "splits were unavailable."
            )
            return

        metric_rows: list[dict[str, Any]] = []
        feature_rows: list[dict[str, Any]] = []
        successful_resamples = 0
        sample_size = max(2, int(round(len(train_frame) * robustness.sample_fraction)))
        if not robustness.sample_with_replacement:
            sample_size = min(sample_size, len(train_frame))
        evaluator = EvaluationStep()

        for resample_id in range(robustness.resample_count):
            resampled_train = train_frame.sample(
                n=sample_size,
                replace=robustness.sample_with_replacement,
                random_state=robustness.random_state + resample_id,
            ).reset_index(drop=True)
            try:
                resampled_model = build_model_adapter(
                    deepcopy(context.config.model),
                    context.config.target.mode,
                    scorecard_config=context.config.scorecard,
                    scorecard_bin_overrides={
                        override.feature_name: override.bin_edges
                        for override in context.config.manual_review.scorecard_bin_overrides
                    },
                )
                resampled_model.fit(
                    resampled_train[context.feature_columns],
                    resampled_train[context.target_column],
                    context.numeric_features,
                    context.categorical_features,
                )
            except Exception as exc:
                context.warn(
                    f"Robustness resample {resample_id + 1} failed during model fitting: {exc}"
                )
                continue

            try:
                if context.config.target.mode == TargetMode.BINARY:
                    _, metrics = evaluator._score_binary_split(
                        evaluation_frame,
                        robustness.evaluation_split,
                        context.target_column,
                        context.feature_columns,
                        resampled_model,
                        context.config.model.threshold,
                        True,
                    )
                    candidate_metrics = [
                        "roc_auc",
                        "average_precision",
                        "ks_statistic",
                        "brier_score",
                        "log_loss",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1_score",
                        "matthews_correlation",
                    ]
                else:
                    _, metrics = evaluator._score_continuous_split(
                        evaluation_frame,
                        robustness.evaluation_split,
                        context.target_column,
                        context.feature_columns,
                        resampled_model,
                        True,
                    )
                    candidate_metrics = ["rmse", "mae", "r2", "explained_variance"]
            except Exception as exc:
                context.warn(
                    f"Robustness resample {resample_id + 1} failed during held-out scoring: {exc}"
                )
                continue

            successful_resamples += 1
            if robustness.metric_stability:
                for metric_name in candidate_metrics:
                    metric_value = metrics.get(metric_name)
                    if metric_value is None or pd.isna(metric_value):
                        continue
                    metric_rows.append(
                        {
                            "resample_id": resample_id + 1,
                            "evaluation_split": robustness.evaluation_split,
                            "metric_name": metric_name,
                            "metric_value": float(metric_value),
                        }
                    )
            if robustness.coefficient_stability:
                feature_rows.extend(
                    self._build_robustness_feature_rows(
                        feature_importance=resampled_model.get_feature_importance(),
                        resample_id=resample_id + 1,
                        context=context,
                    )
                )

        if successful_resamples < 2:
            context.warn(
                "Robustness testing did not complete enough successful resamples to build "
                "stable outputs."
            )
            return

        if metric_rows:
            metric_table = pd.DataFrame(metric_rows)
            metric_summary = self._summarize_robustness_metrics(metric_table)
            context.diagnostics_tables["robustness_metric_distribution"] = metric_table
            context.diagnostics_tables["robustness_metric_summary"] = metric_summary
            context.visualizations["robustness_metric_boxplot"] = px.box(
                metric_table,
                x="metric_name",
                y="metric_value",
                color="metric_name",
                points="all",
                title="Metric Stability Across Resamples",
                labels={"metric_name": "Metric", "metric_value": "Held-Out Value"},
            )
            context.visualizations["robustness_metric_summary_chart"] = px.bar(
                metric_summary,
                x="metric_name",
                y="mean_value",
                error_y="std_value",
                title="Average Held-Out Metric by Resample",
                labels={"metric_name": "Metric", "mean_value": "Average Value"},
            )

        if feature_rows:
            feature_table = pd.DataFrame(feature_rows)
            feature_summary = self._summarize_robustness_features(
                feature_table,
                successful_resamples=successful_resamples,
            )
            context.diagnostics_tables["robustness_feature_distribution"] = feature_table
            context.diagnostics_tables["robustness_feature_stability"] = feature_summary
            chart_frame = (
                feature_summary.sort_values("mean_abs_effect", ascending=False)
                .head(context.config.diagnostics.top_n_features)
                .sort_values("mean_abs_effect", ascending=True)
            )
            if not chart_frame.empty:
                context.visualizations["robustness_feature_stability"] = px.bar(
                    chart_frame,
                    x="mean_abs_effect",
                    y="feature_name",
                    orientation="h",
                    error_x="std_abs_effect",
                    color="effect_basis",
                    title="Feature Stability Profile",
                    labels={
                        "mean_abs_effect": "Average Absolute Effect",
                        "feature_name": "Feature",
                        "effect_basis": "Effect Basis",
                    },
                    hover_data={
                        "selection_frequency": ":.2f",
                        "mean_effect": ":.4f",
                        "std_effect": ":.4f",
                        "sign_consistency": ":.2f",
                    },
                )

        context.metadata["robustness_summary"] = {
            "enabled": True,
            "successful_resamples": successful_resamples,
            "requested_resamples": robustness.resample_count,
            "evaluation_split": robustness.evaluation_split,
        }
        context.log(
            "Completed robustness testing across "
            f"{successful_resamples} held-out resamples."
        )

    def _build_robustness_feature_rows(
        self,
        *,
        feature_importance: pd.DataFrame,
        resample_id: int,
        context: PipelineContext,
    ) -> list[dict[str, Any]]:
        if feature_importance.empty or "feature_name" not in feature_importance.columns:
            return []

        working = feature_importance.copy(deep=True)
        working["source_feature_name"] = working["feature_name"].map(
            lambda value: self._infer_source_feature_name(str(value), context.feature_columns)
        )
        has_coefficients = (
            "coefficient" in working.columns and working["coefficient"].notna().any()
        )
        rows: list[dict[str, Any]] = []
        grouped = working.groupby("source_feature_name", dropna=False)
        for feature_name, group in grouped:
            importance_value = pd.to_numeric(
                group.get("importance_value", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0.0)
            if has_coefficients:
                coefficient_value = pd.to_numeric(
                    group["coefficient"],
                    errors="coerce",
                ).fillna(0.0)
                effect_value = float(coefficient_value.mean())
                signed_effect = True
                effect_basis = "coefficient"
            else:
                effect_value = float(importance_value.sum())
                signed_effect = False
                effect_basis = "importance"
            rows.append(
                {
                    "resample_id": resample_id,
                    "feature_name": str(feature_name),
                    "effect_basis": effect_basis,
                    "effect_value": effect_value,
                    "abs_effect_value": float(abs(effect_value)),
                    "importance_value": float(importance_value.sum()),
                    "signed_effect": signed_effect,
                }
            )
        return rows

    def _summarize_robustness_metrics(self, metric_table: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for metric_name, group in metric_table.groupby("metric_name", dropna=False):
            values = group["metric_value"].astype(float)
            rows.append(
                {
                    "metric_name": metric_name,
                    "resample_count": int(len(values)),
                    "mean_value": float(values.mean()),
                    "std_value": float(values.std(ddof=0)),
                    "min_value": float(values.min()),
                    "p05_value": float(values.quantile(0.05)),
                    "median_value": float(values.quantile(0.50)),
                    "p95_value": float(values.quantile(0.95)),
                    "max_value": float(values.max()),
                }
            )
        return pd.DataFrame(rows).sort_values("metric_name").reset_index(drop=True)

    def _summarize_robustness_features(
        self,
        feature_table: pd.DataFrame,
        *,
        successful_resamples: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for feature_name, group in feature_table.groupby("feature_name", dropna=False):
            effect_basis = str(group["effect_basis"].mode().iloc[0])
            effect_values = pd.to_numeric(group["effect_value"], errors="coerce").fillna(0.0)
            abs_effect_values = pd.to_numeric(
                group["abs_effect_value"],
                errors="coerce",
            ).fillna(0.0)
            importance_values = pd.to_numeric(
                group["importance_value"],
                errors="coerce",
            ).fillna(0.0)
            signed_effect = bool(group["signed_effect"].any())
            positive_share = (
                float((effect_values > 0).mean()) if signed_effect else float("nan")
            )
            negative_share = (
                float((effect_values < 0).mean()) if signed_effect else float("nan")
            )
            sign_consistency = (
                max(positive_share, negative_share) if signed_effect else float("nan")
            )
            rows.append(
                {
                    "feature_name": str(feature_name),
                    "effect_basis": effect_basis,
                    "selection_frequency": float(
                        group["resample_id"].nunique() / max(successful_resamples, 1)
                    ),
                    "mean_effect": float(effect_values.mean()),
                    "std_effect": float(effect_values.std(ddof=0)),
                    "mean_abs_effect": float(abs_effect_values.mean()),
                    "std_abs_effect": float(abs_effect_values.std(ddof=0)),
                    "mean_importance": float(importance_values.mean()),
                    "std_importance": float(importance_values.std(ddof=0)),
                    "positive_share": positive_share,
                    "negative_share": negative_share,
                    "sign_consistency": sign_consistency,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["selection_frequency", "mean_abs_effect", "feature_name"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

    def _add_scorecard_workbench_outputs(self, context: PipelineContext) -> None:
        if context.config.model.model_type != ModelType.SCORECARD_LOGISTIC_REGRESSION:
            return
        woe_table = context.diagnostics_tables.get("scorecard_woe_table", pd.DataFrame())
        points_table = context.diagnostics_tables.get("scorecard_points_table", pd.DataFrame())
        if woe_table.empty or points_table.empty:
            return

        workbench = context.config.scorecard_workbench
        woe_detail = woe_table.copy(deep=True)
        points_detail = points_table.copy(deep=True)
        woe_detail["bucket_rank"] = woe_detail.groupby("feature_name", dropna=False).cumcount() + 1
        points_detail["bucket_rank"] = (
            points_detail.groupby("feature_name", dropna=False).cumcount() + 1
        )
        woe_detail["bad_rate"] = woe_detail["bad"] / woe_detail["total"].replace(0, np.nan)
        context.diagnostics_tables["scorecard_woe_table"] = woe_detail
        context.diagnostics_tables["scorecard_points_table"] = points_detail

        feature_summary = self._build_scorecard_feature_summary(
            woe_detail=woe_detail,
            points_detail=points_detail,
            manual_override_features={
                override.feature_name
                for override in context.config.manual_review.scorecard_bin_overrides
            },
        )
        if feature_summary.empty:
            return

        context.diagnostics_tables["scorecard_feature_summary"] = feature_summary
        featured_features = feature_summary.head(workbench.max_features)["feature_name"].tolist()
        context.visualizations["scorecard_feature_iv"] = px.bar(
            feature_summary.head(workbench.max_features).sort_values(
                "information_value", ascending=True
            ),
            x="information_value",
            y="feature_name",
            orientation="h",
            title="Scorecard Feature Information Value",
            labels={"information_value": "Information Value", "feature_name": "Feature"},
        )

        if workbench.include_score_distribution:
            score_distribution_figure = self._build_scorecard_score_distribution(context)
            if score_distribution_figure is not None:
                context.visualizations["scorecard_score_distribution"] = score_distribution_figure

        if workbench.include_reason_code_analysis:
            reason_code_table = self._build_scorecard_reason_code_frequency(context)
            if not reason_code_table.empty:
                context.diagnostics_tables["scorecard_reason_code_frequency"] = reason_code_table
                context.visualizations["scorecard_reason_code_frequency_chart"] = px.bar(
                    reason_code_table.head(workbench.max_features * 3),
                    x="feature_name",
                    y="count",
                    color="reason_code_rank",
                    barmode="group",
                    title="Reason Code Frequency",
                    labels={
                        "feature_name": "Feature",
                        "count": "Count",
                        "reason_code_rank": "Reason Code Slot",
                    },
                )

        for feature_name in featured_features:
            self._add_scorecard_feature_figures(
                context=context,
                feature_name=feature_name,
                woe_detail=woe_detail,
                points_detail=points_detail,
            )

    def _build_scorecard_feature_summary(
        self,
        *,
        woe_detail: pd.DataFrame,
        points_detail: pd.DataFrame,
        manual_override_features: set[str],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for feature_name, woe_feature in woe_detail.groupby("feature_name", dropna=False):
            points_feature = points_detail.loc[
                points_detail["feature_name"] == feature_name
            ].copy(deep=True)
            if points_feature.empty:
                continue
            bad_rates = pd.to_numeric(woe_feature["bad_rate"], errors="coerce").dropna()
            woe_values = pd.to_numeric(woe_feature["woe"], errors="coerce").dropna()
            point_values = pd.to_numeric(
                points_feature["partial_score_points"],
                errors="coerce",
            ).dropna()
            total_values = pd.to_numeric(woe_feature["total"], errors="coerce").dropna()
            rows.append(
                {
                    "feature_name": str(feature_name),
                    "bin_count": int(len(woe_feature)),
                    "information_value": float(
                        pd.to_numeric(
                            woe_feature["iv_component"],
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .sum()
                    ),
                    "average_bad_rate": float(bad_rates.mean()) if not bad_rates.empty else np.nan,
                    "largest_bin_share": float(total_values.max() / total_values.sum())
                    if not total_values.empty and float(total_values.sum()) > 0
                    else np.nan,
                    "woe_span": float(woe_values.max() - woe_values.min())
                    if not woe_values.empty
                    else np.nan,
                    "points_span": float(point_values.max() - point_values.min())
                    if not point_values.empty
                    else np.nan,
                    "bad_rate_trend": self._describe_scorecard_trend(bad_rates.tolist()),
                    "manual_override_applied": str(feature_name) in manual_override_features,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["information_value", "feature_name"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _build_scorecard_score_distribution(self, context: PipelineContext) -> go.Figure | None:
        score_split = context.predictions.get("test")
        if score_split is None:
            score_split = next(iter(context.predictions.values()), None)
        if score_split is None or "scorecard_points" not in score_split.columns:
            return None
        sampled = self._sample_frame(score_split.copy(deep=True), context)
        color_column = None
        if self._labels_available(context) and context.target_column in sampled.columns:
            color_column = context.target_column
            sampled[color_column] = sampled[color_column].astype(str)
        elif "split" in sampled.columns:
            color_column = "split"
        return px.histogram(
            sampled,
            x="scorecard_points",
            color=color_column,
            nbins=40,
            title="Scorecard Points Distribution",
            labels={"scorecard_points": "Scorecard Points"},
        )

    def _build_scorecard_reason_code_frequency(self, context: PipelineContext) -> pd.DataFrame:
        score_split = context.predictions.get("test")
        if score_split is None:
            score_split = next(iter(context.predictions.values()), None)
        if score_split is None:
            return pd.DataFrame()
        reason_columns = sorted(
            column for column in score_split.columns if column.startswith("reason_code_")
        )
        if not reason_columns:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        denominator = max(len(score_split), 1)
        for reason_column in reason_columns:
            rank_text = reason_column.rsplit("_", 1)[-1]
            rank_value = int(rank_text) if rank_text.isdigit() else reason_column
            counts = (
                score_split[reason_column]
                .replace("", np.nan)
                .dropna()
                .astype(str)
                .value_counts()
            )
            for feature_name, count in counts.items():
                rows.append(
                    {
                        "reason_code_rank": rank_value,
                        "feature_name": feature_name,
                        "count": int(count),
                        "share": float(count / denominator),
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values(["count", "feature_name"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _add_scorecard_feature_figures(
        self,
        *,
        context: PipelineContext,
        feature_name: str,
        woe_detail: pd.DataFrame,
        points_detail: pd.DataFrame,
    ) -> None:
        feature_woe = woe_detail.loc[woe_detail["feature_name"] == feature_name].copy(deep=True)
        feature_points = points_detail.loc[
            points_detail["feature_name"] == feature_name
        ].copy(deep=True)
        if feature_woe.empty or feature_points.empty:
            return
        feature_woe = feature_woe.sort_values("bucket_rank")
        feature_points = feature_points.sort_values("bucket_rank")
        asset_key = self._sanitize_asset_name(feature_name)

        context.visualizations[f"scorecard_bad_rate_{asset_key}"] = px.bar(
            feature_woe,
            x="bucket_label",
            y="bad_rate",
            title=f"Scorecard Bad Rate by Bucket: {feature_name}",
            labels={"bucket_label": "Bucket", "bad_rate": "Bad Rate"},
        )
        context.visualizations[f"scorecard_woe_{asset_key}"] = px.line(
            feature_woe,
            x="bucket_label",
            y="woe",
            title=f"Scorecard WoE by Bucket: {feature_name}",
            labels={"bucket_label": "Bucket", "woe": "WoE"},
            markers=True,
        )
        context.visualizations[f"scorecard_points_{asset_key}"] = px.bar(
            feature_points,
            x="bucket_label",
            y="partial_score_points",
            title=f"Scorecard Points by Bucket: {feature_name}",
            labels={"bucket_label": "Bucket", "partial_score_points": "Partial Score"},
        )

    def _describe_scorecard_trend(self, values: list[float]) -> str:
        finite_values = [float(value) for value in values if pd.notna(value)]
        if len(finite_values) < 2:
            return "not_applicable"
        if all(
            left <= right
            for left, right in zip(finite_values, finite_values[1:], strict=False)
        ):
            return "increasing"
        if all(
            left >= right
            for left, right in zip(finite_values, finite_values[1:], strict=False)
        ):
            return "decreasing"
        return "non_monotonic"

    def _add_explainability_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
        labels_available: bool,
    ) -> None:
        if (
            context.config.explainability.coefficient_breakdown
            and "coefficient" in context.feature_importance.columns
        ):
            coefficient_table = context.feature_importance.copy(deep=True).head(
                context.config.explainability.top_n_features
            )
            coefficient_table["coefficient_direction"] = coefficient_table["coefficient"].map(
                lambda value: "positive" if value > 0 else "negative" if value < 0 else "flat"
            )
            context.diagnostics_tables["coefficient_breakdown"] = coefficient_table

        if context.config.explainability.feature_effect_curves:
            effect_table = self._build_feature_effect_curves(context, top_features)
            if not effect_table.empty:
                context.diagnostics_tables["feature_effect_curves"] = effect_table
                for feature_name in effect_table["feature_name"].drop_duplicates().tolist():
                    feature_table = effect_table.loc[
                        effect_table["feature_name"] == feature_name
                    ].copy(deep=True)
                    if feature_table["curve_type"].iloc[0] == "numeric":
                        figure = px.line(
                            feature_table.sort_values("sort_order"),
                            x="feature_value",
                            y="average_prediction",
                            title=f"Feature Effect: {feature_name}",
                            markers=True,
                        )
                    else:
                        figure = px.bar(
                            feature_table.sort_values("sort_order"),
                            x="feature_value",
                            y="average_prediction",
                            title=f"Feature Effect: {feature_name}",
                        )
                    context.visualizations[
                        f"feature_effect_{self._sanitize_asset_name(feature_name)}"
                    ] = figure

        if context.config.explainability.permutation_importance and labels_available:
            permutation_table = self._build_permutation_importance(context, top_features)
            if not permutation_table.empty:
                context.diagnostics_tables["permutation_importance"] = permutation_table
                context.visualizations["permutation_importance"] = px.bar(
                    permutation_table.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature_name",
                    orientation="h",
                    title="Permutation Importance",
                    labels={"importance": "Importance", "feature_name": "Feature"},
                )

    def _build_feature_effect_curves(
        self,
        context: PipelineContext,
        top_features: list[str],
    ) -> pd.DataFrame:
        scored_frame = context.predictions.get("test")
        if scored_frame is None:
            return pd.DataFrame()
        sampled = self._sample_rows(
            scored_frame[context.feature_columns].copy(deep=True),
            context.config.explainability.sample_size,
            context,
        )
        rows: list[dict[str, Any]] = []
        score_label = (
            "predicted_probability"
            if context.config.target.mode == TargetMode.BINARY
            else "predicted_value"
        )
        for feature_name in top_features[: context.config.explainability.top_n_features]:
            series = sampled[feature_name]
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna().astype(float)
                if non_null.nunique() < 2:
                    continue
                grid = np.unique(
                    np.quantile(
                        non_null.to_numpy(),
                        np.linspace(0.05, 0.95, context.config.explainability.grid_points),
                    )
                )
                for index, grid_value in enumerate(grid):
                    modified = sampled.copy(deep=True)
                    modified[feature_name] = float(grid_value)
                    rows.append(
                        {
                            "feature_name": feature_name,
                            "feature_value": float(grid_value),
                            "average_prediction": float(
                                np.mean(context.model.predict_score(modified[context.feature_columns]))
                            ),
                            "curve_type": "numeric",
                            "score_label": score_label,
                            "sort_order": index,
                        }
                    )
            else:
                categories = (
                    series.fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .head(context.config.diagnostics.top_n_categories)
                    .index.tolist()
                )
                for index, category in enumerate(categories):
                    modified = sampled.copy(deep=True)
                    modified[feature_name] = category
                    rows.append(
                        {
                            "feature_name": feature_name,
                            "feature_value": category,
                            "average_prediction": float(
                                np.mean(context.model.predict_score(modified[context.feature_columns]))
                            ),
                            "curve_type": "categorical",
                            "score_label": score_label,
                            "sort_order": index,
                        }
                    )
        return pd.DataFrame(rows)

    def _build_permutation_importance(
        self,
        context: PipelineContext,
        top_features: list[str],
    ) -> pd.DataFrame:
        scored_test = context.predictions.get("test")
        if scored_test is None or context.target_column not in scored_test.columns:
            return pd.DataFrame()

        sampled = self._sample_rows(
            scored_test[[*context.feature_columns, context.target_column]].copy(deep=True),
            context.config.explainability.sample_size,
            context,
        )
        x_values = sampled[context.feature_columns]
        y_true = sampled[context.target_column]
        baseline_scores = context.model.predict_score(x_values)
        if context.config.target.mode == TargetMode.BINARY:
            if y_true.nunique() < 2:
                return pd.DataFrame()
            metric_name = "roc_auc"
            baseline_metric = float(roc_auc_score(y_true.astype(int), baseline_scores))
            lower_is_better = False
        else:
            metric_name = "rmse"
            baseline_metric = float(
                math.sqrt(mean_squared_error(y_true.astype(float), baseline_scores))
            )
            lower_is_better = True

        rows = []
        for feature_name in top_features[: context.config.explainability.top_n_features]:
            permuted = x_values.copy(deep=True)
            permuted[feature_name] = permuted[feature_name].sample(
                frac=1.0,
                random_state=context.config.split.random_state,
            ).to_numpy()
            permuted_scores = context.model.predict_score(permuted)
            if context.config.target.mode == TargetMode.BINARY:
                permuted_metric = float(roc_auc_score(y_true.astype(int), permuted_scores))
            else:
                permuted_metric = float(
                    math.sqrt(mean_squared_error(y_true.astype(float), permuted_scores))
                )
            importance = (
                permuted_metric - baseline_metric
                if lower_is_better
                else baseline_metric - permuted_metric
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "metric_name": metric_name,
                    "baseline_metric": baseline_metric,
                    "permuted_metric": permuted_metric,
                    "importance": importance,
                }
            )

        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    def _add_scenario_outputs(self, context: PipelineContext) -> None:
        split_name = context.config.scenario_testing.evaluation_split
        scored_frame = context.predictions.get(split_name)
        if scored_frame is None:
            context.warn(
                f"Skipped scenario testing because the '{split_name}' split was unavailable."
            )
            return

        score_column = (
            "predicted_probability"
            if context.config.target.mode == TargetMode.BINARY
            else "predicted_value"
        )
        base_features = scored_frame[context.feature_columns].copy(deep=True)
        summary_rows: list[dict[str, Any]] = []
        definition_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        segment_column = context.config.diagnostics.default_segment_column or (
            context.categorical_features[0] if context.categorical_features else None
        )
        baseline_score = scored_frame[score_column].to_numpy(dtype=float)

        for scenario in context.config.scenario_testing.scenarios:
            if not scenario.enabled:
                continue
            scenario_features = base_features.copy(deep=True)
            for shock in scenario.feature_shocks:
                definition_rows.append(
                    {
                        "scenario_name": scenario.name,
                        "feature_name": shock.feature_name,
                        "operation": shock.operation.value,
                        "value": shock.value,
                    }
                )
                if shock.feature_name not in scenario_features.columns:
                    context.warn(
                        f"Scenario '{scenario.name}' references missing feature "
                        f"'{shock.feature_name}' and was skipped for that shock."
                    )
                    continue
                self._apply_scenario_shock(
                    scenario_features,
                    shock.feature_name,
                    shock.operation.value,
                    shock.value,
                )

            scenario_score = context.model.predict_score(
                scenario_features[context.feature_columns]
            )
            delta = np.asarray(scenario_score) - baseline_score
            summary = {
                "scenario_name": scenario.name,
                "split": split_name,
                "observation_count": int(len(delta)),
                "mean_baseline_score": float(np.mean(baseline_score)),
                "mean_scenario_score": float(np.mean(scenario_score)),
                "mean_delta": float(np.mean(delta)),
            }
            if context.config.target.mode == TargetMode.BINARY:
                threshold = context.config.model.threshold
                summary["baseline_positive_rate"] = float(
                    np.mean(baseline_score >= threshold)
                )
                summary["scenario_positive_rate"] = float(
                    np.mean(np.asarray(scenario_score) >= threshold)
                )
            summary_rows.append(summary)

            if segment_column and segment_column in scored_frame.columns:
                segment_frame = pd.DataFrame(
                    {
                        "segment_value": scored_frame[segment_column].fillna("Missing").astype(str),
                        "baseline_score": baseline_score,
                        "scenario_score": np.asarray(scenario_score),
                    }
                )
                grouped = (
                    segment_frame.groupby("segment_value", dropna=False)
                    .agg(
                        mean_baseline_score=("baseline_score", "mean"),
                        mean_scenario_score=("scenario_score", "mean"),
                    )
                    .reset_index()
                )
                grouped["mean_delta"] = (
                    grouped["mean_scenario_score"] - grouped["mean_baseline_score"]
                )
                grouped.insert(0, "scenario_name", scenario.name)
                segment_rows.append(grouped)

        if summary_rows:
            summary_table = pd.DataFrame(summary_rows)
            context.diagnostics_tables["scenario_summary"] = summary_table
            context.scenario_results["scenario_summary"] = summary_table
            context.visualizations["scenario_summary_chart"] = px.bar(
                summary_table,
                x="scenario_name",
                y="mean_delta",
                title="Scenario Impact on Average Predicted Score",
                labels={"scenario_name": "Scenario", "mean_delta": "Average Score Delta"},
            )
        if definition_rows:
            context.diagnostics_tables["scenario_definitions"] = pd.DataFrame(definition_rows)
        if segment_rows:
            segment_table = pd.concat(segment_rows, ignore_index=True)
            context.diagnostics_tables["scenario_segment_impacts"] = segment_table
            context.scenario_results["scenario_segment_impacts"] = segment_table
            context.visualizations["scenario_segment_impact"] = px.bar(
                segment_table,
                x="segment_value",
                y="mean_delta",
                color="scenario_name",
                barmode="group",
                title="Scenario Impact by Segment",
                labels={"segment_value": "Segment", "mean_delta": "Average Score Delta"},
            )

    def _apply_scenario_shock(
        self,
        dataframe: pd.DataFrame,
        feature_name: str,
        operation: str,
        value: Any,
    ) -> None:
        if operation == "set":
            dataframe[feature_name] = value
            return
        if not pd.api.types.is_numeric_dtype(dataframe[feature_name]):
            raise ValueError(
                f"Scenario operation '{operation}' requires numeric feature '{feature_name}'."
            )
        numeric_value = float(value)
        if operation == "add":
            dataframe[feature_name] = (
                pd.to_numeric(dataframe[feature_name], errors="coerce") + numeric_value
            )
        elif operation == "multiply":
            dataframe[feature_name] = (
                pd.to_numeric(dataframe[feature_name], errors="coerce") * numeric_value
            )
        else:
            raise ValueError(f"Unsupported scenario operation '{operation}'.")

    def _add_feature_policy_outputs(self, context: PipelineContext) -> None:
        policy = context.config.feature_policy
        rows: list[dict[str, Any]] = []
        feature_set = set(context.feature_columns)
        null_ratio = context.metadata.get("null_ratio_by_column", {})
        coefficient_map = self._build_feature_coefficient_map(context)
        effect_table = context.diagnostics_tables.get("feature_effect_curves", pd.DataFrame())
        vif_table = context.diagnostics_tables.get("vif", pd.DataFrame())
        iv_table = context.diagnostics_tables.get("woe_iv_summary", pd.DataFrame())

        for feature_name in policy.required_features:
            rows.append(
                self._policy_row(
                    "required_feature",
                    feature_name,
                    feature_name in feature_set,
                    feature_name if feature_name in feature_set else "missing",
                    "must_exist",
                )
            )
        for feature_name in policy.excluded_features:
            rows.append(
                self._policy_row(
                    "excluded_feature",
                    feature_name,
                    feature_name not in feature_set,
                    "present" if feature_name in feature_set else "not_present",
                    "must_not_exist",
                )
            )
        if policy.max_missing_pct is not None:
            for feature_name in context.feature_columns:
                observed_missing = float(null_ratio.get(feature_name, 0.0)) * 100
                rows.append(
                    self._policy_row(
                        "max_missing_pct",
                        feature_name,
                        observed_missing <= policy.max_missing_pct,
                        observed_missing,
                        policy.max_missing_pct,
                    )
                )
        if policy.max_vif is not None and not vif_table.empty:
            for _, row in vif_table.iterrows():
                rows.append(
                    self._policy_row(
                        "max_vif",
                        str(row["feature_name"]),
                        float(row["vif"]) <= policy.max_vif,
                        float(row["vif"]),
                        policy.max_vif,
                    )
                )
        if policy.minimum_information_value is not None and not iv_table.empty:
            for _, row in iv_table.iterrows():
                rows.append(
                    self._policy_row(
                        "minimum_information_value",
                        str(row["feature_name"]),
                        float(row["information_value"]) >= policy.minimum_information_value,
                        float(row["information_value"]),
                        policy.minimum_information_value,
                    )
                )
        for feature_name, expected_sign in policy.expected_signs.items():
            observed_sign = coefficient_map.get(feature_name)
            passes = self._check_expected_sign(observed_sign, expected_sign)
            rows.append(
                self._policy_row(
                    "expected_sign",
                    feature_name,
                    passes if observed_sign is not None else False,
                    observed_sign,
                    expected_sign,
                )
            )
        for feature_name, direction in policy.monotonic_features.items():
            monotonic_ok = self._check_monotonic_direction(effect_table, feature_name, direction)
            rows.append(
                self._policy_row(
                    "monotonicity",
                    feature_name,
                    monotonic_ok,
                    "monotonic" if monotonic_ok else "non_monotonic",
                    direction,
                )
            )

        if not rows:
            return

        policy_table = pd.DataFrame(rows)
        context.diagnostics_tables["feature_policy_checks"] = policy_table
        failures = policy_table.loc[policy_table["status"] == "fail"]
        if not failures.empty:
            context.warn(
                f"Feature policy recorded {len(failures)} failed checks across "
                f"{failures['feature_name'].nunique()} features."
            )
            if policy.error_on_violation:
                failure_preview = ", ".join(
                    failures.head(5).apply(
                        lambda row: f"{row['policy_name']}:{row['feature_name']}", axis=1
                    )
                )
                raise ValueError(f"Feature policy violations detected: {failure_preview}.")

    def _build_feature_coefficient_map(self, context: PipelineContext) -> dict[str, float]:
        if "coefficient" not in context.feature_importance.columns:
            return {}
        coefficient_table = context.feature_importance.copy(deep=True)
        coefficient_table["source_feature_name"] = coefficient_table["feature_name"].map(
            lambda value: self._infer_source_feature_name(str(value), context.feature_columns)
        )
        aggregated = coefficient_table.groupby("source_feature_name", dropna=False)[
            "coefficient"
        ].mean()
        return {str(key): float(value) for key, value in aggregated.items()}

    def _infer_source_feature_name(
        self,
        feature_name: str,
        feature_columns: list[str],
    ) -> str:
        if feature_name in feature_columns:
            return feature_name
        suffix = feature_name.split("__", 1)[-1]
        if suffix in feature_columns:
            return suffix
        for candidate in sorted(feature_columns, key=len, reverse=True):
            if suffix == candidate or suffix.startswith(f"{candidate}_"):
                return candidate
            if feature_name == candidate or feature_name.startswith(f"{candidate}_"):
                return candidate
        return suffix

    def _check_expected_sign(self, observed_value: float | None, expected_sign: str) -> bool:
        if observed_value is None:
            return False
        if expected_sign == "positive":
            return observed_value > 0
        if expected_sign == "negative":
            return observed_value < 0
        if expected_sign == "nonnegative":
            return observed_value >= 0
        if expected_sign == "nonpositive":
            return observed_value <= 0
        return False

    def _check_monotonic_direction(
        self,
        effect_table: pd.DataFrame,
        feature_name: str,
        direction: str,
    ) -> bool:
        if effect_table.empty:
            return False
        feature_curve = effect_table.loc[
            (effect_table["feature_name"] == feature_name)
            & (effect_table["curve_type"] == "numeric")
        ].sort_values("sort_order")
        if len(feature_curve) < 2:
            return False
        deltas = np.diff(feature_curve["average_prediction"].to_numpy(dtype=float))
        if direction == "increasing":
            return bool(np.all(deltas >= -1e-6))
        if direction == "decreasing":
            return bool(np.all(deltas <= 1e-6))
        return False

    def _policy_row(
        self,
        policy_name: str,
        feature_name: str,
        passed: bool,
        observed_value: Any,
        threshold: Any,
    ) -> dict[str, Any]:
        return {
            "policy_name": policy_name,
            "feature_name": feature_name,
            "status": "pass" if passed else "fail",
            "observed_value": observed_value,
            "threshold": threshold,
        }

    def _sample_rows(
        self,
        dataframe: pd.DataFrame,
        max_rows: int,
        context: PipelineContext,
    ) -> pd.DataFrame:
        if len(dataframe) <= max_rows:
            return dataframe
        return dataframe.sample(max_rows, random_state=context.config.split.random_state)

    def _sanitize_asset_name(self, name: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in name).strip("_")

    def _select_top_features(self, context: PipelineContext) -> list[str]:
        ranked = context.feature_importance.copy(deep=True)
        if "feature_name" in ranked.columns:
            translated = [
                str(feature_name).split("__", 1)[-1] for feature_name in ranked["feature_name"]
            ]
            ranked["source_feature_name"] = translated
            preferred = [
                feature_name
                for feature_name in ranked["source_feature_name"]
                if feature_name in context.feature_columns
            ]
            deduplicated_preferred = list(dict.fromkeys(preferred))
            if deduplicated_preferred:
                return deduplicated_preferred[: context.config.diagnostics.top_n_features]
        return context.feature_columns[: context.config.diagnostics.top_n_features]

    def _apply_visual_theme(self, context: PipelineContext) -> None:
        themed: dict[str, go.Figure] = {}
        for figure_name, figure in context.visualizations.items():
            themed[figure_name] = apply_fintech_figure_theme(
                figure,
                title=friendly_asset_title(figure_name, kind="figure"),
            )
        context.visualizations = themed

    def _labels_available(self, context: PipelineContext) -> bool:
        return bool(context.metadata.get("labels_available", False))

    def _sample_frame(self, dataframe: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
        max_rows = context.config.diagnostics.max_plot_rows
        if len(dataframe) <= max_rows:
            return dataframe
        return dataframe.sample(max_rows, random_state=context.config.split.random_state)

    def _bucket_numeric_series(self, series: pd.Series, bucket_count: int) -> pd.Series:
        ranked = series.rank(method="first")
        return pd.qcut(ranked, q=min(bucket_count, ranked.nunique()), duplicates="drop")

    def _compute_population_stability_index(
        self,
        expected: pd.Series,
        actual: pd.Series,
    ) -> float:
        expected_series = pd.Series(expected).dropna()
        actual_series = pd.Series(actual).dropna()
        if expected_series.empty or actual_series.empty:
            return float("nan")

        if pd.api.types.is_numeric_dtype(expected_series):
            bucket_edges = np.unique(
                np.quantile(
                    expected_series,
                    np.linspace(0, 1, min(11, max(3, expected_series.nunique()))),
                )
            )
            if len(bucket_edges) < 2:
                return 0.0
            bucket_edges = bucket_edges.astype(float)
            bucket_edges[0] = -np.inf
            bucket_edges[-1] = np.inf
            if len(np.unique(bucket_edges)) < 2:
                return 0.0
            expected_bucket = pd.cut(
                expected_series,
                bins=bucket_edges,
                include_lowest=True,
                duplicates="drop",
            )
            actual_bucket = pd.cut(
                actual_series,
                bins=bucket_edges,
                include_lowest=True,
                duplicates="drop",
            )
            expected_dist = expected_bucket.value_counts(normalize=True, sort=False)
            actual_dist = actual_bucket.value_counts(normalize=True, sort=False)
        else:
            expected_dist = expected_series.astype(str).value_counts(normalize=True)
            actual_dist = actual_series.astype(str).value_counts(normalize=True)

        all_buckets = expected_dist.index.union(actual_dist.index)
        psi_value = 0.0
        for bucket in all_buckets:
            expected_pct = max(float(expected_dist.get(bucket, 0.0)), 1e-6)
            actual_pct = max(float(actual_dist.get(bucket, 0.0)), 1e-6)
            psi_value += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
        return float(psi_value)

    def _run_adf_test(
        self,
        series: pd.Series,
        *,
        split_name: str,
        series_name: str,
    ) -> dict[str, Any] | None:
        if len(series) < 10:
            return None
        try:
            statistic, p_value, used_lag, nobs, _, _ = adfuller(series.to_numpy())
        except Exception:
            return None
        return {
            "split": split_name,
            "series_name": series_name,
            "adf_statistic": float(statistic),
            "p_value": float(p_value),
            "used_lag": int(used_lag),
            "observations": int(nobs),
        }
