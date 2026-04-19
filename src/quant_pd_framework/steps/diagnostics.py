"""Builds statistical tests, tables, and interactive visualizations for a run."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from ..base import BasePipelineStep
from ..config import DataStructure, TargetMode
from ..context import PipelineContext
from ..presentation import apply_fintech_figure_theme, friendly_asset_title


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
        if context.config.explainability.enabled:
            self._add_explainability_outputs(context, top_features, labels_available)
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
        scored_test = context.predictions["test"]
        y_true = scored_test[context.target_column].astype(int).to_numpy()
        probability = scored_test["predicted_probability"].to_numpy()

        calibration_true, calibration_pred = calibration_curve(
            y_true,
            probability,
            n_bins=min(10, max(2, len(scored_test) // 20)),
            strategy="quantile",
        )
        calibration_table = pd.DataFrame(
            {
                "mean_predicted_probability": calibration_pred,
                "observed_default_rate": calibration_true,
            }
        )
        context.diagnostics_tables["calibration"] = calibration_table
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=calibration_table["mean_predicted_probability"],
                y=calibration_table["observed_default_rate"],
                mode="lines+markers",
                name="Model",
            )
        )
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

    def _add_explainability_outputs(
        self,
        context: PipelineContext,
        top_features: list[str],
        labels_available: bool,
    ) -> None:
        for artifact_name, artifact in context.model_artifacts.items():
            if isinstance(artifact, pd.DataFrame):
                context.diagnostics_tables[artifact_name] = artifact.copy(deep=True)

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
