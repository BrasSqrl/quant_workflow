"""Feature-subset comparison workflow for development-time candidate selection."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from ..base import BasePipelineStep
from ..config import ExecutionMode
from ..context import PipelineContext
from ..diagnostic_frameworks import _run_delong_test, _run_mcnemar_test
from ..models import build_model_adapter
from ..presentation import apply_fintech_figure_theme
from .evaluation import EvaluationStep


class FeatureSubsetSearchStep(BasePipelineStep):
    """Evaluates candidate feature subsets for a fixed model family."""

    name = "feature_subset_search"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.config.execution.mode != ExecutionMode.SEARCH_FEATURE_SUBSETS:
            return context
        if context.target_column is None:
            raise ValueError("Feature subset search requires a resolved target column.")
        if not bool(context.metadata.get("labels_available", False)):
            raise ValueError(
                "Feature subset search requires realized labels so candidate subsets can be ranked."
            )

        train_frame = context.split_frames.get("train")
        ranking_frame = context.split_frames.get(context.config.subset_search.ranking_split)
        test_frame = context.split_frames.get("test")
        if train_frame is None or ranking_frame is None:
            raise ValueError(
                "Feature subset search requires train data and the configured ranking split."
            )

        context.diagnostics_tables = {}
        context.visualizations = {}
        context.statistical_tests = {}
        context.predictions = {}
        context.feature_importance = None
        context.backtest_summary = None
        context.comparison_results = None

        candidate_features = self._resolve_candidate_features(context)
        subset_sets = self._enumerate_feature_sets(candidate_features, context)
        evaluator = EvaluationStep()

        candidate_rows: list[dict[str, Any]] = []
        ranking_predictions: dict[str, pd.DataFrame] = {}
        ranking_metadata: dict[str, dict[str, Any]] = {}
        candidate_feature_importance: dict[str, pd.DataFrame] = {}

        for candidate_index, subset in enumerate(subset_sets, start=1):
            candidate_id = f"candidate_{candidate_index:03d}"
            try:
                adapter = build_model_adapter(
                    context.config.model,
                    context.config.target.mode,
                    scorecard_config=context.config.scorecard,
                    scorecard_bin_overrides={
                        override.feature_name: override.bin_edges
                        for override in context.config.manual_review.scorecard_bin_overrides
                    },
                )
                adapter.fit(
                    train_frame[list(subset)],
                    train_frame[context.target_column],
                    [feature for feature in context.numeric_features if feature in subset],
                    [feature for feature in context.categorical_features if feature in subset],
                )

                ranking_scored, ranking_metrics = evaluator._score_binary_split(
                    ranking_frame,
                    context.config.subset_search.ranking_split,
                    context.target_column,
                    list(subset),
                    adapter,
                    context.config.model.threshold,
                    True,
                    context,
                )
                test_scored: pd.DataFrame | None = None
                test_metrics: dict[str, float | int | None] = {}
                if test_frame is not None:
                    test_scored, test_metrics = evaluator._score_binary_split(
                        test_frame,
                        "test",
                        context.target_column,
                        list(subset),
                        adapter,
                        context.config.model.threshold,
                        True,
                        context,
                    )

                ranking_predictions[candidate_id] = ranking_scored
                candidate_feature_importance[candidate_id] = adapter.get_feature_importance()
                ranking_metadata[candidate_id] = {
                    "feature_set": list(subset),
                    "feature_count": len(subset),
                    "ranking_metrics": ranking_metrics,
                    "test_metrics": test_metrics,
                }
                candidate_rows.append(
                    self._build_candidate_row(
                        candidate_id=candidate_id,
                        feature_set=list(subset),
                        ranking_split=context.config.subset_search.ranking_split,
                        ranking_metric=context.config.subset_search.ranking_metric,
                        ranking_metrics=ranking_metrics,
                        test_metrics=test_metrics,
                    )
                )
            except Exception as exc:
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "failure_reason": str(exc),
                        "feature_count": len(subset),
                        "feature_set": ", ".join(subset),
                        "ranking_split": context.config.subset_search.ranking_split,
                        "ranking_metric": context.config.subset_search.ranking_metric,
                    }
                )

        candidate_table = pd.DataFrame(candidate_rows)
        successful = candidate_table.loc[candidate_table["status"] == "success"].copy(deep=True)
        if successful.empty:
            raise ValueError(
                "Feature subset search did not produce any successful candidate evaluations."
            )

        ranking_metric = context.config.subset_search.ranking_metric
        lower_is_better = ranking_metric in {"brier_score", "log_loss"}
        successful = successful.sort_values(
            ["ranking_value", "feature_count"],
            ascending=[lower_is_better, True],
            kind="mergesort",
        ).reset_index(drop=True)
        successful["rank"] = np.arange(1, len(successful) + 1)
        candidate_table = candidate_table.merge(
            successful[["candidate_id", "rank"]],
            on="candidate_id",
            how="left",
        )
        candidate_table["rank"] = candidate_table["rank"].astype("Int64")

        top_candidates = successful.head(context.config.subset_search.top_candidate_count)
        frontier = self._build_frontier_table(
            successful=successful,
            ranking_metric=ranking_metric,
            lower_is_better=lower_is_better,
        )
        feature_frequency = self._build_feature_frequency_table(
            top_candidates=top_candidates,
            ranking_metadata=ranking_metadata,
        )
        significance = self._build_significance_table(
            context=context,
            top_candidates=top_candidates,
            ranking_predictions=ranking_predictions,
        )

        scope_table = pd.DataFrame(
            [
                {
                    "model_type": context.config.model.model_type.value,
                    "ranking_split": context.config.subset_search.ranking_split,
                    "ranking_metric": ranking_metric,
                    "candidate_feature_count": len(candidate_features),
                    "enumerated_subset_count": len(subset_sets),
                    "successful_subset_count": int(len(successful)),
                    "failed_subset_count": int((candidate_table["status"] == "failed").sum()),
                    "min_subset_size": context.config.subset_search.min_subset_size,
                    "max_subset_size": context.config.subset_search.max_subset_size
                    if context.config.subset_search.max_subset_size is not None
                    else len(candidate_features),
                }
            ]
        )

        context.diagnostics_tables["subset_search_scope"] = scope_table
        context.diagnostics_tables["subset_search_candidates"] = candidate_table
        context.diagnostics_tables["subset_search_frontier"] = frontier
        context.diagnostics_tables["subset_search_feature_frequency"] = feature_frequency
        if not significance.empty:
            context.diagnostics_tables["subset_search_significance_tests"] = significance
            context.statistical_tests["subset_search_significance"] = significance.to_dict(
                orient="records"
            )

        best_candidate = top_candidates.iloc[0]
        selected_candidate_id = str(best_candidate["candidate_id"])
        selected_coefficients = self._build_selected_coefficient_table(
            feature_importance=candidate_feature_importance.get(
                selected_candidate_id,
                pd.DataFrame(),
            )
        )
        nonwinning_candidates = successful.loc[
            successful["candidate_id"] != selected_candidate_id
        ].reset_index(drop=True)

        context.metrics["subset_search"] = {
            "candidate_feature_count": int(len(candidate_features)),
            "enumerated_subsets": int(len(subset_sets)),
            "successful_subsets": int(len(successful)),
            "failed_subsets": int((candidate_table["status"] == "failed").sum()),
            "best_candidate_id": str(best_candidate["candidate_id"]),
            "best_feature_count": int(best_candidate["feature_count"]),
            "best_validation_roc_auc": float(best_candidate["ranking_roc_auc"]),
            "best_validation_ks_statistic": float(best_candidate["ranking_ks_statistic"]),
            "best_test_roc_auc": self._optional_float(best_candidate.get("test_roc_auc")),
            "best_test_ks_statistic": self._optional_float(
                best_candidate.get("test_ks_statistic")
            ),
        }
        context.metadata["subset_search_best_candidate"] = best_candidate.to_dict()
        context.metadata["subset_search_ranking_metric"] = ranking_metric
        context.metadata["subset_search_ranking_split"] = (
            context.config.subset_search.ranking_split
        )
        context.metadata["subset_search_top_candidates"] = top_candidates.to_dict(orient="records")
        context.diagnostics_tables["subset_search_selected_candidate"] = (
            self._build_selected_candidate_table(best_candidate)
        )
        context.diagnostics_tables["subset_search_selected_coefficients"] = selected_coefficients
        context.diagnostics_tables["subset_search_nonwinning_candidates"] = (
            self._build_nonwinning_candidate_table(nonwinning_candidates)
        )
        if int((candidate_table["status"] == "failed").sum()) > 0:
            context.warn(
                "Feature subset search skipped one or more candidate subsets because they "
                "failed to fit or score. Review `subset_search_candidates` for details."
            )

        self._build_figures(
            context=context,
            top_candidates=top_candidates,
            selected_candidate_id=selected_candidate_id,
            feature_frequency=feature_frequency,
            ranking_predictions=ranking_predictions,
            ranking_metadata=ranking_metadata,
        )
        return context

    def _resolve_candidate_features(self, context: PipelineContext) -> list[str]:
        config = context.config.subset_search
        available_features = list(context.feature_columns)
        requested = config.candidate_feature_names or available_features
        requested = [feature for feature in requested if feature in available_features]
        if not requested:
            raise ValueError("Feature subset search requires at least one eligible feature.")

        locked_include = [
            feature
            for feature in config.locked_include_features
            if feature in available_features
        ]
        requested = [
            feature
            for feature in requested
            if feature not in config.locked_exclude_features
        ]
        for feature_name in locked_include:
            if feature_name not in requested:
                requested.append(feature_name)
        if len(requested) > config.max_candidate_features:
            raise ValueError(
                "Feature subset search candidate count exceeds the configured cap. "
                f"Resolved {len(requested)} features with a cap of {config.max_candidate_features}."
            )
        return requested

    def _enumerate_feature_sets(
        self,
        candidate_features: list[str],
        context: PipelineContext,
    ) -> list[tuple[str, ...]]:
        config = context.config.subset_search
        locked_include = [
            feature
            for feature in config.locked_include_features
            if feature in candidate_features
        ]
        free_features = [
            feature for feature in candidate_features if feature not in locked_include
        ]
        min_subset_size = config.min_subset_size
        max_subset_size = config.max_subset_size or len(candidate_features)
        if len(locked_include) > max_subset_size:
            raise ValueError(
                "Locked include features exceed the maximum subset size for feature subset search."
            )

        additional_min = max(0, min_subset_size - len(locked_include))
        additional_max = max_subset_size - len(locked_include)
        feature_sets: list[tuple[str, ...]] = []
        for subset_size in range(additional_min, additional_max + 1):
            for free_subset in combinations(free_features, subset_size):
                feature_sets.append(tuple([*locked_include, *free_subset]))
        if not feature_sets:
            raise ValueError("Feature subset search could not build any candidate subsets.")
        return feature_sets

    def _build_candidate_row(
        self,
        *,
        candidate_id: str,
        feature_set: list[str],
        ranking_split: str,
        ranking_metric: str,
        ranking_metrics: dict[str, float | int | None],
        test_metrics: dict[str, float | int | None],
    ) -> dict[str, Any]:
        ranking_value = ranking_metrics.get(ranking_metric)
        return {
            "candidate_id": candidate_id,
            "status": "success",
            "failure_reason": "",
            "feature_count": len(feature_set),
            "feature_set": ", ".join(feature_set),
            "ranking_split": ranking_split,
            "ranking_metric": ranking_metric,
            "ranking_value": self._optional_float(ranking_value),
            "ranking_roc_auc": self._optional_float(ranking_metrics.get("roc_auc")),
            "ranking_ks_statistic": self._optional_float(ranking_metrics.get("ks_statistic")),
            "ranking_average_precision": self._optional_float(
                ranking_metrics.get("average_precision")
            ),
            "ranking_brier_score": self._optional_float(ranking_metrics.get("brier_score")),
            "ranking_log_loss": self._optional_float(ranking_metrics.get("log_loss")),
            "test_roc_auc": self._optional_float(test_metrics.get("roc_auc")),
            "test_ks_statistic": self._optional_float(test_metrics.get("ks_statistic")),
            "test_average_precision": self._optional_float(
                test_metrics.get("average_precision")
            ),
            "test_brier_score": self._optional_float(test_metrics.get("brier_score")),
            "test_log_loss": self._optional_float(test_metrics.get("log_loss")),
        }

    def _build_frontier_table(
        self,
        *,
        successful: pd.DataFrame,
        ranking_metric: str,
        lower_is_better: bool,
    ) -> pd.DataFrame:
        frontier_rows: list[pd.Series] = []
        for _feature_count, count_frame in successful.groupby("feature_count", dropna=False):
            best_row = count_frame.sort_values(
                "ranking_value",
                ascending=lower_is_better,
                kind="mergesort",
            ).iloc[0]
            frontier_rows.append(best_row)
        if not frontier_rows:
            return pd.DataFrame()
        frontier = pd.DataFrame(frontier_rows).reset_index(drop=True)
        frontier["frontier_metric"] = ranking_metric
        return frontier

    def _build_feature_frequency_table(
        self,
        *,
        top_candidates: pd.DataFrame,
        ranking_metadata: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        if top_candidates.empty:
            return pd.DataFrame()
        top_candidate_ids = top_candidates["candidate_id"].astype(str).tolist()
        denominator = max(len(top_candidate_ids), 1)
        feature_counts: dict[str, int] = {}
        for candidate_id in top_candidate_ids:
            for feature_name in ranking_metadata.get(candidate_id, {}).get("feature_set", []):
                feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
        return (
            pd.DataFrame(
                [
                    {
                        "feature_name": feature_name,
                        "selected_count": int(count),
                        "selected_share": float(count / denominator),
                    }
                    for feature_name, count in feature_counts.items()
                ]
            )
            .sort_values(["selected_count", "feature_name"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _build_significance_table(
        self,
        *,
        context: PipelineContext,
        top_candidates: pd.DataFrame,
        ranking_predictions: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if (
            not context.config.subset_search.include_significance_tests
            or len(top_candidates) < 2
            or context.target_column is None
        ):
            return pd.DataFrame()
        baseline_id = str(top_candidates.iloc[0]["candidate_id"])
        baseline_frame = ranking_predictions.get(baseline_id)
        if baseline_frame is None:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for _, challenger_row in top_candidates.iloc[1:].iterrows():
            challenger_id = str(challenger_row["candidate_id"])
            challenger_frame = ranking_predictions.get(challenger_id)
            if challenger_frame is None:
                continue
            aligned = baseline_frame[
                [context.target_column, "predicted_probability", "predicted_class"]
            ].rename(
                columns={
                    "predicted_probability": "predicted_probability_primary",
                    "predicted_class": "predicted_class_primary",
                }
            ).join(
                challenger_frame[["predicted_probability", "predicted_class"]].rename(
                    columns={
                        "predicted_probability": "predicted_probability_challenger",
                        "predicted_class": "predicted_class_challenger",
                    }
                ),
                how="inner",
            ).dropna()
            if len(aligned) < 20:
                continue
            delong_result = _run_delong_test(
                y_true=aligned[context.target_column].astype(int).to_numpy(),
                baseline_scores=aligned["predicted_probability_primary"].to_numpy(dtype=float),
                challenger_scores=aligned["predicted_probability_challenger"].to_numpy(
                    dtype=float
                ),
            )
            if delong_result is not None:
                rows.append(
                    {
                        "baseline_candidate_id": baseline_id,
                        "challenger_candidate_id": challenger_id,
                        "test_name": "delong_auc_difference",
                        **delong_result,
                    }
                )
            rows.append(
                {
                    "baseline_candidate_id": baseline_id,
                    "challenger_candidate_id": challenger_id,
                    "test_name": "mcnemar_threshold_difference",
                    **_run_mcnemar_test(
                        y_true=aligned[context.target_column].astype(int).to_numpy(),
                        baseline_class=aligned["predicted_class_primary"].astype(int).to_numpy(),
                        challenger_class=aligned["predicted_class_challenger"]
                        .astype(int)
                        .to_numpy(),
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_figures(
        self,
        *,
        context: PipelineContext,
        top_candidates: pd.DataFrame,
        selected_candidate_id: str,
        feature_frequency: pd.DataFrame,
        ranking_predictions: dict[str, pd.DataFrame],
        ranking_metadata: dict[str, dict[str, Any]],
    ) -> None:
        if top_candidates.empty:
            return
        auc_scatter = px.scatter(
            top_candidates,
            x="feature_count",
            y="ranking_roc_auc",
            hover_name="candidate_id",
            hover_data={"feature_set": True},
            title="Subset Search: ROC AUC vs Feature Count",
            labels={"feature_count": "Feature Count", "ranking_roc_auc": "ROC AUC"},
        )
        context.visualizations["subset_search_auc_frontier"] = apply_fintech_figure_theme(
            auc_scatter,
            title="Subset Search: ROC AUC vs Feature Count",
        )

        ks_scatter = px.scatter(
            top_candidates,
            x="feature_count",
            y="ranking_ks_statistic",
            hover_name="candidate_id",
            hover_data={"feature_set": True},
            title="Subset Search: KS vs Feature Count",
            labels={"feature_count": "Feature Count", "ranking_ks_statistic": "KS Statistic"},
        )
        context.visualizations["subset_search_ks_frontier"] = apply_fintech_figure_theme(
            ks_scatter,
            title="Subset Search: KS vs Feature Count",
        )

        frontier = px.scatter(
            top_candidates,
            x="feature_count",
            y="ranking_value",
            color="ranking_ks_statistic",
            hover_name="candidate_id",
            hover_data={"feature_set": True},
            title="Subset Search: Performance vs Parsimony",
            labels={"feature_count": "Feature Count", "ranking_value": "Ranking Metric"},
        )
        context.visualizations["subset_search_metric_frontier"] = apply_fintech_figure_theme(
            frontier,
            title="Subset Search: Performance vs Parsimony",
        )

        if not feature_frequency.empty:
            frequency_chart = px.bar(
                feature_frequency,
                x="feature_name",
                y="selected_share",
                title="Subset Search: Feature Inclusion Frequency",
                labels={"feature_name": "Feature", "selected_share": "Selection Share"},
            )
            context.visualizations["subset_search_feature_frequency_chart"] = (
                apply_fintech_figure_theme(
                    frequency_chart,
                    title="Subset Search: Feature Inclusion Frequency",
                )
            )

        prediction_frame = ranking_predictions.get(selected_candidate_id)
        candidate_meta = ranking_metadata.get(selected_candidate_id, {})
        if prediction_frame is None or context.target_column is None:
            return
        y_true = prediction_frame[context.target_column].astype(int).to_numpy()
        scores = prediction_frame["predicted_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_chart = px.line(
            pd.DataFrame({"fpr": fpr, "tpr": tpr}),
            x="fpr",
            y="tpr",
            title=(
                "Selected Candidate ROC Curve "
                f"({selected_candidate_id}, {int(candidate_meta.get('feature_count', 0))} features)"
            ),
            labels={"fpr": "False Positive Rate", "tpr": "True Positive Rate"},
        )
        roc_chart.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Random",
                line={"dash": "dash", "color": "#607089"},
                showlegend=True,
            )
        )
        context.visualizations["subset_search_selected_roc_curve"] = apply_fintech_figure_theme(
            roc_chart,
            title="Selected Candidate ROC Curve",
        )

        ks_rows = self._build_ks_curve_rows(selected_candidate_id, y_true, scores)
        if ks_rows:
            ks_chart = px.line(
                pd.DataFrame(ks_rows),
                x="sample_share",
                y="ks_gap",
                title="Selected Candidate KS Curve",
                labels={"sample_share": "Population Share", "ks_gap": "KS Gap"},
            )
            context.visualizations["subset_search_selected_ks_curve"] = (
                apply_fintech_figure_theme(
                    ks_chart,
                    title="Selected Candidate KS Curve",
                )
            )

    def _build_ks_curve_rows(
        self,
        candidate_id: str,
        y_true: np.ndarray,
        scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        order = np.argsort(-scores)
        ordered_target = y_true[order]
        positives = max(int(ordered_target.sum()), 1)
        negatives = max(int(len(ordered_target) - ordered_target.sum()), 1)
        positive_cdf = np.cumsum(ordered_target) / positives
        negative_cdf = np.cumsum(1 - ordered_target) / negatives
        sample_share = np.arange(1, len(ordered_target) + 1, dtype=float) / len(ordered_target)
        return [
            {
                "candidate_id": candidate_id,
                "sample_share": float(sample_share[index]),
                "ks_gap": float(positive_cdf[index] - negative_cdf[index]),
            }
            for index in range(len(sample_share))
        ]

    def _optional_float(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _build_selected_candidate_table(self, candidate_row: pd.Series) -> pd.DataFrame:
        column_order = [
            "candidate_id",
            "feature_set",
            "feature_count",
            "ranking_roc_auc",
            "ranking_ks_statistic",
            "ranking_average_precision",
            "ranking_brier_score",
            "ranking_log_loss",
            "test_roc_auc",
            "test_ks_statistic",
            "test_average_precision",
            "test_brier_score",
            "test_log_loss",
        ]
        selected = pd.DataFrame([candidate_row.to_dict()])
        resolved_columns = [
            column_name for column_name in column_order if column_name in selected.columns
        ]
        return selected.loc[:, resolved_columns].reset_index(drop=True)

    def _build_nonwinning_candidate_table(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame()
        column_order = [
            "rank",
            "candidate_id",
            "feature_set",
            "feature_count",
            "ranking_roc_auc",
            "ranking_ks_statistic",
            "ranking_average_precision",
            "ranking_brier_score",
            "ranking_log_loss",
            "test_roc_auc",
            "test_ks_statistic",
            "test_average_precision",
            "test_brier_score",
            "test_log_loss",
        ]
        resolved_columns = [
            column_name for column_name in column_order if column_name in candidates.columns
        ]
        return (
            candidates.loc[:, resolved_columns]
            .sort_values(["rank", "candidate_id"], ascending=[True, True], kind="mergesort")
            .reset_index(drop=True)
        )

    def _build_selected_coefficient_table(self, feature_importance: pd.DataFrame) -> pd.DataFrame:
        if feature_importance.empty:
            return pd.DataFrame()
        coefficient_like = feature_importance.copy(deep=True)
        sort_column = next(
            (
                column_name
                for column_name in ["abs_coefficient", "importance_value", "coefficient"]
                if column_name in coefficient_like.columns
            ),
            None,
        )
        selected_columns = [
            column_name
            for column_name in [
                "feature_name",
                "coefficient",
                "abs_coefficient",
                "odds_ratio",
                "importance_value",
                "importance_type",
                "std_error",
                "p_value",
            ]
            if column_name in coefficient_like.columns
        ]
        selected = coefficient_like.loc[:, selected_columns].reset_index(drop=True)
        if sort_column is None or sort_column not in selected.columns:
            return selected
        return selected.sort_values(sort_column, ascending=False, kind="mergesort").reset_index(
            drop=True
        )
