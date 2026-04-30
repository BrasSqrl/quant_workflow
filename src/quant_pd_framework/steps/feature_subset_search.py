"""Feature-subset comparison workflow for development-time candidate selection."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from ..base import BasePipelineStep
from ..config import ExecutionMode, ModelType, TargetMode, feature_subset_search_lower_is_better
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
                model_feature_columns = self._model_input_columns(
                    list(subset),
                    train_frame,
                    context,
                )
                adapter.fit(
                    train_frame[model_feature_columns],
                    train_frame[context.target_column],
                    [
                        feature
                        for feature in context.numeric_features
                        if feature in model_feature_columns
                    ],
                    [
                        feature
                        for feature in context.categorical_features
                        if feature in model_feature_columns
                    ],
                )

                ranking_scored, ranking_metrics = self._score_candidate_split(
                    evaluator=evaluator,
                    frame=ranking_frame,
                    split_name=context.config.subset_search.ranking_split,
                    feature_columns=model_feature_columns,
                    adapter=adapter,
                    context=context,
                )
                ranking_calibration_error = self._calibration_error(
                    ranking_scored,
                    context.target_column,
                )
                test_scored: pd.DataFrame | None = None
                test_metrics: dict[str, float | int | None] = {}
                test_calibration_error: float | None = None
                if test_frame is not None:
                    test_scored, test_metrics = self._score_candidate_split(
                        evaluator=evaluator,
                        frame=test_frame,
                        split_name="test",
                        feature_columns=model_feature_columns,
                        adapter=adapter,
                        context=context,
                    )
                    test_calibration_error = self._calibration_error(
                        test_scored,
                        context.target_column,
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
                        ranking_calibration_error=ranking_calibration_error,
                        test_calibration_error=test_calibration_error,
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
        successful = successful.loc[
            pd.to_numeric(successful["ranking_value"], errors="coerce").notna()
        ]
        if successful.empty:
            raise ValueError(
                "Feature subset search did not produce any successful candidate evaluations."
            )

        ranking_metric = context.config.subset_search.ranking_metric
        lower_is_better = feature_subset_search_lower_is_better(ranking_metric)
        successful = successful.sort_values(
            ["ranking_value", "feature_count"],
            ascending=[lower_is_better, True],
            kind="mergesort",
        ).reset_index(drop=True)
        successful["rank"] = np.arange(1, len(successful) + 1)
        successful = self._add_candidate_selection_scores(
            successful,
            ranking_metric=ranking_metric,
        )
        selection_score_columns = [
            column_name
            for column_name in [
                "candidate_id",
                "rank",
                "overall_selection_score",
                "ranking_metric_score",
                "simplicity_score",
                "calibration_score",
            ]
            if column_name in successful.columns
        ]
        candidate_table = candidate_table.merge(
            successful[selection_score_columns],
            on="candidate_id",
            how="left",
        )
        candidate_table["rank"] = candidate_table["rank"].astype("Int64")

        top_candidates = successful.head(context.config.subset_search.top_candidate_count)
        leaderboard = self._build_leaderboard_table(successful)
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
        contribution_consistency = self._build_contribution_consistency_table(
            top_candidates=top_candidates,
            candidate_feature_importance=candidate_feature_importance,
            ranking_metadata=ranking_metadata,
        )
        redundancy = self._build_feature_redundancy_table(
            train_frame=train_frame,
            candidate_features=candidate_features,
            top_candidates=top_candidates,
            ranking_metadata=ranking_metadata,
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
        context.diagnostics_tables["subset_search_leaderboard"] = leaderboard
        context.diagnostics_tables["subset_search_frontier"] = frontier
        context.diagnostics_tables["subset_search_feature_frequency"] = feature_frequency
        if not contribution_consistency.empty:
            context.diagnostics_tables["subset_search_contribution_consistency"] = (
                contribution_consistency
            )
        if not redundancy.empty:
            context.diagnostics_tables["subset_search_redundancy_diagnostics"] = redundancy
        if not significance.empty:
            context.diagnostics_tables["subset_search_significance_tests"] = significance
            context.statistical_tests["subset_search_significance"] = significance.to_dict(
                orient="records"
            )

        best_candidate = top_candidates.iloc[0]
        selected_candidate_id = str(best_candidate["candidate_id"])
        selected_feature_set = ranking_metadata.get(selected_candidate_id, {}).get(
            "feature_set",
            [],
        )
        selected_coefficients = self._build_selected_coefficient_table(
            feature_importance=candidate_feature_importance.get(
                selected_candidate_id,
                pd.DataFrame(),
            )
        )
        nonwinning_candidates = successful.loc[
            successful["candidate_id"] != selected_candidate_id
        ].reset_index(drop=True)
        risk_flags = self._build_candidate_risk_flags_table(
            successful=successful,
            selected_candidate=best_candidate,
            ranking_metadata=ranking_metadata,
            redundancy=redundancy,
            contribution_consistency=contribution_consistency,
        )
        top_candidate_comparison = self._build_top_candidate_comparison_table(
            top_candidates=top_candidates,
            risk_flags=risk_flags,
        )
        selection_rationale = self._build_selection_rationale_table(
            selected_candidate=best_candidate,
            top_candidates=top_candidates,
            risk_flags=risk_flags,
        )
        excluded_features = self._build_excluded_feature_insight_table(
            candidate_features=candidate_features,
            selected_feature_set=selected_feature_set,
            successful=successful,
            ranking_metadata=ranking_metadata,
            feature_frequency=feature_frequency,
        )
        feature_family = self._build_feature_family_table(
            candidate_features=candidate_features,
            selected_feature_set=selected_feature_set,
            feature_frequency=feature_frequency,
        )
        transformation_effectiveness = self._build_transformation_effectiveness_table(
            successful=successful,
            ranking_metadata=ranking_metadata,
        )
        segment_performance = self._build_segment_performance_table(
            context=context,
            top_candidates=top_candidates,
            ranking_predictions=ranking_predictions,
        )
        time_performance = self._build_time_performance_table(
            context=context,
            top_candidates=top_candidates,
            ranking_predictions=ranking_predictions,
        )

        context.metrics["subset_search"] = {
            "candidate_feature_count": int(len(candidate_features)),
            "enumerated_subsets": int(len(subset_sets)),
            "successful_subsets": int(len(successful)),
            "failed_subsets": int((candidate_table["status"] == "failed").sum()),
            "best_candidate_id": str(best_candidate["candidate_id"]),
            "best_feature_count": int(best_candidate["feature_count"]),
            f"best_{context.config.subset_search.ranking_split}_{ranking_metric}": (
                self._optional_float(best_candidate.get("ranking_value"))
            ),
            f"best_test_{ranking_metric}": self._optional_float(
                best_candidate.get(f"test_{ranking_metric}")
            ),
            "best_overall_selection_score": self._optional_float(
                best_candidate.get("overall_selection_score")
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
        context.diagnostics_tables["subset_search_top_candidate_comparison"] = (
            top_candidate_comparison
        )
        context.diagnostics_tables["subset_search_selection_rationale"] = (
            selection_rationale
        )
        if not risk_flags.empty:
            context.diagnostics_tables["subset_search_candidate_risk_flags"] = risk_flags
        if not excluded_features.empty:
            context.diagnostics_tables["subset_search_excluded_feature_insights"] = (
                excluded_features
            )
        if not feature_family.empty:
            context.diagnostics_tables["subset_search_feature_family_view"] = feature_family
        if not transformation_effectiveness.empty:
            context.diagnostics_tables["subset_search_transformation_effectiveness"] = (
                transformation_effectiveness
            )
        if not segment_performance.empty:
            context.diagnostics_tables["subset_search_segment_performance"] = (
                segment_performance
            )
        if not time_performance.empty:
            context.diagnostics_tables["subset_search_time_performance"] = time_performance
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
            leaderboard=leaderboard,
            contribution_consistency=contribution_consistency,
            redundancy=redundancy,
            risk_flags=risk_flags,
            segment_performance=segment_performance,
            time_performance=time_performance,
            transformation_effectiveness=transformation_effectiveness,
            feature_family=feature_family,
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

    def _model_input_columns(
        self,
        feature_set: list[str],
        frame: pd.DataFrame,
        context: PipelineContext,
    ) -> list[str]:
        """Adds required grouping columns without counting them as candidate features."""

        columns = list(feature_set)
        model_type = context.config.model.model_type
        group_column = None
        if model_type == ModelType.GEE_LOGISTIC_REGRESSION:
            group_column = context.config.model.gee_group_column
        elif model_type == ModelType.MIXED_EFFECTS_REGRESSION:
            group_column = context.config.model.mixed_effects_group_column
        if group_column and group_column in frame.columns and group_column not in columns:
            columns.append(group_column)
        return columns

    def _score_candidate_split(
        self,
        *,
        evaluator: EvaluationStep,
        frame: pd.DataFrame,
        split_name: str,
        feature_columns: list[str],
        adapter,
        context: PipelineContext,
    ) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
        if context.target_column is None:
            raise ValueError("Feature subset search requires a resolved target column.")
        labels_available = context.target_column in frame.columns
        if context.config.target.mode == TargetMode.BINARY:
            return evaluator._score_binary_split(
                frame,
                split_name,
                context.target_column,
                feature_columns,
                adapter,
                context.config.model.threshold,
                labels_available,
                context,
            )
        if context.config.target.mode == TargetMode.MULTICLASS:
            return evaluator._score_multiclass_split(
                frame,
                split_name,
                context.target_column,
                feature_columns,
                adapter,
                labels_available,
                context,
            )
        return evaluator._score_continuous_split(
            frame,
            split_name,
            context.target_column,
            feature_columns,
            adapter,
            labels_available,
            context,
        )

    def _build_candidate_row(
        self,
        *,
        candidate_id: str,
        feature_set: list[str],
        ranking_split: str,
        ranking_metric: str,
        ranking_metrics: dict[str, float | int | None],
        test_metrics: dict[str, float | int | None],
        ranking_calibration_error: float | None,
        test_calibration_error: float | None,
    ) -> dict[str, Any]:
        ranking_value = ranking_metrics.get(ranking_metric)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "status": "success",
            "failure_reason": "",
            "feature_count": len(feature_set),
            "feature_set": ", ".join(feature_set),
            "ranking_split": ranking_split,
            "ranking_metric": ranking_metric,
            "ranking_value": self._optional_float(ranking_value),
        }
        for metric_name, metric_value in ranking_metrics.items():
            if isinstance(metric_value, bool):
                continue
            row[f"ranking_{metric_name}"] = self._optional_float(metric_value)
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, bool):
                continue
            row[f"test_{metric_name}"] = self._optional_float(metric_value)
        if ranking_calibration_error is not None:
            row["ranking_calibration_error"] = self._optional_float(
                ranking_calibration_error
            )
        if test_calibration_error is not None:
            row["test_calibration_error"] = self._optional_float(test_calibration_error)
        return row

    def _add_candidate_selection_scores(
        self,
        successful: pd.DataFrame,
        *,
        ranking_metric: str,
    ) -> pd.DataFrame:
        scored = successful.copy(deep=True)
        scored["ranking_metric_score"] = self._rank_score(
            scored["ranking_value"],
            higher_is_better=not feature_subset_search_lower_is_better(ranking_metric),
        )
        metric_score_specs = [
            ("ranking_roc_auc", "auc_score", True),
            ("ranking_ks_statistic", "ks_score", True),
            ("ranking_average_precision", "average_precision_score", True),
            ("ranking_accuracy", "accuracy_score", True),
            ("ranking_f1_score", "f1_score_rank", True),
            ("ranking_macro_f1", "macro_f1_score", True),
            ("ranking_weighted_f1", "weighted_f1_score", True),
            ("ranking_r2", "r2_score", True),
            ("ranking_explained_variance", "explained_variance_score", True),
            ("ranking_brier_score", "brier_score_rank", False),
            ("ranking_log_loss", "log_loss_score", False),
            ("ranking_rmse", "rmse_score", False),
            ("ranking_mae", "mae_score", False),
            ("ranking_calibration_error", "calibration_score", False),
        ]
        for metric_column, score_column, higher_is_better in metric_score_specs:
            if metric_column in scored.columns:
                scored[score_column] = self._rank_score(
                    scored[metric_column],
                    higher_is_better=higher_is_better,
                )
        scored["simplicity_score"] = self._rank_score(
            scored.get("feature_count"),
            higher_is_better=False,
        )
        weights: dict[str, float] = {
            "ranking_metric_score": 0.30,
            "auc_score": 0.18,
            "ks_score": 0.18,
            "average_precision_score": 0.10,
            "accuracy_score": 0.16,
            "f1_score_rank": 0.14,
            "macro_f1_score": 0.16,
            "weighted_f1_score": 0.14,
            "r2_score": 0.16,
            "explained_variance_score": 0.12,
            "brier_score_rank": 0.07,
            "log_loss_score": 0.07,
            "rmse_score": 0.16,
            "mae_score": 0.12,
            "calibration_score": 0.06,
            "simplicity_score": 0.04,
        }
        score_columns = [column_name for column_name in weights if column_name in scored.columns]
        weighted_sum = pd.Series(0.0, index=scored.index)
        weight_sum = pd.Series(0.0, index=scored.index)
        for column_name in score_columns:
            column = pd.to_numeric(scored[column_name], errors="coerce")
            weight = weights[column_name]
            weighted_sum = weighted_sum.add(column.fillna(0.0) * weight, fill_value=0.0)
            weight_sum = weight_sum.add(column.notna().astype(float) * weight, fill_value=0.0)
        scored["overall_selection_score"] = np.where(
            weight_sum > 0,
            weighted_sum / weight_sum,
            np.nan,
        )
        return scored

    def _rank_score(
        self,
        values: pd.Series | Any,
        *,
        higher_is_better: bool = True,
    ) -> pd.Series:
        if not isinstance(values, pd.Series):
            return pd.Series(dtype=float)
        series = pd.to_numeric(values, errors="coerce")
        if series.notna().sum() <= 1:
            return pd.Series(
                np.where(series.notna(), 1.0, np.nan),
                index=series.index,
                dtype=float,
            )
        ranks = series.rank(
            ascending=not higher_is_better,
            method="min",
            na_option="keep",
        )
        denominator = max(float(series.notna().sum() - 1), 1.0)
        return (1.0 - ((ranks - 1.0) / denominator)).clip(lower=0.0, upper=1.0)

    def _build_leaderboard_table(self, successful: pd.DataFrame) -> pd.DataFrame:
        column_order = [
            "rank",
            "candidate_id",
            "overall_selection_score",
            "ranking_metric_score",
            "simplicity_score",
            "calibration_score",
            "feature_count",
            "feature_set",
            "ranking_metric",
            "ranking_value",
            "ranking_roc_auc",
            "ranking_ks_statistic",
            "ranking_average_precision",
            "ranking_accuracy",
            "ranking_f1_score",
            "ranking_macro_f1",
            "ranking_weighted_f1",
            "ranking_brier_score",
            "ranking_log_loss",
            "ranking_rmse",
            "ranking_mae",
            "ranking_r2",
            "ranking_explained_variance",
            "ranking_calibration_error",
            "test_roc_auc",
            "test_ks_statistic",
            "test_accuracy",
            "test_f1_score",
            "test_macro_f1",
            "test_weighted_f1",
            "test_brier_score",
            "test_log_loss",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_explained_variance",
            "test_calibration_error",
        ]
        metric_columns = [
            column_name
            for column_name in successful.columns
            if column_name.startswith(("ranking_", "test_"))
            and column_name not in column_order
        ]
        resolved_columns = [
            column_name
            for column_name in [*column_order, *sorted(metric_columns)]
            if column_name in successful.columns
        ]
        return (
            successful.loc[:, resolved_columns]
            .sort_values(["rank", "candidate_id"], ascending=[True, True], kind="mergesort")
            .reset_index(drop=True)
        )

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
            or context.config.target.mode != TargetMode.BINARY
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

    def _build_contribution_consistency_table(
        self,
        *,
        top_candidates: pd.DataFrame,
        candidate_feature_importance: dict[str, pd.DataFrame],
        ranking_metadata: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, candidate_row in top_candidates.iterrows():
            candidate_id = str(candidate_row["candidate_id"])
            importance = candidate_feature_importance.get(candidate_id, pd.DataFrame())
            if importance.empty or "feature_name" not in importance.columns:
                continue
            for _, importance_row in importance.iterrows():
                feature_name = str(importance_row.get("feature_name", "")).strip()
                if not feature_name:
                    continue
                coefficient = self._optional_float(importance_row.get("coefficient"))
                importance_value = self._optional_float(
                    importance_row.get("importance_value")
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_rank": self._optional_float(candidate_row.get("rank")),
                        "feature_name": feature_name,
                        "coefficient": coefficient,
                        "importance_value": importance_value,
                        "feature_in_candidate": feature_name
                        in ranking_metadata.get(candidate_id, {}).get("feature_set", []),
                    }
                )
        if not rows:
            return pd.DataFrame()
        long_frame = pd.DataFrame(rows)
        summary_rows: list[dict[str, Any]] = []
        for feature_name, feature_frame in long_frame.groupby("feature_name", dropna=False):
            coefficients = pd.to_numeric(feature_frame["coefficient"], errors="coerce").dropna()
            importance_values = pd.to_numeric(
                feature_frame["importance_value"],
                errors="coerce",
            ).dropna()
            positive_count = int((coefficients > 0).sum())
            negative_count = int((coefficients < 0).sum())
            signed_count = positive_count + negative_count
            sign_consistency = (
                max(positive_count, negative_count) / signed_count if signed_count else None
            )
            summary_rows.append(
                {
                    "feature_name": feature_name,
                    "candidate_count": int(feature_frame["candidate_id"].nunique()),
                    "best_candidate_rank": self._optional_float(
                        pd.to_numeric(feature_frame["candidate_rank"], errors="coerce").min()
                    ),
                    "mean_coefficient": self._optional_float(coefficients.mean())
                    if not coefficients.empty
                    else None,
                    "coefficient_std": self._optional_float(coefficients.std(ddof=0))
                    if len(coefficients) > 1
                    else 0.0
                    if len(coefficients) == 1
                    else None,
                    "mean_abs_coefficient": self._optional_float(coefficients.abs().mean())
                    if not coefficients.empty
                    else None,
                    "mean_importance_value": self._optional_float(importance_values.mean())
                    if not importance_values.empty
                    else None,
                    "sign_consistency_share": self._optional_float(sign_consistency),
                    "positive_coefficient_count": positive_count,
                    "negative_coefficient_count": negative_count,
                }
            )
        return (
            pd.DataFrame(summary_rows)
            .sort_values(
                ["candidate_count", "mean_abs_coefficient", "feature_name"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    def _build_feature_redundancy_table(
        self,
        *,
        train_frame: pd.DataFrame,
        candidate_features: list[str],
        top_candidates: pd.DataFrame,
        ranking_metadata: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        numeric_features = [
            feature
            for feature in candidate_features
            if feature in train_frame.columns
            and pd.api.types.is_numeric_dtype(train_frame[feature])
        ]
        if len(numeric_features) < 2 or top_candidates.empty:
            return pd.DataFrame()
        correlation = train_frame.loc[:, numeric_features].corr(numeric_only=True).abs()
        top_candidate_ids = top_candidates["candidate_id"].astype(str).tolist()
        rows: list[dict[str, Any]] = []
        for left_index, left_feature in enumerate(numeric_features):
            for right_feature in numeric_features[left_index + 1 :]:
                value = self._optional_float(correlation.loc[left_feature, right_feature])
                if value is None or value < 0.70:
                    continue
                candidate_ids = [
                    candidate_id
                    for candidate_id in top_candidate_ids
                    if {
                        left_feature,
                        right_feature,
                    }.issubset(set(ranking_metadata.get(candidate_id, {}).get("feature_set", [])))
                ]
                if not candidate_ids:
                    continue
                rows.append(
                    {
                        "feature_a": left_feature,
                        "feature_b": right_feature,
                        "absolute_correlation": value,
                        "appears_in_top_candidate_count": len(candidate_ids),
                        "candidate_ids": ", ".join(candidate_ids),
                        "risk_level": "high" if value >= 0.85 else "watch",
                    }
                )
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["absolute_correlation", "appears_in_top_candidate_count"],
                ascending=[False, False],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    def _build_candidate_risk_flags_table(
        self,
        *,
        successful: pd.DataFrame,
        selected_candidate: pd.Series,
        ranking_metadata: dict[str, dict[str, Any]],
        redundancy: pd.DataFrame,
        contribution_consistency: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_auc = self._optional_float(selected_candidate.get("ranking_roc_auc"))
        selected_ks = self._optional_float(selected_candidate.get("ranking_ks_statistic"))
        selected_brier = self._optional_float(selected_candidate.get("ranking_brier_score"))
        selected_feature_count = int(selected_candidate.get("feature_count", 0) or 0)
        redundant_pairs = self._redundant_pair_lookup(redundancy)
        unstable_features = set()
        if not contribution_consistency.empty:
            unstable = contribution_consistency.loc[
                pd.to_numeric(
                    contribution_consistency.get("sign_consistency_share"),
                    errors="coerce",
                )
                .fillna(1.0)
                < 0.80
            ]
            unstable_features = set(unstable.get("feature_name", pd.Series(dtype=str)).astype(str))

        rows: list[dict[str, Any]] = []
        for _, row in successful.iterrows():
            candidate_id = str(row["candidate_id"])
            feature_set = ranking_metadata.get(candidate_id, {}).get("feature_set", [])
            feature_set_lookup = set(feature_set)
            self._append_risk_flag(
                rows,
                candidate_id,
                "complexity_watch",
                "watch",
                "Candidate uses materially more features than the selected subset.",
                int(row.get("feature_count", 0) or 0) > selected_feature_count + 2,
            )
            self._append_risk_flag(
                rows,
                candidate_id,
                "auc_gap",
                "watch",
                "Ranking AUC trails the selected subset by at least 0.02.",
                selected_auc is not None
                and self._optional_float(row.get("ranking_roc_auc")) is not None
                and float(row["ranking_roc_auc"]) < selected_auc - 0.02,
            )
            self._append_risk_flag(
                rows,
                candidate_id,
                "ks_gap",
                "watch",
                "Ranking KS trails the selected subset by at least 0.03.",
                selected_ks is not None
                and self._optional_float(row.get("ranking_ks_statistic")) is not None
                and float(row["ranking_ks_statistic"]) < selected_ks - 0.03,
            )
            self._append_risk_flag(
                rows,
                candidate_id,
                "calibration_watch",
                "watch",
                "Ranking calibration error is above 0.05.",
                self._optional_float(row.get("ranking_calibration_error")) is not None
                and float(row["ranking_calibration_error"]) > 0.05,
            )
            self._append_risk_flag(
                rows,
                candidate_id,
                "brier_gap",
                "watch",
                "Brier score is more than 10% worse than the selected subset.",
                selected_brier is not None
                and selected_brier > 0
                and self._optional_float(row.get("ranking_brier_score")) is not None
                and float(row["ranking_brier_score"]) > selected_brier * 1.10,
            )
            redundant_hits = [
                pair
                for pair in redundant_pairs
                if {pair[0], pair[1]}.issubset(feature_set_lookup)
            ]
            self._append_risk_flag(
                rows,
                candidate_id,
                "redundant_features",
                "watch",
                "Candidate contains high-correlation feature pairs.",
                bool(redundant_hits),
                detail=", ".join(f"{left} + {right}" for left, right in redundant_hits[:3]),
            )
            unstable_hits = sorted(feature_set_lookup & unstable_features)
            self._append_risk_flag(
                rows,
                candidate_id,
                "coefficient_instability",
                "watch",
                "Candidate contains features with unstable coefficient signs across top subsets.",
                bool(unstable_hits),
                detail=", ".join(unstable_hits[:5]),
            )
        return pd.DataFrame(rows)

    def _append_risk_flag(
        self,
        rows: list[dict[str, Any]],
        candidate_id: str,
        flag_name: str,
        severity: str,
        description: str,
        condition: bool,
        *,
        detail: str = "",
    ) -> None:
        if condition:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "flag_name": flag_name,
                    "severity": severity,
                    "description": description,
                    "detail": detail,
                }
            )

    def _redundant_pair_lookup(self, redundancy: pd.DataFrame) -> list[tuple[str, str]]:
        if redundancy.empty or not {"feature_a", "feature_b"}.issubset(redundancy.columns):
            return []
        return [
            (str(row["feature_a"]), str(row["feature_b"]))
            for _, row in redundancy.iterrows()
        ]

    def _build_top_candidate_comparison_table(
        self,
        *,
        top_candidates: pd.DataFrame,
        risk_flags: pd.DataFrame,
    ) -> pd.DataFrame:
        if top_candidates.empty:
            return pd.DataFrame()
        risk_counts = (
            risk_flags.groupby("candidate_id", dropna=False)
            .size()
            .rename("risk_flag_count")
            .reset_index()
            if not risk_flags.empty
            else pd.DataFrame(columns=["candidate_id", "risk_flag_count"])
        )
        columns = [
            "rank",
            "candidate_id",
            "overall_selection_score",
            "feature_count",
            "feature_set",
            "ranking_metric",
            "ranking_value",
            "ranking_roc_auc",
            "ranking_ks_statistic",
            "ranking_accuracy",
            "ranking_macro_f1",
            "ranking_rmse",
            "ranking_mae",
            "ranking_r2",
            "ranking_brier_score",
            "ranking_calibration_error",
            "test_roc_auc",
            "test_ks_statistic",
            "test_accuracy",
            "test_macro_f1",
            "test_rmse",
            "test_mae",
            "test_r2",
        ]
        comparison = top_candidates.loc[
            :,
            [column for column in columns if column in top_candidates.columns],
        ].merge(risk_counts, on="candidate_id", how="left")
        comparison["risk_flag_count"] = comparison["risk_flag_count"].fillna(0).astype(int)
        return comparison.reset_index(drop=True)

    def _build_selection_rationale_table(
        self,
        *,
        selected_candidate: pd.Series,
        top_candidates: pd.DataFrame,
        risk_flags: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_id = str(selected_candidate.get("candidate_id", "n/a"))
        selected_flags = (
            risk_flags.loc[risk_flags["candidate_id"].astype(str) == selected_id]
            if not risk_flags.empty
            else pd.DataFrame()
        )
        runner_up = top_candidates.iloc[1] if len(top_candidates) > 1 else None
        rationale = [
            (
                f"{selected_id} ranked first on the configured ranking metric "
                f"({selected_candidate.get('ranking_metric', 'n/a')})."
            ),
            (
                "It uses "
                f"{int(selected_candidate.get('feature_count', 0) or 0)} feature(s), "
                "which supports parsimony review before final model development."
            ),
        ]
        if runner_up is not None:
            rationale.append(
                f"The next-ranked candidate is {runner_up.get('candidate_id', 'n/a')} "
                f"with {runner_up.get('feature_count', 'n/a')} feature(s)."
            )
        if selected_flags.empty:
            rationale.append("No automated risk flags were raised for the selected subset.")
        else:
            rationale.append(
                f"{len(selected_flags)} automated risk flag(s) were raised for review."
            )
        return pd.DataFrame(
            [
                {
                    "selected_candidate_id": selected_id,
                    "selected_feature_set": selected_candidate.get("feature_set", ""),
                    "selected_rank": selected_candidate.get("rank", ""),
                    "overall_selection_score": selected_candidate.get(
                        "overall_selection_score",
                        None,
                    ),
                    "selection_rationale": " ".join(rationale),
                    "risk_flag_count": int(len(selected_flags)),
                }
            ]
        )

    def _build_excluded_feature_insight_table(
        self,
        *,
        candidate_features: list[str],
        selected_feature_set: list[str],
        successful: pd.DataFrame,
        ranking_metadata: dict[str, dict[str, Any]],
        feature_frequency: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_lookup = set(selected_feature_set)
        frequency_lookup = (
            feature_frequency.set_index("feature_name").to_dict(orient="index")
            if not feature_frequency.empty and "feature_name" in feature_frequency.columns
            else {}
        )
        rows: list[dict[str, Any]] = []
        for feature_name in candidate_features:
            if feature_name in selected_lookup:
                continue
            containing = [
                candidate_id
                for candidate_id, metadata in ranking_metadata.items()
                if feature_name in metadata.get("feature_set", [])
            ]
            subset_rows = successful.loc[successful["candidate_id"].isin(containing)]
            frequency = frequency_lookup.get(feature_name, {})
            rows.append(
                {
                    "feature_name": feature_name,
                    "included_in_candidate_count": len(containing),
                    "top_candidate_selected_count": int(frequency.get("selected_count", 0)),
                    "top_candidate_selected_share": self._optional_float(
                        frequency.get("selected_share")
                    ),
                    "best_rank_when_included": self._optional_float(subset_rows["rank"].min())
                    if not subset_rows.empty
                    else None,
                    "best_ranking_value_when_included": self._optional_float(
                        subset_rows.sort_values("rank", kind="mergesort").iloc[0][
                            "ranking_value"
                        ]
                    )
                    if not subset_rows.empty
                    else None,
                    "best_auc_when_included": self._optional_float(
                        subset_rows["ranking_roc_auc"].max()
                    )
                    if not subset_rows.empty and "ranking_roc_auc" in subset_rows.columns
                    else None,
                    "best_ks_when_included": self._optional_float(
                        subset_rows["ranking_ks_statistic"].max()
                    )
                    if not subset_rows.empty and "ranking_ks_statistic" in subset_rows.columns
                    else None,
                }
            )
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["best_rank_when_included", "feature_name"],
                ascending=[True, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    def _build_feature_family_table(
        self,
        *,
        candidate_features: list[str],
        selected_feature_set: list[str],
        feature_frequency: pd.DataFrame,
    ) -> pd.DataFrame:
        frequency_lookup = (
            feature_frequency.set_index("feature_name").to_dict(orient="index")
            if not feature_frequency.empty and "feature_name" in feature_frequency.columns
            else {}
        )
        selected_lookup = set(selected_feature_set)
        rows = []
        for feature_name in candidate_features:
            frequency = frequency_lookup.get(feature_name, {})
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_family": self._infer_feature_family(feature_name),
                    "transformation_type": self._infer_transformation_type(feature_name),
                    "selected_candidate_feature": feature_name in selected_lookup,
                    "top_candidate_selected_count": int(frequency.get("selected_count", 0)),
                    "top_candidate_selected_share": self._optional_float(
                        frequency.get("selected_share")
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_transformation_effectiveness_table(
        self,
        *,
        successful: pd.DataFrame,
        ranking_metadata: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        if successful.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for transformation_type in sorted(
            {
                self._infer_transformation_type(feature)
                for metadata in ranking_metadata.values()
                for feature in metadata.get("feature_set", [])
            }
        ):
            candidate_ids = [
                candidate_id
                for candidate_id, metadata in ranking_metadata.items()
                if any(
                    self._infer_transformation_type(feature) == transformation_type
                    for feature in metadata.get("feature_set", [])
                )
            ]
            candidate_rows = successful.loc[successful["candidate_id"].isin(candidate_ids)]
            if candidate_rows.empty:
                continue
            best_row = candidate_rows.sort_values("rank", kind="mergesort").iloc[0]
            rows.append(
                {
                    "transformation_type": transformation_type,
                    "candidate_count": int(len(candidate_rows)),
                    "best_candidate_id": best_row.get("candidate_id"),
                    "best_rank": best_row.get("rank"),
                    "ranking_metric": best_row.get("ranking_metric"),
                    "best_ranking_value": best_row.get("ranking_value"),
                    "best_ranking_roc_auc": best_row.get("ranking_roc_auc"),
                    "best_ranking_ks_statistic": best_row.get("ranking_ks_statistic"),
                    "mean_overall_selection_score": self._optional_float(
                        pd.to_numeric(
                            candidate_rows.get("overall_selection_score"),
                            errors="coerce",
                        ).mean()
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["best_rank", "transformation_type"],
            ascending=[True, True],
            kind="mergesort",
        )

    def _build_segment_performance_table(
        self,
        *,
        context: PipelineContext,
        top_candidates: pd.DataFrame,
        ranking_predictions: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if top_candidates.empty or context.target_column is None:
            return pd.DataFrame()
        first_candidate_id = str(top_candidates.iloc[0]["candidate_id"])
        first_frame = ranking_predictions.get(first_candidate_id, pd.DataFrame())
        segment_columns = self._candidate_segment_columns(first_frame, context)
        if not segment_columns:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        top_curve_candidates = top_candidates.head(
            context.config.subset_search.top_curve_count
        )
        for _, candidate_row in top_curve_candidates.iterrows():
            candidate_id = str(candidate_row["candidate_id"])
            frame = ranking_predictions.get(candidate_id, pd.DataFrame())
            if frame.empty:
                continue
            for segment_column in segment_columns:
                if segment_column not in frame.columns:
                    continue
                top_values = (
                    frame[segment_column]
                    .astype(str)
                    .value_counts(dropna=False)
                    .head(8)
                    .index.tolist()
                )
                for segment_value in top_values:
                    segment_frame = frame.loc[frame[segment_column].astype(str) == segment_value]
                    metrics = self._metric_summary(
                        segment_frame,
                        context.target_column,
                        context.config.target.mode,
                    )
                    if metrics["row_count"] < 20:
                        continue
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "candidate_rank": candidate_row.get("rank"),
                            "segment_column": segment_column,
                            "segment_value": segment_value,
                            **metrics,
                        }
                    )
        return pd.DataFrame(rows)

    def _candidate_segment_columns(
        self,
        frame: pd.DataFrame,
        context: PipelineContext,
    ) -> list[str]:
        configured = context.config.diagnostics.default_segment_column
        candidates = [configured] if configured else []
        candidates.extend(
            feature
            for feature in context.categorical_features
            if feature in frame.columns
            and frame[feature].nunique(dropna=True)
            <= context.config.performance.max_categorical_cardinality
        )
        excluded = {
            context.target_column,
            "split",
            "predicted_probability",
            "predicted_class",
        }
        return list(
            dict.fromkeys(
                column
                for column in candidates
                if column and column in frame.columns and column not in excluded
            )
        )[:3]

    def _build_time_performance_table(
        self,
        *,
        context: PipelineContext,
        top_candidates: pd.DataFrame,
        ranking_predictions: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        date_column = context.config.split.date_column
        if not date_column or top_candidates.empty or context.target_column is None:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        top_curve_candidates = top_candidates.head(
            context.config.subset_search.top_curve_count
        )
        for _, candidate_row in top_curve_candidates.iterrows():
            candidate_id = str(candidate_row["candidate_id"])
            frame = ranking_predictions.get(candidate_id, pd.DataFrame())
            if frame.empty or date_column not in frame.columns:
                continue
            working = frame.copy(deep=False)
            working["_period"] = pd.to_datetime(
                working[date_column],
                errors="coerce",
            ).dt.to_period("M").astype(str)
            for period, period_frame in working.dropna(subset=["_period"]).groupby("_period"):
                metrics = self._metric_summary(
                    period_frame,
                    context.target_column,
                    context.config.target.mode,
                )
                if metrics["row_count"] < 20:
                    continue
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_rank": candidate_row.get("rank"),
                        "period": period,
                        **metrics,
                    }
                )
        return pd.DataFrame(rows)

    def _metric_summary(
        self,
        frame: pd.DataFrame,
        target_column: str,
        target_mode: TargetMode,
    ) -> dict[str, Any]:
        if target_mode == TargetMode.BINARY:
            return self._binary_metric_summary(frame, target_column)
        if target_mode == TargetMode.MULTICLASS:
            return self._multiclass_metric_summary(frame, target_column)
        return self._continuous_metric_summary(frame, target_column)

    def _binary_metric_summary(
        self,
        frame: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Any]:
        if frame.empty or target_column not in frame.columns:
            return {
                "row_count": 0,
                "observed_event_rate": None,
                "average_score": None,
                "roc_auc": None,
                "ks_statistic": None,
                "brier_score": None,
                "log_loss": None,
                "calibration_error": None,
            }
        y_true = pd.to_numeric(frame[target_column], errors="coerce")
        scores = pd.to_numeric(frame["predicted_probability"], errors="coerce")
        aligned = pd.DataFrame({"target": y_true, "score": scores}).dropna()
        if aligned.empty:
            return {
                "row_count": 0,
                "observed_event_rate": None,
                "average_score": None,
                "roc_auc": None,
                "ks_statistic": None,
                "brier_score": None,
                "log_loss": None,
                "calibration_error": None,
            }
        target = aligned["target"].astype(int)
        score = aligned["score"].clip(0.0, 1.0)
        has_two_classes = target.nunique(dropna=True) == 2
        return {
            "row_count": int(len(aligned)),
            "observed_event_rate": self._optional_float(target.mean()),
            "average_score": self._optional_float(score.mean()),
            "roc_auc": self._safe_metric_value(
                lambda: roc_auc_score(target, score),
                enabled=has_two_classes,
            ),
            "ks_statistic": self._safe_metric_value(
                lambda: self._ks_statistic(target.to_numpy(), score.to_numpy()),
                enabled=has_two_classes,
            ),
            "brier_score": self._safe_metric_value(lambda: brier_score_loss(target, score)),
            "log_loss": self._safe_metric_value(lambda: log_loss(target, score, labels=[0, 1])),
            "calibration_error": self._optional_float(abs(target.mean() - score.mean())),
        }

    def _multiclass_metric_summary(
        self,
        frame: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Any]:
        if frame.empty or target_column not in frame.columns or "predicted_class" not in frame:
            return {
                "row_count": 0,
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "average_predicted_confidence": None,
            }
        aligned = frame[[target_column, "predicted_class"]].dropna()
        if aligned.empty:
            return {
                "row_count": 0,
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "average_predicted_confidence": None,
            }
        y_true = aligned[target_column].astype(int)
        y_pred = aligned["predicted_class"].astype(int)
        confidence = (
            pd.to_numeric(frame.get("predicted_class_probability"), errors="coerce")
            if "predicted_class_probability" in frame
            else pd.Series(dtype=float)
        )
        return {
            "row_count": int(len(aligned)),
            "accuracy": self._safe_metric_value(lambda: accuracy_score(y_true, y_pred)),
            "macro_f1": self._safe_metric_value(
                lambda: f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "weighted_f1": self._safe_metric_value(
                lambda: f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "average_predicted_confidence": self._optional_float(confidence.mean())
            if not confidence.empty
            else None,
        }

    def _continuous_metric_summary(
        self,
        frame: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Any]:
        if frame.empty or target_column not in frame.columns or "predicted_value" not in frame:
            return {
                "row_count": 0,
                "mean_actual": None,
                "mean_predicted": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "explained_variance": None,
            }
        aligned = pd.DataFrame(
            {
                "target": pd.to_numeric(frame[target_column], errors="coerce"),
                "prediction": pd.to_numeric(frame["predicted_value"], errors="coerce"),
            }
        ).dropna()
        if aligned.empty:
            return {
                "row_count": 0,
                "mean_actual": None,
                "mean_predicted": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "explained_variance": None,
            }
        y_true = aligned["target"].to_numpy(dtype=float)
        y_pred = aligned["prediction"].to_numpy(dtype=float)
        return {
            "row_count": int(len(aligned)),
            "mean_actual": self._optional_float(np.mean(y_true)),
            "mean_predicted": self._optional_float(np.mean(y_pred)),
            "rmse": self._safe_metric_value(
                lambda: float(np.sqrt(mean_squared_error(y_true, y_pred)))
            ),
            "mae": self._safe_metric_value(lambda: mean_absolute_error(y_true, y_pred)),
            "r2": self._safe_metric_value(lambda: r2_score(y_true, y_pred)),
            "explained_variance": self._safe_metric_value(
                lambda: explained_variance_score(y_true, y_pred)
            ),
        }

    def _calibration_error(
        self,
        scored_frame: pd.DataFrame,
        target_column: str,
    ) -> float | None:
        if (
            scored_frame.empty
            or target_column not in scored_frame.columns
            or "predicted_probability" not in scored_frame.columns
        ):
            return None
        aligned = scored_frame[[target_column, "predicted_probability"]].dropna().copy()
        if len(aligned) < 10:
            return None
        aligned[target_column] = pd.to_numeric(aligned[target_column], errors="coerce")
        aligned["predicted_probability"] = pd.to_numeric(
            aligned["predicted_probability"],
            errors="coerce",
        ).clip(0.0, 1.0)
        aligned = aligned.dropna()
        if aligned.empty:
            return None
        try:
            aligned["_bucket"] = pd.qcut(
                aligned["predicted_probability"],
                q=min(10, aligned["predicted_probability"].nunique()),
                duplicates="drop",
            )
        except ValueError:
            return self._optional_float(
                abs(aligned[target_column].mean() - aligned["predicted_probability"].mean())
            )
        grouped = aligned.groupby("_bucket", observed=True).agg(
            observed_rate=(target_column, "mean"),
            average_score=("predicted_probability", "mean"),
            row_count=(target_column, "size"),
        )
        if grouped.empty:
            return None
        weighted_error = (
            (grouped["observed_rate"] - grouped["average_score"]).abs()
            * grouped["row_count"]
        ).sum() / grouped["row_count"].sum()
        return self._optional_float(weighted_error)

    def _safe_metric_value(self, metric_callable, *, enabled: bool = True) -> float | None:
        if not enabled:
            return None
        try:
            return self._optional_float(metric_callable())
        except Exception:
            return None

    def _ks_statistic(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        order = np.argsort(-scores)
        ordered_target = np.asarray(y_true)[order]
        positives = max(int(ordered_target.sum()), 1)
        negatives = max(int(len(ordered_target) - ordered_target.sum()), 1)
        positive_cdf = np.cumsum(ordered_target) / positives
        negative_cdf = np.cumsum(1 - ordered_target) / negatives
        return float(np.max(np.abs(positive_cdf - negative_cdf)))

    def _infer_feature_family(self, feature_name: str) -> str:
        name = feature_name.lower()
        family_keywords = {
            "delinquency": ("delinq", "dpd", "past_due", "arrears"),
            "utilization": ("util", "usage", "line"),
            "income": ("income", "revenue", "sales", "ebitda", "cash_flow"),
            "balance_sheet": ("asset", "liability", "equity", "debt", "balance"),
            "profitability": ("margin", "profit", "roa", "roe", "expense"),
            "macro": ("unemployment", "gdp", "cpi", "rate", "macro"),
            "behavioral": ("inquiry", "transaction", "payment", "activity"),
            "scorecard": ("woe", "iv", "scorecard", "bin"),
            "time": ("lag", "rolling", "month", "quarter", "year"),
        }
        for family, keywords in family_keywords.items():
            if any(keyword in name for keyword in keywords):
                return family
        return "other"

    def _infer_transformation_type(self, feature_name: str) -> str:
        name = feature_name.lower()
        if "woe" in name:
            return "woe"
        if "bin" in name or "bucket" in name:
            return "binned"
        if "winsor" in name:
            return "winsorized"
        if name.startswith("log_") or name.endswith("_log") or "log_" in name:
            return "log"
        if "lag" in name:
            return "lag"
        if "rolling" in name or "window" in name:
            return "rolling"
        if "interaction" in name or "_x_" in name or "_by_" in name:
            return "interaction"
        if "ratio" in name or "pct" in name or "percent" in name:
            return "ratio"
        if "sqrt" in name:
            return "sqrt"
        if "zscore" in name or "standard" in name:
            return "standardized"
        return "raw"

    def _display_metric_for_target_mode(self, target_mode: TargetMode) -> str:
        if target_mode == TargetMode.BINARY:
            return "roc_auc"
        if target_mode == TargetMode.MULTICLASS:
            return "accuracy"
        return "rmse"

    def _build_figures(
        self,
        *,
        context: PipelineContext,
        top_candidates: pd.DataFrame,
        selected_candidate_id: str,
        feature_frequency: pd.DataFrame,
        ranking_predictions: dict[str, pd.DataFrame],
        ranking_metadata: dict[str, dict[str, Any]],
        leaderboard: pd.DataFrame,
        contribution_consistency: pd.DataFrame,
        redundancy: pd.DataFrame,
        risk_flags: pd.DataFrame,
        segment_performance: pd.DataFrame,
        time_performance: pd.DataFrame,
        transformation_effectiveness: pd.DataFrame,
        feature_family: pd.DataFrame,
    ) -> None:
        if top_candidates.empty:
            return
        if "ranking_roc_auc" in top_candidates.columns:
            auc_frame = top_candidates.dropna(subset=["ranking_roc_auc"])
            if not auc_frame.empty:
                auc_scatter = px.scatter(
                    auc_frame,
                    x="feature_count",
                    y="ranking_roc_auc",
                    hover_name="candidate_id",
                    hover_data={"feature_set": True},
                    title="Subset Search: ROC AUC vs Feature Count",
                    labels={"feature_count": "Feature Count", "ranking_roc_auc": "ROC AUC"},
                )
                context.visualizations["subset_search_auc_frontier"] = (
                    apply_fintech_figure_theme(
                        auc_scatter,
                        title="Subset Search: ROC AUC vs Feature Count",
                    )
                )

        if "ranking_ks_statistic" in top_candidates.columns:
            ks_frame = top_candidates.dropna(subset=["ranking_ks_statistic"])
            if not ks_frame.empty:
                ks_scatter = px.scatter(
                    ks_frame,
                    x="feature_count",
                    y="ranking_ks_statistic",
                    hover_name="candidate_id",
                    hover_data={"feature_set": True},
                    title="Subset Search: KS vs Feature Count",
                    labels={
                        "feature_count": "Feature Count",
                        "ranking_ks_statistic": "KS Statistic",
                    },
                )
                context.visualizations["subset_search_ks_frontier"] = (
                    apply_fintech_figure_theme(
                        ks_scatter,
                        title="Subset Search: KS vs Feature Count",
                    )
                )

        frontier = px.scatter(
            top_candidates,
            x="feature_count",
            y="ranking_value",
            color="ranking_ks_statistic"
            if "ranking_ks_statistic" in top_candidates.columns
            else "ranking_metric",
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

        if not leaderboard.empty and "overall_selection_score" in leaderboard.columns:
            score_chart = px.bar(
                leaderboard.head(15),
                x="candidate_id",
                y="overall_selection_score",
                color="ranking_roc_auc"
                if "ranking_roc_auc" in leaderboard.columns
                else "ranking_value"
                if "ranking_value" in leaderboard.columns
                else None,
                title="Subset Search: Candidate Leaderboard Score",
                labels={
                    "candidate_id": "Candidate",
                    "overall_selection_score": "Selection Score",
                    "ranking_roc_auc": "ROC AUC",
                },
            )
            context.visualizations["subset_search_leaderboard_score_chart"] = (
                apply_fintech_figure_theme(
                    score_chart,
                    title="Subset Search: Candidate Leaderboard Score",
                )
            )

        metric_columns = [
            column
            for column in [
                "ranking_roc_auc",
                "ranking_ks_statistic",
                "ranking_average_precision",
                "ranking_accuracy",
                "ranking_f1_score",
                "ranking_macro_f1",
                "ranking_weighted_f1",
                "ranking_rmse",
                "ranking_mae",
                "ranking_r2",
                "ranking_explained_variance",
                "overall_selection_score",
                "simplicity_score",
                "calibration_score",
            ]
            if column in top_candidates.columns
        ]
        if metric_columns:
            heatmap_data = (
                top_candidates.head(12)
                .set_index("candidate_id")
                .loc[:, metric_columns]
                .apply(pd.to_numeric, errors="coerce")
            )
            heatmap = px.imshow(
                heatmap_data,
                aspect="auto",
                title="Subset Search: Top Candidate Metric Heatmap",
                labels={"x": "Metric", "y": "Candidate", "color": "Value"},
            )
            context.visualizations["subset_search_metric_comparison_heatmap"] = (
                apply_fintech_figure_theme(
                    heatmap,
                    title="Subset Search: Top Candidate Metric Heatmap",
                )
            )

        if "ranking_calibration_error" in top_candidates.columns:
            calibration_frame = top_candidates.dropna(subset=["ranking_calibration_error"])
            if not calibration_frame.empty:
                calibration_chart = px.bar(
                    calibration_frame.head(15),
                    x="candidate_id",
                    y="ranking_calibration_error",
                    color="feature_count",
                    title="Subset Search: Calibration Error by Candidate",
                    labels={
                        "candidate_id": "Candidate",
                        "ranking_calibration_error": "Calibration Error",
                    },
                )
                context.visualizations["subset_search_calibration_comparison"] = (
                    apply_fintech_figure_theme(
                        calibration_chart,
                        title="Subset Search: Calibration Error by Candidate",
                    )
                )

        if not risk_flags.empty:
            risk_summary = (
                risk_flags.groupby(["candidate_id", "severity"], dropna=False)
                .size()
                .rename("flag_count")
                .reset_index()
            )
            risk_chart = px.bar(
                risk_summary,
                x="candidate_id",
                y="flag_count",
                color="severity",
                title="Subset Search: Candidate Risk Flags",
                labels={"candidate_id": "Candidate", "flag_count": "Risk Flags"},
            )
            context.visualizations["subset_search_risk_flag_summary"] = (
                apply_fintech_figure_theme(
                    risk_chart,
                    title="Subset Search: Candidate Risk Flags",
                )
            )

        if not contribution_consistency.empty:
            consistency_chart = px.bar(
                contribution_consistency.head(20),
                x="feature_name",
                y="candidate_count",
                color="sign_consistency_share"
                if "sign_consistency_share" in contribution_consistency.columns
                else None,
                title="Subset Search: Feature Contribution Consistency",
                labels={
                    "feature_name": "Feature",
                    "candidate_count": "Top Candidate Appearances",
                },
            )
            context.visualizations["subset_search_contribution_consistency_chart"] = (
                apply_fintech_figure_theme(
                    consistency_chart,
                    title="Subset Search: Feature Contribution Consistency",
                )
            )

        if not redundancy.empty:
            redundancy_chart = px.bar(
                redundancy.head(20),
                x="feature_a",
                y="absolute_correlation",
                color="feature_b",
                title="Subset Search: Redundant Feature Pair Watchlist",
                labels={
                    "feature_a": "Feature A",
                    "absolute_correlation": "Absolute Correlation",
                },
            )
            context.visualizations["subset_search_redundancy_watchlist"] = (
                apply_fintech_figure_theme(
                    redundancy_chart,
                    title="Subset Search: Redundant Feature Pair Watchlist",
                )
            )

        segment_metric = self._display_metric_for_target_mode(context.config.target.mode)
        if not segment_performance.empty and segment_metric in segment_performance.columns:
            segment_chart_frame = segment_performance.dropna(subset=[segment_metric]).head(80)
            if not segment_chart_frame.empty:
                segment_chart = px.bar(
                    segment_chart_frame,
                    x="segment_value",
                    y=segment_metric,
                    color="candidate_id",
                    facet_col="segment_column",
                    title="Subset Search: Segment-Level Candidate Performance",
                    labels={
                        "segment_value": "Segment",
                        segment_metric: segment_metric.replace("_", " ").title(),
                    },
                )
                context.visualizations["subset_search_segment_performance_chart"] = (
                    apply_fintech_figure_theme(
                        segment_chart,
                        title="Subset Search: Segment-Level Candidate Performance",
                    )
                )

        if not time_performance.empty and segment_metric in time_performance.columns:
            time_chart_frame = time_performance.dropna(subset=[segment_metric])
            if not time_chart_frame.empty:
                time_chart = px.line(
                    time_chart_frame,
                    x="period",
                    y=segment_metric,
                    color="candidate_id",
                    markers=True,
                    title="Subset Search: Time-Split Candidate Performance",
                    labels={
                        "period": "Period",
                        segment_metric: segment_metric.replace("_", " ").title(),
                    },
                )
                context.visualizations["subset_search_time_performance_chart"] = (
                    apply_fintech_figure_theme(
                        time_chart,
                        title="Subset Search: Time-Split Candidate Performance",
                    )
                )

        if not transformation_effectiveness.empty:
            transformation_chart = px.bar(
                transformation_effectiveness,
                x="transformation_type",
                y="mean_overall_selection_score",
                color="best_rank",
                title="Subset Search: Transformation Effectiveness",
                labels={
                    "transformation_type": "Transformation",
                    "mean_overall_selection_score": "Mean Selection Score",
                },
            )
            context.visualizations["subset_search_transformation_effectiveness_chart"] = (
                apply_fintech_figure_theme(
                    transformation_chart,
                    title="Subset Search: Transformation Effectiveness",
                )
            )

        if not feature_family.empty:
            family_summary = (
                feature_family.groupby("feature_family", dropna=False)
                .agg(
                    feature_count=("feature_name", "count"),
                    selected_candidate_features=("selected_candidate_feature", "sum"),
                    top_candidate_selected_count=("top_candidate_selected_count", "sum"),
                )
                .reset_index()
            )
            family_chart = px.bar(
                family_summary,
                x="feature_family",
                y="top_candidate_selected_count",
                color="selected_candidate_features",
                title="Subset Search: Feature Family Selection Pattern",
                labels={
                    "feature_family": "Feature Family",
                    "top_candidate_selected_count": "Top Candidate Appearances",
                },
            )
            context.visualizations["subset_search_feature_family_chart"] = (
                apply_fintech_figure_theme(
                    family_chart,
                    title="Subset Search: Feature Family Selection Pattern",
                )
            )

        prediction_frame = ranking_predictions.get(selected_candidate_id)
        candidate_meta = ranking_metadata.get(selected_candidate_id, {})
        if (
            prediction_frame is None
            or context.target_column is None
            or context.config.target.mode != TargetMode.BINARY
            or "predicted_probability" not in prediction_frame.columns
        ):
            return
        aligned_curve = prediction_frame[
            [context.target_column, "predicted_probability"]
        ].dropna()
        if aligned_curve.empty or aligned_curve[context.target_column].nunique() < 2:
            return
        y_true = aligned_curve[context.target_column].astype(int).to_numpy()
        scores = aligned_curve["predicted_probability"].to_numpy(dtype=float)
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
            "ranking_accuracy",
            "ranking_f1_score",
            "ranking_macro_f1",
            "ranking_weighted_f1",
            "ranking_brier_score",
            "ranking_log_loss",
            "ranking_rmse",
            "ranking_mae",
            "ranking_r2",
            "ranking_explained_variance",
            "ranking_metric",
            "ranking_value",
            "ranking_calibration_error",
            "overall_selection_score",
            "test_roc_auc",
            "test_ks_statistic",
            "test_average_precision",
            "test_accuracy",
            "test_f1_score",
            "test_macro_f1",
            "test_weighted_f1",
            "test_brier_score",
            "test_log_loss",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_explained_variance",
            "test_calibration_error",
        ]
        selected = pd.DataFrame([candidate_row.to_dict()])
        metric_columns = [
            column_name
            for column_name in selected.columns
            if column_name.startswith(("ranking_", "test_"))
            and column_name not in column_order
        ]
        resolved_columns = [
            column_name
            for column_name in [*column_order, *sorted(metric_columns)]
            if column_name in selected.columns
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
            "ranking_accuracy",
            "ranking_f1_score",
            "ranking_macro_f1",
            "ranking_weighted_f1",
            "ranking_brier_score",
            "ranking_log_loss",
            "ranking_rmse",
            "ranking_mae",
            "ranking_r2",
            "ranking_explained_variance",
            "ranking_metric",
            "ranking_value",
            "ranking_calibration_error",
            "overall_selection_score",
            "test_roc_auc",
            "test_ks_statistic",
            "test_average_precision",
            "test_accuracy",
            "test_f1_score",
            "test_macro_f1",
            "test_weighted_f1",
            "test_brier_score",
            "test_log_loss",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_explained_variance",
            "test_calibration_error",
        ]
        metric_columns = [
            column_name
            for column_name in candidates.columns
            if column_name.startswith(("ranking_", "test_"))
            and column_name not in column_order
        ]
        resolved_columns = [
            column_name
            for column_name in [*column_order, *sorted(metric_columns)]
            if column_name in candidates.columns
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
