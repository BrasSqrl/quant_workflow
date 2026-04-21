"""Fits optional challenger models for development-time model comparison."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from ..base import BasePipelineStep
from ..config import ExecutionMode, TargetMode
from ..context import PipelineContext
from ..models import build_model_adapter
from .evaluation import EvaluationStep


class ModelComparisonStep(BasePipelineStep):
    """Trains configured challengers on the same splits and compares held-out metrics."""

    name = "model_comparison"

    def run(self, context: PipelineContext) -> PipelineContext:
        comparison_config = context.config.comparison
        labels_available = bool(context.metadata.get("labels_available", False))
        if (
            not comparison_config.enabled
            or not comparison_config.challenger_model_types
            or context.config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL
        ):
            return context
        if not labels_available:
            context.warn(
                "Skipped model comparison because held-out labels are unavailable for ranking."
            )
            return context

        train_frame = context.split_frames.get("train")
        if train_frame is None or context.target_column is None:
            raise ValueError("Model comparison requires train data and a target column.")

        evaluator = EvaluationStep()
        target_mode = context.config.target.mode
        ranking_metric = comparison_config.ranking_metric or (
            "roc_auc" if target_mode == TargetMode.BINARY else "rmse"
        )
        rows: list[dict[str, object]] = []
        prediction_snapshots: dict[str, pd.DataFrame] = {}

        for split_name in ["validation", "test"]:
            if split_name in context.metrics:
                rows.append(
                    {
                        "model_type": context.config.model.model_type.value,
                        "split": split_name,
                        "is_primary": True,
                        **context.metrics[split_name],
                    }
                )
        primary_snapshot = context.predictions.get(comparison_config.ranking_split)
        if primary_snapshot is not None:
            prediction_snapshots[context.config.model.model_type.value] = primary_snapshot.copy(
                deep=True
            )

        x_train = train_frame[context.feature_columns]
        y_train = train_frame[context.target_column]
        for challenger_type in comparison_config.challenger_model_types:
            challenger_config = deepcopy(context.config.model)
            challenger_config.model_type = challenger_type
            challenger = build_model_adapter(
                challenger_config,
                target_mode,
                scorecard_config=context.config.scorecard,
                scorecard_bin_overrides={
                    override.feature_name: override.bin_edges
                    for override in context.config.manual_review.scorecard_bin_overrides
                },
            )
            challenger.fit(
                x_train,
                y_train,
                context.numeric_features,
                context.categorical_features,
            )
            for split_name, frame in context.split_frames.items():
                if split_name not in {"validation", "test"}:
                    continue
                if target_mode == TargetMode.BINARY:
                    scored_frame, metrics = evaluator._score_binary_split(
                        frame,
                        split_name,
                        context.target_column,
                        context.feature_columns,
                        challenger,
                        challenger_config.threshold,
                        True,
                    )
                else:
                    scored_frame, metrics = evaluator._score_continuous_split(
                        frame,
                        split_name,
                        context.target_column,
                        context.feature_columns,
                        challenger,
                        True,
                    )
                if split_name == comparison_config.ranking_split:
                    prediction_snapshots[challenger_type.value] = scored_frame.copy(deep=True)
                rows.append(
                    {
                        "model_type": challenger_type.value,
                        "split": split_name,
                        "is_primary": False,
                        **metrics,
                    }
                )

        comparison_table = pd.DataFrame(rows)
        if comparison_table.empty:
            return context
        if ranking_metric not in comparison_table.columns:
            ranking_metric = "roc_auc" if target_mode == TargetMode.BINARY else "rmse"

        comparison_table["ranking_metric"] = ranking_metric
        comparison_table["ranking_value"] = comparison_table[ranking_metric]
        ranking_frame = comparison_table.loc[
            comparison_table["split"] == comparison_config.ranking_split
        ].dropna(subset=["ranking_value"])
        if not ranking_frame.empty:
            lower_is_better = ranking_metric in {"rmse", "mae", "brier_score", "log_loss"}
            best_row = ranking_frame.sort_values(
                "ranking_value",
                ascending=lower_is_better,
            ).iloc[0]
            context.metadata["comparison_recommended_model"] = str(best_row["model_type"])
            if str(best_row["model_type"]) != context.config.model.model_type.value:
                context.warn(
                    "A challenger outperformed the primary model on the "
                    f"{comparison_config.ranking_split} split using {ranking_metric}: "
                    f"{best_row['model_type']}."
                )

        context.comparison_results = comparison_table
        if prediction_snapshots:
            context.metadata["comparison_prediction_snapshots"] = prediction_snapshots
        return context
