"""Optional fold-based validation diagnostics for fitted workflows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ..base import BasePipelineStep
from ..config import (
    CrossValidationStrategy,
    DataStructure,
    ExecutionMode,
    TargetMode,
)
from ..context import PipelineContext
from ..models import build_model_adapter
from ..presentation import apply_fintech_figure_theme
from .evaluation import EvaluationStep


class CrossValidationStep(BasePipelineStep):
    """
    Runs optional fold-based validation diagnostics without replacing the final model.

    The production artifact remains the model fitted by ``ModelTrainingStep`` on
    the configured training split. This step fits temporary fold models only to
    assess metric and feature stability.
    """

    name = "cross_validation"

    def run(self, context: PipelineContext) -> PipelineContext:
        cross_validation = context.config.cross_validation
        if not cross_validation.enabled:
            return context
        if context.config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
            context.warn(
                "Skipped cross-validation because existing-model scoring does not refit "
                "fold-specific models."
            )
            return context
        if context.target_column is None or not bool(context.metadata.get("labels_available")):
            context.warn("Skipped cross-validation because labeled targets are unavailable.")
            return context

        train_frame = context.split_frames.get("train")
        if train_frame is None or train_frame.empty:
            context.warn("Skipped cross-validation because the train split is unavailable.")
            return context
        if len(train_frame) < 4:
            context.warn("Skipped cross-validation because the train split is too small.")
            return context

        working_train = self._ordered_training_frame(context, train_frame)
        try:
            folds, validation_method = self._build_fold_indices(context, working_train)
        except ValueError as exc:
            context.warn(f"Skipped cross-validation: {exc}")
            return context

        evaluator = EvaluationStep()
        fold_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        feature_rows: list[dict[str, Any]] = []
        successful_folds = 0

        for fold_id, (train_index, validation_index) in enumerate(folds, start=1):
            fold_train = working_train.iloc[train_index].reset_index(drop=True)
            fold_validation = working_train.iloc[validation_index].reset_index(drop=True)
            if not self._fold_has_valid_training_target(context, fold_train, fold_id):
                continue

            try:
                fold_model = self._fit_fold_model(context, fold_train)
            except Exception as exc:
                context.warn(f"Cross-validation fold {fold_id} failed during fit: {exc}")
                continue

            try:
                metrics = self._score_fold(
                    context=context,
                    evaluator=evaluator,
                    fold_validation=fold_validation,
                    fold_model=fold_model,
                )
            except Exception as exc:
                context.warn(f"Cross-validation fold {fold_id} failed during scoring: {exc}")
                continue

            successful_folds += 1
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "validation_method": validation_method,
                    "train_rows": int(len(fold_train)),
                    "validation_rows": int(len(fold_validation)),
                    **metrics,
                }
            )
            if cross_validation.metric_stability:
                metric_rows.extend(
                    self._build_metric_rows(
                        fold_id=fold_id,
                        validation_method=validation_method,
                        metrics=metrics,
                        target_mode=context.config.target.mode,
                    )
                )
            if cross_validation.coefficient_stability:
                feature_rows.extend(
                    self._build_feature_rows(
                        fold_id=fold_id,
                        feature_importance=fold_model.get_feature_importance(),
                    )
                )

        if successful_folds < 2:
            context.warn(
                "Cross-validation did not complete enough successful folds to build "
                "stability diagnostics."
            )
            return context

        self._publish_outputs(
            context=context,
            fold_rows=fold_rows,
            metric_rows=metric_rows,
            feature_rows=feature_rows,
            validation_method=validation_method,
            successful_folds=successful_folds,
        )
        return context

    def _ordered_training_frame(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if context.config.split.data_structure == DataStructure.CROSS_SECTIONAL:
            return train_frame.reset_index(drop=True)

        date_column = context.config.split.date_column
        if date_column and date_column in train_frame.columns:
            working = train_frame.copy(deep=True)
            working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
            return working.sort_values(date_column).reset_index(drop=True)
        return train_frame.reset_index(drop=True)

    def _build_fold_indices(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
        strategy = self._resolve_strategy(context)
        requested_folds = context.config.cross_validation.fold_count

        if strategy == CrossValidationStrategy.TIME_SERIES:
            effective_folds = min(requested_folds, len(train_frame) - 1)
            if effective_folds < 2:
                raise ValueError("time-aware cross-validation requires at least 3 train rows.")
            splitter = TimeSeriesSplit(n_splits=effective_folds)
            return list(splitter.split(train_frame)), "time_series_expanding_window"

        if strategy == CrossValidationStrategy.STRATIFIED_KFOLD:
            target = train_frame[context.target_column].astype(int)
            class_counts = target.value_counts(dropna=True)
            if class_counts.empty or int(class_counts.min()) < 2:
                raise ValueError(
                    "stratified k-fold cross-validation requires at least two rows per class."
                )
            effective_folds = min(requested_folds, int(class_counts.min()))
            splitter = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=context.config.cross_validation.shuffle,
                random_state=(
                    context.config.cross_validation.random_state
                    if context.config.cross_validation.shuffle
                    else None
                ),
            )
            return list(splitter.split(train_frame, target)), "stratified_kfold"

        effective_folds = min(requested_folds, len(train_frame))
        if effective_folds < 2:
            raise ValueError("k-fold cross-validation requires at least two train rows.")
        splitter = KFold(
            n_splits=effective_folds,
            shuffle=context.config.cross_validation.shuffle,
            random_state=(
                context.config.cross_validation.random_state
                if context.config.cross_validation.shuffle
                else None
            ),
        )
        return list(splitter.split(train_frame)), "kfold"

    def _resolve_strategy(self, context: PipelineContext) -> CrossValidationStrategy:
        configured = context.config.cross_validation.strategy
        if configured != CrossValidationStrategy.AUTO:
            return configured
        if context.config.split.data_structure != DataStructure.CROSS_SECTIONAL:
            return CrossValidationStrategy.TIME_SERIES
        if context.config.target.mode == TargetMode.BINARY:
            return CrossValidationStrategy.STRATIFIED_KFOLD
        return CrossValidationStrategy.KFOLD

    def _fold_has_valid_training_target(
        self,
        context: PipelineContext,
        fold_train: pd.DataFrame,
        fold_id: int,
    ) -> bool:
        if context.config.target.mode != TargetMode.BINARY:
            return True
        if fold_train[context.target_column].nunique(dropna=True) >= 2:
            return True
        context.warn(
            f"Skipped cross-validation fold {fold_id} because the fold train sample "
            "contains only one target class."
        )
        return False

    def _fit_fold_model(self, context: PipelineContext, fold_train: pd.DataFrame):
        fold_model = build_model_adapter(
            deepcopy(context.config.model),
            context.config.target.mode,
            scorecard_config=context.config.scorecard,
            scorecard_bin_overrides={
                override.feature_name: override.bin_edges
                for override in context.config.manual_review.scorecard_bin_overrides
            },
        )
        fold_model.fit(
            fold_train[context.feature_columns],
            fold_train[context.target_column],
            context.numeric_features,
            context.categorical_features,
        )
        return fold_model

    def _score_fold(
        self,
        *,
        context: PipelineContext,
        evaluator: EvaluationStep,
        fold_validation: pd.DataFrame,
        fold_model,
    ) -> dict[str, float | int | None]:
        if context.config.target.mode == TargetMode.BINARY:
            _, metrics = evaluator._score_binary_split(
                fold_validation,
                "cross_validation",
                context.target_column,
                context.feature_columns,
                fold_model,
                context.config.model.threshold,
                True,
            )
            return metrics

        _, metrics = evaluator._score_continuous_split(
            fold_validation,
            "cross_validation",
            context.target_column,
            context.feature_columns,
            fold_model,
            True,
        )
        return metrics

    def _build_metric_rows(
        self,
        *,
        fold_id: int,
        validation_method: str,
        metrics: dict[str, float | int | None],
        target_mode: TargetMode,
    ) -> list[dict[str, Any]]:
        metric_names = (
            [
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
            if target_mode == TargetMode.BINARY
            else ["rmse", "mae", "r2", "explained_variance"]
        )
        rows: list[dict[str, Any]] = []
        for metric_name in metric_names:
            metric_value = metrics.get(metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            rows.append(
                {
                    "fold_id": fold_id,
                    "validation_method": validation_method,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                }
            )
        return rows

    def _build_feature_rows(
        self,
        *,
        fold_id: int,
        feature_importance: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        if feature_importance.empty or "feature_name" not in feature_importance.columns:
            return []

        rows: list[dict[str, Any]] = []
        for _, row in feature_importance.iterrows():
            feature_name = str(row["feature_name"])
            if "coefficient" in feature_importance.columns and pd.notna(row.get("coefficient")):
                effect_value = float(row["coefficient"])
                effect_basis = "coefficient"
            else:
                effect_value = float(row.get("importance_value", 0.0) or 0.0)
                effect_basis = "importance_value"
            rows.append(
                {
                    "fold_id": fold_id,
                    "feature_name": feature_name,
                    "effect_basis": effect_basis,
                    "effect_value": effect_value,
                    "abs_effect_value": abs(effect_value),
                    "selected_in_fold": abs(effect_value) > 0,
                }
            )
        return rows

    def _publish_outputs(
        self,
        *,
        context: PipelineContext,
        fold_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
        feature_rows: list[dict[str, Any]],
        validation_method: str,
        successful_folds: int,
    ) -> None:
        fold_table = pd.DataFrame(fold_rows)
        context.diagnostics_tables["cross_validation_fold_metrics"] = fold_table
        context.statistical_tests["cross_validation"] = {
            "validation_method": validation_method,
            "requested_folds": context.config.cross_validation.fold_count,
            "successful_folds": successful_folds,
        }

        if metric_rows:
            metric_table = pd.DataFrame(metric_rows)
            metric_summary = self._summarize_metrics(metric_table)
            context.diagnostics_tables["cross_validation_metric_distribution"] = metric_table
            context.diagnostics_tables["cross_validation_metric_summary"] = metric_summary
            context.visualizations["cross_validation_metric_boxplot"] = (
                apply_fintech_figure_theme(
                    px.box(
                        metric_table,
                        x="metric_name",
                        y="metric_value",
                        color="metric_name",
                        points="all",
                        title="Cross-Validation Metric Distribution",
                        labels={
                            "metric_name": "Metric",
                            "metric_value": "Fold Validation Value",
                        },
                    )
                )
            )
            context.visualizations["cross_validation_metric_summary_chart"] = (
                apply_fintech_figure_theme(
                    px.bar(
                        metric_summary,
                        x="metric_name",
                        y="mean_value",
                        error_y="std_value",
                        title="Cross-Validation Metric Summary",
                        labels={
                            "metric_name": "Metric",
                            "mean_value": "Average Fold Value",
                        },
                    )
                )
            )

        if feature_rows:
            feature_table = pd.DataFrame(feature_rows)
            feature_summary = self._summarize_features(feature_table, successful_folds)
            context.diagnostics_tables["cross_validation_feature_distribution"] = feature_table
            context.diagnostics_tables["cross_validation_feature_stability"] = feature_summary
            chart_frame = (
                feature_summary.sort_values("mean_abs_effect", ascending=False)
                .head(context.config.diagnostics.top_n_features)
                .sort_values("mean_abs_effect", ascending=True)
            )
            if not chart_frame.empty:
                context.visualizations["cross_validation_feature_stability"] = (
                    apply_fintech_figure_theme(
                        px.bar(
                            chart_frame,
                            x="mean_abs_effect",
                            y="feature_name",
                            orientation="h",
                            error_x="std_abs_effect",
                            color="effect_basis",
                            title="Cross-Validation Feature Stability",
                            labels={
                                "mean_abs_effect": "Average Absolute Effect",
                                "feature_name": "Feature",
                                "effect_basis": "Effect Basis",
                            },
                        )
                    )
                )

        context.metadata["cross_validation_summary"] = {
            "enabled": True,
            "validation_method": validation_method,
            "successful_folds": successful_folds,
            "requested_folds": context.config.cross_validation.fold_count,
            "final_model_refit": False,
        }
        context.log(
            "Completed cross-validation diagnostics across "
            f"{successful_folds} folds using {validation_method}."
        )

    def _summarize_metrics(self, metric_table: pd.DataFrame) -> pd.DataFrame:
        grouped = metric_table.groupby("metric_name")["metric_value"]
        return grouped.agg(
            fold_count="count",
            mean_value="mean",
            std_value="std",
            min_value="min",
            median_value="median",
            max_value="max",
        ).reset_index()

    def _summarize_features(
        self,
        feature_table: pd.DataFrame,
        successful_folds: int,
    ) -> pd.DataFrame:
        grouped = feature_table.groupby(["feature_name", "effect_basis"], dropna=False)
        summary = grouped.agg(
            fold_count=("fold_id", "nunique"),
            selection_count=("selected_in_fold", "sum"),
            mean_effect=("effect_value", "mean"),
            std_effect=("effect_value", "std"),
            mean_abs_effect=("abs_effect_value", "mean"),
            std_abs_effect=("abs_effect_value", "std"),
        ).reset_index()
        summary["selection_frequency"] = summary["selection_count"] / max(successful_folds, 1)
        summary["sign_consistency"] = grouped["effect_value"].apply(
            lambda values: self._sign_consistency(values)
        ).reset_index(drop=True)
        return summary.sort_values("mean_abs_effect", ascending=False).reset_index(drop=True)

    def _sign_consistency(self, values: pd.Series) -> float:
        non_zero_signs = np.sign(pd.to_numeric(values, errors="coerce").dropna())
        non_zero_signs = non_zero_signs[non_zero_signs != 0]
        if len(non_zero_signs) == 0:
            return 0.0
        positive_share = float((non_zero_signs > 0).mean())
        negative_share = float((non_zero_signs < 0).mean())
        return max(positive_share, negative_share)
