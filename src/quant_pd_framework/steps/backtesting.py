"""Builds a simple score-band performance table on an evaluation split."""

from __future__ import annotations

import pandas as pd

from ..base import BasePipelineStep
from ..config import TargetMode
from ..context import PipelineContext


class BacktestStep(BasePipelineStep):
    """
    Summarizes holdout performance by score band when a holdout split exists.

    For binary models, comparing predicted probability bands against observed
    event rates is a useful first-pass backtesting and calibration check.
    """

    name = "backtesting"

    def run(self, context: PipelineContext) -> PipelineContext:
        backtest_split = self._select_backtest_split(context)
        if backtest_split is None:
            raise ValueError("Backtesting requires at least one scored split.")
        if context.target_column is None:
            raise ValueError("Backtesting requires a target column.")

        scored_test = context.predictions[backtest_split].copy(deep=False)
        context.metadata["backtest_split"] = backtest_split
        labels_available = bool(context.metadata.get("labels_available", False)) and (
            context.target_column in scored_test.columns
        )
        score_column = (
            "predicted_probability"
            if context.config.target.mode == TargetMode.BINARY
            else "predicted_value"
        )
        band_count = min(
            context.config.diagnostics.quantile_bucket_count,
            max(1, len(scored_test)),
        )
        scored_test["risk_band"] = (
            pd.qcut(
                scored_test[score_column].rank(method="first"),
                q=band_count,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

        if context.config.target.mode == TargetMode.BINARY and labels_available:
            summary = (
                scored_test.groupby("risk_band", dropna=False)
                .agg(
                    observation_count=(score_column, "size"),
                    default_count=(context.target_column, "sum"),
                    average_predicted_pd=(score_column, "mean"),
                    observed_default_rate=(context.target_column, "mean"),
                    min_predicted_pd=(score_column, "min"),
                    max_predicted_pd=(score_column, "max"),
                )
                .reset_index()
                .sort_values("risk_band", ascending=False)
            )
        elif context.config.target.mode == TargetMode.BINARY:
            summary = (
                scored_test.groupby("risk_band", dropna=False)
                .agg(
                    observation_count=(score_column, "size"),
                    average_predicted_pd=(score_column, "mean"),
                    min_predicted_pd=(score_column, "min"),
                    max_predicted_pd=(score_column, "max"),
                )
                .reset_index()
                .sort_values("risk_band", ascending=False)
            )
        elif labels_available:
            summary = (
                scored_test.groupby("risk_band", dropna=False)
                .agg(
                    observation_count=(score_column, "size"),
                    average_predicted_value=(score_column, "mean"),
                    observed_average=(context.target_column, "mean"),
                    min_predicted_value=(score_column, "min"),
                    max_predicted_value=(score_column, "max"),
                )
                .reset_index()
                .sort_values("risk_band", ascending=False)
            )
        else:
            summary = (
                scored_test.groupby("risk_band", dropna=False)
                .agg(
                    observation_count=(score_column, "size"),
                    average_predicted_value=(score_column, "mean"),
                    min_predicted_value=(score_column, "min"),
                    max_predicted_value=(score_column, "max"),
                )
                .reset_index()
                .sort_values("risk_band", ascending=False)
            )

        context.backtest_summary = summary
        return context

    def _select_backtest_split(self, context: PipelineContext) -> str | None:
        for split_name in ("test", "validation", "train"):
            if split_name in context.predictions and not context.predictions[split_name].empty:
                return split_name
        return None
