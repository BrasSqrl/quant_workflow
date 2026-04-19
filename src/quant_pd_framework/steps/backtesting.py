"""Builds a simple score-band performance table on the test set."""

from __future__ import annotations

import pandas as pd

from ..base import BasePipelineStep
from ..config import TargetMode
from ..context import PipelineContext


class BacktestStep(BasePipelineStep):
    """
    Summarizes out-of-sample performance by score band.

    For binary models, comparing predicted probability bands against observed
    event rates is a useful first-pass backtesting and calibration check.
    """

    name = "backtesting"

    def run(self, context: PipelineContext) -> PipelineContext:
        if "test" not in context.predictions:
            raise ValueError("Backtesting requires scored test-set predictions.")
        if context.target_column is None:
            raise ValueError("Backtesting requires a target column.")

        scored_test = context.predictions["test"].copy(deep=True)
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
