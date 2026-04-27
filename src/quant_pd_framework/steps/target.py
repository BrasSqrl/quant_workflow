"""Builds the configured target column used by the active model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import BasePipelineStep
from ..config import ExecutionMode, TargetMode
from ..context import PipelineContext


class TargetConstructionStep(BasePipelineStep):
    """
    Converts the user-selected target source into the configured modeling target.

    This step isolates label construction because target logic is often one of
    the first things that changes between quantitative modeling projects.
    """

    name = "target_construction"

    def run(self, context: PipelineContext) -> PipelineContext:
        dataframe = context.working_data
        if dataframe is None:
            raise ValueError("Target construction requires a dataframe in the context.")

        source_column = context.config.target.source_column
        output_column = context.config.target.output_column

        if source_column not in dataframe.columns:
            if context.config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
                context.target_column = output_column
                context.metadata["target_source_column"] = source_column
                context.metadata["target_mode"] = context.config.target.mode.value
                context.metadata["labels_available"] = False
                context.warn(
                    f"Target source column '{source_column}' is missing from the dataframe. "
                    "The run will continue in score-only mode without label-based metrics."
                )
                return context
            raise ValueError(
                f"Target source column '{source_column}' is missing from the dataframe."
            )

        if context.config.target.mode == TargetMode.BINARY:
            target_series = self._build_binary_target(
                dataframe[source_column],
                context.config.target.positive_values,
            ).astype("Int64")
        else:
            target_series = pd.to_numeric(dataframe[source_column], errors="coerce").astype(
                "float64"
            )

        working = dataframe.copy(deep=False)
        working[output_column] = target_series

        if context.config.target.drop_source_column and source_column != output_column:
            working = working.drop(columns=source_column)
            context.dropped_columns.append(source_column)

        context.working_data = working
        context.target_column = output_column
        context.metadata["target_source_column"] = source_column
        context.metadata["target_mode"] = context.config.target.mode.value
        context.metadata["labels_available"] = bool(working[output_column].notna().any())
        context.metadata["target_distribution"] = (
            working[output_column].value_counts(dropna=False).to_dict()
        )
        return context

    def _build_binary_target(
        self,
        source: pd.Series,
        positive_values: list[Any] | None,
    ) -> pd.Series:
        if positive_values:
            normalized_positive_values = {self._normalize_value(value) for value in positive_values}
            return source.map(
                lambda value: (
                    1
                    if self._normalize_value(value) in normalized_positive_values
                    else (pd.NA if pd.isna(value) else 0)
                )
            )

        unique_values = {self._normalize_value(value) for value in source.dropna().unique()}
        if unique_values.issubset({"0", "1"}):
            return pd.to_numeric(source, errors="coerce")
        if unique_values.issubset({"false", "true"}):
            return source.map(lambda value: 1 if self._normalize_value(value) == "true" else 0)

        raise ValueError(
            "The target source is not already binary. Provide TargetConfig.positive_values "
            "to identify which source values should map to the positive class."
        )

    def _normalize_value(self, value: Any) -> str | None:
        if pd.isna(value):
            return None
        return str(value).strip().lower()
