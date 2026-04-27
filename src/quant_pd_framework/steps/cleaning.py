"""Performs general-purpose data hygiene before feature creation."""

from __future__ import annotations

import pandas as pd

from ..base import BasePipelineStep
from ..config import ColumnRole
from ..context import PipelineContext


class CleaningStep(BasePipelineStep):
    """
    Applies reusable cleaning rules that are common across tabular credit data.

    This is intentionally conservative: heavy transformations belong in feature
    engineering or custom model-specific steps, not in generic cleaning.
    """

    name = "cleaning"

    def run(self, context: PipelineContext) -> PipelineContext:
        dataframe = context.working_data
        target_column = context.target_column
        if dataframe is None or target_column is None:
            raise ValueError("Cleaning requires a dataframe and target column.")
        labels_available = (
            bool(context.metadata.get("labels_available", False))
            and target_column in dataframe.columns
        )

        config = context.config.cleaning
        working = dataframe.copy(deep=False)

        # Standardize blank strings early so missing-value logic sees a single null marker.
        if config.blank_strings_as_null:
            working = working.replace(r"^\s*$", pd.NA, regex=True)

        if config.trim_string_columns:
            object_columns = working.select_dtypes(include=["object", "string", "category"]).columns
            for column in object_columns:
                working[column] = working[column].map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )

        if config.drop_duplicate_rows:
            before = len(working)
            working = working.drop_duplicates()
            context.metadata["duplicate_rows_removed"] = int(before - len(working))

        if config.drop_rows_with_missing_target and labels_available:
            before = len(working)
            working = working.loc[working[target_column].notna()].copy()
            context.metadata["rows_removed_missing_target"] = int(before - len(working))
        elif config.drop_rows_with_missing_target:
            context.metadata["rows_removed_missing_target"] = 0

        if config.drop_all_null_feature_columns:
            protected_columns = {target_column}
            for spec in context.config.schema.column_specs:
                if spec.role == ColumnRole.IDENTIFIER and spec.enabled:
                    protected_columns.add(spec.name)

            drop_candidates = [
                column
                for column in working.columns
                if column not in protected_columns and working[column].isna().all()
            ]
            if drop_candidates:
                working = working.drop(columns=drop_candidates)
                context.dropped_columns.extend(drop_candidates)
                context.warn(
                    "Dropped columns that were entirely null after cleaning: "
                    + ", ".join(sorted(drop_candidates))
                )

        context.working_data = working
        context.metadata["post_clean_shape"] = {
            "rows": int(working.shape[0]),
            "columns": int(working.shape[1]),
        }
        return context
