"""Applies user-controlled column toggles, renames, and dtype coercions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import BasePipelineStep
from ..config import ColumnSpec
from ..context import PipelineContext


class SchemaManagementStep(BasePipelineStep):
    """
    Standardizes the incoming schema so the rest of the pipeline has stable inputs.

    This is where the user can turn columns on or off, create new columns,
    rename columns, and enforce data types before modeling starts.
    """

    name = "schema_management"

    def run(self, context: PipelineContext) -> PipelineContext:
        dataframe = context.working_data
        if dataframe is None:
            raise ValueError("Schema management requires a dataframe in the context.")

        working = dataframe.copy(deep=False)
        dropped_columns: list[str] = []
        role_map: dict[str, str] = {}

        for spec in context.config.schema.column_specs:
            working, dropped = self._apply_column_spec(working, spec, context)
            dropped_columns.extend(dropped)
            role_map[spec.name] = spec.role.value

        if not context.config.schema.pass_through_unconfigured_columns:
            allowed = {spec.name for spec in context.config.schema.column_specs if spec.enabled}
            working = working.loc[:, [column for column in working.columns if column in allowed]]

        context.working_data = working
        context.dropped_columns.extend(sorted(set(dropped_columns)))
        context.metadata["column_roles"] = role_map
        context.metadata["post_schema_columns"] = list(working.columns)
        return context

    def _apply_column_spec(
        self,
        dataframe: pd.DataFrame,
        spec: ColumnSpec,
        context: PipelineContext,
    ) -> tuple[pd.DataFrame, list[str]]:
        working = dataframe
        dropped_columns: list[str] = []
        source_name = spec.source_name or spec.name
        source_exists = source_name in working.columns
        output_exists = spec.name in working.columns

        if not spec.enabled:
            for column_name in {source_name, spec.name}:
                if column_name in working.columns:
                    working = working.drop(columns=column_name)
                    dropped_columns.append(column_name)
            return working, dropped_columns

        if source_exists and spec.name != source_name:
            # This behaves like a configurable rename while still allowing the
            # user to keep the original source column when desired.
            working[spec.name] = working[source_name]
            if not spec.keep_source:
                working = working.drop(columns=source_name)
                dropped_columns.append(source_name)
        elif not source_exists and spec.create_if_missing:
            working[spec.name] = spec.default_value
        elif not source_exists and not output_exists:
            raise ValueError(
                f"Configured column '{spec.name}' does not exist in the input data and "
                "was not marked with create_if_missing=True."
            )

        if spec.dtype:
            working[spec.name] = self._cast_series(working[spec.name], spec.dtype)

        return working, dropped_columns

    def _cast_series(self, series: pd.Series, dtype: str) -> pd.Series:
        normalized = dtype.lower()

        if normalized in {"string", "str", "text"}:
            return series.astype("string")
        if normalized in {"category", "categorical"}:
            return series.astype("category")
        if normalized in {"float", "float64", "double"}:
            return pd.to_numeric(series, errors="coerce").astype("float64")
        if normalized in {"int", "int64", "integer"}:
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if normalized in {"bool", "boolean"}:
            return self._coerce_boolean_series(series)
        if normalized in {"datetime", "datetime64", "date"}:
            return pd.to_datetime(series, errors="coerce")

        raise ValueError(
            f"Unsupported dtype '{dtype}'. Use one of string/category/float/int/bool/datetime."
        )

    def _coerce_boolean_series(self, series: pd.Series) -> pd.Series:
        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n"}

        def convert(value: Any) -> Any:
            if pd.isna(value):
                return pd.NA
            text = str(value).strip().lower()
            if text in truthy:
                return True
            if text in falsy:
                return False
            return pd.NA

        return series.map(convert).astype("boolean")
