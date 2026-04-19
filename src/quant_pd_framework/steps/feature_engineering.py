"""Creates model-ready features and classifies them by type."""

from __future__ import annotations

import pandas as pd

from ..base import BasePipelineStep
from ..config import ColumnRole, ModelType
from ..context import PipelineContext


class FeatureEngineeringStep(BasePipelineStep):
    """
    Creates lightweight derived features and finalizes the model input columns.

    V1 keeps feature engineering intentionally transparent: date expansion is
    useful in many quant settings, while more complex transforms can be added by
    subclassing this step later.
    """

    name = "feature_engineering"

    def run(self, context: PipelineContext) -> PipelineContext:
        dataframe = context.working_data
        target_column = context.target_column
        if dataframe is None or target_column is None:
            raise ValueError("Feature engineering requires a dataframe and target column.")

        working = dataframe.copy(deep=True)
        config = context.config.feature_engineering
        split_config = context.config.split

        ignored_columns = {target_column, context.metadata.get("target_source_column")}
        identifier_columns = {
            spec.name
            for spec in context.config.schema.column_specs
            if spec.role in {ColumnRole.IDENTIFIER, ColumnRole.IGNORE} and spec.enabled
        }
        if (
            context.config.model.model_type == ModelType.PANEL_REGRESSION
            and split_config.entity_column
        ):
            identifier_columns.discard(split_config.entity_column)
        date_columns = {
            spec.name
            for spec in context.config.schema.column_specs
            if spec.role == ColumnRole.DATE and spec.enabled and spec.name in working.columns
        }
        if split_config.date_column and split_config.date_column in working.columns:
            date_columns.add(split_config.date_column)

        for column in sorted(date_columns):
            working[column] = pd.to_datetime(working[column], errors="coerce")
            if config.derive_date_parts:
                self._add_date_parts(working, column, config.date_parts)
            if (
                config.drop_raw_date_columns
                and column in working.columns
                and column != split_config.date_column
            ):
                working = working.drop(columns=column)
                context.dropped_columns.append(column)

        excluded_columns = {column for column in ignored_columns if column}
        excluded_columns.update(identifier_columns)
        excluded_columns.update(column for column in date_columns if column in working.columns)
        feature_columns = [column for column in working.columns if column not in excluded_columns]

        numeric_features: list[str] = []
        categorical_features: list[str] = []
        for column in feature_columns:
            series = working[column]
            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                numeric_features.append(column)
            else:
                categorical_features.append(column)

        if not feature_columns:
            raise ValueError("No feature columns remain after feature engineering.")

        context.working_data = working
        context.feature_columns = feature_columns
        context.numeric_features = numeric_features
        context.categorical_features = categorical_features
        context.metadata["feature_summary"] = {
            "feature_count": len(feature_columns),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
        }
        return context

    def _add_date_parts(
        self,
        dataframe: pd.DataFrame,
        column: str,
        date_parts: list[str],
    ) -> None:
        accessor = dataframe[column].dt
        for part in date_parts:
            feature_name = f"{column}_{part}"
            if part == "year":
                dataframe[feature_name] = accessor.year.astype("Int64")
            elif part == "month":
                dataframe[feature_name] = accessor.month.astype("Int64")
            elif part == "quarter":
                dataframe[feature_name] = accessor.quarter.astype("Int64")
            elif part == "day":
                dataframe[feature_name] = accessor.day.astype("Int64")
            elif part == "dayofweek":
                dataframe[feature_name] = accessor.dayofweek.astype("Int64")
            else:
                raise ValueError(
                    f"Unsupported date part '{part}'. Add a handler in FeatureEngineeringStep."
                )
