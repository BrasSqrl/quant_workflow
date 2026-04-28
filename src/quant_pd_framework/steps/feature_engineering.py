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

        working = dataframe.copy(deep=False)
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

        created_date_part_features: list[str] = []
        removed_raw_date_columns: list[str] = []
        retained_raw_date_columns: list[str] = []
        for column in sorted(date_columns):
            working[column] = pd.to_datetime(working[column], errors="coerce")
            if config.derive_date_parts:
                created_date_part_features.extend(
                    self._add_date_parts(working, column, config.date_parts)
                )
            if (
                config.drop_raw_date_columns
                and column in working.columns
                and column != split_config.date_column
            ):
                working = working.drop(columns=column)
                context.dropped_columns.append(column)
                removed_raw_date_columns.append(column)
            elif column in working.columns:
                retained_raw_date_columns.append(column)

        if context.config.model.model_type == ModelType.DISCRETE_TIME_HAZARD_MODEL:
            self._add_hazard_time_features(working, split_config, context)

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

        self._profile_categorical_cardinality(context, working, categorical_features)
        context.working_data = working
        context.feature_columns = feature_columns
        context.numeric_features = numeric_features
        context.categorical_features = categorical_features
        context.metadata["feature_summary"] = {
            "feature_count": len(feature_columns),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
        }
        context.metadata["date_feature_engineering"] = {
            "derive_date_parts": bool(config.derive_date_parts),
            "date_parts": list(config.date_parts),
            "created_date_part_features": created_date_part_features,
            "drop_raw_date_columns": bool(config.drop_raw_date_columns),
            "removed_raw_date_columns": removed_raw_date_columns,
            "retained_raw_date_columns": retained_raw_date_columns,
            "raw_date_columns_excluded_from_model_features": sorted(
                column for column in date_columns if column not in feature_columns
            ),
        }
        return context

    def _profile_categorical_cardinality(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
        categorical_features: list[str],
    ) -> None:
        if not categorical_features:
            return
        performance = context.config.performance
        rows: list[dict[str, object]] = []
        high_cardinality_features: list[str] = []
        row_count = max(int(len(dataframe)), 1)
        for feature_name in categorical_features:
            series = dataframe[feature_name]
            unique_count = int(series.nunique(dropna=True))
            unique_ratio = unique_count / row_count
            exceeds_threshold = (
                unique_count > performance.max_categorical_cardinality
                and unique_ratio > performance.max_categorical_cardinality_ratio
            )
            if exceeds_threshold:
                high_cardinality_features.append(feature_name)
            rows.append(
                {
                    "feature_name": feature_name,
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                    "max_allowed_unique_count": performance.max_categorical_cardinality,
                    "max_allowed_unique_ratio": performance.max_categorical_cardinality_ratio,
                    "status": "high_cardinality" if exceeds_threshold else "ok",
                }
            )

        profile = pd.DataFrame(rows).sort_values(
            ["status", "unique_count"],
            ascending=[True, False],
            kind="stable",
        )
        context.diagnostics_tables["categorical_cardinality_profile"] = profile
        context.metadata["categorical_cardinality_profile"] = {
            "feature_count": len(categorical_features),
            "high_cardinality_feature_count": len(high_cardinality_features),
            "high_cardinality_features": high_cardinality_features,
            "max_categorical_cardinality": performance.max_categorical_cardinality,
            "max_categorical_cardinality_ratio": performance.max_categorical_cardinality_ratio,
            "allow_high_cardinality_categoricals": (
                performance.allow_high_cardinality_categoricals
            ),
        }
        if not high_cardinality_features:
            return

        message = (
            "High-cardinality categorical model features were detected: "
            + ", ".join(high_cardinality_features[:10])
            + ". Group rare levels, bin/encode the feature compactly, or explicitly allow "
            "high-cardinality categoricals after confirming memory capacity."
        )
        if performance.allow_high_cardinality_categoricals:
            context.warn(message)
            return
        raise ValueError(message)

    def _add_hazard_time_features(
        self,
        dataframe: pd.DataFrame,
        split_config,
        context: PipelineContext,
    ) -> None:
        date_column = split_config.date_column
        if date_column is None or date_column not in dataframe.columns:
            return
        if split_config.entity_column and split_config.entity_column in dataframe.columns:
            ordered = dataframe.sort_values([split_config.entity_column, date_column])
            period_index = ordered.groupby(split_config.entity_column).cumcount() + 1
            dataframe.loc[ordered.index, "hazard_period_index"] = period_index.to_numpy()
        else:
            ordered = dataframe.sort_values(date_column)
            period_index = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
            dataframe.loc[ordered.index, "hazard_period_index"] = period_index.to_numpy()
        dataframe["hazard_period_index"] = pd.to_numeric(
            dataframe["hazard_period_index"],
            errors="coerce",
        ).astype("Int64")
        dataframe["hazard_period_index_sq"] = (
            dataframe["hazard_period_index"].astype(float) ** 2
        )
        context.metadata["hazard_time_features"] = [
            "hazard_period_index",
            "hazard_period_index_sq",
        ]

    def _add_date_parts(
        self,
        dataframe: pd.DataFrame,
        column: str,
        date_parts: list[str],
    ) -> list[str]:
        accessor = dataframe[column].dt
        created_features: list[str] = []
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
            created_features.append(feature_name)
        return created_features
