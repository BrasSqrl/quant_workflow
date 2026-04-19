"""Checks that the standardized dataset is valid for the selected model flow."""

from __future__ import annotations

import pandas as pd

from ..base import BasePipelineStep
from ..config import ColumnRole, DataStructure, ExecutionMode, ModelType, TargetMode
from ..context import PipelineContext


class ValidationStep(BasePipelineStep):
    """
    Performs pre-model quality gates.

    Quant workflows fail expensively when assumptions are wrong, so this step
    checks target quality, required split fields, and basic dataframe integrity.
    """

    name = "validation"

    def run(self, context: PipelineContext) -> PipelineContext:
        dataframe = context.working_data
        target_column = context.target_column
        execution_mode = context.config.execution.mode

        if dataframe is None or target_column is None:
            raise ValueError("Validation requires both a dataframe and target column.")
        if dataframe.empty:
            raise ValueError("No rows remain after preprocessing.")
        labels_available = (
            bool(context.metadata.get("labels_available", False))
            and target_column in dataframe.columns
        )

        target_mode = context.config.target.mode
        model_type = context.config.model.model_type
        if target_mode == TargetMode.BINARY and model_type == ModelType.TOBIT_REGRESSION:
            raise ValueError(
                "Tobit regression is only supported for continuous censored "
                "targets, not binary PD labels."
            )
        if target_mode == TargetMode.CONTINUOUS and model_type in {
            ModelType.LOGISTIC_REGRESSION,
            ModelType.PROBIT_REGRESSION,
        }:
            raise ValueError(f"{model_type.value} requires a binary target.")

        if not labels_available:
            if execution_mode == ExecutionMode.FIT_NEW_MODEL:
                raise ValueError(f"Target column '{target_column}' is missing from the dataframe.")
            context.metadata["labels_available"] = False
            context.metadata["target_unique_values"] = 0
        else:
            non_null_target = dataframe[target_column].dropna()
            if non_null_target.empty:
                if execution_mode == ExecutionMode.FIT_NEW_MODEL:
                    raise ValueError(
                        "The target column does not contain any non-null observations."
                    )
                context.metadata["labels_available"] = False
                context.metadata["target_unique_values"] = 0
                context.warn(
                    "The target column does not contain any non-null observations. "
                    "The run will continue in score-only mode."
                )
            else:
                unique_count = int(non_null_target.nunique())
                context.metadata["target_unique_values"] = unique_count
                if target_mode == TargetMode.BINARY and unique_count > 2:
                    raise ValueError(
                        "Binary target mode requires exactly two distinct target values."
                    )
                if target_mode == TargetMode.CONTINUOUS and not pd.api.types.is_numeric_dtype(
                    non_null_target
                ):
                    raise ValueError("Continuous target mode requires a numeric target column.")
                if unique_count < 2:
                    if execution_mode == ExecutionMode.FIT_NEW_MODEL:
                        raise ValueError(
                            "The target column must contain at least two distinct values."
                        )
                    context.warn(
                        "The target column contains fewer than two distinct non-null values. "
                        "Score distributions and score-only diagnostics will "
                        "still run, but class-separation "
                        "metrics and some plots may be unavailable."
                    )
                if model_type in {
                    ModelType.BETA_REGRESSION,
                    ModelType.TWO_STAGE_LGD_MODEL,
                }:
                    min_value = float(non_null_target.min())
                    max_value = float(non_null_target.max())
                    if min_value < 0 or max_value > 1:
                        raise ValueError(
                            f"{model_type.value} requires target values bounded within [0, 1]."
                        )

        split_config = context.config.split
        date_column = split_config.date_column or self._infer_role_column(context, ColumnRole.DATE)
        if split_config.data_structure in {DataStructure.TIME_SERIES, DataStructure.PANEL}:
            if not date_column:
                raise ValueError(
                    "Time-series and panel workflows require a date column in "
                    "SplitConfig.date_column "
                    "or a schema column marked with role=DATE."
                )
            if date_column not in dataframe.columns:
                raise ValueError(
                    f"Configured date column '{date_column}' is missing from the dataframe."
                )

        if split_config.data_structure == DataStructure.PANEL and split_config.entity_column:
            if split_config.entity_column not in dataframe.columns:
                raise ValueError(
                    f"Configured entity column '{split_config.entity_column}' "
                    "is missing from the dataframe."
                )
        if model_type == ModelType.PANEL_REGRESSION:
            if split_config.data_structure != DataStructure.PANEL:
                raise ValueError(
                    "panel_regression requires DataStructure.PANEL so entity effects are defined."
                )
            if not split_config.entity_column:
                raise ValueError(
                    "panel_regression requires SplitConfig.entity_column to identify panels."
                )

        null_ratio = dataframe.isna().mean().sort_values(ascending=False).to_dict()
        context.metadata["null_ratio_by_column"] = {
            key: float(value) for key, value in null_ratio.items()
        }
        return context

    def _infer_role_column(self, context: PipelineContext, role: ColumnRole) -> str | None:
        for spec in context.config.schema.column_specs:
            if spec.role == role and spec.enabled:
                return spec.name
        return None
