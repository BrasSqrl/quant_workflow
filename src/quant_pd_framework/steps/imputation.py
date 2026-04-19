"""Applies documented missing-value rules after splitting and before model fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..base import BasePipelineStep
from ..config import DataStructure, MissingValuePolicy
from ..context import PipelineContext


@dataclass(slots=True)
class ImputationRule:
    """Resolved imputation contract for one feature column."""

    feature_name: str
    configured_policy: MissingValuePolicy
    applied_policy: MissingValuePolicy
    fill_value: Any = None
    learned_from_train: bool = False
    train_missing_count: int = 0


class ImputationStep(BasePipelineStep):
    """
    Fits and applies per-column missing-value treatment.

    The rule contract is learned from the training split so validation, test,
    and score-only runs stay aligned with the model-development preprocessing
    that produced the fitted estimator.
    """

    name = "imputation"

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.split_frames:
            raise ValueError("Imputation requires populated split frames.")
        if not context.feature_columns:
            raise ValueError("Imputation requires resolved feature columns.")

        train_frame = context.split_frames["train"]
        spec_map = {
            spec.name: spec
            for spec in context.config.schema.column_specs
            if spec.enabled and spec.name in context.feature_columns
        }

        rules = [
            self._build_rule(
                feature_name=feature_name,
                train_series=train_frame[feature_name],
                context=context,
                spec_map=spec_map,
            )
            for feature_name in context.feature_columns
            if feature_name in train_frame.columns
        ]

        updated_splits: dict[str, pd.DataFrame] = {}
        missing_after_imputation: dict[str, dict[str, int]] = {}
        for split_name, split_frame in context.split_frames.items():
            updated_frame = split_frame.copy(deep=True)
            for rule in rules:
                if rule.feature_name not in updated_frame.columns:
                    continue
                updated_frame[rule.feature_name] = self._apply_rule(
                    updated_frame,
                    column_name=rule.feature_name,
                    rule=rule,
                    context=context,
                )

            remaining_missing = {
                feature_name: int(updated_frame[feature_name].isna().sum())
                for feature_name in context.feature_columns
                if feature_name in updated_frame.columns
                and updated_frame[feature_name].isna().any()
            }
            if remaining_missing:
                missing_after_imputation[split_name] = remaining_missing

            updated_splits[split_name] = updated_frame

        if missing_after_imputation:
            details = []
            for split_name, missing_columns in missing_after_imputation.items():
                rendered = ", ".join(
                    f"{column} ({count})" for column, count in sorted(missing_columns.items())
                )
                details.append(f"{split_name}: {rendered}")
            raise ValueError(
                "Missing values remain in model features after the configured imputation step. "
                "Configure a missing-value policy in the column designer or disable the affected "
                "feature columns. Remaining missingness -> " + " | ".join(details)
            )

        context.split_frames = updated_splits
        context.metadata["imputation_summary"] = {
            "feature_count": len(rules),
            "policies": {
                policy.value: sum(1 for rule in rules if rule.applied_policy == policy)
                for policy in MissingValuePolicy
                if any(rule.applied_policy == policy for rule in rules)
            },
        }
        context.diagnostics_tables["imputation_rules"] = pd.DataFrame(
            [
                {
                    "feature_name": rule.feature_name,
                    "configured_policy": rule.configured_policy.value,
                    "applied_policy": rule.applied_policy.value,
                    "fill_value": self._render_fill_value(rule.fill_value),
                    "learned_from_train": rule.learned_from_train,
                    "train_missing_count": rule.train_missing_count,
                }
                for rule in rules
            ]
        )
        context.log(f"Applied missing-value rules to {len(rules)} feature columns.")
        return context

    def _build_rule(
        self,
        *,
        feature_name: str,
        train_series: pd.Series,
        context: PipelineContext,
        spec_map: dict[str, Any],
    ) -> ImputationRule:
        spec = spec_map.get(feature_name)
        configured_policy = (
            spec.missing_value_policy
            if spec is not None
            else MissingValuePolicy.INHERIT_DEFAULT
        )
        applied_policy = self._resolve_policy(configured_policy, train_series)
        train_missing_count = int(train_series.isna().sum())

        if applied_policy in {
            MissingValuePolicy.NONE,
            MissingValuePolicy.FORWARD_FILL,
            MissingValuePolicy.BACKWARD_FILL,
        }:
            self._validate_directional_policy(feature_name, applied_policy, context)
            return ImputationRule(
                feature_name=feature_name,
                configured_policy=configured_policy,
                applied_policy=applied_policy,
                train_missing_count=train_missing_count,
            )

        if applied_policy == MissingValuePolicy.CONSTANT:
            if spec is None:
                raise ValueError(
                    "Column "
                    f"'{feature_name}' uses constant imputation but no schema spec was found."
                )
            fill_value = self._coerce_constant_value(
                raw_value=spec.missing_value_fill_value,
                reference_series=train_series,
            )
            return ImputationRule(
                feature_name=feature_name,
                configured_policy=configured_policy,
                applied_policy=applied_policy,
                fill_value=fill_value,
                train_missing_count=train_missing_count,
            )

        fill_value = self._fit_scalar_fill_value(feature_name, train_series, applied_policy)
        return ImputationRule(
            feature_name=feature_name,
            configured_policy=configured_policy,
            applied_policy=applied_policy,
            fill_value=fill_value,
            learned_from_train=True,
            train_missing_count=train_missing_count,
        )

    def _resolve_policy(
        self,
        configured_policy: MissingValuePolicy,
        series: pd.Series,
    ) -> MissingValuePolicy:
        if configured_policy != MissingValuePolicy.INHERIT_DEFAULT:
            return configured_policy
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return MissingValuePolicy.MEDIAN
        return MissingValuePolicy.MODE

    def _fit_scalar_fill_value(
        self,
        feature_name: str,
        series: pd.Series,
        policy: MissingValuePolicy,
    ) -> Any:
        non_null_series = series.dropna()
        if non_null_series.empty:
            raise ValueError(
                f"Column '{feature_name}' cannot fit {policy.value} imputation because the "
                "training split contains no non-missing values. Use constant imputation or "
                "disable the feature."
            )

        if policy == MissingValuePolicy.MEAN:
            if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                raise ValueError(
                    f"Column '{feature_name}' uses mean imputation but is not numeric."
                )
            return float(non_null_series.astype(float).mean())

        if policy == MissingValuePolicy.MEDIAN:
            if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                raise ValueError(
                    f"Column '{feature_name}' uses median imputation but is not numeric."
                )
            return float(non_null_series.astype(float).median())

        if policy == MissingValuePolicy.MODE:
            mode_values = non_null_series.mode(dropna=True)
            if mode_values.empty:
                raise ValueError(
                    f"Column '{feature_name}' could not determine a mode-based imputation value."
                )
            return mode_values.iloc[0]

        raise ValueError(f"Unsupported scalar imputation policy '{policy.value}'.")

    def _coerce_constant_value(self, *, raw_value: Any, reference_series: pd.Series) -> Any:
        if raw_value is None:
            raise ValueError("Constant imputation requires a fill value.")
        if pd.api.types.is_bool_dtype(reference_series):
            truthy = {"1", "true", "t", "yes", "y"}
            falsy = {"0", "false", "f", "no", "n"}
            text = str(raw_value).strip().lower()
            if text in truthy:
                return True
            if text in falsy:
                return False
            raise ValueError(f"Could not coerce constant value '{raw_value}' to boolean.")
        if pd.api.types.is_numeric_dtype(reference_series) and not pd.api.types.is_bool_dtype(
            reference_series
        ):
            coerced = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(coerced):
                raise ValueError(f"Could not coerce constant value '{raw_value}' to numeric.")
            return float(coerced)
        if pd.api.types.is_datetime64_any_dtype(reference_series):
            coerced = pd.to_datetime(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(coerced):
                raise ValueError(f"Could not coerce constant value '{raw_value}' to datetime.")
            return coerced
        return raw_value

    def _validate_directional_policy(
        self,
        feature_name: str,
        policy: MissingValuePolicy,
        context: PipelineContext,
    ) -> None:
        if policy in {MissingValuePolicy.FORWARD_FILL, MissingValuePolicy.BACKWARD_FILL} and (
            context.config.split.data_structure == DataStructure.CROSS_SECTIONAL
        ):
            raise ValueError(
                f"Column '{feature_name}' uses {policy.value}, which is only supported for "
                "time-series or panel workflows."
            )

    def _apply_rule(
        self,
        frame: pd.DataFrame,
        *,
        column_name: str,
        rule: ImputationRule,
        context: PipelineContext,
    ) -> pd.Series:
        series = frame[column_name]
        if rule.applied_policy == MissingValuePolicy.NONE:
            return series
        if rule.applied_policy == MissingValuePolicy.FORWARD_FILL:
            return self._directional_fill(series, frame, context, forward=True)
        if rule.applied_policy == MissingValuePolicy.BACKWARD_FILL:
            return self._directional_fill(series, frame, context, forward=False)
        return self._fillna_scalar(series, rule.fill_value)

    def _directional_fill(
        self,
        series: pd.Series,
        frame: pd.DataFrame,
        context: PipelineContext,
        *,
        forward: bool,
    ) -> pd.Series:
        entity_column = context.config.split.entity_column
        if context.config.split.data_structure == DataStructure.PANEL and entity_column:
            grouped = frame.groupby(entity_column, sort=False)[series.name]
            return grouped.ffill() if forward else grouped.bfill()
        return series.ffill() if forward else series.bfill()

    def _render_fill_value(self, value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def _fillna_scalar(self, series: pd.Series, fill_value: Any) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            categorical_series = series
            if fill_value not in categorical_series.cat.categories:
                categorical_series = categorical_series.cat.add_categories([fill_value])
            return categorical_series.fillna(fill_value)
        return series.fillna(fill_value)
