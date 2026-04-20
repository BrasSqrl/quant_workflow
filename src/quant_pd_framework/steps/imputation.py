"""Applies documented missing-value rules after splitting and before model fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..base import BasePipelineStep
from ..config import DataStructure, MissingValuePolicy
from ..context import PipelineContext

MISSING_INDICATOR_SUFFIX = "__missing_indicator"
GROUP_KEY_MISSING_TOKEN = "__group_missing__"


@dataclass(slots=True)
class ImputationRule:
    """Resolved imputation contract for one feature column."""

    feature_name: str
    configured_policy: MissingValuePolicy
    applied_policy: MissingValuePolicy
    fill_value: Any = None
    learned_from_train: bool = False
    train_missing_count: int = 0
    group_columns: tuple[str, ...] = ()
    group_fill_lookup: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    group_rule_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_indicator_column: str | None = None


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

        context.metadata["pre_imputation_split_frames"] = {
            split_name: split_frame.copy(deep=True)
            for split_name, split_frame in context.split_frames.items()
        }
        context.metadata["pre_imputation_feature_columns"] = list(context.feature_columns)
        context.metadata["pre_imputation_numeric_features"] = list(context.numeric_features)
        context.metadata["pre_imputation_categorical_features"] = list(context.categorical_features)
        train_frame = context.split_frames["train"]
        spec_map = {
            spec.name: spec
            for spec in context.config.schema.column_specs
            if spec.enabled and spec.name in context.feature_columns
        }

        rules = [
            self._build_rule(
                feature_name=feature_name,
                train_frame=train_frame,
                context=context,
                spec_map=spec_map,
            )
            for feature_name in context.feature_columns
            if feature_name in train_frame.columns
        ]
        self._validate_generated_indicator_columns(context, rules)

        updated_splits: dict[str, pd.DataFrame] = {}
        missing_after_imputation: dict[str, dict[str, int]] = {}
        for split_name, split_frame in context.split_frames.items():
            updated_frame = split_frame.copy(deep=True)
            for rule in rules:
                if rule.feature_name not in updated_frame.columns:
                    continue
                original_series = updated_frame[rule.feature_name]
                if rule.missing_indicator_column is not None:
                    updated_frame[rule.missing_indicator_column] = original_series.isna().astype(
                        int
                    )
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

        generated_indicator_columns = [
            rule.missing_indicator_column
            for rule in rules
            if rule.missing_indicator_column is not None
        ]
        if generated_indicator_columns:
            context.feature_columns = self._deduplicate_preserve_order(
                [*context.feature_columns, *generated_indicator_columns]
            )
            context.numeric_features = self._deduplicate_preserve_order(
                [*context.numeric_features, *generated_indicator_columns]
            )
            context.metadata["generated_missing_indicator_columns"] = generated_indicator_columns

        context.split_frames = updated_splits
        context.working_data = pd.concat(updated_splits.values(), ignore_index=True)
        context.metadata["imputation_summary"] = {
            "feature_count": len(rules),
            "policies": {
                policy.value: sum(1 for rule in rules if rule.applied_policy == policy)
                for policy in MissingValuePolicy
                if any(rule.applied_policy == policy for rule in rules)
            },
            "grouped_feature_count": sum(1 for rule in rules if rule.group_columns),
            "group_rule_count": sum(len(rule.group_fill_lookup) for rule in rules),
            "generated_missing_indicator_count": len(generated_indicator_columns),
        }
        context.metadata["imputation_rules_by_feature"] = {
            rule.feature_name: {
                "configured_policy": rule.configured_policy.value,
                "applied_policy": rule.applied_policy.value,
                "fill_value": self._render_fill_value(rule.fill_value),
                "train_missing_count": rule.train_missing_count,
                "group_columns": list(rule.group_columns),
                "missing_indicator_column": rule.missing_indicator_column,
            }
            for rule in rules
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
                    "group_columns": ", ".join(rule.group_columns),
                    "group_rule_count": len(rule.group_fill_lookup),
                    "missing_indicator_column": rule.missing_indicator_column or "",
                }
                for rule in rules
            ]
        )
        group_rule_rows = [
            row for rule in rules for row in rule.group_rule_rows if rule.group_rule_rows
        ]
        if group_rule_rows:
            context.diagnostics_tables["imputation_group_rules"] = pd.DataFrame(group_rule_rows)
        context.log(f"Applied missing-value rules to {len(rules)} feature columns.")
        return context

    def _build_rule(
        self,
        *,
        feature_name: str,
        train_frame: pd.DataFrame,
        context: PipelineContext,
        spec_map: dict[str, Any],
    ) -> ImputationRule:
        spec = spec_map.get(feature_name)
        configured_policy = (
            spec.missing_value_policy if spec is not None else MissingValuePolicy.INHERIT_DEFAULT
        )
        train_series = train_frame[feature_name]
        applied_policy = self._resolve_policy(configured_policy, train_series)
        train_missing_count = int(train_series.isna().sum())
        group_columns = tuple(spec.missing_value_group_columns) if spec is not None else ()
        missing_indicator_column = (
            self._indicator_column_name(feature_name)
            if spec is not None and spec.create_missing_indicator
            else None
        )
        self._validate_group_columns(feature_name, group_columns, train_frame)

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
                group_columns=group_columns,
                missing_indicator_column=missing_indicator_column,
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
                missing_indicator_column=missing_indicator_column,
            )

        fill_value = self._fit_scalar_fill_value(feature_name, train_series, applied_policy)
        group_fill_lookup: dict[tuple[Any, ...], Any] = {}
        group_rule_rows: list[dict[str, Any]] = []
        if group_columns:
            group_fill_lookup, group_rule_rows = self._fit_group_fill_rules(
                feature_name=feature_name,
                train_frame=train_frame,
                group_columns=group_columns,
                policy=applied_policy,
            )
        return ImputationRule(
            feature_name=feature_name,
            configured_policy=configured_policy,
            applied_policy=applied_policy,
            fill_value=fill_value,
            learned_from_train=True,
            train_missing_count=train_missing_count,
            group_columns=group_columns,
            group_fill_lookup=group_fill_lookup,
            group_rule_rows=group_rule_rows,
            missing_indicator_column=missing_indicator_column,
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

    def _fit_group_fill_rules(
        self,
        *,
        feature_name: str,
        train_frame: pd.DataFrame,
        group_columns: tuple[str, ...],
        policy: MissingValuePolicy,
    ) -> tuple[dict[tuple[Any, ...], Any], list[dict[str, Any]]]:
        lookup: dict[tuple[Any, ...], Any] = {}
        rows: list[dict[str, Any]] = []
        grouped = train_frame.loc[:, [*group_columns, feature_name]].groupby(
            list(group_columns),
            dropna=False,
            sort=False,
        )
        for raw_group_values, group_frame in grouped:
            group_values = (
                raw_group_values if isinstance(raw_group_values, tuple) else (raw_group_values,)
            )
            if group_frame[feature_name].dropna().empty:
                continue
            group_fill_value = self._fit_scalar_fill_value(
                feature_name,
                group_frame[feature_name],
                policy,
            )
            normalized_group_values = self._normalize_group_tuple(group_values)
            lookup[normalized_group_values] = group_fill_value
            row = {
                "feature_name": feature_name,
                "applied_policy": policy.value,
                "group_columns": ", ".join(group_columns),
                "group_fill_value": self._render_fill_value(group_fill_value),
                "group_train_count": int(len(group_frame)),
                "group_missing_count": int(group_frame[feature_name].isna().sum()),
            }
            for column_name, raw_value in zip(group_columns, group_values, strict=False):
                row[column_name] = self._render_fill_value(raw_value)
            rows.append(row)
        return lookup, rows

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

    def _validate_group_columns(
        self,
        feature_name: str,
        group_columns: tuple[str, ...],
        train_frame: pd.DataFrame,
    ) -> None:
        if not group_columns:
            return
        missing_group_columns = [
            column_name for column_name in group_columns if column_name not in train_frame.columns
        ]
        if missing_group_columns:
            raise ValueError(
                f"Column '{feature_name}' references missing imputation group columns: "
                + ", ".join(missing_group_columns)
            )

    def _validate_generated_indicator_columns(
        self,
        context: PipelineContext,
        rules: list[ImputationRule],
    ) -> None:
        generated_columns = [
            rule.missing_indicator_column
            for rule in rules
            if rule.missing_indicator_column is not None
        ]
        if len(generated_columns) != len(set(generated_columns)):
            raise ValueError("Advanced imputation generated duplicate missing-indicator columns.")

        existing_columns = {
            column_name
            for split_frame in context.split_frames.values()
            for column_name in split_frame.columns
        }
        conflicting_columns = [
            column_name for column_name in generated_columns if column_name in existing_columns
        ]
        if conflicting_columns:
            raise ValueError(
                "Advanced imputation would overwrite existing columns: "
                + ", ".join(sorted(conflicting_columns))
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
        if rule.group_fill_lookup:
            group_fill_series = self._build_group_fill_series(frame, rule)
            series = self._fillna_series(series, group_fill_series)
        if series.isna().any():
            return self._fillna_scalar(series, rule.fill_value)
        return series

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

    def _build_group_fill_series(
        self,
        frame: pd.DataFrame,
        rule: ImputationRule,
    ) -> pd.Series:
        normalized_frame = pd.DataFrame(index=frame.index)
        for column_name in rule.group_columns:
            normalized_frame[column_name] = frame[column_name].map(self._normalize_group_value)
        normalized_keys = pd.MultiIndex.from_frame(normalized_frame).tolist()
        key_series = pd.Series(normalized_keys, index=frame.index, dtype="object")
        return key_series.map(rule.group_fill_lookup)

    def _normalize_group_tuple(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(self._normalize_group_value(value) for value in values)

    def _normalize_group_value(self, value: Any) -> Any:
        if pd.isna(value):
            return GROUP_KEY_MISSING_TOKEN
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def _indicator_column_name(self, feature_name: str) -> str:
        return f"{feature_name}{MISSING_INDICATOR_SUFFIX}"

    def _render_fill_value(self, value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def _fillna_series(self, series: pd.Series, fill_values: pd.Series) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            categorical_series = series
            candidate_values = [
                value
                for value in pd.Series(fill_values).dropna().unique().tolist()
                if value is not None
            ]
            missing_categories = [
                value
                for value in candidate_values
                if value not in categorical_series.cat.categories
            ]
            if missing_categories:
                categorical_series = categorical_series.cat.add_categories(missing_categories)
            return categorical_series.where(categorical_series.notna(), fill_values)
        return series.where(series.notna(), fill_values)

    def _fillna_scalar(self, series: pd.Series, fill_value: Any) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            categorical_series = series
            if fill_value not in categorical_series.cat.categories:
                categorical_series = categorical_series.cat.add_categories([fill_value])
            return categorical_series.fillna(fill_value)
        return series.fillna(fill_value)

    def _deduplicate_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered
