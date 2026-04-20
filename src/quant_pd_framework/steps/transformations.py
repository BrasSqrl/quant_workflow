"""Applies governed feature transformations after split-time imputation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..base import BasePipelineStep
from ..config import TransformationSpec, TransformationType
from ..context import PipelineContext


@dataclass(slots=True)
class ResolvedTransformation:
    """Transformation spec plus the fitted parameters learned from train."""

    spec: TransformationSpec
    output_feature: str
    learned_parameters: dict[str, Any]


class TransformationStep(BasePipelineStep):
    """
    Applies explicit, exportable transformations to the modeled feature set.

    The transformation contract is fit on the training split and then applied
    to every downstream split and to the working dataframe used for diagnostics.
    """

    name = "transformations"

    def run(self, context: PipelineContext) -> PipelineContext:
        config = context.config.transformations
        if not config.enabled or not config.transformations:
            return context
        if not context.split_frames or context.working_data is None:
            raise ValueError("Governed transformations require split and working dataframes.")

        working_splits = {
            split_name: frame.copy(deep=True)
            for split_name, frame in context.split_frames.items()
        }
        working_dataframe = context.working_data.copy(deep=True)
        resolved_transformations: list[ResolvedTransformation] = []
        audit_rows: list[dict[str, Any]] = []

        for transformation in config.transformations:
            if not transformation.enabled:
                continue
            try:
                train_frame = working_splits["train"]
                resolved = self._fit_transformation(
                    context=context,
                    train_frame=train_frame,
                    spec=transformation,
                )
                for split_name, split_frame in working_splits.items():
                    working_splits[split_name] = self._apply_transformation(
                        split_frame,
                        resolved,
                    )
                working_dataframe = self._apply_transformation(working_dataframe, resolved)
                self._update_feature_contract(context, working_dataframe, resolved.output_feature)
                resolved_transformations.append(resolved)
                audit_rows.append(
                    {
                        "transform_type": resolved.spec.transform_type.value,
                        "source_feature": resolved.spec.source_feature,
                        "secondary_feature": resolved.spec.secondary_feature or "",
                        "output_feature": resolved.output_feature,
                        "learned_parameters": self._render_parameters(
                            resolved.learned_parameters
                        ),
                        "notes": resolved.spec.notes,
                        "status": "applied",
                    }
                )
            except Exception as exc:
                audit_rows.append(
                    {
                        "transform_type": transformation.transform_type.value,
                        "source_feature": transformation.source_feature,
                        "secondary_feature": transformation.secondary_feature or "",
                        "output_feature": self._resolve_output_feature_name(transformation),
                        "learned_parameters": "",
                        "notes": transformation.notes,
                        "status": f"failed: {exc}",
                    }
                )
                if config.error_on_failure:
                    raise
                context.warn(
                    f"Skipped governed transformation '{transformation.transform_type.value}' "
                    f"for '{transformation.source_feature}': {exc}"
                )

        if not resolved_transformations:
            return context

        context.split_frames = working_splits
        context.working_data = working_dataframe
        context.diagnostics_tables["governed_transformations"] = pd.DataFrame(audit_rows)
        context.metadata["transformation_summary"] = {
            "count": len(resolved_transformations),
            "output_features": [item.output_feature for item in resolved_transformations],
        }
        context.log(
            "Applied governed transformations to "
            f"{len(resolved_transformations)} configured outputs."
        )
        return context

    def _fit_transformation(
        self,
        *,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        spec: TransformationSpec,
    ) -> ResolvedTransformation:
        if spec.source_feature not in context.feature_columns:
            raise ValueError(
                f"Transformation source '{spec.source_feature}' is not in the feature set."
            )
        if spec.source_feature not in train_frame.columns:
            raise ValueError(
                f"Transformation source '{spec.source_feature}' is missing from the train split."
            )

        output_feature = self._resolve_output_feature_name(spec)
        if output_feature == context.target_column:
            raise ValueError("Governed transformations cannot overwrite the target column.")

        source_series = train_frame[spec.source_feature]
        learned_parameters: dict[str, Any] = {}

        if spec.transform_type == TransformationType.WINSORIZE:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            lower_quantile = 0.01 if spec.lower_quantile is None else spec.lower_quantile
            upper_quantile = 0.99 if spec.upper_quantile is None else spec.upper_quantile
            learned_parameters = {
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_value": float(numeric_series.quantile(lower_quantile)),
                "upper_value": float(numeric_series.quantile(upper_quantile)),
            }
        elif spec.transform_type == TransformationType.LOG1P:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            minimum_value = float(numeric_series.min())
            if minimum_value <= -1.0:
                raise ValueError(
                    f"log1p requires values greater than -1 for '{spec.source_feature}'."
                )
            learned_parameters = {"min_train_value": minimum_value}
        elif spec.transform_type in {
            TransformationType.RATIO,
            TransformationType.INTERACTION,
        }:
            secondary_feature = spec.secondary_feature or ""
            if secondary_feature not in context.feature_columns:
                raise ValueError(
                    f"Secondary feature '{secondary_feature}' is not in the feature set."
                )
            if secondary_feature not in train_frame.columns:
                raise ValueError(
                    f"Secondary feature '{secondary_feature}' is missing from the train split."
                )
            self._numeric_series(train_frame[secondary_feature], secondary_feature)
        elif spec.transform_type == TransformationType.MANUAL_BINS:
            self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {"bin_edges": list(spec.bin_edges)}
        else:
            raise ValueError(f"Unsupported transformation type '{spec.transform_type.value}'.")

        return ResolvedTransformation(
            spec=spec,
            output_feature=output_feature,
            learned_parameters=learned_parameters,
        )

    def _apply_transformation(
        self,
        frame: pd.DataFrame,
        resolved: ResolvedTransformation,
    ) -> pd.DataFrame:
        working = frame.copy(deep=True)
        spec = resolved.spec
        source_name = spec.source_feature
        output_name = resolved.output_feature

        if spec.transform_type == TransformationType.WINSORIZE:
            values = pd.to_numeric(working[source_name], errors="coerce")
            working[output_name] = values.clip(
                lower=resolved.learned_parameters["lower_value"],
                upper=resolved.learned_parameters["upper_value"],
            )
        elif spec.transform_type == TransformationType.LOG1P:
            values = pd.to_numeric(working[source_name], errors="coerce")
            working[output_name] = np.where(values > -1, np.log1p(values), np.nan)
        elif spec.transform_type == TransformationType.RATIO:
            numerator = pd.to_numeric(working[source_name], errors="coerce")
            denominator = pd.to_numeric(
                working[spec.secondary_feature or ""],
                errors="coerce",
            )
            working[output_name] = numerator.div(denominator.replace({0: np.nan}))
        elif spec.transform_type == TransformationType.INTERACTION:
            left = pd.to_numeric(working[source_name], errors="coerce")
            right = pd.to_numeric(working[spec.secondary_feature or ""], errors="coerce")
            working[output_name] = left * right
        elif spec.transform_type == TransformationType.MANUAL_BINS:
            values = pd.to_numeric(working[source_name], errors="coerce")
            bin_edges = [-np.inf, *resolved.learned_parameters["bin_edges"], np.inf]
            working[output_name] = pd.cut(
                values,
                bins=bin_edges,
                include_lowest=True,
                duplicates="drop",
            ).astype("category")

        return working

    def _update_feature_contract(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
        output_feature: str,
    ) -> None:
        if output_feature not in context.feature_columns:
            context.feature_columns.append(output_feature)
        context.numeric_features = []
        context.categorical_features = []
        for feature_name in context.feature_columns:
            if feature_name not in dataframe.columns:
                continue
            series = dataframe[feature_name]
            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                context.numeric_features.append(feature_name)
            else:
                context.categorical_features.append(feature_name)
        context.metadata["feature_summary"] = {
            "feature_count": len(context.feature_columns),
            "numeric_feature_count": len(context.numeric_features),
            "categorical_feature_count": len(context.categorical_features),
        }

    def _resolve_output_feature_name(self, spec: TransformationSpec) -> str:
        configured_output = (spec.output_feature or "").strip()
        if configured_output:
            return configured_output
        if spec.transform_type == TransformationType.MANUAL_BINS:
            return f"{spec.source_feature}_binned"
        if spec.transform_type == TransformationType.RATIO:
            return f"{spec.source_feature}_over_{spec.secondary_feature}"
        if spec.transform_type == TransformationType.INTERACTION:
            return f"{spec.source_feature}_x_{spec.secondary_feature}"
        return spec.source_feature

    def _numeric_series(self, series: pd.Series, feature_name: str) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.dropna().empty:
            raise ValueError(
                f"Transformation feature '{feature_name}' does not contain usable numeric values."
            )
        return numeric_series

    def _render_parameters(self, parameters: dict[str, Any]) -> str:
        rendered: list[str] = []
        for key, value in parameters.items():
            if isinstance(value, float):
                rendered.append(f"{key}={value:.6g}")
            else:
                rendered.append(f"{key}={value}")
        return ", ".join(rendered)
