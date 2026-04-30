"""Applies governed feature transformations after split-time imputation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import boxcox, boxcox_normmax, norm, yeojohnson, yeojohnson_normmax

from ..base import BasePipelineStep
from ..config import (
    DataStructure,
    ExecutionMode,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
)
from ..context import PipelineContext


@dataclass(slots=True)
class ResolvedTransformation:
    """Transformation spec plus the fitted parameters learned from train."""

    spec: TransformationSpec
    output_feature: str
    learned_parameters: dict[str, Any]


NUMERIC_SCALING_TRANSFORMS = {
    TransformationType.STANDARD_SCALE,
    TransformationType.ROBUST_SCALE,
    TransformationType.MIN_MAX_SCALE,
    TransformationType.PERCENTILE_RANK,
    TransformationType.NORMAL_SCORE,
}

NUMERIC_DIRECT_TRANSFORMS = {
    TransformationType.SIGNED_LOG1P,
    TransformationType.SQRT,
    TransformationType.RECIPROCAL,
    TransformationType.SQUARE,
    TransformationType.POWER,
    TransformationType.ABSOLUTE_VALUE,
    TransformationType.CENTER_MEAN,
    TransformationType.CENTER_MEDIAN,
}

PAIR_NUMERIC_TRANSFORMS = {
    TransformationType.RATIO,
    TransformationType.SAFE_RATIO,
    TransformationType.MARGIN_RATIO,
    TransformationType.DEBT_SERVICE_RATIO,
    TransformationType.ADD,
    TransformationType.SUBTRACT,
    TransformationType.PRODUCT,
}

TEMPORAL_NUMERIC_TRANSFORMS = {
    TransformationType.LAG,
    TransformationType.DIFFERENCE,
    TransformationType.EWMA,
    TransformationType.ROLLING_MEAN,
    TransformationType.ROLLING_MEDIAN,
    TransformationType.ROLLING_MIN,
    TransformationType.ROLLING_MAX,
    TransformationType.ROLLING_STD,
    TransformationType.ROLLING_SUM,
    TransformationType.ROLLING_RANGE,
    TransformationType.ROLLING_CV,
    TransformationType.ROLLING_SLOPE,
    TransformationType.EXPANDING_MEAN,
    TransformationType.CUMULATIVE_SUM,
    TransformationType.CUMULATIVE_COUNT,
    TransformationType.MONTHS_SINCE_EVENT,
    TransformationType.CHANGE_FROM_BASELINE,
    TransformationType.PCT_CHANGE,
}

BINNING_TRANSFORMS = {
    TransformationType.MANUAL_BINS,
    TransformationType.QUANTILE_BINS,
    TransformationType.EQUAL_WIDTH_BINS,
    TransformationType.MONOTONIC_BINS,
}

ENCODING_TRANSFORMS = {
    TransformationType.WOE_ENCODING,
    TransformationType.BAD_RATE_ENCODING,
    TransformationType.RARE_CATEGORY_COLLAPSE,
    TransformationType.FREQUENCY_ENCODING,
    TransformationType.ORDINAL_ENCODING,
    TransformationType.TARGET_ENCODING,
}

DATE_TRANSFORMS = {
    TransformationType.DATE_YEAR,
    TransformationType.DATE_MONTH,
    TransformationType.DATE_QUARTER,
    TransformationType.DATE_MONTH_END_FLAG,
    TransformationType.DATE_FISCAL_QUARTER,
    TransformationType.DATE_AGE_DAYS,
    TransformationType.DATE_AGE_MONTHS,
}

ROW_MISSINGNESS_TRANSFORMS = {
    TransformationType.ROW_MISSING_COUNT,
    TransformationType.ROW_MISSING_SHARE,
    TransformationType.ANY_MISSING_FLAG,
}


class TransformationStep(BasePipelineStep):
    """
    Applies explicit, exportable transformations to the modeled feature set.

    The transformation contract is fit on the training split and then applied
    to every downstream split and to the working dataframe used for diagnostics.
    """

    name = "transformations"

    def run(self, context: PipelineContext) -> PipelineContext:
        config = context.config.transformations
        if not context.split_frames or context.working_data is None:
            raise ValueError("Governed transformations require split and working dataframes.")

        working_splits = {
            split_name: frame.copy(deep=False) for split_name, frame in context.split_frames.items()
        }
        working_dataframe = context.working_data.copy(deep=False)
        resolved_transformations: list[ResolvedTransformation] = []
        audit_rows: list[dict[str, Any]] = []

        generated_specs = self._resolve_auto_interactions(
            context=context,
            config=config,
            working_splits=working_splits,
        )
        if not config.enabled and not generated_specs:
            return context

        for transformation in list(config.transformations):
            if not transformation.enabled:
                continue
            try:
                resolved = self._fit_transformation(
                    context=context,
                    train_frame=working_splits["train"],
                    spec=transformation,
                )
                for split_name, split_frame in working_splits.items():
                    working_splits[split_name] = self._apply_transformation(
                        split_frame,
                        resolved,
                        context=context,
                        split_name=split_name,
                    )
                working_dataframe = self._apply_transformation(
                    working_dataframe,
                    resolved,
                    context=context,
                )
                resolved_output_features = list(
                    resolved.learned_parameters.get("output_features", [resolved.output_feature])
                )
                self._update_feature_contract(
                    context,
                    working_dataframe,
                    resolved_output_features,
                )
                resolved_transformations.append(resolved)
                audit_rows.append(
                    {
                        "transform_type": resolved.spec.transform_type.value,
                        "source_feature": resolved.spec.source_feature,
                        "secondary_feature": resolved.spec.secondary_feature or "",
                        "categorical_value": resolved.spec.categorical_value or "",
                        "output_feature": ", ".join(resolved_output_features),
                        "learned_parameters": self._render_parameters(resolved.learned_parameters),
                        "generated_automatically": resolved.spec.generated_automatically,
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
                        "categorical_value": transformation.categorical_value or "",
                        "output_feature": self._resolve_output_feature_name(transformation),
                        "learned_parameters": "",
                        "generated_automatically": transformation.generated_automatically,
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
        context.metadata["resolved_transformation_objects"] = resolved_transformations
        context.diagnostics_tables["governed_transformations"] = pd.DataFrame(audit_rows)
        context.metadata["transformation_summary"] = {
            "count": len(resolved_transformations),
            "output_features": [
                output_feature
                for item in resolved_transformations
                for output_feature in item.learned_parameters.get(
                    "output_features",
                    [item.output_feature],
                )
            ],
            "generated_interaction_count": len(generated_specs),
        }
        if generated_specs:
            context.metadata["generated_interaction_features"] = [
                self._resolve_output_feature_name(spec) for spec in generated_specs
            ]
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
        source_features = self._source_feature_names(spec)
        self._validate_source_columns(context, train_frame, spec, source_features)

        output_feature = self._resolve_output_feature_name(spec)
        if output_feature == context.target_column:
            raise ValueError("Governed transformations cannot overwrite the target column.")

        source_series = train_frame[source_features[0]]
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
        elif spec.transform_type == TransformationType.BOX_COX:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            minimum_value = float(numeric_series.min())
            if minimum_value <= 0.0:
                raise ValueError(
                    f"box_cox requires strictly positive values for '{spec.source_feature}'."
                )
            learned_parameters = {
                "lambda": float(boxcox_normmax(numeric_series.dropna().to_numpy(dtype=float)))
            }
        elif spec.transform_type == TransformationType.NATURAL_SPLINE:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            spline_df = 4 if spec.parameter_value is None else int(spec.parameter_value)
            non_null = numeric_series.dropna()
            if len(non_null) < max(spline_df + 1, 8):
                raise ValueError(
                    f"natural_spline requires at least {max(spline_df + 1, 8)} usable "
                    f"rows for '{spec.source_feature}'."
                )
            spline_design = dmatrix(
                f"cr(x, df={spline_df}) - 1",
                {"x": non_null.to_numpy(dtype=float)},
                return_type="dataframe",
            )
            output_features = tuple(
                f"{output_feature}_basis_{index + 1}" for index in range(spline_design.shape[1])
            )
            learned_parameters = {
                "degrees_of_freedom": spline_df,
                "design_info": spline_design.design_info,
                "output_features": output_features,
            }
        elif spec.transform_type == TransformationType.YEO_JOHNSON:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {
                "lambda": float(yeojohnson_normmax(numeric_series.dropna().to_numpy(dtype=float)))
            }
        elif spec.transform_type == TransformationType.CAPPED_ZSCORE:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            std_value = float(numeric_series.std(ddof=0))
            learned_parameters = {
                "mean": float(numeric_series.mean()),
                "std": std_value if np.isfinite(std_value) and std_value > 0 else 0.0,
                "z_cap": 3.0 if spec.parameter_value is None else float(spec.parameter_value),
            }
        elif spec.transform_type == TransformationType.PIECEWISE_LINEAR:
            self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {
                "hinge_point": float(spec.parameter_value),
            }
        elif spec.transform_type in NUMERIC_SCALING_TRANSFORMS:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            non_null = numeric_series.dropna()
            if spec.transform_type == TransformationType.STANDARD_SCALE:
                std_value = float(non_null.std(ddof=0))
                learned_parameters = {
                    "mean": float(non_null.mean()),
                    "std": std_value if np.isfinite(std_value) and std_value > 0 else 0.0,
                }
            elif spec.transform_type == TransformationType.ROBUST_SCALE:
                lower_quartile = float(non_null.quantile(0.25))
                upper_quartile = float(non_null.quantile(0.75))
                iqr = upper_quartile - lower_quartile
                learned_parameters = {
                    "median": float(non_null.median()),
                    "iqr": float(iqr) if np.isfinite(iqr) and iqr > 0 else 0.0,
                }
            elif spec.transform_type == TransformationType.MIN_MAX_SCALE:
                min_value = float(non_null.min())
                max_value = float(non_null.max())
                learned_parameters = {
                    "min": min_value,
                    "max": max_value,
                    "range": max_value - min_value,
                }
            else:
                quantile_grid = np.linspace(0.0, 1.0, 1001)
                quantile_values = (
                    non_null.quantile(quantile_grid)
                    .drop_duplicates()
                    .to_numpy(dtype=float)
                )
                quantile_positions = np.linspace(0.0, 1.0, len(quantile_values))
                learned_parameters = {
                    "rank_values": tuple(float(value) for value in quantile_values),
                    "rank_positions": tuple(float(value) for value in quantile_positions),
                    "non_null_count": int(non_null.shape[0]),
                }
        elif spec.transform_type in NUMERIC_DIRECT_TRANSFORMS:
            numeric_series = self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {
                "epsilon": 1e-12,
                "power": None if spec.parameter_value is None else float(spec.parameter_value),
            }
            if spec.transform_type == TransformationType.CENTER_MEAN:
                learned_parameters["center"] = float(numeric_series.mean())
            elif spec.transform_type == TransformationType.CENTER_MEDIAN:
                learned_parameters["center"] = float(numeric_series.median())
        elif spec.transform_type in PAIR_NUMERIC_TRANSFORMS:
            secondary_feature = spec.secondary_feature or ""
            self._numeric_series(source_series, spec.source_feature)
            self._numeric_series(train_frame[secondary_feature], secondary_feature)
            learned_parameters = {"epsilon": 1e-12}
        elif spec.transform_type == TransformationType.INTERACTION:
            left_role, right_role = self._interaction_roles(train_frame, spec)
            learned_parameters = {
                "left_role": left_role,
                "right_role": right_role,
            }
        elif spec.transform_type in TEMPORAL_NUMERIC_TRANSFORMS:
            self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {
                "lag_periods": 1 if spec.lag_periods is None else int(spec.lag_periods),
                "window_size": 3 if spec.window_size is None else int(spec.window_size),
            }
        elif spec.transform_type in BINNING_TRANSFORMS:
            self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {
                "bin_edges": self._fit_bin_edges(
                    train_frame=train_frame,
                    source_feature=spec.source_feature,
                    spec=spec,
                    target_column=context.target_column,
                )
            }
        elif spec.transform_type in ENCODING_TRANSFORMS:
            learned_parameters = self._fit_encoding_parameters(
                context=context,
                train_frame=train_frame,
                spec=spec,
            )
        elif spec.transform_type in DATE_TRANSFORMS:
            date_values = pd.to_datetime(source_series, errors="coerce")
            if date_values.dropna().empty:
                raise ValueError(
                    f"Date transformation source '{spec.source_feature}' has no valid dates."
                )
            learned_parameters = self._fit_date_parameters(
                train_frame=train_frame,
                source_feature=spec.source_feature,
                spec=spec,
                context=context,
            )
        elif spec.transform_type in ROW_MISSINGNESS_TRANSFORMS:
            learned_parameters = {"source_features": tuple(source_features)}
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
        *,
        context: PipelineContext,
        split_name: str | None = None,
    ) -> pd.DataFrame:
        working = frame.copy(deep=False)
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
        elif spec.transform_type == TransformationType.BOX_COX:
            values = pd.to_numeric(working[source_name], errors="coerce")
            transformed = pd.Series(np.nan, index=working.index, dtype=float)
            mask = values.notna() & values.gt(0)
            if mask.any():
                transformed.loc[mask] = boxcox(
                    values.loc[mask].to_numpy(dtype=float),
                    lmbda=float(resolved.learned_parameters["lambda"]),
                )
            working[output_name] = transformed
        elif spec.transform_type == TransformationType.NATURAL_SPLINE:
            values = pd.to_numeric(working[source_name], errors="coerce")
            output_features = list(resolved.learned_parameters["output_features"])
            transformed = pd.DataFrame(
                np.nan,
                index=working.index,
                columns=output_features,
                dtype=float,
            )
            mask = values.notna()
            if mask.any():
                design_matrix = build_design_matrices(
                    [resolved.learned_parameters["design_info"]],
                    {"x": values.loc[mask].to_numpy(dtype=float)},
                )[0]
                transformed.loc[mask, output_features] = np.asarray(
                    design_matrix,
                    dtype=float,
                )
            for feature_name in output_features:
                working[feature_name] = transformed[feature_name]
        elif spec.transform_type == TransformationType.YEO_JOHNSON:
            values = pd.to_numeric(working[source_name], errors="coerce")
            transformed = pd.Series(np.nan, index=working.index, dtype=float)
            mask = values.notna()
            if mask.any():
                transformed.loc[mask] = yeojohnson(
                    values.loc[mask].to_numpy(dtype=float),
                    lmbda=float(resolved.learned_parameters["lambda"]),
                )
            working[output_name] = transformed
        elif spec.transform_type == TransformationType.CAPPED_ZSCORE:
            values = pd.to_numeric(working[source_name], errors="coerce")
            std_value = float(resolved.learned_parameters["std"])
            if std_value == 0.0:
                working[output_name] = 0.0
            else:
                z_values = (values - float(resolved.learned_parameters["mean"])) / std_value
                working[output_name] = z_values.clip(
                    lower=-float(resolved.learned_parameters["z_cap"]),
                    upper=float(resolved.learned_parameters["z_cap"]),
                )
        elif spec.transform_type == TransformationType.PIECEWISE_LINEAR:
            values = pd.to_numeric(working[source_name], errors="coerce")
            hinge_point = float(resolved.learned_parameters["hinge_point"])
            working[output_name] = np.maximum(values - hinge_point, 0.0)
        elif spec.transform_type in NUMERIC_SCALING_TRANSFORMS:
            working[output_name] = self._apply_numeric_scaling_transform(
                working[source_name],
                spec.transform_type,
                resolved.learned_parameters,
            )
        elif spec.transform_type in NUMERIC_DIRECT_TRANSFORMS:
            working[output_name] = self._apply_numeric_direct_transform(
                working[source_name],
                spec.transform_type,
                resolved.learned_parameters,
            )
        elif spec.transform_type in PAIR_NUMERIC_TRANSFORMS:
            working[output_name] = self._apply_pair_numeric_transform(
                working[source_name],
                working[spec.secondary_feature or ""],
                spec.transform_type,
                resolved.learned_parameters,
            )
        elif spec.transform_type == TransformationType.INTERACTION:
            left = self._interaction_operand(
                working,
                spec.source_feature,
                resolved.learned_parameters.get("left_role", "numeric"),
                spec.categorical_value,
            )
            right = self._interaction_operand(
                working,
                spec.secondary_feature or "",
                resolved.learned_parameters.get("right_role", "numeric"),
                spec.categorical_value,
            )
            working[output_name] = left * right
        elif spec.transform_type in TEMPORAL_NUMERIC_TRANSFORMS:
            working[output_name] = self._apply_temporal_numeric_transform(
                working,
                source_name,
                context=context,
                transform_type=spec.transform_type,
                lag_periods=int(resolved.learned_parameters["lag_periods"]),
                window_size=int(resolved.learned_parameters["window_size"]),
            ).fillna(0.0)
        elif spec.transform_type in BINNING_TRANSFORMS:
            values = pd.to_numeric(working[source_name], errors="coerce")
            working[output_name] = self._apply_binned_feature(
                values,
                resolved.learned_parameters["bin_edges"],
            )
        elif spec.transform_type in ENCODING_TRANSFORMS:
            working[output_name] = self._apply_encoding_transform(
                working,
                spec,
                resolved.learned_parameters,
                context=context,
                split_name=split_name,
            )
        elif spec.transform_type in DATE_TRANSFORMS:
            working[output_name] = self._apply_date_transform(
                working,
                spec,
                resolved.learned_parameters,
                context=context,
            )
        elif spec.transform_type in ROW_MISSINGNESS_TRANSFORMS:
            working[output_name] = self._apply_missingness_transform(
                working,
                spec.transform_type,
                list(resolved.learned_parameters["source_features"]),
            )

        return working

    def _source_feature_names(self, spec: TransformationSpec) -> list[str]:
        if spec.transform_type in ROW_MISSINGNESS_TRANSFORMS:
            return [
                feature_name.strip()
                for feature_name in spec.source_feature.split(",")
                if feature_name.strip()
            ]
        return [spec.source_feature.strip()]

    def _validate_source_columns(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        spec: TransformationSpec,
        source_features: list[str],
    ) -> None:
        if not source_features:
            raise ValueError("Transformation source_feature cannot be blank.")
        if spec.transform_type not in ROW_MISSINGNESS_TRANSFORMS and len(source_features) != 1:
            raise ValueError(
                f"{spec.transform_type.value} supports exactly one source_feature. "
                "Use comma-separated source features only for row missingness transforms."
            )
        for source_feature in source_features:
            if source_feature not in train_frame.columns:
                raise ValueError(
                    f"Transformation source '{source_feature}' is missing from the train split."
                )
            if context.target_column and source_feature == context.target_column:
                raise ValueError("Governed transformations cannot use the target as source.")

        secondary_feature = (spec.secondary_feature or "").strip()
        if secondary_feature:
            if secondary_feature not in train_frame.columns:
                raise ValueError(
                    f"Secondary feature '{secondary_feature}' is missing from the train split."
                )
            if context.target_column and secondary_feature == context.target_column:
                raise ValueError("Governed transformations cannot use the target as secondary.")

    def _apply_numeric_scaling_transform(
        self,
        series: pd.Series,
        transform_type: TransformationType,
        parameters: dict[str, Any],
    ) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if transform_type == TransformationType.STANDARD_SCALE:
            std_value = float(parameters["std"])
            if std_value == 0.0:
                return pd.Series(0.0, index=series.index)
            return (values - float(parameters["mean"])) / std_value
        if transform_type == TransformationType.ROBUST_SCALE:
            iqr = float(parameters["iqr"])
            if iqr == 0.0:
                return pd.Series(0.0, index=series.index)
            return (values - float(parameters["median"])) / iqr
        if transform_type == TransformationType.MIN_MAX_SCALE:
            value_range = float(parameters["range"])
            if value_range == 0.0:
                return pd.Series(0.0, index=series.index)
            return (values - float(parameters["min"])) / value_range

        rank_values = np.asarray(parameters["rank_values"], dtype=float)
        rank_positions = np.asarray(parameters["rank_positions"], dtype=float)
        if rank_values.size <= 1:
            percentile = pd.Series(0.5, index=series.index)
        else:
            percentile = pd.Series(
                np.interp(
                    values.to_numpy(dtype=float),
                    rank_values,
                    rank_positions,
                    left=0.0,
                    right=1.0,
                ),
                index=series.index,
            )
            percentile.loc[values.isna()] = np.nan
        if transform_type == TransformationType.PERCENTILE_RANK:
            return percentile

        non_null_count = max(int(parameters.get("non_null_count", 1)), 1)
        epsilon = min(0.49, 1.0 / (2.0 * non_null_count))
        return pd.Series(norm.ppf(percentile.clip(epsilon, 1.0 - epsilon)), index=series.index)

    def _apply_numeric_direct_transform(
        self,
        series: pd.Series,
        transform_type: TransformationType,
        parameters: dict[str, Any],
    ) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if transform_type == TransformationType.SIGNED_LOG1P:
            return np.sign(values) * np.log1p(np.abs(values))
        if transform_type == TransformationType.SQRT:
            return pd.Series(np.where(values >= 0, np.sqrt(values), np.nan), index=series.index)
        if transform_type == TransformationType.RECIPROCAL:
            epsilon = float(parameters.get("epsilon", 1e-12))
            return pd.Series(
                np.where(values.abs() > epsilon, 1.0 / values, np.nan),
                index=series.index,
            )
        if transform_type == TransformationType.SQUARE:
            return values**2
        if transform_type == TransformationType.POWER:
            power = float(parameters["power"])
            with np.errstate(invalid="ignore"):
                transformed = np.power(values.to_numpy(dtype=float), power)
            return pd.Series(transformed, index=series.index)
        if transform_type == TransformationType.ABSOLUTE_VALUE:
            return values.abs()
        if transform_type in {TransformationType.CENTER_MEAN, TransformationType.CENTER_MEDIAN}:
            return values - float(parameters["center"])
        raise ValueError(f"Unsupported numeric transformation '{transform_type.value}'.")

    def _apply_pair_numeric_transform(
        self,
        left_series: pd.Series,
        right_series: pd.Series,
        transform_type: TransformationType,
        parameters: dict[str, Any],
    ) -> pd.Series:
        left = pd.to_numeric(left_series, errors="coerce")
        right = pd.to_numeric(right_series, errors="coerce")
        epsilon = float(parameters.get("epsilon", 1e-12))
        if transform_type in {
            TransformationType.RATIO,
            TransformationType.SAFE_RATIO,
            TransformationType.DEBT_SERVICE_RATIO,
        }:
            return left.div(right.where(right.abs() > epsilon))
        if transform_type == TransformationType.MARGIN_RATIO:
            return (left - right).div(left.where(left.abs() > epsilon).abs())
        if transform_type == TransformationType.ADD:
            return left + right
        if transform_type == TransformationType.SUBTRACT:
            return left - right
        if transform_type == TransformationType.PRODUCT:
            return left * right
        raise ValueError(f"Unsupported pair transformation '{transform_type.value}'.")

    def _fit_bin_edges(
        self,
        *,
        train_frame: pd.DataFrame,
        source_feature: str,
        spec: TransformationSpec,
        target_column: str | None,
    ) -> list[float]:
        if spec.transform_type == TransformationType.MANUAL_BINS and spec.bin_edges:
            return [float(value) for value in spec.bin_edges]

        values = pd.to_numeric(train_frame[source_feature], errors="coerce")
        non_null = values.dropna()
        if non_null.nunique() <= 1:
            return []

        bin_count = 5 if spec.parameter_value is None else int(spec.parameter_value)
        bin_count = max(2, min(bin_count, int(non_null.nunique())))
        if spec.transform_type == TransformationType.EQUAL_WIDTH_BINS:
            edges = np.linspace(float(non_null.min()), float(non_null.max()), bin_count + 1)[1:-1]
            return self._dedupe_edges(edges)

        if spec.transform_type == TransformationType.MONOTONIC_BINS and target_column:
            target = pd.to_numeric(train_frame[target_column], errors="coerce")
            for candidate_count in range(bin_count, 1, -1):
                candidate_edges = self._quantile_edges(non_null, candidate_count)
                if self._bins_are_monotonic(values, target, candidate_edges):
                    return candidate_edges

        return self._quantile_edges(non_null, bin_count)

    def _quantile_edges(self, non_null: pd.Series, bin_count: int) -> list[float]:
        _, raw_edges = pd.qcut(
            non_null,
            q=bin_count,
            retbins=True,
            duplicates="drop",
        )
        return self._dedupe_edges(raw_edges[1:-1])

    def _dedupe_edges(self, edges: Any) -> list[float]:
        return [
            float(value)
            for value in pd.Series(edges, dtype="float64")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        ]

    def _bins_are_monotonic(
        self,
        values: pd.Series,
        target: pd.Series,
        bin_edges: list[float],
    ) -> bool:
        binned = self._apply_binned_feature(values, bin_edges)
        grouped = (
            pd.DataFrame({"bin": binned.astype("string"), "target": target})
            .dropna()
            .groupby("bin", observed=False)["target"]
            .mean()
        )
        if grouped.shape[0] < 2:
            return False
        differences = grouped.diff().dropna()
        return bool((differences.ge(0).all()) or (differences.le(0).all()))

    def _apply_binned_feature(self, values: pd.Series, bin_edges: list[float]) -> pd.Series:
        bins = [-np.inf, *[float(edge) for edge in bin_edges], np.inf]
        return pd.cut(values, bins=bins, include_lowest=True, duplicates="drop").astype(
            "category"
        )

    def _fit_encoding_parameters(
        self,
        *,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        spec: TransformationSpec,
    ) -> dict[str, Any]:
        source_series = train_frame[spec.source_feature]
        if spec.transform_type == TransformationType.RARE_CATEGORY_COLLAPSE:
            keys = self._category_keys(source_series)
            min_share = 0.01 if spec.parameter_value is None else float(spec.parameter_value)
            shares = keys.value_counts(normalize=True, dropna=False)
            return {
                "min_share": min_share,
                "retained_levels": tuple(shares.loc[shares >= min_share].index.astype(str)),
            }
        if spec.transform_type == TransformationType.FREQUENCY_ENCODING:
            keys = self._category_keys(source_series)
            shares = keys.value_counts(normalize=True, dropna=False)
            return {
                "mapping": {str(key): float(value) for key, value in shares.items()},
                "default": 0.0,
            }
        if spec.transform_type == TransformationType.ORDINAL_ENCODING:
            keys = self._category_keys(source_series)
            levels = sorted(str(value) for value in keys.dropna().unique())
            return {
                "mapping": {level: float(index) for index, level in enumerate(levels)},
                "default": -1.0,
            }

        if not context.target_column or context.target_column not in train_frame.columns:
            raise ValueError(f"{spec.transform_type.value} requires a modeled target column.")
        target = pd.to_numeric(train_frame[context.target_column], errors="coerce")

        source_kind = "category"
        bin_edges: list[float] = []
        if spec.transform_type in {
            TransformationType.WOE_ENCODING,
            TransformationType.BAD_RATE_ENCODING,
        } and self._is_numeric_feature(source_series):
            bin_edges = (
                [float(edge) for edge in spec.bin_edges]
                if spec.bin_edges
                else self._fit_bin_edges(
                    train_frame=train_frame,
                    source_feature=spec.source_feature,
                    spec=TransformationSpec(
                        transform_type=TransformationType.QUANTILE_BINS,
                        source_feature=spec.source_feature,
                        parameter_value=spec.parameter_value,
                    ),
                    target_column=context.target_column,
                )
            )
            keys = self._binned_keys(pd.to_numeric(source_series, errors="coerce"), bin_edges)
            source_kind = "numeric_bin"
        else:
            keys = self._category_keys(source_series)

        if spec.transform_type == TransformationType.WOE_ENCODING:
            return self._fit_woe_mapping(keys, target, source_kind=source_kind, bin_edges=bin_edges)
        if spec.transform_type == TransformationType.BAD_RATE_ENCODING:
            return self._fit_mean_encoding_mapping(
                keys,
                target,
                source_kind=source_kind,
                bin_edges=bin_edges,
                output_name="bad_rate",
            )
        if spec.transform_type == TransformationType.TARGET_ENCODING:
            parameters = self._fit_mean_encoding_mapping(
                keys,
                target,
                source_kind=source_kind,
                bin_edges=bin_edges,
                output_name="target_mean",
            )
            parameters["smoothing"] = 20.0 if spec.parameter_value is None else float(
                spec.parameter_value
            )
            return parameters
        raise ValueError(f"Unsupported encoding transformation '{spec.transform_type.value}'.")

    def _fit_woe_mapping(
        self,
        keys: pd.Series,
        target: pd.Series,
        *,
        source_kind: str,
        bin_edges: list[float],
    ) -> dict[str, Any]:
        target_non_null = target.dropna()
        unique_values = set(float(value) for value in target_non_null.unique())
        if not unique_values.issubset({0.0, 1.0}):
            raise ValueError("woe_encoding requires a binary 0/1 target.")
        data = pd.DataFrame({"key": keys.astype(str), "target": target}).dropna()
        if data.empty:
            raise ValueError("woe_encoding requires non-missing source and target values.")
        grouped = data.groupby("key", observed=False)["target"].agg(["sum", "count"])
        event_total = float(grouped["sum"].sum())
        nonevent_total = float((grouped["count"] - grouped["sum"]).sum())
        if event_total <= 0 or nonevent_total <= 0:
            raise ValueError("woe_encoding requires at least one event and one non-event.")
        smoothing = 0.5
        bin_count = max(int(grouped.shape[0]), 1)
        mapping: dict[str, float] = {}
        for key, row in grouped.iterrows():
            events = float(row["sum"])
            nonevents = float(row["count"] - row["sum"])
            event_rate = (events + smoothing) / (event_total + smoothing * bin_count)
            nonevent_rate = (nonevents + smoothing) / (nonevent_total + smoothing * bin_count)
            mapping[str(key)] = float(np.log(nonevent_rate / event_rate))
        return {
            "source_kind": source_kind,
            "bin_edges": bin_edges,
            "mapping": mapping,
            "default": 0.0,
            "smoothing": smoothing,
        }

    def _fit_mean_encoding_mapping(
        self,
        keys: pd.Series,
        target: pd.Series,
        *,
        source_kind: str,
        bin_edges: list[float],
        output_name: str,
    ) -> dict[str, Any]:
        data = pd.DataFrame({"key": keys.astype(str), "target": target}).dropna()
        if data.empty:
            raise ValueError(f"{output_name} encoding requires usable source and target values.")
        grouped = data.groupby("key", observed=False)["target"].agg(["sum", "count", "mean"])
        global_mean = float(data["target"].mean())
        return {
            "source_kind": source_kind,
            "bin_edges": bin_edges,
            "mapping": {str(key): float(value) for key, value in grouped["mean"].items()},
            "counts": {str(key): float(value) for key, value in grouped["count"].items()},
            "sums": {str(key): float(value) for key, value in grouped["sum"].items()},
            "global_mean": global_mean,
            "default": global_mean,
        }

    def _apply_encoding_transform(
        self,
        frame: pd.DataFrame,
        spec: TransformationSpec,
        parameters: dict[str, Any],
        *,
        context: PipelineContext,
        split_name: str | None,
    ) -> pd.Series:
        keys = self._encoding_keys_for_frame(frame, spec, parameters)
        if spec.transform_type == TransformationType.RARE_CATEGORY_COLLAPSE:
            retained = set(str(value) for value in parameters["retained_levels"])
            collapsed = keys.astype(str).where(keys.astype(str).isin(retained), "Other")
            collapsed = collapsed.where(keys.astype(str).ne("Missing"), "Missing")
            return collapsed.astype("category")

        mapping = parameters.get("mapping", {})
        default_value = float(parameters.get("default", 0.0))
        encoded = keys.astype(str).map(mapping).astype(float).fillna(default_value)
        if spec.transform_type != TransformationType.TARGET_ENCODING:
            return encoded

        if (
            split_name != "train"
            or not context.target_column
            or context.target_column not in frame.columns
        ):
            return encoded

        counts = keys.astype(str).map(parameters.get("counts", {})).astype(float)
        sums = keys.astype(str).map(parameters.get("sums", {})).astype(float)
        target = pd.to_numeric(frame[context.target_column], errors="coerce")
        smoothing = float(parameters.get("smoothing", 20.0))
        global_mean = float(parameters.get("global_mean", default_value))
        denominator = (counts - 1.0 + smoothing).where(counts > 1.0)
        loo_values = ((sums - target) + smoothing * global_mean).div(denominator)
        return loo_values.fillna(global_mean)

    def _encoding_keys_for_frame(
        self,
        frame: pd.DataFrame,
        spec: TransformationSpec,
        parameters: dict[str, Any],
    ) -> pd.Series:
        if parameters.get("source_kind") == "numeric_bin":
            return self._binned_keys(
                pd.to_numeric(frame[spec.source_feature], errors="coerce"),
                list(parameters.get("bin_edges", [])),
            )
        return self._category_keys(frame[spec.source_feature])

    def _category_keys(self, series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("Missing").astype(str)

    def _binned_keys(self, values: pd.Series, bin_edges: list[float]) -> pd.Series:
        return self._apply_binned_feature(values, bin_edges).astype("string").fillna("Missing")

    def _fit_date_parameters(
        self,
        *,
        train_frame: pd.DataFrame,
        source_feature: str,
        spec: TransformationSpec,
        context: PipelineContext,
    ) -> dict[str, Any]:
        fiscal_start_month = 1 if spec.parameter_value is None else int(spec.parameter_value)
        fiscal_start_month = min(max(fiscal_start_month, 1), 12)
        reference_column = (spec.secondary_feature or "").strip()
        split_date_column = context.config.split.date_column
        if (
            not reference_column
            and split_date_column
            and split_date_column != source_feature
            and split_date_column in train_frame.columns
        ):
            reference_column = split_date_column
        reference_date = None
        if not reference_column:
            source_dates = pd.to_datetime(train_frame[source_feature], errors="coerce")
            reference_date = source_dates.max()
        return {
            "fiscal_start_month": fiscal_start_month,
            "reference_column": reference_column,
            "reference_date": reference_date,
        }

    def _apply_date_transform(
        self,
        frame: pd.DataFrame,
        spec: TransformationSpec,
        parameters: dict[str, Any],
        *,
        context: PipelineContext,
    ) -> pd.Series:
        del context
        values = pd.to_datetime(frame[spec.source_feature], errors="coerce")
        if spec.transform_type == TransformationType.DATE_YEAR:
            return values.dt.year.astype("Float64")
        if spec.transform_type == TransformationType.DATE_MONTH:
            return values.dt.month.astype("Float64")
        if spec.transform_type == TransformationType.DATE_QUARTER:
            return values.dt.quarter.astype("Float64")
        if spec.transform_type == TransformationType.DATE_MONTH_END_FLAG:
            return values.dt.is_month_end.astype(float)
        if spec.transform_type == TransformationType.DATE_FISCAL_QUARTER:
            fiscal_start = int(parameters.get("fiscal_start_month", 1))
            fiscal_month = ((values.dt.month - fiscal_start) % 12) + 1
            return (((fiscal_month - 1) // 3) + 1).astype("Float64")

        reference_column = str(parameters.get("reference_column") or "")
        if reference_column and reference_column in frame.columns:
            reference_values = pd.to_datetime(frame[reference_column], errors="coerce")
        else:
            reference_values = pd.Series(parameters.get("reference_date"), index=frame.index)
        age_days = (reference_values - values).dt.days.astype("Float64")
        if spec.transform_type == TransformationType.DATE_AGE_DAYS:
            return age_days
        return (age_days / 30.4375).astype("Float64")

    def _apply_missingness_transform(
        self,
        frame: pd.DataFrame,
        transform_type: TransformationType,
        source_features: list[str],
    ) -> pd.Series:
        missing_frame = frame.loc[:, source_features].isna()
        missing_count = missing_frame.sum(axis=1).astype(float)
        if transform_type == TransformationType.ROW_MISSING_COUNT:
            return missing_count
        if transform_type == TransformationType.ROW_MISSING_SHARE:
            return missing_count / max(len(source_features), 1)
        if transform_type == TransformationType.ANY_MISSING_FLAG:
            return missing_count.gt(0).astype(float)
        raise ValueError(f"Unsupported missingness transformation '{transform_type.value}'.")

    def _resolve_auto_interactions(
        self,
        *,
        context: PipelineContext,
        config: TransformationConfig,
        working_splits: dict[str, pd.DataFrame],
    ) -> list[TransformationSpec]:
        existing_generated_specs = [
            spec for spec in config.transformations if spec.generated_automatically
        ]
        if existing_generated_specs:
            return existing_generated_specs
        if not config.auto_interactions_enabled:
            return []
        if context.config.execution.mode != ExecutionMode.FIT_NEW_MODEL:
            context.warn(
                "Auto interaction generation is skipped for score-existing-model runs. "
                "Saved generated interactions from the original development run are still applied."
            )
            return []
        train_frame = working_splits.get("train")
        if train_frame is None or context.target_column not in train_frame.columns:
            return []
        target_series = pd.to_numeric(train_frame[context.target_column], errors="coerce")
        if target_series.dropna().nunique() < 2:
            return []

        candidate_rows: list[dict[str, Any]] = []
        if config.include_numeric_numeric_interactions:
            candidate_rows.extend(
                self._screen_numeric_interactions(context, train_frame, target_series)
            )
        if config.include_categorical_numeric_interactions:
            candidate_rows.extend(
                self._screen_categorical_numeric_interactions(context, train_frame, target_series)
            )
        if not candidate_rows:
            return []

        candidate_frame = pd.DataFrame(candidate_rows).sort_values(
            ["score", "candidate_type"],
            ascending=[False, True],
        )
        candidate_frame = candidate_frame.loc[
            candidate_frame["score"] >= float(config.min_interaction_score)
        ].copy(deep=True)

        selected_specs: list[TransformationSpec] = []
        used_outputs = {self._resolve_output_feature_name(spec) for spec in config.transformations}
        candidate_frame["selected"] = False
        for index, row in candidate_frame.iterrows():
            if len(selected_specs) >= config.max_auto_interactions:
                break
            spec = TransformationSpec(
                transform_type=TransformationType.INTERACTION,
                source_feature=str(row["source_feature"]),
                secondary_feature=str(row["secondary_feature"]),
                categorical_value=(
                    None
                    if not str(row.get("categorical_value", "")).strip()
                    else str(row["categorical_value"]).strip()
                ),
                output_feature=str(row["output_feature"]),
                enabled=True,
                generated_automatically=True,
                notes=f"Auto-screened interaction (score={float(row['score']):.4f}).",
            )
            output_name = self._resolve_output_feature_name(spec)
            if output_name in used_outputs:
                continue
            used_outputs.add(output_name)
            selected_specs.append(spec)
            candidate_frame.loc[index, "selected"] = True

        if selected_specs:
            config.transformations.extend(selected_specs)
            config.enabled = True
            context.diagnostics_tables["interaction_candidates"] = candidate_frame
        return selected_specs

    def _screen_numeric_interactions(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        target_series: pd.Series,
    ) -> list[dict[str, Any]]:
        numeric_candidates = [
            feature_name
            for feature_name in context.numeric_features
            if feature_name in train_frame.columns and feature_name != context.target_column
        ]
        rows: list[dict[str, Any]] = []
        for left_feature, right_feature in combinations(numeric_candidates, 2):
            interaction_series = pd.to_numeric(
                train_frame[left_feature], errors="coerce"
            ) * pd.to_numeric(
                train_frame[right_feature],
                errors="coerce",
            )
            score = self._interaction_score(interaction_series, target_series)
            if score is None:
                continue
            rows.append(
                {
                    "candidate_type": "numeric_numeric",
                    "source_feature": left_feature,
                    "secondary_feature": right_feature,
                    "categorical_value": "",
                    "output_feature": f"{left_feature}_x_{right_feature}",
                    "score": score,
                }
            )
        return rows

    def _screen_categorical_numeric_interactions(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        target_series: pd.Series,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        numeric_candidates = [
            feature_name
            for feature_name in context.numeric_features
            if feature_name in train_frame.columns and feature_name != context.target_column
        ]
        categorical_candidates = [
            feature_name
            for feature_name in context.categorical_features
            if feature_name in train_frame.columns
        ]
        for categorical_feature in categorical_candidates:
            levels = (
                train_frame[categorical_feature]
                .astype("string")
                .fillna("Missing")
                .value_counts(dropna=False)
                .head(context.config.transformations.max_categorical_levels)
                .index.tolist()
            )
            for level in levels:
                indicator = (
                    train_frame[categorical_feature]
                    .astype("string")
                    .fillna("Missing")
                    .eq(level)
                    .astype(float)
                )
                for numeric_feature in numeric_candidates:
                    interaction_series = indicator * pd.to_numeric(
                        train_frame[numeric_feature],
                        errors="coerce",
                    )
                    score = self._interaction_score(interaction_series, target_series)
                    if score is None:
                        continue
                    rows.append(
                        {
                            "candidate_type": "categorical_numeric",
                            "source_feature": numeric_feature,
                            "secondary_feature": categorical_feature,
                            "categorical_value": str(level),
                            "output_feature": (
                                f"{numeric_feature}_x_{categorical_feature}_"
                                f"{self._sanitize_token(str(level))}"
                            ),
                            "score": score,
                        }
                    )
        return rows

    def _interaction_score(
        self,
        interaction_series: pd.Series,
        target_series: pd.Series,
    ) -> float | None:
        aligned = pd.DataFrame(
            {"interaction": interaction_series, "target": target_series}
        ).dropna()
        if len(aligned) < 10 or aligned["interaction"].nunique() < 2:
            return None
        correlation = aligned["interaction"].corr(aligned["target"])
        if pd.isna(correlation):
            return None
        return float(abs(correlation))

    def _interaction_roles(
        self,
        frame: pd.DataFrame,
        spec: TransformationSpec,
    ) -> tuple[str, str]:
        source_is_numeric = self._is_numeric_feature(frame[spec.source_feature])
        secondary_series = frame[spec.secondary_feature or ""]
        secondary_is_numeric = self._is_numeric_feature(secondary_series)
        if source_is_numeric and secondary_is_numeric:
            return "numeric", "numeric"
        if (
            source_is_numeric
            and not secondary_is_numeric
            and (spec.categorical_value or "").strip()
        ):
            return "numeric", "categorical"
        if (
            secondary_is_numeric
            and not source_is_numeric
            and (spec.categorical_value or "").strip()
        ):
            return "categorical", "numeric"
        raise ValueError(
            "interaction transformations support numeric-numeric pairs or one categorical "
            "feature when categorical_value is provided."
        )

    def _interaction_operand(
        self,
        frame: pd.DataFrame,
        feature_name: str,
        role: str,
        categorical_value: str | None,
    ) -> pd.Series:
        if role == "numeric":
            return pd.to_numeric(frame[feature_name], errors="coerce")
        category_value = "" if categorical_value is None else categorical_value
        return (
            frame[feature_name].astype("string").fillna("Missing").eq(category_value).astype(float)
        )

    def _apply_temporal_numeric_transform(
        self,
        frame: pd.DataFrame,
        feature_name: str,
        *,
        context: PipelineContext,
        transform_type: TransformationType,
        lag_periods: int,
        window_size: int,
    ) -> pd.Series:
        ordered = frame.copy(deep=False)
        ordered["_transform_value"] = pd.to_numeric(frame[feature_name], errors="coerce")
        ordered["_transform_row_position"] = np.arange(len(frame), dtype=np.int64)
        ordered = self._sort_temporal_frame(ordered, context)
        split_config = context.config.split
        group_column = (
            split_config.entity_column
            if split_config.data_structure == DataStructure.PANEL
            and split_config.entity_column
            and split_config.entity_column in ordered.columns
            else None
        )
        values = ordered["_transform_value"]
        grouped = (
            ordered.groupby(group_column, sort=False)["_transform_value"]
            if group_column
            else None
        )

        if transform_type == TransformationType.LAG:
            transformed = grouped.shift(lag_periods) if grouped is not None else values.shift(
                lag_periods
            )
        elif transform_type == TransformationType.DIFFERENCE:
            transformed = grouped.diff(periods=lag_periods) if grouped is not None else values.diff(
                periods=lag_periods
            )
        elif transform_type == TransformationType.EWMA:
            shifted = grouped.shift(1) if grouped is not None else values.shift(1)
            if group_column:
                transformed = shifted.groupby(ordered[group_column], sort=False).transform(
                    lambda series: series.ewm(span=window_size, adjust=False, min_periods=1).mean()
                )
            else:
                transformed = shifted.ewm(span=window_size, adjust=False, min_periods=1).mean()
        elif transform_type in {
            TransformationType.ROLLING_MEAN,
            TransformationType.ROLLING_MEDIAN,
            TransformationType.ROLLING_MIN,
            TransformationType.ROLLING_MAX,
            TransformationType.ROLLING_STD,
            TransformationType.ROLLING_SUM,
            TransformationType.ROLLING_RANGE,
            TransformationType.ROLLING_CV,
            TransformationType.ROLLING_SLOPE,
        }:
            shifted = grouped.shift(1) if grouped is not None else values.shift(1)
            transformed = self._rolling_temporal_transform(
                shifted,
                transform_type,
                window_size=window_size,
                group_keys=ordered[group_column] if group_column else None,
            )
        elif transform_type == TransformationType.EXPANDING_MEAN:
            shifted = grouped.shift(1) if grouped is not None else values.shift(1)
            if group_column:
                transformed = shifted.groupby(ordered[group_column], sort=False).transform(
                    lambda series: series.expanding(min_periods=1).mean()
                )
            else:
                transformed = shifted.expanding(min_periods=1).mean()
        elif transform_type == TransformationType.CUMULATIVE_SUM:
            shifted = grouped.shift(1) if grouped is not None else values.shift(1)
            if group_column:
                transformed = shifted.groupby(ordered[group_column], sort=False).cumsum()
            else:
                transformed = shifted.cumsum()
        elif transform_type == TransformationType.CUMULATIVE_COUNT:
            if group_column:
                transformed = ordered.groupby(group_column, sort=False).cumcount().astype(float)
            else:
                transformed = pd.Series(np.arange(len(ordered), dtype=float), index=ordered.index)
        elif transform_type == TransformationType.MONTHS_SINCE_EVENT:
            transformed = self._months_since_event(
                ordered,
                value_column="_transform_value",
                context=context,
                group_column=group_column,
            )
        elif transform_type == TransformationType.CHANGE_FROM_BASELINE:
            if group_column:
                baseline = grouped.transform("first")
            else:
                baseline_value = values.dropna().iloc[0] if values.notna().any() else np.nan
                baseline = pd.Series(baseline_value, index=ordered.index)
            transformed = values - baseline
        elif transform_type == TransformationType.PCT_CHANGE:
            transformed = (
                grouped.pct_change(periods=lag_periods)
                if grouped is not None
                else values.pct_change(periods=lag_periods)
            )
            transformed = transformed.replace([np.inf, -np.inf], np.nan)
        else:
            raise ValueError(f"Unsupported temporal transformation '{transform_type.value}'.")
        ordered["_transformed"] = transformed
        restored = ordered.sort_values("_transform_row_position", kind="mergesort")[
            "_transformed"
        ]
        restored.index = frame.index
        return restored

    def _rolling_temporal_transform(
        self,
        values: pd.Series,
        transform_type: TransformationType,
        *,
        window_size: int,
        group_keys: pd.Series | None,
    ) -> pd.Series:
        if group_keys is not None:
            rolling = values.groupby(group_keys, sort=False).rolling(window_size, min_periods=1)
            transformed = self._apply_rolling_operation(rolling, transform_type)
            return transformed.reset_index(level=0, drop=True)
        rolling = values.rolling(window_size, min_periods=1)
        return self._apply_rolling_operation(rolling, transform_type)

    def _apply_rolling_operation(
        self,
        rolling: Any,
        transform_type: TransformationType,
    ) -> pd.Series:
        if transform_type == TransformationType.ROLLING_MEAN:
            return rolling.mean()
        if transform_type == TransformationType.ROLLING_MEDIAN:
            return rolling.median()
        if transform_type == TransformationType.ROLLING_MIN:
            return rolling.min()
        if transform_type == TransformationType.ROLLING_MAX:
            return rolling.max()
        if transform_type == TransformationType.ROLLING_STD:
            return rolling.std(ddof=0)
        if transform_type == TransformationType.ROLLING_SUM:
            return rolling.sum()
        if transform_type == TransformationType.ROLLING_RANGE:
            return rolling.max() - rolling.min()
        if transform_type == TransformationType.ROLLING_CV:
            mean_values = rolling.mean()
            std_values = rolling.std(ddof=0)
            return std_values.div(mean_values.replace({0: np.nan}))
        if transform_type == TransformationType.ROLLING_SLOPE:
            return rolling.apply(self._rolling_slope, raw=True)
        raise ValueError(f"Unsupported rolling transformation '{transform_type.value}'.")

    @staticmethod
    def _rolling_slope(values: np.ndarray) -> float:
        valid_values = pd.Series(values).dropna().to_numpy(dtype=float)
        if valid_values.size < 2:
            return np.nan
        x_values = np.arange(valid_values.size, dtype=float)
        return float(np.polyfit(x_values, valid_values, 1)[0])

    def _months_since_event(
        self,
        frame: pd.DataFrame,
        *,
        value_column: str,
        context: PipelineContext,
        group_column: str | None,
    ) -> pd.Series:
        date_column = context.config.split.date_column

        def calculate(group: pd.DataFrame) -> pd.Series:
            event_mask = pd.to_numeric(group[value_column], errors="coerce").fillna(0).gt(0)
            if date_column and date_column in group.columns:
                dates = pd.to_datetime(group[date_column], errors="coerce")
                last_event_dates = dates.where(event_mask).ffill()
                months = (dates - last_event_dates).dt.days / 30.4375
                return months.fillna(np.nan)
            event_positions = pd.Series(np.arange(len(group), dtype=float), index=group.index)
            last_event_position = event_positions.where(event_mask).ffill()
            return event_positions - last_event_position

        if group_column:
            return frame.groupby(group_column, sort=False, group_keys=False).apply(calculate)
        return calculate(frame)

    def _sort_temporal_frame(
        self,
        frame: pd.DataFrame,
        context: PipelineContext,
    ) -> pd.DataFrame:
        split_config = context.config.split
        sort_columns: list[str] = []
        if (
            split_config.data_structure == DataStructure.PANEL
            and split_config.entity_column
            and split_config.entity_column in frame.columns
        ):
            sort_columns.append(split_config.entity_column)
        if split_config.date_column and split_config.date_column in frame.columns:
            sort_columns.append(split_config.date_column)
        if not sort_columns:
            return frame
        return frame.sort_values(sort_columns, kind="mergesort")

    def _group_temporal_series(
        self,
        frame: pd.DataFrame,
        context: PipelineContext,
    ):
        split_config = context.config.split
        if (
            split_config.data_structure == DataStructure.PANEL
            and split_config.entity_column
            and split_config.entity_column in frame.columns
        ):
            return frame.groupby(split_config.entity_column, sort=False)["_transform_value"]
        return frame["_transform_value"]

    def _update_feature_contract(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
        output_feature: str | list[str] | tuple[str, ...],
    ) -> None:
        output_features = (
            [output_feature]
            if isinstance(output_feature, str)
            else list(output_feature)
        )
        for feature_name in output_features:
            if feature_name not in context.feature_columns:
                context.feature_columns.append(feature_name)
        context.numeric_features = []
        context.categorical_features = []
        for feature_name in context.feature_columns:
            if feature_name not in dataframe.columns:
                continue
            series = dataframe[feature_name]
            if self._is_numeric_feature(series):
                context.numeric_features.append(feature_name)
            else:
                context.categorical_features.append(feature_name)
        context.metadata["feature_summary"] = {
            "feature_count": len(context.feature_columns),
            "numeric_feature_count": len(context.numeric_features),
            "categorical_feature_count": len(context.categorical_features),
        }

    def _resolve_output_feature_name(self, spec: TransformationSpec) -> str:
        return TransformationConfig()._resolve_output_name(spec)

    def _numeric_series(self, series: pd.Series, feature_name: str) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.dropna().empty:
            raise ValueError(
                f"Transformation feature '{feature_name}' does not contain usable numeric values."
            )
        return numeric_series

    def _is_numeric_feature(self, series: pd.Series) -> bool:
        return pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)

    def _render_parameters(self, parameters: dict[str, Any]) -> str:
        rendered: list[str] = []
        for key, value in parameters.items():
            if key == "design_info":
                continue
            if key == "output_features":
                rendered.append(f"{key}={', '.join(str(item) for item in value)}")
                continue
            if isinstance(value, dict):
                preview = ", ".join(str(item) for item in list(value)[:5])
                suffix = "" if len(value) <= 5 else "..."
                rendered.append(f"{key}={len(value)} entries [{preview}{suffix}]")
                continue
            if isinstance(value, (list, tuple, set)) and len(value) > 12:
                preview = ", ".join(str(item) for item in list(value)[:5])
                rendered.append(f"{key}={len(value)} values [{preview}...]")
                continue
            if isinstance(value, float):
                rendered.append(f"{key}={value:.6g}")
            else:
                rendered.append(f"{key}={value}")
        return ", ".join(rendered)

    def _sanitize_token(self, value: str) -> str:
        sanitized = "".join(character if character.isalnum() else "_" for character in value)
        return sanitized.strip("_").lower() or "value"
