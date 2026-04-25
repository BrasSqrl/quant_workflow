"""Applies governed feature transformations after split-time imputation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import boxcox, boxcox_normmax, yeojohnson, yeojohnson_normmax

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
            split_name: frame.copy(deep=True) for split_name, frame in context.split_frames.items()
        }
        working_dataframe = context.working_data.copy(deep=True)
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
            if spec.transform_type == TransformationType.RATIO:
                self._numeric_series(source_series, spec.source_feature)
                self._numeric_series(train_frame[secondary_feature], secondary_feature)
            else:
                left_role, right_role = self._interaction_roles(train_frame, spec)
                learned_parameters = {
                    "left_role": left_role,
                    "right_role": right_role,
                }
        elif spec.transform_type in {
            TransformationType.LAG,
            TransformationType.DIFFERENCE,
            TransformationType.EWMA,
            TransformationType.ROLLING_MEAN,
            TransformationType.ROLLING_MEDIAN,
            TransformationType.ROLLING_MIN,
            TransformationType.ROLLING_MAX,
            TransformationType.ROLLING_STD,
            TransformationType.PCT_CHANGE,
        }:
            self._numeric_series(source_series, spec.source_feature)
            learned_parameters = {
                "lag_periods": 1 if spec.lag_periods is None else int(spec.lag_periods),
                "window_size": 3 if spec.window_size is None else int(spec.window_size),
            }
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
        *,
        context: PipelineContext,
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
        elif spec.transform_type == TransformationType.RATIO:
            numerator = pd.to_numeric(working[source_name], errors="coerce")
            denominator = pd.to_numeric(
                working[spec.secondary_feature or ""],
                errors="coerce",
            )
            working[output_name] = numerator.div(denominator.replace({0: np.nan}))
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
        elif spec.transform_type in {
            TransformationType.LAG,
            TransformationType.DIFFERENCE,
            TransformationType.EWMA,
            TransformationType.ROLLING_MEAN,
            TransformationType.ROLLING_MEDIAN,
            TransformationType.ROLLING_MIN,
            TransformationType.ROLLING_MAX,
            TransformationType.ROLLING_STD,
            TransformationType.PCT_CHANGE,
        }:
            working[output_name] = self._apply_temporal_numeric_transform(
                working,
                source_name,
                context=context,
                transform_type=spec.transform_type,
                lag_periods=int(resolved.learned_parameters["lag_periods"]),
                window_size=int(resolved.learned_parameters["window_size"]),
            ).fillna(0.0)
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
        ordered = frame.copy(deep=True)
        ordered["_transform_value"] = pd.to_numeric(frame[feature_name], errors="coerce")
        ordered["_original_index"] = frame.index
        ordered = self._sort_temporal_frame(ordered, context)
        grouped = self._group_temporal_series(ordered, context)
        if transform_type == TransformationType.LAG:
            transformed = grouped.shift(lag_periods)
        elif transform_type == TransformationType.DIFFERENCE:
            transformed = grouped.diff(periods=lag_periods)
        elif transform_type == TransformationType.EWMA:
            transformed = grouped.shift(1).ewm(
                span=window_size,
                adjust=False,
                min_periods=1,
            ).mean()
        elif transform_type == TransformationType.ROLLING_MEAN:
            transformed = grouped.shift(1).rolling(window_size, min_periods=1).mean()
        elif transform_type == TransformationType.ROLLING_MEDIAN:
            transformed = grouped.shift(1).rolling(window_size, min_periods=1).median()
        elif transform_type == TransformationType.ROLLING_MIN:
            transformed = grouped.shift(1).rolling(window_size, min_periods=1).min()
        elif transform_type == TransformationType.ROLLING_MAX:
            transformed = grouped.shift(1).rolling(window_size, min_periods=1).max()
        elif transform_type == TransformationType.ROLLING_STD:
            transformed = grouped.shift(1).rolling(window_size, min_periods=1).std(ddof=0)
        elif transform_type == TransformationType.PCT_CHANGE:
            transformed = grouped.pct_change(periods=lag_periods).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        else:
            raise ValueError(f"Unsupported temporal transformation '{transform_type.value}'.")
        ordered["_transformed"] = transformed
        restored = ordered.sort_values("_original_index", kind="mergesort")["_transformed"]
        restored.index = frame.index
        return restored

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
        configured_output = (spec.output_feature or "").strip()
        if configured_output:
            return configured_output
        if spec.transform_type == TransformationType.MANUAL_BINS:
            return f"{spec.source_feature}_binned"
        if spec.transform_type == TransformationType.RATIO:
            return f"{spec.source_feature}_over_{spec.secondary_feature}"
        if spec.transform_type == TransformationType.INTERACTION:
            if (spec.categorical_value or "").strip():
                return (
                    f"{spec.source_feature}_x_{spec.secondary_feature}_"
                    f"{self._sanitize_token(spec.categorical_value)}"
                )
            return f"{spec.source_feature}_x_{spec.secondary_feature}"
        if spec.transform_type == TransformationType.YEO_JOHNSON:
            return f"{spec.source_feature}_yeo_johnson"
        if spec.transform_type == TransformationType.BOX_COX:
            return f"{spec.source_feature}_box_cox"
        if spec.transform_type == TransformationType.NATURAL_SPLINE:
            spline_df = 4 if spec.parameter_value is None else int(spec.parameter_value)
            return f"{spec.source_feature}_spline_df_{spline_df}"
        if spec.transform_type == TransformationType.CAPPED_ZSCORE:
            return f"{spec.source_feature}_zscore"
        if spec.transform_type == TransformationType.PIECEWISE_LINEAR:
            hinge_value = "hinge" if spec.parameter_value is None else str(spec.parameter_value)
            return f"{spec.source_feature}_piecewise_{self._sanitize_token(hinge_value)}"
        if spec.transform_type == TransformationType.LAG:
            return (
                f"{spec.source_feature}_lag_{1 if spec.lag_periods is None else spec.lag_periods}"
            )
        if spec.transform_type == TransformationType.DIFFERENCE:
            return (
                f"{spec.source_feature}_diff_{1 if spec.lag_periods is None else spec.lag_periods}"
            )
        if spec.transform_type == TransformationType.EWMA:
            return (
                f"{spec.source_feature}_ewma_{3 if spec.window_size is None else spec.window_size}"
            )
        if spec.transform_type == TransformationType.ROLLING_MEAN:
            return (
                f"{spec.source_feature}_rollmean_"
                f"{3 if spec.window_size is None else spec.window_size}"
            )
        if spec.transform_type == TransformationType.ROLLING_MEDIAN:
            return (
                f"{spec.source_feature}_rollmedian_"
                f"{3 if spec.window_size is None else spec.window_size}"
            )
        if spec.transform_type == TransformationType.ROLLING_MIN:
            return (
                f"{spec.source_feature}_rollmin_"
                f"{3 if spec.window_size is None else spec.window_size}"
            )
        if spec.transform_type == TransformationType.ROLLING_MAX:
            return (
                f"{spec.source_feature}_rollmax_"
                f"{3 if spec.window_size is None else spec.window_size}"
            )
        if spec.transform_type == TransformationType.ROLLING_STD:
            return (
                f"{spec.source_feature}_rollstd_"
                f"{3 if spec.window_size is None else spec.window_size}"
            )
        if spec.transform_type == TransformationType.PCT_CHANGE:
            return (
                f"{spec.source_feature}_pct_change_"
                f"{1 if spec.lag_periods is None else spec.lag_periods}"
            )
        return spec.source_feature

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
            if isinstance(value, float):
                rendered.append(f"{key}={value:.6g}")
            else:
                rendered.append(f"{key}={value}")
        return ", ".join(rendered)

    def _sanitize_token(self, value: str) -> str:
        sanitized = "".join(character if character.isalnum() else "_" for character in value)
        return sanitized.strip("_").lower() or "value"
