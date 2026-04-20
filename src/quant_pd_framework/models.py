"""Model adapters that provide a common interface across supported estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.othermod.betareg import BetaModel
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    ModelConfig,
    ModelType,
    ScorecardConfig,
    ScorecardMonotonicity,
    TargetMode,
)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Builds a dense preprocessing pipeline shared by several model adapters."""

    transformers = []
    if numeric_features:
        numeric_steps: list[tuple[str, Any]] = []
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps) if numeric_steps else "passthrough",
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("Training requires at least one numeric or categorical feature.")

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


class BaseModelAdapter(ABC):
    """Common interface used by training, evaluation, and export steps."""

    model_type: ModelType
    target_mode: TargetMode

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        self.model_config = model_config
        self.target_mode = target_mode
        self.feature_names_: list[str] = []
        self.raw_feature_names_: list[str] = []
        self.raw_numeric_features_: list[str] = []
        self.raw_categorical_features_: list[str] = []

    @property
    def is_binary_classifier(self) -> bool:
        return self.target_mode == TargetMode.BINARY

    @property
    def summary_text(self) -> str:
        return f"Model type: {self.model_type.value}"

    @abstractmethod
    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> BaseModelAdapter:
        """Fits the model on the provided dataframe and target."""

    @abstractmethod
    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        """Returns probability-like scores for binary targets or predicted values otherwise."""

    def predict_class(self, x_frame: pd.DataFrame, threshold: float) -> np.ndarray | None:
        """Returns thresholded class predictions for binary targets when available."""

        if not self.is_binary_classifier:
            return None
        return (self.predict_score(x_frame) >= threshold).astype(int)

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Returns a table describing feature importance or coefficients."""

    def get_model_artifacts(self) -> dict[str, pd.DataFrame]:
        """Returns optional model-specific tables that support documentation/export."""

        return {}

    def get_prediction_outputs(self, x_frame: pd.DataFrame) -> dict[str, np.ndarray | list[str]]:
        """Returns optional prediction-side outputs such as scorecard points."""

        return {}


class SklearnAdapter(BaseModelAdapter):
    """Base adapter for sklearn-style estimators with an external preprocessor."""

    def __init__(
        self,
        model_config: ModelConfig,
        target_mode: TargetMode,
        estimator: Any,
        *,
        scale_numeric: bool = True,
    ) -> None:
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.estimator = estimator
        self.scale_numeric = scale_numeric

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> SklearnAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            scale_numeric=self.scale_numeric,
        )
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        self.estimator.fit(x_matrix, y_values)
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def get_feature_importance(self) -> pd.DataFrame:
        if hasattr(self.estimator, "coef_"):
            coefficients = np.ravel(self.estimator.coef_)
            table = pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(coefficients),
                    "importance_type": "absolute_coefficient",
                    "coefficient": coefficients,
                    "abs_coefficient": np.abs(coefficients),
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.exp(coefficients) if self.is_binary_classifier else np.nan,
                }
            )
        else:
            importances = getattr(self.estimator, "feature_importances_", None)
            if importances is None:
                raise ValueError(
                    "This estimator does not expose coefficients or feature importances."
                )
            table = pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": importances,
                    "importance_type": "model_importance",
                    "coefficient": np.nan,
                    "abs_coefficient": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.nan,
                }
            )

        return table.sort_values("importance_value", ascending=False).reset_index(drop=True)


class LogisticRegressionAdapter(SklearnAdapter):
    """Sklearn logistic regression model for binary default prediction."""

    model_type = ModelType.LOGISTIC_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.BINARY:
            raise ValueError("Logistic regression requires a binary target.")
        super().__init__(
            model_config,
            target_mode,
            LogisticRegression(
                max_iter=model_config.max_iter,
                C=model_config.C,
                solver=model_config.solver,
                class_weight=model_config.class_weight,
            ),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._transform(x_frame))[:, 1]

    @property
    def summary_text(self) -> str:
        return (
            "Logistic regression fitted with sklearn.\n"
            f"Solver: {self.model_config.solver}\n"
            f"Max iterations: {self.model_config.max_iter}\n"
            f"C: {self.model_config.C}\n"
            f"Class weight: {self.model_config.class_weight}\n"
        )


class DiscreteTimeHazardModelAdapter(LogisticRegressionAdapter):
    """Pooled-logit discrete-time hazard model for lifetime PD development."""

    model_type = ModelType.DISCRETE_TIME_HAZARD_MODEL

    @property
    def summary_text(self) -> str:
        return (
            "Discrete-time hazard model fitted as a pooled logistic regression.\n"
            "This is intended for lifetime PD and CECL-style person-period development.\n"
            f"Solver: {self.model_config.solver}\n"
            f"Max iterations: {self.model_config.max_iter}\n"
            f"C: {self.model_config.C}\n"
            f"Class weight: {self.model_config.class_weight}\n"
        )


class ElasticNetLogisticRegressionAdapter(SklearnAdapter):
    """Elastic-net logistic regression for sparse and collinear binary PD features."""

    model_type = ModelType.ELASTIC_NET_LOGISTIC_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.BINARY:
            raise ValueError("Elastic-net logistic regression requires a binary target.")
        super().__init__(
            model_config,
            target_mode,
            LogisticRegression(
                solver="saga",
                l1_ratio=model_config.l1_ratio,
                max_iter=model_config.max_iter,
                C=model_config.C,
                class_weight=model_config.class_weight,
            ),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._transform(x_frame))[:, 1]

    @property
    def summary_text(self) -> str:
        return (
            "Elastic-net logistic regression fitted with sklearn.\n"
            f"L1 ratio: {self.model_config.l1_ratio}\n"
            f"Max iterations: {self.model_config.max_iter}\n"
            f"C: {self.model_config.C}\n"
            f"Class weight: {self.model_config.class_weight}\n"
        )


@dataclass(slots=True)
class ScorecardFeatureSpec:
    """Definition of a scorecard-transformed raw feature."""

    feature_name: str
    kind: str
    bin_edges: list[float] | None
    woe_mapping: dict[str, float]
    fallback_woe: float
    coefficient: float = 0.0


class ScorecardLogisticRegressionAdapter(BaseModelAdapter):
    """WoE-binned logistic regression for transparent scorecard-style PD development."""

    model_type = ModelType.SCORECARD_LOGISTIC_REGRESSION

    def __init__(
        self,
        model_config: ModelConfig,
        target_mode: TargetMode,
        scorecard_config: ScorecardConfig | None = None,
        scorecard_bin_overrides: dict[str, list[float]] | None = None,
    ) -> None:
        if target_mode != TargetMode.BINARY:
            raise ValueError("Scorecard logistic regression requires a binary target.")
        super().__init__(model_config, target_mode)
        self.scorecard_config = scorecard_config or ScorecardConfig()
        self.scorecard_bin_overrides = scorecard_bin_overrides or {}
        self.estimator = LogisticRegression(
            max_iter=model_config.max_iter,
            C=model_config.C,
            solver="lbfgs",
            class_weight=model_config.class_weight,
        )
        self.feature_specs_: list[ScorecardFeatureSpec] = []
        self.scorecard_table_: pd.DataFrame = pd.DataFrame()
        self.scorecard_points_table_: pd.DataFrame = pd.DataFrame()
        self.scorecard_scaling_summary_: pd.DataFrame = pd.DataFrame()
        self.score_factor_: float = 0.0
        self.score_offset_: float = 0.0
        self.base_points_: float = 0.0

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ScorecardLogisticRegressionAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        transformed = pd.DataFrame(index=x_frame.index)
        detail_rows: list[pd.DataFrame] = []
        specs: list[ScorecardFeatureSpec] = []
        for feature_name in x_frame.columns:
            kind = "numeric" if feature_name in numeric_features else "categorical"
            spec, transformed_feature, detail = self._fit_feature(
                x_frame[feature_name],
                y_values.astype(int),
                feature_name=feature_name,
                kind=kind,
            )
            transformed[feature_name] = transformed_feature
            specs.append(spec)
            detail_rows.append(detail)

        self.feature_specs_ = specs
        self.feature_names_ = list(transformed.columns)
        self.estimator.fit(transformed.to_numpy(dtype=float), y_values.astype(int))
        coefficients = np.ravel(self.estimator.coef_)
        for spec, coefficient in zip(self.feature_specs_, coefficients, strict=False):
            spec.coefficient = float(coefficient)
        self._fit_score_scaling()
        self.scorecard_table_ = pd.concat(detail_rows, ignore_index=True)
        self.scorecard_points_table_ = self._build_points_table()
        return self

    def _fit_feature(
        self,
        series: pd.Series,
        y_values: pd.Series,
        *,
        feature_name: str,
        kind: str,
    ) -> tuple[ScorecardFeatureSpec, pd.Series, pd.DataFrame]:
        if kind == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            bin_edges = self._build_numeric_bin_edges(numeric_series, y_values.astype(int))
            bucket_labels = self._bucket_numeric_series(numeric_series, bin_edges)
        else:
            bin_edges = None
            bucket_labels = series.fillna("MISSING").astype(str)

        summary = (
            pd.DataFrame({"bucket": bucket_labels, "target": y_values.astype(int)})
            .groupby("bucket", dropna=False)
            .agg(bad=("target", "sum"), total=("target", "size"))
            .reset_index()
        )
        summary["good"] = summary["total"] - summary["bad"]
        total_bad = float(summary["bad"].sum())
        total_good = float(summary["good"].sum())
        summary["bad_pct"] = (summary["bad"] + 0.5) / max(total_bad + 0.5 * len(summary), 1.0)
        summary["good_pct"] = (summary["good"] + 0.5) / max(
            total_good + 0.5 * len(summary), 1.0
        )
        summary["woe"] = np.log(summary["good_pct"] / summary["bad_pct"])
        summary["iv_component"] = (summary["good_pct"] - summary["bad_pct"]) * summary["woe"]
        summary.insert(0, "feature_name", feature_name)
        summary["bucket"] = summary["bucket"].astype(str)

        mapping = {
            str(bucket): float(woe)
            for bucket, woe in summary.loc[:, ["bucket", "woe"]].itertuples(index=False)
        }
        fallback = float(
            np.average(summary["woe"].to_numpy(dtype=float), weights=summary["total"].to_numpy())
        )
        transformed = bucket_labels.astype(str).map(mapping).fillna(fallback).astype(float)
        spec = ScorecardFeatureSpec(
            feature_name=feature_name,
            kind=kind,
            bin_edges=bin_edges,
            woe_mapping=mapping,
            fallback_woe=fallback,
        )
        return spec, transformed, summary.rename(columns={"bucket": "bucket_label"})

    def _build_numeric_bin_edges(
        self,
        series: pd.Series,
        y_values: pd.Series,
    ) -> list[float] | None:
        if series.name in self.scorecard_bin_overrides:
            manual_edges = np.asarray(self.scorecard_bin_overrides[series.name], dtype=float)
            if manual_edges.size == 0:
                return None
            manual_edges = np.unique(manual_edges)
            if manual_edges.size == 0:
                return None
            manual_edges = manual_edges.astype(float)
            return [-np.inf, *manual_edges.tolist(), np.inf]

        non_null = series.dropna().astype(float)
        if non_null.nunique() < 2:
            return None
        max_bins_from_share = max(
            2,
            int(1.0 / self.scorecard_config.min_bin_share),
        )
        requested_bins = min(
            self.model_config.scorecard_bins,
            int(non_null.nunique()),
            max_bins_from_share,
        )
        quantiles = np.linspace(
            0,
            1,
            requested_bins + 1,
        )
        edges = np.unique(np.quantile(non_null.to_numpy(), quantiles))
        if len(edges) < 2:
            return None
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        if len(np.unique(edges)) < 2:
            return None
        optimized_edges = self._optimize_numeric_bin_edges(
            series=series,
            y_values=y_values,
            edges=edges.tolist(),
        )
        return optimized_edges

    def _optimize_numeric_bin_edges(
        self,
        *,
        series: pd.Series,
        y_values: pd.Series,
        edges: list[float],
    ) -> list[float] | None:
        if len(edges) <= 2:
            return edges
        monotonicity = self.scorecard_config.monotonicity
        current_edges = list(edges)
        while len(current_edges) > 2:
            summary = self._summarize_numeric_bins(series, y_values, current_edges)
            if summary.empty:
                return current_edges
            too_small = summary["share"] < self.scorecard_config.min_bin_share
            if too_small.any() and len(current_edges) > 3:
                smallest_position = int(summary.loc[too_small, "position"].iloc[0])
                current_edges.pop(self._edge_to_remove(smallest_position, len(current_edges)))
                continue
            if monotonicity == ScorecardMonotonicity.NONE:
                break
            direction = self._resolve_monotonic_direction(summary)
            if direction is None or self._is_monotonic(summary["bad_rate"].tolist(), direction):
                break
            violation_index = self._first_monotonic_violation(
                summary["bad_rate"].tolist(),
                direction,
            )
            if violation_index is None or len(current_edges) <= 3:
                break
            current_edges.pop(self._edge_to_remove(violation_index, len(current_edges)))
        return current_edges

    def _summarize_numeric_bins(
        self,
        series: pd.Series,
        y_values: pd.Series,
        edges: list[float],
    ) -> pd.DataFrame:
        numeric_series = pd.to_numeric(series, errors="coerce")
        bucketed = pd.cut(
            numeric_series,
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )
        summary = (
            pd.DataFrame({"bucket": bucketed, "target": y_values.astype(int)})
            .groupby("bucket", dropna=False)
            .agg(bad=("target", "sum"), total=("target", "size"))
            .reset_index()
        )
        summary["bucket"] = summary["bucket"].astype(str)
        summary["bad_rate"] = summary["bad"] / summary["total"].replace(0, np.nan)
        summary["share"] = summary["total"] / max(float(summary["total"].sum()), 1.0)
        summary["position"] = range(len(summary))
        return summary

    def _resolve_monotonic_direction(self, summary: pd.DataFrame) -> str | None:
        if self.scorecard_config.monotonicity == ScorecardMonotonicity.INCREASING:
            return "increasing"
        if self.scorecard_config.monotonicity == ScorecardMonotonicity.DECREASING:
            return "decreasing"
        if self.scorecard_config.monotonicity == ScorecardMonotonicity.NONE:
            return None
        if len(summary) < 2:
            return None
        return (
            "increasing"
            if float(summary["bad_rate"].iloc[-1]) >= float(summary["bad_rate"].iloc[0])
            else "decreasing"
        )

    def _is_monotonic(self, values: list[float], direction: str) -> bool:
        if direction == "increasing":
            return all(left <= right for left, right in zip(values, values[1:], strict=False))
        return all(left >= right for left, right in zip(values, values[1:], strict=False))

    def _first_monotonic_violation(self, values: list[float], direction: str) -> int | None:
        for index, (left, right) in enumerate(zip(values, values[1:], strict=False)):
            if direction == "increasing" and left > right:
                return index
            if direction == "decreasing" and left < right:
                return index
        return None

    def _edge_to_remove(self, violation_index: int, edge_count: int) -> int:
        candidate = min(violation_index + 1, edge_count - 2)
        return max(candidate, 1)

    def _bucket_numeric_series(self, series: pd.Series, bin_edges: list[float] | None) -> pd.Series:
        if not bin_edges:
            return pd.Series(["ALL"] * len(series), index=series.index, dtype="string")
        bucketed = pd.cut(
            series,
            bins=bin_edges,
            include_lowest=True,
            duplicates="drop",
        )
        labels = pd.Series(bucketed, index=series.index).astype("object")
        return labels.map(lambda value: str(value) if pd.notna(value) else "MISSING")

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = pd.DataFrame(index=x_frame.index)
        for spec in self.feature_specs_:
            series = x_frame[spec.feature_name]
            if spec.kind == "numeric":
                numeric_series = pd.to_numeric(series, errors="coerce")
                labels = self._bucket_numeric_series(numeric_series, spec.bin_edges)
            else:
                labels = series.fillna("MISSING").astype(str)
            transformed[spec.feature_name] = (
                labels.astype(str).map(spec.woe_mapping).fillna(spec.fallback_woe).astype(float)
            )
        return transformed.to_numpy(dtype=float)

    def _transform_frame(self, x_frame: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=x_frame.index)
        for spec in self.feature_specs_:
            series = x_frame[spec.feature_name]
            if spec.kind == "numeric":
                numeric_series = pd.to_numeric(series, errors="coerce")
                labels = self._bucket_numeric_series(numeric_series, spec.bin_edges)
            else:
                labels = series.fillna("MISSING").astype(str)
            transformed[spec.feature_name] = (
                labels.astype(str).map(spec.woe_mapping).fillna(spec.fallback_woe).astype(float)
            )
        return transformed

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._transform(x_frame))[:, 1]

    def get_prediction_outputs(self, x_frame: pd.DataFrame) -> dict[str, np.ndarray | list[str]]:
        transformed = self._transform_frame(x_frame)
        contribution_frame = pd.DataFrame(index=transformed.index)
        for spec in self.feature_specs_:
            contribution_frame[spec.feature_name] = (
                -self.score_factor_ * spec.coefficient * transformed[spec.feature_name]
            )
        score_points = self.base_points_ + contribution_frame.sum(axis=1)
        outputs: dict[str, np.ndarray | list[str]] = {
            "scorecard_points": score_points.to_numpy(dtype=float),
        }
        reason_code_count = min(
            self.scorecard_config.reason_code_count,
            len(contribution_frame.columns),
        )
        if reason_code_count <= 0:
            return outputs
        reason_code_frame = contribution_frame.apply(
            lambda row: row.nsmallest(reason_code_count).index.tolist(),
            axis=1,
        )
        for index in range(reason_code_count):
            outputs[f"reason_code_{index + 1}"] = reason_code_frame.map(
                lambda values, position=index: values[position] if len(values) > position else ""
            ).tolist()
        return outputs

    def get_feature_importance(self) -> pd.DataFrame:
        coefficients = np.ravel(self.estimator.coef_)
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(coefficients),
                    "importance_type": "absolute_coefficient",
                    "coefficient": coefficients,
                    "abs_coefficient": np.abs(coefficients),
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.exp(coefficients),
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    def get_model_artifacts(self) -> dict[str, pd.DataFrame]:
        return {
            "scorecard_woe_table": self.scorecard_table_.copy(deep=True),
            "scorecard_points_table": self.scorecard_points_table_.copy(deep=True),
            "scorecard_scaling_summary": self.scorecard_scaling_summary_.copy(deep=True),
        }

    def _fit_score_scaling(self) -> None:
        self.score_factor_ = self.scorecard_config.points_to_double_odds / np.log(2.0)
        self.score_offset_ = self.scorecard_config.base_score - self.score_factor_ * np.log(
            self.scorecard_config.odds_reference
        )
        intercept = float(np.ravel(self.estimator.intercept_)[0])
        self.base_points_ = self.score_offset_ - self.score_factor_ * intercept
        self.scorecard_scaling_summary_ = pd.DataFrame(
            [
                {"metric": "base_score", "value": self.scorecard_config.base_score},
                {
                    "metric": "points_to_double_odds",
                    "value": self.scorecard_config.points_to_double_odds,
                },
                {"metric": "odds_reference", "value": self.scorecard_config.odds_reference},
                {"metric": "score_factor", "value": self.score_factor_},
                {"metric": "score_offset", "value": self.score_offset_},
                {"metric": "base_points", "value": self.base_points_},
            ]
        )

    def _build_points_table(self) -> pd.DataFrame:
        if self.scorecard_table_.empty:
            return pd.DataFrame()
        coefficient_map = {spec.feature_name: spec.coefficient for spec in self.feature_specs_}
        points_table = self.scorecard_table_.copy(deep=True)
        points_table["coefficient"] = points_table["feature_name"].map(coefficient_map).fillna(0.0)
        points_table["partial_score_points"] = (
            -self.score_factor_ * points_table["coefficient"] * points_table["woe"]
        )
        return points_table

    @property
    def summary_text(self) -> str:
        return (
            "Scorecard logistic regression fitted on WoE-transformed features.\n"
            f"Bins per numeric feature: {self.model_config.scorecard_bins}\n"
            f"Monotonicity: {self.scorecard_config.monotonicity.value}\n"
            f"Max iterations: {self.model_config.max_iter}\n"
            f"C: {self.model_config.C}\n"
        )


class LinearRegressionAdapter(SklearnAdapter):
    """Linear regression adapter used for continuous targets or linear-probability modeling."""

    model_type = ModelType.LINEAR_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        super().__init__(
            model_config,
            target_mode,
            LinearRegression(),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        predictions = self.estimator.predict(self._transform(x_frame))
        if self.is_binary_classifier:
            return np.clip(predictions, 0.0, 1.0)
        return predictions

    @property
    def summary_text(self) -> str:
        if self.is_binary_classifier:
            return (
                "Linear regression fitted as a linear-probability model.\n"
                "Predicted scores are clipped into the [0, 1] range for "
                "classification diagnostics.\n"
            )
        return "Ordinary least squares linear regression fitted with sklearn.\n"


class QuantileRegressionAdapter(SklearnAdapter):
    """Median or user-selected quantile regression for forecast tail analysis."""

    model_type = ModelType.QUANTILE_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Quantile regression requires a continuous target.")
        super().__init__(
            model_config,
            target_mode,
            QuantileRegressor(
                quantile=model_config.quantile_alpha,
                alpha=0.0,
                fit_intercept=True,
                solver="highs",
            ),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._transform(x_frame))

    @property
    def summary_text(self) -> str:
        return (
            "Quantile regression fitted with sklearn.\n"
            f"Target quantile: {self.model_config.quantile_alpha}\n"
        )


class BetaRegressionAdapter(BaseModelAdapter):
    """Beta regression for bounded LGD-style targets in the (0, 1) interval."""

    model_type = ModelType.BETA_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Beta regression requires a continuous target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> BetaRegressionAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        x_matrix = self.preprocessor.fit_transform(x_frame)
        x_with_const = sm.add_constant(x_matrix, has_constant="add")
        clipped_target = np.clip(
            y_values.astype(float).to_numpy(),
            self.model_config.beta_clip_epsilon,
            1 - self.model_config.beta_clip_epsilon,
        )
        self.results = BetaModel(clipped_target, x_with_const).fit(
            maxiter=self.model_config.max_iter,
            disp=False,
        )
        self.feature_names_ = list(self.results.model.exog_names)
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        transformed = self.preprocessor.transform(x_frame)
        return sm.add_constant(transformed, has_constant="add")

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        prediction = np.asarray(self.results.predict(self._transform(x_frame)))
        return np.clip(prediction, 0.0, 1.0)

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        pvalues = np.asarray(self.results.pvalues)
        bse = np.asarray(self.results.bse)
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(params),
                    "importance_type": "absolute_coefficient",
                    "coefficient": params,
                    "abs_coefficient": np.abs(params),
                    "std_error": bse,
                    "p_value": pvalues,
                    "odds_ratio": np.nan,
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        return self.results.summary().as_text()


class TwoStageLGDModelAdapter(BaseModelAdapter):
    """Two-stage LGD model: positive-loss probability times severity expectation."""

    model_type = ModelType.TWO_STAGE_LGD_MODEL

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Two-stage LGD modeling requires a continuous target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.constant_positive_probability_: float | None = None
        self.stage_one_estimator = LogisticRegression(
            max_iter=model_config.max_iter,
            C=model_config.C,
            solver=model_config.solver,
            class_weight=model_config.class_weight,
        )
        self.stage_two_results = None
        self.stage_feature_names_: list[str] = []

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> TwoStageLGDModelAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        self.stage_feature_names_ = ["const", *self.feature_names_]

        y_array = y_values.astype(float).to_numpy()
        positive_indicator = (y_array > self.model_config.lgd_positive_threshold).astype(int)
        unique_positive_values = np.unique(positive_indicator)
        if len(unique_positive_values) == 1:
            self.constant_positive_probability_ = float(unique_positive_values[0])
        else:
            self.stage_one_estimator.fit(x_matrix, positive_indicator)

        positive_mask = positive_indicator == 1
        if positive_mask.sum() < 10:
            raise ValueError("Two-stage LGD requires at least 10 positive-loss observations.")

        clipped_positive_target = np.clip(
            y_array[positive_mask],
            self.model_config.beta_clip_epsilon,
            1 - self.model_config.beta_clip_epsilon,
        )
        x_positive = sm.add_constant(x_matrix[positive_mask], has_constant="add")
        self.stage_two_results = BetaModel(clipped_positive_target, x_positive).fit(
            maxiter=self.model_config.max_iter,
            disp=False,
        )
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        if self.constant_positive_probability_ is not None:
            positive_probability = np.full(
                len(transformed),
                self.constant_positive_probability_,
                dtype=float,
            )
        else:
            positive_probability = self.stage_one_estimator.predict_proba(transformed)[:, 1]
        severity_prediction = np.asarray(
            self.stage_two_results.predict(sm.add_constant(transformed, has_constant="add"))
        )
        return np.clip(positive_probability * severity_prediction, 0.0, 1.0)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.constant_positive_probability_ is not None:
            stage_one_coefficients = np.zeros(len(self.feature_names_), dtype=float)
        else:
            stage_one_coefficients = np.ravel(self.stage_one_estimator.coef_)
        stage_two_params = np.asarray(self.stage_two_results.params)[
            1 : len(self.feature_names_) + 1
        ]
        table = pd.DataFrame(
            {
                "feature_name": self.feature_names_,
                "importance_value": np.abs(stage_one_coefficients) + np.abs(stage_two_params),
                "importance_type": "two_stage_absolute_coefficient",
                "coefficient": stage_one_coefficients + stage_two_params,
                "abs_coefficient": np.abs(stage_one_coefficients) + np.abs(stage_two_params),
                "std_error": np.nan,
                "p_value": np.nan,
                "odds_ratio": np.exp(stage_one_coefficients),
                "stage_one_coefficient": stage_one_coefficients,
                "stage_two_coefficient": stage_two_params,
            }
        )
        return table.sort_values("importance_value", ascending=False).reset_index(drop=True)

    def get_model_artifacts(self) -> dict[str, pd.DataFrame]:
        if self.constant_positive_probability_ is not None:
            stage_one_coefficients = np.zeros(len(self.feature_names_), dtype=float)
        else:
            stage_one_coefficients = np.ravel(self.stage_one_estimator.coef_)
        stage_two_params = np.asarray(self.stage_two_results.params)
        stage_two_names = list(self.stage_two_results.model.exog_names)
        return {
            "lgd_stage_one_coefficients": pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "coefficient": stage_one_coefficients,
                    "odds_ratio": np.exp(stage_one_coefficients),
                }
            ),
            "lgd_stage_two_coefficients": pd.DataFrame(
                {
                    "feature_name": stage_two_names,
                    "coefficient": stage_two_params,
                }
            ),
        }

    @property
    def summary_text(self) -> str:
        return (
            "Two-stage LGD model fitted.\n"
            "Stage one: logistic regression for positive loss.\n"
            "Stage two: beta regression for positive-loss severity.\n"
        )


class XGBoostAdapter(SklearnAdapter):
    """XGBoost adapter for binary classification or continuous regression."""

    model_type = ModelType.XGBOOST

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        estimator = (
            XGBClassifier(
                objective="binary:logistic",
                n_estimators=model_config.xgboost_n_estimators,
                learning_rate=model_config.xgboost_learning_rate,
                max_depth=model_config.xgboost_max_depth,
                subsample=model_config.xgboost_subsample,
                colsample_bytree=model_config.xgboost_colsample_bytree,
                eval_metric="logloss",
                n_jobs=1,
            )
            if target_mode == TargetMode.BINARY
            else XGBRegressor(
                objective="reg:squarederror",
                n_estimators=model_config.xgboost_n_estimators,
                learning_rate=model_config.xgboost_learning_rate,
                max_depth=model_config.xgboost_max_depth,
                subsample=model_config.xgboost_subsample,
                colsample_bytree=model_config.xgboost_colsample_bytree,
                n_jobs=1,
            )
        )
        super().__init__(model_config, target_mode, estimator, scale_numeric=False)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        if self.is_binary_classifier:
            return self.estimator.predict_proba(transformed)[:, 1]
        return self.estimator.predict(transformed)

    @property
    def summary_text(self) -> str:
        return (
            "XGBoost model fitted.\n"
            f"Estimators: {self.model_config.xgboost_n_estimators}\n"
            f"Learning rate: {self.model_config.xgboost_learning_rate}\n"
            f"Max depth: {self.model_config.xgboost_max_depth}\n"
        )


class StatsmodelsAdapter(BaseModelAdapter):
    """Base adapter for statsmodels-style estimators on dense matrices."""

    def __init__(
        self,
        model_config: ModelConfig,
        target_mode: TargetMode,
        *,
        scale_numeric: bool = True,
    ) -> None:
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None
        self.scale_numeric = scale_numeric

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> StatsmodelsAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            scale_numeric=self.scale_numeric,
        )
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = ["const", *self.preprocessor.get_feature_names_out().tolist()]
        x_with_const = sm.add_constant(x_matrix, has_constant="add")
        self.results = self._fit_statsmodel(x_with_const, y_values.astype(float))
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        transformed = self.preprocessor.transform(x_frame)
        return sm.add_constant(transformed, has_constant="add")

    @abstractmethod
    def _fit_statsmodel(self, x_matrix: np.ndarray, y_values: pd.Series):
        """Fits the specific statsmodels estimator."""

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        pvalues = np.asarray(getattr(self.results, "pvalues", np.full_like(params, np.nan)))
        bse = np.asarray(getattr(self.results, "bse", np.full_like(params, np.nan)))

        table = pd.DataFrame(
            {
                "feature_name": self.feature_names_,
                "importance_value": np.abs(params),
                "importance_type": "absolute_coefficient",
                "coefficient": params,
                "abs_coefficient": np.abs(params),
                "std_error": bse,
                "p_value": pvalues,
                "odds_ratio": np.exp(params) if self.is_binary_classifier else np.nan,
            }
        )
        return table.sort_values("importance_value", ascending=False).reset_index(drop=True)

    @property
    def summary_text(self) -> str:
        return self.results.summary().as_text()


class PanelRegressionAdapter(StatsmodelsAdapter):
    """Panel-style fixed-effect regression using dense OLS with entity dummies."""

    model_type = ModelType.PANEL_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Panel regression requires a continuous target.")
        super().__init__(model_config, target_mode, scale_numeric=True)

    def _fit_statsmodel(self, x_matrix: np.ndarray, y_values: pd.Series):
        return sm.OLS(y_values.astype(float), x_matrix).fit()

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.results.predict(self._transform(x_frame)))

    @property
    def summary_text(self) -> str:
        return "Panel regression fitted as an OLS model with encoded panel effects.\n"


class ProbitRegressionAdapter(StatsmodelsAdapter):
    """Binary probit regression via statsmodels."""

    model_type = ModelType.PROBIT_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.BINARY:
            raise ValueError("Probit regression requires a binary target.")
        super().__init__(model_config, target_mode, scale_numeric=True)

    def _fit_statsmodel(self, x_matrix: np.ndarray, y_values: pd.Series):
        model = sm.GLM(
            y_values.astype(float),
            x_matrix,
            family=sm.families.Binomial(link=sm.families.links.Probit()),
        )
        return model.fit(maxiter=self.model_config.max_iter, disp=False)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.results.predict(self._transform(x_frame)))


@dataclass
class TobitResults:
    """Stores the fitted parameters for the custom Tobit model."""

    params: np.ndarray
    bse: np.ndarray
    pvalues: np.ndarray
    success: bool
    message: str
    nobs: int

    def summary(self) -> TobitSummary:
        return TobitSummary(self)


class TobitSummary:
    """Simple summary wrapper that mimics statsmodels' summary text API."""

    def __init__(self, results: TobitResults) -> None:
        self.results = results

    def as_text(self) -> str:
        lines = [
            "Custom Tobit regression fit",
            f"Observations: {self.results.nobs}",
            f"Optimization success: {self.results.success}",
            f"Message: {self.results.message}",
        ]
        return "\n".join(lines)


class TobitRegressionAdapter(StatsmodelsAdapter):
    """Custom Tobit regression for continuous censored targets."""

    model_type = ModelType.TOBIT_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError(
                "Tobit regression requires a continuous target. Use logistic "
                "or probit for binary PD labels."
            )
        super().__init__(model_config, target_mode, scale_numeric=True)

    def _fit_statsmodel(self, x_matrix: np.ndarray, y_values: pd.Series):
        lower = self.model_config.tobit_left_censoring
        upper = self.model_config.tobit_right_censoring
        y_array = y_values.to_numpy(dtype=float)

        initial_params = np.zeros(x_matrix.shape[1] + 1, dtype=float)
        initial_params[-1] = np.log(np.std(y_array) + 1e-6)

        def neg_log_likelihood(params: np.ndarray) -> float:
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            mu = x_matrix @ beta

            lower_mask = np.zeros_like(y_array, dtype=bool)
            upper_mask = np.zeros_like(y_array, dtype=bool)
            if lower is not None:
                lower_mask = np.isclose(y_array, lower) | (y_array < lower)
            if upper is not None:
                upper_mask = np.isclose(y_array, upper) | (y_array > upper)
            uncensored_mask = ~(lower_mask | upper_mask)

            log_likelihood = np.zeros_like(y_array, dtype=float)
            if lower is not None and lower_mask.any():
                log_likelihood[lower_mask] = norm.logcdf((lower - mu[lower_mask]) / sigma)
            if upper is not None and upper_mask.any():
                log_likelihood[upper_mask] = norm.logsf((upper - mu[upper_mask]) / sigma)
            if uncensored_mask.any():
                z_values = (y_array[uncensored_mask] - mu[uncensored_mask]) / sigma
                log_likelihood[uncensored_mask] = norm.logpdf(z_values) - log_sigma

            return float(-np.sum(log_likelihood))

        optimization = minimize(
            neg_log_likelihood,
            initial_params,
            method="BFGS",
            options={"maxiter": self.model_config.max_iter},
        )
        if not optimization.success and (
            not np.isfinite(optimization.fun) or not np.all(np.isfinite(optimization.x))
        ):
            raise ValueError(f"Tobit optimization failed: {optimization.message}")

        covariance = np.asarray(getattr(optimization, "hess_inv", np.eye(len(optimization.x))))
        if covariance.ndim != 2:
            covariance = np.eye(len(optimization.x))
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        z_scores = optimization.x / standard_errors
        pvalues = 2 * (1 - norm.cdf(np.abs(z_scores)))
        return TobitResults(
            params=optimization.x[:-1],
            bse=standard_errors[:-1],
            pvalues=pvalues[:-1],
            success=bool(optimization.success),
            message=str(optimization.message),
            nobs=len(y_array),
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        x_matrix = self._transform(x_frame)
        predictions = x_matrix @ self.results.params
        lower = self.model_config.tobit_left_censoring
        upper = self.model_config.tobit_right_censoring
        if lower is not None:
            predictions = np.maximum(predictions, lower)
        if upper is not None:
            predictions = np.minimum(predictions, upper)
        return predictions


def build_model_adapter(
    model_config: ModelConfig,
    target_mode: TargetMode,
    *,
    scorecard_config: ScorecardConfig | None = None,
    scorecard_bin_overrides: dict[str, list[float]] | None = None,
) -> BaseModelAdapter:
    """Factory for the supported model adapters."""

    if model_config.model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.DISCRETE_TIME_HAZARD_MODEL:
        return DiscreteTimeHazardModelAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.ELASTIC_NET_LOGISTIC_REGRESSION:
        return ElasticNetLogisticRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.SCORECARD_LOGISTIC_REGRESSION:
        return ScorecardLogisticRegressionAdapter(
            model_config,
            target_mode,
            scorecard_config=scorecard_config,
            scorecard_bin_overrides=scorecard_bin_overrides,
        )
    if model_config.model_type == ModelType.PROBIT_REGRESSION:
        return ProbitRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.BETA_REGRESSION:
        return BetaRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.TWO_STAGE_LGD_MODEL:
        return TwoStageLGDModelAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.PANEL_REGRESSION:
        return PanelRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.QUANTILE_REGRESSION:
        return QuantileRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.TOBIT_REGRESSION:
        return TobitRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.XGBOOST:
        return XGBoostAdapter(model_config, target_mode)

    raise ValueError(f"Unsupported model type '{model_config.model_type}'.")
