"""Model adapters that provide a common interface across supported estimators."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from warnings import WarningMessage

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.othermod.betareg import BetaModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    ModelConfig,
    ModelType,
    ScorecardConfig,
    ScorecardMonotonicity,
    TargetMode,
)

ODDS_RATIO_CLIP_BOUND = 40.0


def _resolved_n_jobs(model_config: ModelConfig) -> int | None:
    """Normalizes GUI CPU-thread settings for estimators that support n_jobs."""

    if model_config.n_jobs == 0:
        return -1
    return model_config.n_jobs


def _safe_odds_ratio(values: np.ndarray | list[float] | float) -> np.ndarray:
    """Exponentiates coefficients without emitting overflow warnings."""

    numeric_values = np.asarray(values, dtype=float)
    return np.exp(np.clip(numeric_values, -ODDS_RATIO_CLIP_BOUND, ODDS_RATIO_CLIP_BOUND))


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = True,
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Builds preprocessing shared by model adapters.

    Sklearn-style estimators can keep one-hot encoded categoricals sparse, while
    statsmodels-style estimators request dense output because statsmodels expects
    dense design matrices.
    """

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
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=sparse_output,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("Training requires at least one numeric or categorical feature.")

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0 if sparse_output else 0.0,
    )


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
        self.numerical_warning_records_: list[dict[str, Any]] = []
        self.numerical_diagnostics_: list[dict[str, Any]] = []

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

        return self._base_model_artifacts()

    def get_prediction_outputs(self, x_frame: pd.DataFrame) -> dict[str, np.ndarray | list[str]]:
        """Returns optional prediction-side outputs such as scorecard points."""

        return {}

    def get_numerical_warning_table(self) -> pd.DataFrame:
        """Returns normalized warning records captured during fitting."""

        if not self.numerical_warning_records_:
            return pd.DataFrame(
                columns=[
                    "source",
                    "stage",
                    "warning_code",
                    "category",
                    "message",
                    "occurrence_count",
                ]
            )
        return pd.DataFrame(self.numerical_warning_records_).copy(deep=True)

    def get_numerical_diagnostics_table(self) -> pd.DataFrame:
        """Returns normalized estimation-health diagnostics captured during fitting."""

        if not self.numerical_diagnostics_:
            return pd.DataFrame(
                columns=["source", "diagnostic_name", "value", "status", "detail"]
            )
        return pd.DataFrame(self.numerical_diagnostics_).copy(deep=True)

    def _base_model_artifacts(self) -> dict[str, pd.DataFrame]:
        artifacts: dict[str, pd.DataFrame] = {}
        warning_table = self.get_numerical_warning_table()
        if not warning_table.empty:
            artifacts["numerical_warning_summary"] = warning_table
        diagnostics_table = self.get_numerical_diagnostics_table()
        if not diagnostics_table.empty:
            artifacts["model_numerical_diagnostics"] = diagnostics_table
        return artifacts

    def _merge_model_artifacts(
        self, artifacts: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        return {**self._base_model_artifacts(), **artifacts}

    def _run_with_warning_capture(
        self,
        callback,
        *,
        source: str,
        stage: str = "fit",
    ):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = callback()
        self._record_warning_batch(captured, source=source, stage=stage)
        return result

    def _record_warning_batch(
        self,
        captured: list[WarningMessage],
        *,
        source: str,
        stage: str,
    ) -> None:
        grouped: dict[tuple[str, str, str, str, str], int] = {}
        for record in captured:
            category = record.category.__name__
            message = " ".join(str(record.message).split())
            warning_code = self._normalize_warning_code(category, message)
            key = (source, stage, warning_code, category, message)
            grouped[key] = grouped.get(key, 0) + 1
        for (
            warning_source,
            warning_stage,
            warning_code,
            category,
            message,
        ), count in grouped.items():
            self.numerical_warning_records_.append(
                {
                    "source": warning_source,
                    "stage": warning_stage,
                    "warning_code": warning_code,
                    "category": category,
                    "message": message,
                    "occurrence_count": count,
                }
            )

    def _normalize_warning_code(self, category: str, message: str) -> str:
        normalized_message = message.lower()
        if category == ConvergenceWarning.__name__:
            return "convergence_max_iter"
        if category == HessianInversionWarning.__name__:
            return "hessian_inversion"
        if "divide by zero" in normalized_message:
            return "divide_by_zero"
        if "invalid value" in normalized_message:
            return "invalid_numeric_value"
        if "overflow" in normalized_message:
            return "overflow"
        return category.lower()

    def _add_numerical_diagnostic(
        self,
        *,
        source: str,
        diagnostic_name: str,
        value: Any,
        status: str = "ok",
        detail: str = "",
    ) -> None:
        if hasattr(value, "item"):
            value = value.item()
        self.numerical_diagnostics_.append(
            {
                "source": source,
                "diagnostic_name": diagnostic_name,
                "value": value,
                "status": status,
                "detail": detail,
            }
        )

    def _update_sklearn_fit_diagnostics(self, estimator: Any, *, source: str) -> None:
        solver = getattr(estimator, "solver", None)
        if solver:
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="solver",
                value=str(solver),
            )

        configured_max_iter = getattr(estimator, "max_iter", None)
        if configured_max_iter is not None:
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="configured_max_iter",
                value=int(configured_max_iter),
            )

        observed_iterations = getattr(estimator, "n_iter_", None)
        if observed_iterations is not None:
            max_observed_iterations = int(np.max(np.asarray(observed_iterations, dtype=float)))
            converged = (
                configured_max_iter is None or max_observed_iterations < int(configured_max_iter)
            )
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="observed_max_iterations",
                value=max_observed_iterations,
                status="ok" if converged else "warn",
                detail="Maximum iterations observed across fitted classes.",
            )
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="converged_before_max_iter",
                value=bool(converged),
                status="ok" if converged else "warn",
                detail="False indicates the estimator reached the configured iteration cap.",
            )

    def _update_statsmodels_fit_diagnostics(self, results: Any, *, source: str) -> None:
        converged = getattr(results, "converged", None)
        if converged is not None:
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="converged",
                value=bool(converged),
                status="ok" if converged else "warn",
            )

        mle_retvals = getattr(results, "mle_retvals", None)
        if isinstance(mle_retvals, dict):
            for key in ("iterations", "fopt", "warnflag"):
                if key in mle_retvals:
                    self._add_numerical_diagnostic(
                        source=source,
                        diagnostic_name=f"mle_{key}",
                        value=mle_retvals[key],
                        status="warn" if key == "warnflag" and mle_retvals[key] else "ok",
                    )

        optimizer_success = getattr(results, "success", None)
        if optimizer_success is not None:
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="optimizer_success",
                value=bool(optimizer_success),
                status="ok" if optimizer_success else "warn",
            )

        optimizer_message = getattr(results, "message", None)
        if optimizer_message:
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="optimizer_message",
                value=str(optimizer_message),
                status="ok" if getattr(results, "success", True) else "warn",
            )

        standard_errors = getattr(results, "bse", None)
        if standard_errors is not None:
            finite_standard_errors = bool(
                np.isfinite(np.asarray(standard_errors, dtype=float)).all()
            )
            self._add_numerical_diagnostic(
                source=source,
                diagnostic_name="finite_standard_errors",
                value=finite_standard_errors,
                status="ok" if finite_standard_errors else "warn",
                detail="False indicates covariance estimation did not return finite values.",
            )


class SklearnAdapter(BaseModelAdapter):
    """Base adapter for sklearn-style estimators with an external preprocessor."""

    def __init__(
        self,
        model_config: ModelConfig,
        target_mode: TargetMode,
        estimator: Any,
        *,
        scale_numeric: bool = True,
        sparse_preprocessor: bool = True,
    ) -> None:
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.estimator = estimator
        self.scale_numeric = scale_numeric
        self.sparse_preprocessor = sparse_preprocessor

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
            sparse_output=self.sparse_preprocessor,
        )
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        self._run_with_warning_capture(
            lambda: self.estimator.fit(x_matrix, y_values),
            source=f"{self.model_type.value}.estimator",
        )
        self._update_sklearn_fit_diagnostics(
            self.estimator,
            source=f"{self.model_type.value}.estimator",
        )
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
                    "odds_ratio": _safe_odds_ratio(coefficients)
                    if self.is_binary_classifier
                    else np.nan,
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
        self._run_with_warning_capture(
            lambda: self.estimator.fit(transformed.to_numpy(dtype=float), y_values.astype(int)),
            source=f"{self.model_type.value}.estimator",
        )
        self._update_sklearn_fit_diagnostics(
            self.estimator,
            source=f"{self.model_type.value}.estimator",
        )
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
                    "odds_ratio": _safe_odds_ratio(coefficients),
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    def get_model_artifacts(self) -> dict[str, pd.DataFrame]:
        return self._merge_model_artifacts(
            {
            "scorecard_woe_table": self.scorecard_table_.copy(deep=True),
            "scorecard_points_table": self.scorecard_points_table_.copy(deep=True),
            "scorecard_scaling_summary": self.scorecard_scaling_summary_.copy(deep=True),
            }
        )

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


class RidgeRegressionAdapter(SklearnAdapter):
    """Ridge regression for continuous targets with coefficient shrinkage."""

    model_type = ModelType.RIDGE_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Ridge regression requires a continuous target.")
        super().__init__(
            model_config,
            target_mode,
            Ridge(alpha=model_config.regularization_alpha),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._transform(x_frame))

    @property
    def summary_text(self) -> str:
        return (
            "Ridge regression fitted with sklearn.\n"
            f"Regularization alpha: {self.model_config.regularization_alpha}\n"
        )


class LassoRegressionAdapter(SklearnAdapter):
    """Lasso regression for continuous targets with sparse coefficient pressure."""

    model_type = ModelType.LASSO_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Lasso regression requires a continuous target.")
        super().__init__(
            model_config,
            target_mode,
            Lasso(
                alpha=model_config.regularization_alpha,
                max_iter=model_config.max_iter,
            ),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._transform(x_frame))

    @property
    def summary_text(self) -> str:
        return (
            "Lasso regression fitted with sklearn.\n"
            f"Regularization alpha: {self.model_config.regularization_alpha}\n"
        )


class ElasticNetRegressionAdapter(SklearnAdapter):
    """Elastic-net regression for continuous targets with mixed L1/L2 shrinkage."""

    model_type = ModelType.ELASTIC_NET_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Elastic-net regression requires a continuous target.")
        super().__init__(
            model_config,
            target_mode,
            ElasticNet(
                alpha=model_config.regularization_alpha,
                l1_ratio=model_config.l1_ratio,
                max_iter=model_config.max_iter,
            ),
            scale_numeric=True,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._transform(x_frame))

    @property
    def summary_text(self) -> str:
        return (
            "Elastic-net regression fitted with sklearn.\n"
            f"Regularization alpha: {self.model_config.regularization_alpha}\n"
            f"L1 ratio: {self.model_config.l1_ratio}\n"
        )


class MultinomialLogisticRegressionAdapter(SklearnAdapter):
    """Multinomial logistic model for unordered multiclass outcomes."""

    model_type = ModelType.MULTINOMIAL_LOGISTIC_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.MULTICLASS:
            raise ValueError("Multinomial logistic regression requires a multiclass target.")
        super().__init__(
            model_config,
            target_mode,
            LogisticRegression(
                max_iter=model_config.max_iter,
                C=model_config.C,
                solver="lbfgs",
                class_weight=model_config.class_weight,
            ),
            scale_numeric=True,
        )

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> MultinomialLogisticRegressionAdapter:
        return super().fit(x_frame, y_values.astype(int), numeric_features, categorical_features)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._transform(x_frame)).astype(float)

    def predict_class(self, x_frame: pd.DataFrame, threshold: float) -> np.ndarray | None:
        return self.estimator.predict(self._transform(x_frame)).astype(int)

    def predict_proba(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._transform(x_frame))

    def get_feature_importance(self) -> pd.DataFrame:
        coefficients = np.asarray(self.estimator.coef_)
        importance = np.mean(np.abs(coefficients), axis=0)
        signed_average = np.mean(coefficients, axis=0)
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": importance,
                    "importance_type": "multinomial_mean_absolute_coefficient",
                    "coefficient": signed_average,
                    "abs_coefficient": importance,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.nan,
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        return (
            "Multinomial logistic regression fitted with sklearn.\n"
            f"Classes: {', '.join(str(value) for value in self.estimator.classes_)}\n"
        )


class GAMSplineAdapter(BaseModelAdapter):
    """Spline-expanded regression/logistic adapter approximating a lightweight GAM."""

    model_type = ModelType.GAM_SPLINE_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode not in {TargetMode.BINARY, TargetMode.CONTINUOUS}:
            raise ValueError("Spline GAM-style models support binary or continuous targets.")
        super().__init__(model_config, target_mode)
        self.model_type = model_config.model_type
        self.preprocessor = None
        self.estimator = (
            LogisticRegression(
                max_iter=model_config.max_iter,
                C=model_config.C,
                solver=model_config.solver,
                class_weight=model_config.class_weight,
            )
            if target_mode == TargetMode.BINARY
            else Ridge(alpha=model_config.regularization_alpha)
        )

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> GAMSplineAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = self._build_spline_preprocessor(numeric_features, categorical_features)
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        target_values = y_values.astype(int if self.is_binary_classifier else float)
        self._run_with_warning_capture(
            lambda: self.estimator.fit(x_matrix, target_values),
            source=f"{self.model_type.value}.estimator",
        )
        self._update_sklearn_fit_diagnostics(
            self.estimator,
            source=f"{self.model_type.value}.estimator",
        )
        return self

    def _build_spline_preprocessor(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ColumnTransformer:
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_features:
            transformers.append(
                (
                    "numeric_spline",
                    Pipeline(
                        steps=[
                            (
                                "spline",
                                SplineTransformer(
                                    n_knots=self.model_config.spline_n_knots,
                                    degree=self.model_config.spline_degree,
                                    include_bias=False,
                                ),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            )
        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                )
            )
        if not transformers:
            raise ValueError("Spline GAM-style modeling requires at least one feature.")
        return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        if self.is_binary_classifier:
            return self.estimator.predict_proba(transformed)[:, 1]
        return self.estimator.predict(transformed)

    def get_feature_importance(self) -> pd.DataFrame:
        if hasattr(self.estimator, "coef_"):
            coefficients = np.ravel(self.estimator.coef_)
        else:
            coefficients = np.zeros(len(self.feature_names_), dtype=float)
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(coefficients),
                    "importance_type": "spline_absolute_coefficient",
                    "coefficient": coefficients,
                    "abs_coefficient": np.abs(coefficients),
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": _safe_odds_ratio(coefficients)
                    if self.is_binary_classifier
                    else np.nan,
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        family = "logistic" if self.is_binary_classifier else "ridge regression"
        return (
            f"Spline-expanded GAM-style {family} model fitted with sklearn.\n"
            f"Spline knots: {self.model_config.spline_n_knots}\n"
            f"Spline degree: {self.model_config.spline_degree}\n"
        )


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
        self.results = self._run_with_warning_capture(
            lambda: BetaModel(clipped_target, x_with_const).fit(
                maxiter=self.model_config.max_iter,
                disp=False,
            ),
            source=f"{self.model_type.value}.beta_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.beta_fit",
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
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


class ZeroOneInflatedBetaAdapter(BaseModelAdapter):
    """Three-part LGD model for exact-zero, exact-one, and interior beta severity."""

    model_type = ModelType.ZERO_ONE_INFLATED_BETA

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Zero-one inflated beta requires a continuous bounded target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.zero_estimator = LogisticRegression(
            max_iter=model_config.max_iter,
            C=model_config.C,
            solver=model_config.solver,
            class_weight=model_config.class_weight,
        )
        self.one_estimator = LogisticRegression(
            max_iter=model_config.max_iter,
            C=model_config.C,
            solver=model_config.solver,
            class_weight=model_config.class_weight,
        )
        self.beta_results = None
        self.constant_zero_probability_: float | None = None
        self.constant_one_probability_: float | None = None

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ZeroOneInflatedBetaAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        y_array = np.clip(y_values.astype(float).to_numpy(), 0.0, 1.0)
        zero_indicator = y_array <= self.model_config.beta_clip_epsilon
        one_indicator = y_array >= 1 - self.model_config.beta_clip_epsilon
        self._fit_boundary_estimator(
            self.zero_estimator,
            zero_indicator.astype(int),
            x_matrix,
            boundary_name="zero",
        )
        self._fit_boundary_estimator(
            self.one_estimator,
            one_indicator.astype(int),
            x_matrix,
            boundary_name="one",
        )
        interior_mask = ~(zero_indicator | one_indicator)
        if interior_mask.sum() < 10:
            raise ValueError(
                "Zero-one inflated beta requires at least 10 interior observations in (0, 1)."
            )
        clipped_target = np.clip(
            y_array[interior_mask],
            self.model_config.beta_clip_epsilon,
            1 - self.model_config.beta_clip_epsilon,
        )
        x_interior = sm.add_constant(x_matrix[interior_mask], has_constant="add")
        self.beta_results = self._run_with_warning_capture(
            lambda: BetaModel(clipped_target, x_interior).fit(
                maxiter=self.model_config.max_iter,
                disp=False,
            ),
            source=f"{self.model_type.value}.beta_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.beta_results,
            source=f"{self.model_type.value}.beta_fit",
        )
        return self

    def _fit_boundary_estimator(
        self,
        estimator: LogisticRegression,
        target: np.ndarray,
        x_matrix: np.ndarray,
        *,
        boundary_name: str,
    ) -> None:
        unique_values = np.unique(target)
        if len(unique_values) == 1:
            probability = float(unique_values[0])
            if boundary_name == "zero":
                self.constant_zero_probability_ = probability
            else:
                self.constant_one_probability_ = probability
            self._add_numerical_diagnostic(
                source=f"{self.model_type.value}.{boundary_name}_classifier",
                diagnostic_name=f"constant_{boundary_name}_probability",
                value=probability,
                status="warn",
                detail=f"The {boundary_name} boundary indicator had one class.",
            )
            return
        self._run_with_warning_capture(
            lambda: estimator.fit(x_matrix, target),
            source=f"{self.model_type.value}.{boundary_name}_classifier",
        )
        self._update_sklearn_fit_diagnostics(
            estimator,
            source=f"{self.model_type.value}.{boundary_name}_classifier",
        )

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def _boundary_probability(
        self,
        estimator: LogisticRegression,
        transformed: np.ndarray,
        constant_probability: float | None,
    ) -> np.ndarray:
        if constant_probability is not None:
            return np.full(len(transformed), constant_probability, dtype=float)
        return estimator.predict_proba(transformed)[:, 1]

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        p_zero = self._boundary_probability(
            self.zero_estimator,
            transformed,
            self.constant_zero_probability_,
        )
        p_one = self._boundary_probability(
            self.one_estimator,
            transformed,
            self.constant_one_probability_,
        )
        boundary_total = np.maximum(p_zero + p_one, 1e-9)
        overfull = boundary_total > 0.95
        p_zero = np.where(overfull, p_zero / boundary_total * 0.95, p_zero)
        p_one = np.where(overfull, p_one / boundary_total * 0.95, p_one)
        p_interior = np.clip(1.0 - p_zero - p_one, 0.0, 1.0)
        interior_mean = np.asarray(
            self.beta_results.predict(sm.add_constant(transformed, has_constant="add"))
        )
        return np.clip(p_one + p_interior * interior_mean, 0.0, 1.0)

    def get_feature_importance(self) -> pd.DataFrame:
        zero_coef = (
            np.zeros(len(self.feature_names_), dtype=float)
            if self.constant_zero_probability_ is not None
            else np.ravel(self.zero_estimator.coef_)
        )
        one_coef = (
            np.zeros(len(self.feature_names_), dtype=float)
            if self.constant_one_probability_ is not None
            else np.ravel(self.one_estimator.coef_)
        )
        beta_params = np.asarray(self.beta_results.params)[1 : len(self.feature_names_) + 1]
        combined = zero_coef + one_coef + beta_params
        table = pd.DataFrame(
            {
                "feature_name": self.feature_names_,
                "importance_value": np.abs(zero_coef) + np.abs(one_coef) + np.abs(beta_params),
                "importance_type": "zero_one_inflated_absolute_coefficient",
                "coefficient": combined,
                "abs_coefficient": np.abs(combined),
                "std_error": np.nan,
                "p_value": np.nan,
                "odds_ratio": np.nan,
                "zero_boundary_coefficient": zero_coef,
                "one_boundary_coefficient": one_coef,
                "interior_beta_coefficient": beta_params,
            }
        )
        return table.sort_values("importance_value", ascending=False).reset_index(drop=True)

    def get_model_artifacts(self) -> dict[str, pd.DataFrame]:
        return self._merge_model_artifacts(
            {
                "zero_one_inflated_beta_components": self.get_feature_importance(),
            }
        )

    @property
    def summary_text(self) -> str:
        return (
            "Zero-one inflated beta LGD model fitted.\n"
            "Stage one: exact-zero probability.\n"
            "Stage two: exact-one probability.\n"
            "Stage three: beta regression for interior severity in (0, 1).\n"
        )


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
            self._add_numerical_diagnostic(
                source=f"{self.model_type.value}.stage_one",
                diagnostic_name="constant_positive_probability",
                value=self.constant_positive_probability_,
                status="warn",
                detail=(
                    "Stage one collapsed to a constant because the positive-loss "
                    "flag had one class."
                ),
            )
        else:
            self._run_with_warning_capture(
                lambda: self.stage_one_estimator.fit(x_matrix, positive_indicator),
                source=f"{self.model_type.value}.stage_one_logit",
            )
            self._update_sklearn_fit_diagnostics(
                self.stage_one_estimator,
                source=f"{self.model_type.value}.stage_one_logit",
            )

        positive_mask = positive_indicator == 1
        if positive_mask.sum() < 10:
            raise ValueError("Two-stage LGD requires at least 10 positive-loss observations.")

        clipped_positive_target = np.clip(
            y_array[positive_mask],
            self.model_config.beta_clip_epsilon,
            1 - self.model_config.beta_clip_epsilon,
        )
        x_positive = sm.add_constant(x_matrix[positive_mask], has_constant="add")
        self.stage_two_results = self._run_with_warning_capture(
            lambda: BetaModel(clipped_positive_target, x_positive).fit(
                maxiter=self.model_config.max_iter,
                disp=False,
            ),
            source=f"{self.model_type.value}.stage_two_beta",
        )
        self._update_statsmodels_fit_diagnostics(
            self.stage_two_results,
            source=f"{self.model_type.value}.stage_two_beta",
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
                "odds_ratio": _safe_odds_ratio(stage_one_coefficients),
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
        return self._merge_model_artifacts(
            {
            "lgd_stage_one_coefficients": pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "coefficient": stage_one_coefficients,
                    "odds_ratio": _safe_odds_ratio(stage_one_coefficients),
                }
            ),
            "lgd_stage_two_coefficients": pd.DataFrame(
                {
                    "feature_name": stage_two_names,
                    "coefficient": stage_two_params,
                }
            ),
            }
        )

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
                n_jobs=_resolved_n_jobs(model_config),
            )
            if target_mode == TargetMode.BINARY
            else XGBRegressor(
                objective="reg:squarederror",
                n_estimators=model_config.xgboost_n_estimators,
                learning_rate=model_config.xgboost_learning_rate,
                max_depth=model_config.xgboost_max_depth,
                subsample=model_config.xgboost_subsample,
                colsample_bytree=model_config.xgboost_colsample_bytree,
                n_jobs=_resolved_n_jobs(model_config),
            )
        )
        super().__init__(
            model_config,
            target_mode,
            estimator,
            scale_numeric=False,
            sparse_preprocessor=False,
        )

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


class RandomForestAdapter(SklearnAdapter):
    """Random forest challenger for nonlinear benchmark comparisons."""

    model_type = ModelType.RANDOM_FOREST

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        estimator = (
            RandomForestClassifier(
                n_estimators=model_config.tree_n_estimators,
                max_depth=model_config.tree_max_depth,
                class_weight=model_config.class_weight,
                random_state=42,
                n_jobs=_resolved_n_jobs(model_config),
            )
            if target_mode == TargetMode.BINARY
            else RandomForestRegressor(
                n_estimators=model_config.tree_n_estimators,
                max_depth=model_config.tree_max_depth,
                random_state=42,
                n_jobs=_resolved_n_jobs(model_config),
            )
        )
        super().__init__(
            model_config,
            target_mode,
            estimator,
            scale_numeric=False,
            sparse_preprocessor=False,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        if self.is_binary_classifier:
            return self.estimator.predict_proba(transformed)[:, 1]
        return self.estimator.predict(transformed)

    @property
    def summary_text(self) -> str:
        return (
            "Random forest model fitted with sklearn.\n"
            f"Trees: {self.model_config.tree_n_estimators}\n"
            f"Max depth: {self.model_config.tree_max_depth}\n"
        )


class ExtraTreesAdapter(SklearnAdapter):
    """Extremely randomized trees challenger for nonlinear benchmark comparisons."""

    model_type = ModelType.EXTRA_TREES

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        estimator = (
            ExtraTreesClassifier(
                n_estimators=model_config.tree_n_estimators,
                max_depth=model_config.tree_max_depth,
                class_weight=model_config.class_weight,
                random_state=42,
                n_jobs=_resolved_n_jobs(model_config),
            )
            if target_mode == TargetMode.BINARY
            else ExtraTreesRegressor(
                n_estimators=model_config.tree_n_estimators,
                max_depth=model_config.tree_max_depth,
                random_state=42,
                n_jobs=_resolved_n_jobs(model_config),
            )
        )
        super().__init__(
            model_config,
            target_mode,
            estimator,
            scale_numeric=False,
            sparse_preprocessor=False,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        if self.is_binary_classifier:
            return self.estimator.predict_proba(transformed)[:, 1]
        return self.estimator.predict(transformed)

    @property
    def summary_text(self) -> str:
        return (
            "Extra Trees model fitted with sklearn.\n"
            f"Trees: {self.model_config.tree_n_estimators}\n"
            f"Max depth: {self.model_config.tree_max_depth}\n"
        )


class DecisionTreeAdapter(SklearnAdapter):
    """Single decision tree for transparent SAS-style tree benchmarking."""

    model_type = ModelType.DECISION_TREE

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        estimator = (
            DecisionTreeClassifier(
                max_depth=model_config.tree_max_depth,
                class_weight=model_config.class_weight
                if target_mode == TargetMode.BINARY
                else None,
                random_state=42,
            )
            if target_mode in {TargetMode.BINARY, TargetMode.MULTICLASS}
            else DecisionTreeRegressor(
                max_depth=model_config.tree_max_depth,
                random_state=42,
            )
        )
        super().__init__(
            model_config,
            target_mode,
            estimator,
            scale_numeric=False,
            sparse_preprocessor=False,
        )

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> DecisionTreeAdapter:
        target = (
            y_values.astype(int)
            if self.target_mode in {TargetMode.BINARY, TargetMode.MULTICLASS}
            else y_values.astype(float)
        )
        return super().fit(x_frame, target, numeric_features, categorical_features)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        transformed = self._transform(x_frame)
        if self.target_mode == TargetMode.BINARY:
            return self.estimator.predict_proba(transformed)[:, 1]
        if self.target_mode == TargetMode.MULTICLASS:
            return self.estimator.predict(transformed).astype(float)
        return self.estimator.predict(transformed)

    def predict_class(self, x_frame: pd.DataFrame, threshold: float) -> np.ndarray | None:
        transformed = self._transform(x_frame)
        if self.target_mode == TargetMode.BINARY:
            return (self.estimator.predict_proba(transformed)[:, 1] >= threshold).astype(int)
        if self.target_mode == TargetMode.MULTICLASS:
            return self.estimator.predict(transformed).astype(int)
        return None

    def predict_proba(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._transform(x_frame))

    @property
    def summary_text(self) -> str:
        return (
            "Single decision tree fitted with sklearn.\n"
            f"Max depth: {self.model_config.tree_max_depth}\n"
        )


class ExplainableBoostingMachineAdapter(SklearnAdapter):
    """No-new-dependency EBM-style shallow boosted-tree challenger."""

    model_type = ModelType.EXPLAINABLE_BOOSTING_MACHINE

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        estimator = (
            GradientBoostingClassifier(
                n_estimators=model_config.tree_n_estimators,
                learning_rate=model_config.xgboost_learning_rate,
                max_depth=min(model_config.tree_max_depth or 2, 2),
                random_state=42,
            )
            if target_mode == TargetMode.BINARY
            else GradientBoostingRegressor(
                n_estimators=model_config.tree_n_estimators,
                learning_rate=model_config.xgboost_learning_rate,
                max_depth=min(model_config.tree_max_depth or 2, 2),
                random_state=42,
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
            "Explainable boosting-style model fitted with sklearn shallow boosted trees.\n"
            "This avoids adding the external interpret package, so review PDP/ALE and "
            "feature-importance outputs as the explanation layer.\n"
            f"Estimators: {self.model_config.tree_n_estimators}\n"
            f"Learning rate: {self.model_config.xgboost_learning_rate}\n"
            f"Max tree depth: {min(self.model_config.tree_max_depth or 2, 2)}\n"
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
        self.results = self._run_with_warning_capture(
            lambda: self._fit_statsmodel(x_with_const, y_values.astype(float)),
            source=f"{self.model_type.value}.statsmodel_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.statsmodel_fit",
        )
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
                "odds_ratio": _safe_odds_ratio(params) if self.is_binary_classifier else np.nan,
            }
        )
        return table.sort_values("importance_value", ascending=False).reset_index(drop=True)

    @property
    def summary_text(self) -> str:
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


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


class FractionalLogitAdapter(StatsmodelsAdapter):
    """Fractional logit / quasi-binomial model for bounded LGD rates."""

    model_type = ModelType.FRACTIONAL_LOGIT

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Fractional logit requires a continuous bounded target.")
        super().__init__(model_config, target_mode, scale_numeric=True)

    def _fit_statsmodel(self, x_matrix: np.ndarray, y_values: pd.Series):
        clipped_target = np.clip(y_values.astype(float), 0.0, 1.0)
        model = sm.GLM(
            clipped_target,
            x_matrix,
            family=sm.families.Binomial(),
        )
        return model.fit(maxiter=self.model_config.max_iter, disp=False)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.results.predict(self._transform(x_frame))), 0.0, 1.0)


class GLMRegressionAdapter(StatsmodelsAdapter):
    """Statsmodels GLM adapter for SAS GENMOD-style continuous/count models."""

    model_type = ModelType.POISSON_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("GLM regression families require a continuous numeric target.")
        super().__init__(model_config, target_mode, scale_numeric=True)
        self.model_type = model_config.model_type

    def _fit_statsmodel(self, x_matrix: np.ndarray, y_values: pd.Series):
        target = y_values.astype(float)
        family = self._resolve_family()
        if self.model_config.model_type in {
            ModelType.POISSON_REGRESSION,
            ModelType.NEGATIVE_BINOMIAL_REGRESSION,
        }:
            target = np.clip(target, 0.0, None)
        elif self.model_config.model_type in {
            ModelType.GAMMA_REGRESSION,
            ModelType.TWEEDIE_REGRESSION,
        }:
            target = np.clip(target, 1e-9, None)
        return sm.GLM(target, x_matrix, family=family).fit(maxiter=self.model_config.max_iter)

    def _resolve_family(self):
        if self.model_config.model_type == ModelType.POISSON_REGRESSION:
            return sm.families.Poisson()
        if self.model_config.model_type == ModelType.NEGATIVE_BINOMIAL_REGRESSION:
            return sm.families.NegativeBinomial()
        if self.model_config.model_type == ModelType.GAMMA_REGRESSION:
            return sm.families.Gamma(link=sm.families.links.Log())
        if self.model_config.model_type == ModelType.TWEEDIE_REGRESSION:
            return sm.families.Tweedie(
                var_power=self.model_config.tweedie_variance_power,
                link=sm.families.links.Log(),
            )
        raise ValueError(f"Unsupported GLM model type: {self.model_config.model_type.value}")

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.results.predict(self._transform(x_frame))), 0.0, None)


class OrdinalLogisticRegressionAdapter(BaseModelAdapter):
    """Ordered logit model for ordinal multiclass outcomes."""

    model_type = ModelType.ORDINAL_LOGISTIC_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.MULTICLASS:
            raise ValueError("Ordinal logistic regression requires a multiclass target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None
        self.class_labels_: np.ndarray = np.array([], dtype=int)

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> OrdinalLogisticRegressionAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = self._build_ordinal_preprocessor(
            numeric_features,
            categorical_features,
        )
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        target = y_values.astype(int).to_numpy()
        self.class_labels_ = np.sort(np.unique(target))
        self.results = self._run_with_warning_capture(
            lambda: OrderedModel(target, x_matrix, distr="logit").fit(
                method="bfgs",
                maxiter=self.model_config.max_iter,
                disp=False,
            ),
            source=f"{self.model_type.value}.ordered_logit_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.ordered_logit_fit",
        )
        return self

    def _build_ordinal_preprocessor(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ColumnTransformer:
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_features:
            transformers.append(("numeric", StandardScaler(), numeric_features))
        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    categorical_features,
                )
            )
        if not transformers:
            raise ValueError("Ordinal logistic regression requires at least one feature.")
        return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def predict_proba(self, x_frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(
            self.results.model.predict(self.results.params, self._transform(x_frame))
        )
        return probabilities.reshape((len(x_frame), len(self.class_labels_)))

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return self.predict_class(x_frame, threshold=0.5).astype(float)

    def predict_class(self, x_frame: pd.DataFrame, threshold: float) -> np.ndarray | None:
        probabilities = self.predict_proba(x_frame)
        return self.class_labels_[np.argmax(probabilities, axis=1)].astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        feature_params = params[: len(self.feature_names_)]
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(feature_params),
                    "importance_type": "ordinal_logit_absolute_coefficient",
                    "coefficient": feature_params,
                    "abs_coefficient": np.abs(feature_params),
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": _safe_odds_ratio(feature_params),
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


class GEELogisticRegressionAdapter(BaseModelAdapter):
    """GEE logistic model for clustered or repeated-observation binary PD data."""

    model_type = ModelType.GEE_LOGISTIC_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.BINARY:
            raise ValueError("GEE logistic regression requires a binary target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None
        self.model_feature_columns_: list[str] = []

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> GEELogisticRegressionAdapter:
        group_column = self.model_config.gee_group_column
        groups = (
            x_frame[group_column].astype(str).to_numpy()
            if group_column and group_column in x_frame.columns
            else np.arange(len(x_frame))
        )
        self.model_feature_columns_ = [
            column for column in x_frame.columns if column != group_column
        ]
        model_numeric = [
            feature for feature in numeric_features if feature in self.model_feature_columns_
        ]
        model_categorical = [
            feature for feature in categorical_features if feature in self.model_feature_columns_
        ]
        self.raw_feature_names_ = list(self.model_feature_columns_)
        self.raw_numeric_features_ = model_numeric
        self.raw_categorical_features_ = model_categorical
        self.preprocessor = build_preprocessor(model_numeric, model_categorical)
        x_matrix = self.preprocessor.fit_transform(x_frame.loc[:, self.model_feature_columns_])
        x_with_const = sm.add_constant(x_matrix, has_constant="add")
        self.feature_names_ = ["const", *self.preprocessor.get_feature_names_out().tolist()]
        self.results = self._run_with_warning_capture(
            lambda: sm.GEE(
                y_values.astype(float),
                x_with_const,
                groups=groups,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit(maxiter=self.model_config.max_iter),
            source=f"{self.model_type.value}.gee_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.gee_fit",
        )
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        transformed = self.preprocessor.transform(x_frame.loc[:, self.model_feature_columns_])
        return sm.add_constant(transformed, has_constant="add")

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.results.predict(self._transform(x_frame))), 0.0, 1.0)

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        pvalues = np.asarray(getattr(self.results, "pvalues", np.full_like(params, np.nan)))
        bse = np.asarray(getattr(self.results, "bse", np.full_like(params, np.nan)))
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(params),
                    "importance_type": "gee_absolute_coefficient",
                    "coefficient": params,
                    "abs_coefficient": np.abs(params),
                    "std_error": bse,
                    "p_value": pvalues,
                    "odds_ratio": _safe_odds_ratio(params),
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


class MixedEffectsRegressionAdapter(BaseModelAdapter):
    """Random-intercept mixed-effects regression using statsmodels MixedLM."""

    model_type = ModelType.MIXED_EFFECTS_REGRESSION

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Mixed-effects regression requires a continuous target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None
        self.model_feature_columns_: list[str] = []

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> MixedEffectsRegressionAdapter:
        group_column = self.model_config.mixed_effects_group_column
        groups = (
            x_frame[group_column].astype(str).to_numpy()
            if group_column and group_column in x_frame.columns
            else np.arange(len(x_frame))
        )
        self.model_feature_columns_ = [
            column for column in x_frame.columns if column != group_column
        ]
        model_numeric = [
            feature for feature in numeric_features if feature in self.model_feature_columns_
        ]
        model_categorical = [
            feature for feature in categorical_features if feature in self.model_feature_columns_
        ]
        self.raw_feature_names_ = list(self.model_feature_columns_)
        self.raw_numeric_features_ = model_numeric
        self.raw_categorical_features_ = model_categorical
        self.preprocessor = build_preprocessor(model_numeric, model_categorical)
        x_matrix = self.preprocessor.fit_transform(x_frame.loc[:, self.model_feature_columns_])
        x_with_const = sm.add_constant(x_matrix, has_constant="add")
        self.feature_names_ = ["const", *self.preprocessor.get_feature_names_out().tolist()]
        self.results = self._run_with_warning_capture(
            lambda: sm.MixedLM(y_values.astype(float), x_with_const, groups=groups).fit(
                maxiter=self.model_config.max_iter,
                reml=False,
                method="lbfgs",
            ),
            source=f"{self.model_type.value}.mixedlm_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.mixedlm_fit",
        )
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        transformed = self.preprocessor.transform(x_frame.loc[:, self.model_feature_columns_])
        return sm.add_constant(transformed, has_constant="add")

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        fixed_effects = np.asarray(self.results.fe_params)
        return self._transform(x_frame) @ fixed_effects

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.fe_params)
        pvalues = np.asarray(getattr(self.results, "pvalues", np.full_like(params, np.nan)))[
            : len(params)
        ]
        bse = np.asarray(getattr(self.results, "bse_fe", np.full_like(params, np.nan)))
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(params),
                    "importance_type": "mixed_effects_absolute_fixed_effect",
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
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


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


class CoxProportionalHazardsAdapter(BaseModelAdapter):
    """Cox proportional hazards model using statsmodels PHReg with observed events."""

    model_type = ModelType.COX_PROPORTIONAL_HAZARDS

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Cox proportional hazards requires a continuous duration target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> CoxProportionalHazardsAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        x_matrix = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        durations = np.clip(y_values.astype(float).to_numpy(), 1e-9, None)
        status = np.ones(len(durations), dtype=int)
        self.results = self._run_with_warning_capture(
            lambda: PHReg(durations, x_matrix, status=status).fit(
                maxiter=self.model_config.max_iter,
                disp=False,
            ),
            source=f"{self.model_type.value}.phreg_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.phreg_fit",
        )
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        linear_predictor = self._transform(x_frame) @ np.asarray(self.results.params)
        return np.exp(np.clip(linear_predictor, -20, 20))

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        pvalues = np.asarray(getattr(self.results, "pvalues", np.full_like(params, np.nan)))
        bse = np.asarray(getattr(self.results, "bse", np.full_like(params, np.nan)))
        return (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_,
                    "importance_value": np.abs(params),
                    "importance_type": "cox_absolute_log_hazard",
                    "coefficient": params,
                    "abs_coefficient": np.abs(params),
                    "std_error": bse,
                    "p_value": pvalues,
                    "odds_ratio": np.nan,
                    "hazard_ratio": _safe_odds_ratio(params),
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


class AFTSurvivalModelAdapter(SklearnAdapter):
    """Log-normal AFT-style duration model using linear regression on log duration."""

    model_type = ModelType.AFT_SURVIVAL_MODEL

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("AFT survival modeling requires a continuous duration target.")
        super().__init__(
            model_config,
            target_mode,
            LinearRegression(),
            scale_numeric=True,
        )

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> AFTSurvivalModelAdapter:
        log_duration = np.log(np.clip(y_values.astype(float).to_numpy(), 1e-9, None))
        return super().fit(
            x_frame,
            pd.Series(log_duration, index=y_values.index),
            numeric_features,
            categorical_features,
        )

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.estimator.predict(self._transform(x_frame)))

    @property
    def summary_text(self) -> str:
        return "AFT-style log-normal duration model fitted with sklearn linear regression.\n"


class SARIMAXForecastAdapter(BaseModelAdapter):
    """SARIMAX forecasting model with optional exogenous feature matrix."""

    model_type = ModelType.SARIMAX_FORECAST

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("SARIMAX forecasting requires a continuous target.")
        super().__init__(model_config, target_mode)
        self.preprocessor = None
        self.results = None
        self.train_row_count_: int = 0
        self.prediction_calls_: int = 0

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> SARIMAXForecastAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            scale_numeric=True,
        )
        exog = self.preprocessor.fit_transform(x_frame)
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        self.train_row_count_ = len(y_values)
        order = (
            self.model_config.sarimax_order_p,
            self.model_config.sarimax_order_d,
            self.model_config.sarimax_order_q,
        )
        self.results = self._run_with_warning_capture(
            lambda: SARIMAX(
                y_values.astype(float).to_numpy(),
                exog=exog,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=self.model_config.max_iter),
            source=f"{self.model_type.value}.sarimax_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.sarimax_fit",
        )
        return self

    def _transform(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("The model adapter has not been fitted yet.")
        return self.preprocessor.transform(x_frame)

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        exog = self._transform(x_frame)
        if self.prediction_calls_ == 0 and len(x_frame) == self.train_row_count_:
            self.prediction_calls_ += 1
            return np.asarray(self.results.fittedvalues)
        self.prediction_calls_ += 1
        return np.asarray(self.results.get_forecast(steps=len(x_frame), exog=exog).predicted_mean)

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        names = list(getattr(self.results, "param_names", [])) or [
            f"parameter_{index}" for index in range(len(params))
        ]
        return (
            pd.DataFrame(
                {
                    "feature_name": names,
                    "importance_value": np.abs(params),
                    "importance_type": "sarimax_absolute_parameter",
                    "coefficient": params,
                    "abs_coefficient": np.abs(params),
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.nan,
                }
            )
            .sort_values("importance_value", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def summary_text(self) -> str:
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


class ExponentialSmoothingForecastAdapter(BaseModelAdapter):
    """SAS/ETS-style exponential smoothing forecast over the target series."""

    model_type = ModelType.EXPONENTIAL_SMOOTHING_FORECAST

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Exponential smoothing requires a continuous target.")
        super().__init__(model_config, target_mode)
        self.results = None
        self.train_row_count_: int = 0
        self.prediction_calls_: int = 0

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ExponentialSmoothingForecastAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.feature_names_ = ["level", "trend"]
        self.train_row_count_ = len(y_values)
        seasonal_periods = self.model_config.seasonal_periods
        seasonal = (
            "add"
            if seasonal_periods and self.train_row_count_ >= seasonal_periods * 2
            else None
        )
        self.results = self._run_with_warning_capture(
            lambda: ExponentialSmoothing(
                y_values.astype(float).to_numpy(),
                trend="add",
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True),
            source=f"{self.model_type.value}.exponential_smoothing_fit",
        )
        return self

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.prediction_calls_ == 0 and len(x_frame) == self.train_row_count_:
            self.prediction_calls_ += 1
            return np.asarray(self.results.fittedvalues)
        self.prediction_calls_ += 1
        return np.asarray(self.results.forecast(len(x_frame)))

    def get_feature_importance(self) -> pd.DataFrame:
        params = getattr(self.results, "params", {})
        rows = [
            {
                "feature_name": str(name),
                "importance_value": abs(float(value))
                if np.isscalar(value) and pd.notna(value)
                else np.nan,
                "importance_type": "exponential_smoothing_parameter",
                "coefficient": float(value) if np.isscalar(value) and pd.notna(value) else np.nan,
                "abs_coefficient": abs(float(value))
                if np.isscalar(value) and pd.notna(value)
                else np.nan,
                "std_error": np.nan,
                "p_value": np.nan,
                "odds_ratio": np.nan,
            }
            for name, value in params.items()
            if np.isscalar(value)
        ]
        return pd.DataFrame(rows) if rows else self._fallback_importance()

    def _fallback_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature_name": "time_series_level",
                    "importance_value": 1.0,
                    "importance_type": "forecast_component",
                    "coefficient": np.nan,
                    "abs_coefficient": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.nan,
                }
            ]
        )

    @property
    def summary_text(self) -> str:
        return "Exponential smoothing forecast fitted with statsmodels.\n"


class UnobservedComponentsForecastAdapter(BaseModelAdapter):
    """Unobserved-components forecast with local linear trend."""

    model_type = ModelType.UNOBSERVED_COMPONENTS_FORECAST

    def __init__(self, model_config: ModelConfig, target_mode: TargetMode) -> None:
        if target_mode != TargetMode.CONTINUOUS:
            raise ValueError("Unobserved components forecasting requires a continuous target.")
        super().__init__(model_config, target_mode)
        self.results = None
        self.train_row_count_: int = 0
        self.prediction_calls_: int = 0

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> UnobservedComponentsForecastAdapter:
        self.raw_feature_names_ = list(x_frame.columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.feature_names_ = ["level", "trend"]
        self.train_row_count_ = len(y_values)
        seasonal = self.model_config.seasonal_periods
        self.results = self._run_with_warning_capture(
            lambda: UnobservedComponents(
                y_values.astype(float).to_numpy(),
                level="local linear trend",
                seasonal=seasonal,
            ).fit(disp=False, maxiter=self.model_config.max_iter),
            source=f"{self.model_type.value}.ucm_fit",
        )
        self._update_statsmodels_fit_diagnostics(
            self.results,
            source=f"{self.model_type.value}.ucm_fit",
        )
        return self

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.prediction_calls_ == 0 and len(x_frame) == self.train_row_count_:
            self.prediction_calls_ += 1
            return np.asarray(self.results.fittedvalues)
        self.prediction_calls_ += 1
        return np.asarray(self.results.get_forecast(steps=len(x_frame)).predicted_mean)

    def get_feature_importance(self) -> pd.DataFrame:
        params = np.asarray(self.results.params)
        names = list(getattr(self.results, "param_names", [])) or [
            f"parameter_{index}" for index in range(len(params))
        ]
        return pd.DataFrame(
            {
                "feature_name": names,
                "importance_value": np.abs(params),
                "importance_type": "unobserved_components_absolute_parameter",
                "coefficient": params,
                "abs_coefficient": np.abs(params),
                "std_error": np.nan,
                "p_value": np.nan,
                "odds_ratio": np.nan,
            }
        )

    @property
    def summary_text(self) -> str:
        return self._run_with_warning_capture(
            lambda: self.results.summary().as_text(),
            source=f"{self.model_type.value}.summary",
            stage="summary",
        )


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
    if model_config.model_type == ModelType.MULTINOMIAL_LOGISTIC_REGRESSION:
        return MultinomialLogisticRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.ORDINAL_LOGISTIC_REGRESSION:
        return OrdinalLogisticRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.FRACTIONAL_LOGIT:
        return FractionalLogitAdapter(model_config, target_mode)
    if model_config.model_type in {
        ModelType.POISSON_REGRESSION,
        ModelType.NEGATIVE_BINOMIAL_REGRESSION,
        ModelType.GAMMA_REGRESSION,
        ModelType.TWEEDIE_REGRESSION,
    }:
        return GLMRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.ZERO_ONE_INFLATED_BETA:
        return ZeroOneInflatedBetaAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.RIDGE_REGRESSION:
        return RidgeRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.LASSO_REGRESSION:
        return LassoRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.ELASTIC_NET_REGRESSION:
        return ElasticNetRegressionAdapter(model_config, target_mode)
    if model_config.model_type in {
        ModelType.GAM_SPLINE_REGRESSION,
        ModelType.GAM_SPLINE_LOGISTIC,
    }:
        return GAMSplineAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.EXPLAINABLE_BOOSTING_MACHINE:
        return ExplainableBoostingMachineAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.GEE_LOGISTIC_REGRESSION:
        return GEELogisticRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.MIXED_EFFECTS_REGRESSION:
        return MixedEffectsRegressionAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.COX_PROPORTIONAL_HAZARDS:
        return CoxProportionalHazardsAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.AFT_SURVIVAL_MODEL:
        return AFTSurvivalModelAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.RANDOM_FOREST:
        return RandomForestAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.EXTRA_TREES:
        return ExtraTreesAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.DECISION_TREE:
        return DecisionTreeAdapter(model_config, target_mode)
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
    if model_config.model_type == ModelType.SARIMAX_FORECAST:
        return SARIMAXForecastAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.EXPONENTIAL_SMOOTHING_FORECAST:
        return ExponentialSmoothingForecastAdapter(model_config, target_mode)
    if model_config.model_type == ModelType.UNOBSERVED_COMPONENTS_FORECAST:
        return UnobservedComponentsForecastAdapter(model_config, target_mode)

    raise ValueError(f"Unsupported model type '{model_config.model_type}'.")
