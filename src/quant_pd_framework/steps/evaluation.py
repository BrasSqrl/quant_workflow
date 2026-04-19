"""Scores each split and computes standard PD-model diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..base import BasePipelineStep
from ..config import TargetMode
from ..context import PipelineContext


class EvaluationStep(BasePipelineStep):
    """
    Measures how well the fitted model separates defaults from non-defaults.

    The framework keeps training and validation/test diagnostics together so the
    user can review discrimination, calibration, and threshold-based metrics.
    """

    name = "evaluation"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.model is None:
            raise ValueError("Evaluation requires a fitted model.")
        if context.target_column is None:
            raise ValueError("Evaluation requires a target column.")

        threshold = context.config.model.threshold
        target_mode = context.config.target.mode
        labels_available = bool(context.metadata.get("labels_available", False))

        for split_name, frame in context.split_frames.items():
            if target_mode == TargetMode.BINARY:
                scored_frame, metrics = self._score_binary_split(
                    frame,
                    split_name,
                    context.target_column,
                    context.feature_columns,
                    context.model,
                    threshold,
                    labels_available and context.target_column in frame.columns,
                )
            else:
                scored_frame, metrics = self._score_continuous_split(
                    frame,
                    split_name,
                    context.target_column,
                    context.feature_columns,
                    context.model,
                    labels_available and context.target_column in frame.columns,
                )
            context.predictions[split_name] = scored_frame
            context.metrics[split_name] = metrics

        context.feature_importance = context.model.get_feature_importance()
        context.model_artifacts = context.model.get_model_artifacts()
        return context

    def _score_binary_split(
        self,
        frame: pd.DataFrame,
        split_name: str,
        target_column: str,
        feature_columns: list[str],
        model,
        threshold: float,
        labels_available: bool,
    ) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
        x_values = frame[feature_columns]
        probability = np.asarray(model.predict_score(x_values))
        predicted_class = np.asarray(model.predict_class(x_values, threshold))

        scored_frame = frame.copy(deep=True).reset_index(drop=True)
        scored_frame["split"] = split_name
        scored_frame["predicted_probability"] = probability
        scored_frame["predicted_class"] = predicted_class

        if not labels_available:
            metrics = {
                "row_count": int(len(frame)),
                "labels_available": False,
                "average_predicted_probability": float(np.mean(probability))
                if len(probability)
                else None,
                "predicted_positive_rate": float(np.mean(predicted_class))
                if len(predicted_class)
                else None,
                "roc_auc": None,
                "average_precision": None,
                "brier_score": None,
                "log_loss": None,
                "ks_statistic": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "matthews_correlation": None,
                "true_negative": None,
                "false_positive": None,
                "false_negative": None,
                "true_positive": None,
            }
            return scored_frame, metrics

        y_true = frame[target_column].astype(int)
        confusion = confusion_matrix(y_true, predicted_class, labels=[0, 1])
        metrics = {
            "row_count": int(len(frame)),
            "labels_available": True,
            "default_rate": float(y_true.mean()),
            "roc_auc": self._safe_metric(lambda: roc_auc_score(y_true, probability)),
            "average_precision": self._safe_metric(
                lambda: average_precision_score(y_true, probability)
            ),
            "brier_score": self._safe_metric(lambda: brier_score_loss(y_true, probability)),
            "log_loss": self._safe_metric(lambda: log_loss(y_true, probability, labels=[0, 1])),
            "ks_statistic": self._safe_metric(lambda: self._ks_statistic(y_true, probability)),
            "accuracy": self._safe_metric(lambda: accuracy_score(y_true, predicted_class)),
            "precision": self._safe_metric(
                lambda: precision_score(y_true, predicted_class, zero_division=0)
            ),
            "recall": self._safe_metric(
                lambda: recall_score(y_true, predicted_class, zero_division=0)
            ),
            "f1_score": self._safe_metric(
                lambda: f1_score(y_true, predicted_class, zero_division=0)
            ),
            "matthews_correlation": self._safe_metric(
                lambda: matthews_corrcoef(y_true, predicted_class)
            ),
            "true_negative": int(confusion[0, 0]),
            "false_positive": int(confusion[0, 1]),
            "false_negative": int(confusion[1, 0]),
            "true_positive": int(confusion[1, 1]),
        }

        return scored_frame, metrics

    def _score_continuous_split(
        self,
        frame: pd.DataFrame,
        split_name: str,
        target_column: str,
        feature_columns: list[str],
        model,
        labels_available: bool,
    ) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
        x_values = frame[feature_columns]
        prediction = np.asarray(model.predict_score(x_values))

        scored_frame = frame.copy(deep=True).reset_index(drop=True)
        scored_frame["split"] = split_name
        scored_frame["predicted_value"] = prediction

        if not labels_available:
            metrics = {
                "row_count": int(len(frame)),
                "labels_available": False,
                "mean_predicted": float(np.mean(prediction)) if len(prediction) else None,
                "prediction_std": float(np.std(prediction)) if len(prediction) else None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "explained_variance": None,
            }
            return scored_frame, metrics

        y_true = frame[target_column].astype(float)
        residual = y_true - prediction
        scored_frame["residual"] = residual

        metrics = {
            "row_count": int(len(frame)),
            "labels_available": True,
            "mean_actual": float(y_true.mean()),
            "mean_predicted": float(np.mean(prediction)),
            "rmse": self._safe_metric(lambda: math.sqrt(mean_squared_error(y_true, prediction))),
            "mae": self._safe_metric(lambda: mean_absolute_error(y_true, prediction)),
            "r2": self._safe_metric(lambda: r2_score(y_true, prediction)),
            "explained_variance": self._safe_metric(
                lambda: explained_variance_score(y_true, prediction)
            ),
        }
        return scored_frame, metrics

    def _safe_metric(self, metric_callable) -> float | None:
        try:
            value = metric_callable()
        except ValueError:
            return None

        if value is None or pd.isna(value):
            return None
        return float(value)

    def _ks_statistic(self, y_true: pd.Series, y_score: np.ndarray) -> float:
        positives = y_score[y_true == 1]
        negatives = y_score[y_true == 0]
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("KS statistic requires both classes.")

        all_scores = np.sort(np.unique(y_score))
        positive_cdf = np.searchsorted(np.sort(positives), all_scores, side="right") / len(
            positives
        )
        negative_cdf = np.searchsorted(np.sort(negatives), all_scores, side="right") / len(
            negatives
        )
        return float(np.max(np.abs(positive_cdf - negative_cdf)))

    def build_classification_curves(
        self,
        y_true: pd.Series,
        probability: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Returns reusable arrays for precision-recall and calibration diagnostics."""

        precision, recall, pr_thresholds = precision_recall_curve(y_true, probability)
        calibration_true, calibration_pred = calibration_curve(
            y_true,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        return {
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
            "calibration_true": calibration_true,
            "calibration_pred": calibration_pred,
        }
