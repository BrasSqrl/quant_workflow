"""Segmented model router used for governed per-segment model builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelConfig, ModelType, TargetMode

MISSING_SEGMENT_TOKEN = "<missing>"


def build_segment_key_series(frame: pd.DataFrame, segment_columns: list[str]) -> pd.Series:
    """Builds a stable composite segment key from one or more segment columns."""

    if not segment_columns:
        return pd.Series(["__global__"] * len(frame), index=frame.index, dtype="string")
    missing_columns = [column for column in segment_columns if column not in frame.columns]
    if missing_columns:
        return pd.Series([MISSING_SEGMENT_TOKEN] * len(frame), index=frame.index, dtype="string")

    parts: list[pd.Series] = []
    for column_name in segment_columns:
        values = frame[column_name].astype("object")
        normalized = values.where(~pd.isna(values), MISSING_SEGMENT_TOKEN).astype(str)
        normalized = normalized.str.strip().replace("", MISSING_SEGMENT_TOKEN)
        parts.append(column_name + "=" + normalized)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " | " + part
    return combined.astype("string")


@dataclass(slots=True)
class SegmentRouteSummary:
    """Routing outputs for a prediction request."""

    segment_key: np.ndarray
    segment_model_id: np.ndarray
    segment_model_status: np.ndarray
    used_global_fallback: np.ndarray


class SegmentedModelBundle:
    """Routes each scored row to its segment model or to the global fallback model."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        target_mode: TargetMode,
        segment_columns: list[str],
        feature_columns: list[str],
        numeric_features: list[str],
        categorical_features: list[str],
        global_model: Any,
        segment_models: dict[str, Any],
        segment_inventory: pd.DataFrame,
    ) -> None:
        self.model_config = model_config
        self.model_type: ModelType = model_config.model_type
        self.target_mode = target_mode
        self.segment_columns_ = list(segment_columns)
        self.raw_feature_names_ = list(feature_columns)
        self.raw_numeric_features_ = list(numeric_features)
        self.raw_categorical_features_ = list(categorical_features)
        self.global_model_ = global_model
        self.segment_models_ = dict(segment_models)
        self.segment_inventory_ = segment_inventory.copy(deep=True).reset_index(drop=True)
        self.feature_names_ = getattr(global_model, "feature_names_", [])
        self.numerical_warning_records_: list[dict[str, Any]] = []
        self.numerical_diagnostics_: list[dict[str, Any]] = []

    @property
    def is_binary_classifier(self) -> bool:
        return self.target_mode == TargetMode.BINARY

    @property
    def summary_text(self) -> str:
        fitted = int((self.segment_inventory_["status"] == "fitted").sum())
        fallback = int((self.segment_inventory_["status"] != "fitted").sum())
        return (
            f"Segmented {self.model_type.value} router\n"
            f"Segment columns: {', '.join(self.segment_columns_)}\n"
            f"Fitted segment models: {fitted}\n"
            f"Fallback segments: {fallback}\n"
            "Fallback policy: global model"
        )

    def fit(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> SegmentedModelBundle:
        """Segmented bundles are fit by ModelTrainingStep before construction."""

        return self

    def predict_score(self, x_frame: pd.DataFrame) -> np.ndarray:
        route = self._route(x_frame)
        predictions = np.empty(len(x_frame), dtype=float)
        feature_frame = self._feature_frame(x_frame)
        key_series = pd.Series(route.segment_key, index=x_frame.index)
        for segment_key in pd.unique(key_series):
            mask = key_series == segment_key
            model = self.segment_models_.get(str(segment_key), self.global_model_)
            predictions[mask.to_numpy()] = np.asarray(
                model.predict_score(feature_frame.loc[mask, self.raw_feature_names_])
            )
        return predictions

    def predict_class(self, x_frame: pd.DataFrame, threshold: float) -> np.ndarray | None:
        if not self.is_binary_classifier:
            return None
        return (self.predict_score(x_frame) >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        global_importance = self.global_model_.get_feature_importance().copy(deep=True)
        global_importance.insert(0, "model_status", "global_fallback")
        global_importance.insert(0, "model_id", "global")
        global_importance.insert(0, "segment_key", "__global__")
        frames.append(global_importance)

        for segment_key, model in self.segment_models_.items():
            table = model.get_feature_importance().copy(deep=True)
            table.insert(0, "model_status", "fitted")
            table.insert(0, "model_id", f"segment::{segment_key}")
            table.insert(0, "segment_key", segment_key)
            frames.append(table)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_model_artifacts(self) -> dict[str, pd.DataFrame]:
        artifacts: dict[str, pd.DataFrame] = {
            "segment_model_inventory": self.segment_inventory_.copy(deep=True),
        }
        fallback_segments = self.segment_inventory_.loc[
            self.segment_inventory_["status"] != "fitted"
        ].copy(deep=True)
        if not fallback_segments.empty:
            artifacts["fallback_segments"] = fallback_segments
        importance = self.get_feature_importance()
        if not importance.empty:
            artifacts["segment_coefficients_or_importance"] = importance
        return artifacts

    def get_prediction_outputs(self, x_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        route = self._route(x_frame)
        return {
            "segment_key": route.segment_key,
            "segment_model_id": route.segment_model_id,
            "segment_model_status": route.segment_model_status,
            "used_global_fallback": route.used_global_fallback,
        }

    def get_numerical_warning_table(self) -> pd.DataFrame:
        frames = []
        for model_id, model in self._iter_models():
            getter = getattr(model, "get_numerical_warning_table", None)
            if not callable(getter):
                continue
            table = getter()
            if not table.empty:
                copied = table.copy(deep=True)
                copied.insert(0, "model_id", model_id)
                frames.append(copied)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_numerical_diagnostics_table(self) -> pd.DataFrame:
        frames = []
        for model_id, model in self._iter_models():
            getter = getattr(model, "get_numerical_diagnostics_table", None)
            if not callable(getter):
                continue
            table = getter()
            if not table.empty:
                copied = table.copy(deep=True)
                copied.insert(0, "model_id", model_id)
                frames.append(copied)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _route(self, x_frame: pd.DataFrame) -> SegmentRouteSummary:
        keys = build_segment_key_series(x_frame, self.segment_columns_).astype(str).to_numpy()
        statuses: list[str] = []
        model_ids: list[str] = []
        fallback_flags: list[bool] = []
        inventory_status = {
            str(row["segment_key"]): str(row["status"])
            for _, row in self.segment_inventory_.iterrows()
        }
        for segment_key in keys:
            if segment_key in self.segment_models_:
                statuses.append("fitted")
                model_ids.append(f"segment::{segment_key}")
                fallback_flags.append(False)
            else:
                statuses.append(inventory_status.get(segment_key, "unseen_or_missing_segment"))
                model_ids.append("global")
                fallback_flags.append(True)
        return SegmentRouteSummary(
            segment_key=keys,
            segment_model_id=np.asarray(model_ids, dtype=object),
            segment_model_status=np.asarray(statuses, dtype=object),
            used_global_fallback=np.asarray(fallback_flags, dtype=bool),
        )

    def _feature_frame(self, x_frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.raw_feature_names_ if column not in x_frame.columns]
        if missing:
            raise ValueError(
                "Segmented model scoring is missing required feature columns: "
                + ", ".join(missing)
            )
        return x_frame.loc[:, self.raw_feature_names_]

    def _iter_models(self):
        yield "global", self.global_model_
        for segment_key, model in self.segment_models_.items():
            yield f"segment::{segment_key}", model
