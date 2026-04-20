"""Fits the selected model family or loads an existing fitted artifact."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd

from ..base import BasePipelineStep
from ..config import ExecutionMode
from ..context import PipelineContext
from ..models import build_model_adapter


class ModelTrainingStep(BasePipelineStep):
    """
    Resolves the model that downstream steps should use.

    In the default flow this fits a fresh estimator on the training split.
    In existing-model mode it loads a saved artifact and validates that the
    prepared dataframe still contains the feature columns that model expects.
    """

    name = "training"

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.split_frames:
            raise ValueError("Training requires populated split frames.")
        if not context.feature_columns or context.target_column is None:
            raise ValueError("Training requires resolved feature columns and target column.")

        if context.config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
            return self._load_existing_model(context)
        return self._fit_new_model(context)

    def _fit_new_model(self, context: PipelineContext) -> PipelineContext:
        train_frame = context.split_frames["train"]
        x_train = train_frame[context.feature_columns]
        y_train = train_frame[context.target_column]
        target_mode = context.config.target.mode
        model_adapter = build_model_adapter(
            context.config.model,
            target_mode,
            scorecard_config=context.config.scorecard,
            scorecard_bin_overrides={
                override.feature_name: override.bin_edges
                for override in context.config.manual_review.scorecard_bin_overrides
            },
        )
        model_adapter.fit(
            x_train,
            y_train,
            context.numeric_features,
            context.categorical_features,
        )

        context.model = model_adapter
        context.model_summary = model_adapter.summary_text
        context.metadata["model_reused"] = False
        self._publish_model_numerical_findings(context, model_adapter, loaded_model=False)
        return context

    def _load_existing_model(self, context: PipelineContext) -> PipelineContext:
        model_path = context.config.execution.existing_model_path
        if model_path is None:
            raise ValueError(
                "Existing-model execution requires ExecutionConfig.existing_model_path."
            )

        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Existing model artifact not found: {resolved_path}")

        model_adapter = joblib.load(resolved_path)
        self._validate_loaded_model(context, model_adapter, resolved_path)
        self._apply_loaded_feature_contract(context, model_adapter)

        context.model = model_adapter
        context.model_summary = getattr(
            model_adapter, "summary_text", f"Loaded model from {resolved_path}"
        )
        context.metadata["loaded_model_path"] = str(resolved_path)
        context.metadata["model_reused"] = True
        self._publish_model_numerical_findings(context, model_adapter, loaded_model=True)
        return context

    def _publish_model_numerical_findings(
        self,
        context: PipelineContext,
        model_adapter,
        *,
        loaded_model: bool,
    ) -> None:
        warning_table_getter = getattr(model_adapter, "get_numerical_warning_table", None)
        if not callable(warning_table_getter):
            return
        warning_table = warning_table_getter()
        if warning_table.empty:
            return

        if loaded_model:
            context.warn(
                "Loaded model artifact carries normalized fit-time numerical warnings. "
                "Review `numerical_warning_summary` in the exported outputs."
            )
            return

        for _, row in warning_table.iterrows():
            count_suffix = (
                f" ({int(row['occurrence_count'])}x)"
                if int(row.get("occurrence_count", 1)) > 1
                else ""
            )
            context.warn(
                f"{row['source']}: {row['message']}{count_suffix}. "
                "Review `numerical_warning_summary` for the normalized record."
            )

    def _validate_loaded_model(
        self, context: PipelineContext, model_adapter, model_path: Path
    ) -> None:
        required_methods = ["predict_score", "get_feature_importance"]
        missing_methods = [name for name in required_methods if not hasattr(model_adapter, name)]
        if missing_methods:
            raise ValueError(
                f"Loaded model '{model_path}' is not a compatible framework artifact. "
                f"Missing methods: {', '.join(missing_methods)}."
            )

        loaded_target_mode = getattr(model_adapter, "target_mode", None)
        if loaded_target_mode is not None and loaded_target_mode != context.config.target.mode:
            raise ValueError(
                "Loaded model target mode does not match the current run configuration. "
                f"Model target mode: {loaded_target_mode.value}. "
                f"Configured target mode: {context.config.target.mode.value}."
            )

        loaded_model_type = getattr(model_adapter, "model_type", None)
        if loaded_model_type is not None and loaded_model_type != context.config.model.model_type:
            context.warn(
                "Loaded model type differs from the current model configuration. "
                f"Using loaded model type '{loaded_model_type.value}' for scoring."
            )
            context.config.model.model_type = loaded_model_type

    def _apply_loaded_feature_contract(self, context: PipelineContext, model_adapter) -> None:
        expected_features, numeric_features, categorical_features = self._extract_feature_contract(
            model_adapter
        )
        if not expected_features:
            context.warn(
                "The loaded model did not expose its original raw feature list. "
                "The framework will use the current feature-engineering output as-is."
            )
            return

        dataframe = context.split_frames.get("train")
        if dataframe is None:
            dataframe = next(iter(context.split_frames.values()), None)
        if dataframe is None:
            dataframe = context.working_data
        if dataframe is None:
            raise ValueError("Feature compatibility checks require a working dataframe.")

        missing_features = [
            feature for feature in expected_features if feature not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(
                "The loaded model expects feature columns that are not present "
                "after preprocessing: "
                + ", ".join(missing_features)
            )

        context.feature_columns = expected_features
        if numeric_features:
            context.numeric_features = [
                feature for feature in numeric_features if feature in expected_features
            ]
        if categorical_features:
            context.categorical_features = [
                feature for feature in categorical_features if feature in expected_features
            ]

        extra_features = [
            feature for feature in dataframe.columns if feature in context.feature_columns
        ]
        context.metadata["loaded_model_feature_count"] = len(expected_features)
        context.metadata["loaded_model_features"] = expected_features
        context.metadata["available_loaded_features"] = extra_features

    def _extract_feature_contract(self, model_adapter) -> tuple[list[str], list[str], list[str]]:
        raw_features = self._normalize_feature_list(
            getattr(model_adapter, "raw_feature_names_", None)
        )
        raw_numeric_features = self._normalize_feature_list(
            getattr(model_adapter, "raw_numeric_features_", None)
        )
        raw_categorical_features = self._normalize_feature_list(
            getattr(model_adapter, "raw_categorical_features_", None)
        )
        if raw_features:
            return raw_features, raw_numeric_features, raw_categorical_features

        preprocessor = getattr(model_adapter, "preprocessor", None)
        transformers = getattr(preprocessor, "transformers_", None)
        if not transformers:
            return [], [], []

        inferred_features: list[str] = []
        inferred_numeric: list[str] = []
        inferred_categorical: list[str] = []
        for transformer_name, _, columns in transformers:
            if transformer_name == "remainder":
                continue
            if isinstance(columns, str) and columns in {"drop", "passthrough"}:
                continue
            normalized_columns = self._normalize_feature_list(columns)
            for feature_name in normalized_columns:
                if feature_name not in inferred_features:
                    inferred_features.append(feature_name)
            if transformer_name == "numeric":
                inferred_numeric.extend(
                    feature_name
                    for feature_name in normalized_columns
                    if feature_name not in inferred_numeric
                )
            elif transformer_name == "categorical":
                inferred_categorical.extend(
                    feature_name
                    for feature_name in normalized_columns
                    if feature_name not in inferred_categorical
                )

        return inferred_features, inferred_numeric, inferred_categorical

    def _normalize_feature_list(self, values: Iterable[str] | str | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, pd.Index):
            values = values.tolist()
        return [str(value) for value in values]
