"""Fits the selected model family or loads an existing fitted artifact."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..base import BasePipelineStep
from ..config import ExecutionMode, ModelType
from ..context import PipelineContext, PipelineMetadataKey
from ..large_data_policy import resolve_large_data_certification
from ..models import build_model_adapter
from ..safe_serialization import load_joblib_verified
from ..segmented_model import SegmentedModelBundle, build_segment_key_series

Meta = PipelineMetadataKey


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
        if context.config.segmented_model.enabled:
            return self._fit_segmented_model(context)

        model_adapter = self._fit_single_model(
            context,
            train_frame=context.split_frames["train"],
            numeric_features=context.numeric_features,
            categorical_features=context.categorical_features,
        )

        context.model = model_adapter
        context.model_summary = model_adapter.summary_text
        context.metadata["model_reused"] = False
        self._record_large_data_fit_metadata(context, context.split_frames["train"])
        self._publish_model_numerical_findings(context, model_adapter, loaded_model=False)
        return context

    def _fit_single_model(
        self,
        context: PipelineContext,
        *,
        train_frame: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
    ):
        x_train = train_frame[context.feature_columns]
        if (
            context.config.model.model_type == ModelType.GEE_LOGISTIC_REGRESSION
            and context.config.model.gee_group_column
            and context.config.model.gee_group_column in train_frame.columns
            and context.config.model.gee_group_column not in x_train.columns
        ):
            x_train = pd.concat(
                [x_train, train_frame[[context.config.model.gee_group_column]]],
                axis=1,
            )
        if (
            context.config.model.model_type == ModelType.MIXED_EFFECTS_REGRESSION
            and context.config.model.mixed_effects_group_column
            and context.config.model.mixed_effects_group_column in train_frame.columns
            and context.config.model.mixed_effects_group_column not in x_train.columns
        ):
            x_train = pd.concat(
                [x_train, train_frame[[context.config.model.mixed_effects_group_column]]],
                axis=1,
            )
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
            numeric_features,
            categorical_features,
        )
        return model_adapter

    def _fit_segmented_model(self, context: PipelineContext) -> PipelineContext:
        train_frame = context.split_frames["train"]
        segment_config = context.config.segmented_model
        missing_segment_columns = [
            column for column in segment_config.segment_columns if column not in train_frame.columns
        ]
        if missing_segment_columns:
            raise ValueError(
                "Segmented modeling requires segment columns in the training data: "
                + ", ".join(missing_segment_columns)
            )

        segment_keys = build_segment_key_series(train_frame, segment_config.segment_columns)
        segment_counts = segment_keys.value_counts(dropna=False)
        if int(len(segment_counts)) > segment_config.max_segments:
            raise ValueError(
                "Segmented modeling resolved "
                f"{len(segment_counts)} segment combinations, which exceeds the configured "
                f"maximum of {segment_config.max_segments}. Reduce segment columns or raise "
                "the max segment count after confirming model governance capacity."
            )

        global_model = self._fit_single_model(
            context,
            train_frame=train_frame,
            numeric_features=context.numeric_features,
            categorical_features=context.categorical_features,
        )

        segment_models: dict[str, object] = {}
        inventory_rows: list[dict[str, object]] = []
        y_train = train_frame[context.target_column]
        for segment_key, row_count in segment_counts.items():
            segment_key_text = str(segment_key)
            mask = segment_keys == segment_key
            segment_frame = train_frame.loc[mask]
            status = "fitted"
            reason = ""
            event_count: int | None = None
            if int(row_count) < segment_config.min_segment_rows:
                status = "fallback_global"
                reason = (
                    f"Segment row count {int(row_count)} is below the configured minimum "
                    f"{segment_config.min_segment_rows}."
                )
            elif context.config.target.mode.value == "binary":
                event_count = int(pd.to_numeric(y_train.loc[mask], errors="coerce").fillna(0).sum())
                non_event_count = int(row_count) - event_count
                if event_count < segment_config.min_segment_events:
                    status = "fallback_global"
                    reason = (
                        f"Segment event count {event_count} is below the configured minimum "
                        f"{segment_config.min_segment_events}."
                    )
                elif non_event_count < segment_config.min_segment_events:
                    status = "fallback_global"
                    reason = (
                        f"Segment non-event count {non_event_count} is below the configured "
                        f"minimum {segment_config.min_segment_events}."
                    )

            if status == "fitted":
                try:
                    segment_models[segment_key_text] = self._fit_single_model(
                        context,
                        train_frame=segment_frame,
                        numeric_features=[
                            feature
                            for feature in context.numeric_features
                            if feature in segment_frame.columns
                        ],
                        categorical_features=[
                            feature
                            for feature in context.categorical_features
                            if feature in segment_frame.columns
                        ],
                    )
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    status = "fallback_global"
                    reason = f"Segment model fit failed: {exc}"

            inventory_rows.append(
                {
                    "segment_key": segment_key_text,
                    "status": status,
                    "model_id": (
                        f"segment::{segment_key_text}" if status == "fitted" else "global"
                    ),
                    "row_count": int(row_count),
                    "event_count": event_count,
                    "fallback_reason": reason,
                }
            )

        segment_inventory = pd.DataFrame(inventory_rows).sort_values(
            ["status", "row_count"],
            ascending=[False, False],
            kind="stable",
        )
        model_adapter = SegmentedModelBundle(
            model_config=context.config.model,
            target_mode=context.config.target.mode,
            segment_columns=segment_config.segment_columns,
            feature_columns=context.feature_columns,
            numeric_features=context.numeric_features,
            categorical_features=context.categorical_features,
            global_model=global_model,
            segment_models=segment_models,
            segment_inventory=segment_inventory,
        )

        context.model = model_adapter
        context.model_summary = model_adapter.summary_text
        context.metadata["model_reused"] = False
        context.metadata["segmented_model"] = {
            "enabled": True,
            "segment_columns": list(segment_config.segment_columns),
            "segment_count": int(len(segment_inventory)),
            "fitted_segment_count": int((segment_inventory["status"] == "fitted").sum()),
            "fallback_segment_count": int((segment_inventory["status"] != "fitted").sum()),
            "fallback_policy": segment_config.fallback_policy.value,
        }
        context.diagnostics_tables["segment_model_inventory"] = segment_inventory
        self._record_large_data_fit_metadata(context, train_frame)
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

        model_adapter = load_joblib_verified(resolved_path)
        self._validate_loaded_model(context, model_adapter, resolved_path)
        self._apply_loaded_feature_contract(context, model_adapter)

        context.model = model_adapter
        context.model_summary = getattr(
            model_adapter, "summary_text", f"Loaded model from {resolved_path}"
        )
        context.metadata["loaded_model_path"] = str(resolved_path)
        context.metadata["model_reused"] = True
        fit_frame = context.split_frames.get("train")
        if fit_frame is None:
            fit_frame = next(iter(context.split_frames.values()), None)
        if fit_frame is None:
            fit_frame = context.working_data
        if fit_frame is not None:
            self._record_large_data_fit_metadata(context, fit_frame, loaded_model=True)
        self._publish_model_numerical_findings(context, model_adapter, loaded_model=True)
        return context

    def _record_large_data_fit_metadata(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        *,
        loaded_model: bool = False,
    ) -> None:
        performance = context.config.performance
        if not performance.large_data_mode:
            return
        certification = resolve_large_data_certification(
            context.config.model.model_type,
            performance,
        )
        sample_info = context.get_metadata_dict(Meta.LARGE_DATA_SAMPLE)
        payload = {
            **certification.to_metadata(),
            "certified_fit_enabled": bool(performance.large_data_certified_fit_enabled),
            "loaded_model": bool(loaded_model),
            "fit_rows_observed_by_estimator": int(len(train_frame)),
            "fit_feature_count": int(len(context.feature_columns)),
            "sample_rows_requested": sample_info.get("sample_rows_requested"),
            "sample_rows_loaded": sample_info.get("sample_rows_loaded"),
            "fit_basis": (
                "loaded_existing_model"
                if loaded_model
                else (
                    "governed_sample_with_full_file_scoring"
                    if certification.status.value != "full_data_certified"
                    else "certified_large_data_candidate_with_governed_sample_fallback"
                )
            ),
        }
        context.set_metadata(Meta.LARGE_DATA_FIT_RECORD, payload)
        context.diagnostics_tables["large_data_fit_record"] = pd.DataFrame([payload])

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
