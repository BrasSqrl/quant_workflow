"""Shared state container passed between pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FrameworkConfig


class PipelineMetadataKey(StrEnum):
    """Known cross-step metadata keys stored on a pipeline context."""

    INPUT_TYPE = "input_type"
    INPUT_SOURCE = "input_source"
    INPUT_SHAPE = "input_shape"
    CSV_TO_PARQUET_CONVERSION = "csv_to_parquet_conversion"
    DTYPE_OPTIMIZATION = "dtype_optimization"
    LARGE_DATA_HANDLE = "large_data_handle"
    LARGE_DATA_SAMPLE = "large_data_sample"
    LARGE_DATA_PROFILE = "large_data_profile"
    LARGE_DATA_PROFILE_PATH = "large_data_profile_path"
    LARGE_DATA_DATASET_PROFILE = "large_data_dataset_profile"
    LARGE_DATA_MODEL_CERTIFICATION = "large_data_model_certification"
    LARGE_DATA_OVERRIDE_AUDIT = "large_data_override_audit"
    LARGE_DATA_PROJECTED_DATASET = "large_data_projected_dataset"
    LARGE_DATA_TRANSFORMATION_CONTRACT = "large_data_transformation_contract"
    LARGE_DATA_EXECUTION_PLAN = "large_data_execution_plan"
    LARGE_DATA_FEATURE_SCREENING_MANIFEST = "large_data_feature_screening_manifest"
    LARGE_DATA_FEATURE_SCREENING_RECORDED = "large_data_feature_screening_recorded"
    LARGE_DATA_PRESCREEN_EXCLUDED_FEATURES = "large_data_prescreen_excluded_features"
    LARGE_DATA_FIT_RECORD = "large_data_fit_record"
    LARGE_DATA_MEMORY_ESTIMATE = "large_data_memory_estimate"
    LARGE_DATA_FULL_SCORING = "large_data_full_scoring"
    PARTITIONED_DATASET_MANIFEST = "partitioned_dataset_manifest"
    PREPARED_DATASET_MANIFEST = "prepared_dataset_manifest"


PIPELINE_METADATA_SCHEMA: dict[PipelineMetadataKey, type | tuple[type, ...]] = {
    PipelineMetadataKey.INPUT_TYPE: str,
    PipelineMetadataKey.INPUT_SOURCE: dict,
    PipelineMetadataKey.INPUT_SHAPE: dict,
    PipelineMetadataKey.CSV_TO_PARQUET_CONVERSION: dict,
    PipelineMetadataKey.DTYPE_OPTIMIZATION: dict,
    PipelineMetadataKey.LARGE_DATA_HANDLE: dict,
    PipelineMetadataKey.LARGE_DATA_SAMPLE: dict,
    PipelineMetadataKey.LARGE_DATA_PROFILE: dict,
    PipelineMetadataKey.LARGE_DATA_PROFILE_PATH: str,
    PipelineMetadataKey.LARGE_DATA_DATASET_PROFILE: dict,
    PipelineMetadataKey.LARGE_DATA_MODEL_CERTIFICATION: dict,
    PipelineMetadataKey.LARGE_DATA_OVERRIDE_AUDIT: dict,
    PipelineMetadataKey.LARGE_DATA_PROJECTED_DATASET: dict,
    PipelineMetadataKey.LARGE_DATA_TRANSFORMATION_CONTRACT: dict,
    PipelineMetadataKey.LARGE_DATA_EXECUTION_PLAN: dict,
    PipelineMetadataKey.LARGE_DATA_FEATURE_SCREENING_MANIFEST: dict,
    PipelineMetadataKey.LARGE_DATA_FEATURE_SCREENING_RECORDED: bool,
    PipelineMetadataKey.LARGE_DATA_PRESCREEN_EXCLUDED_FEATURES: list,
    PipelineMetadataKey.LARGE_DATA_FIT_RECORD: dict,
    PipelineMetadataKey.LARGE_DATA_MEMORY_ESTIMATE: dict,
    PipelineMetadataKey.LARGE_DATA_FULL_SCORING: dict,
    PipelineMetadataKey.PARTITIONED_DATASET_MANIFEST: dict,
    PipelineMetadataKey.PREPARED_DATASET_MANIFEST: dict,
}


def set_pipeline_metadata(context: Any, key: PipelineMetadataKey | str, value: Any) -> None:
    """Stores schema-validated metadata on PipelineContext-compatible objects."""

    metadata_key = _known_metadata_key(key)
    if metadata_key is not None:
        _validate_metadata_value(metadata_key, value)
        key_value = metadata_key.value
    else:
        key_value = str(key)

    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        context.metadata = metadata
    metadata[key_value] = value


def get_pipeline_metadata(
    context: Any,
    key: PipelineMetadataKey | str,
    default: Any = None,
) -> Any:
    """Returns metadata from PipelineContext-compatible objects."""

    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return default
    metadata_key = _known_metadata_key(key)
    return metadata.get(metadata_key.value if metadata_key else str(key), default)


def get_pipeline_metadata_dict(context: Any, key: PipelineMetadataKey | str) -> dict[str, Any]:
    """Returns a metadata mapping from PipelineContext-compatible objects."""

    value = get_pipeline_metadata(context, key, {})
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ModelArtifactState:
    """Small grouped view of fitted model outputs on a pipeline context."""

    model: Any
    model_summary: str | pd.DataFrame | None
    feature_importance: pd.DataFrame | None
    model_artifacts: dict[str, Any]


@dataclass(frozen=True)
class DiagnosticResultState:
    """Small grouped view of diagnostics, metrics, and generated visuals."""

    predictions: dict[str, pd.DataFrame]
    metrics: dict[str, dict[str, float | int | None]]
    backtest_summary: pd.DataFrame | None
    comparison_results: pd.DataFrame | None
    scenario_results: dict[str, pd.DataFrame]
    diagnostics_tables: dict[str, pd.DataFrame]
    statistical_tests: dict[str, Any]
    visualizations: dict[str, Any]


@dataclass(frozen=True)
class LargeDataState:
    """Small grouped view of large-data source and staging state."""

    large_data_handle: Any
    raw_input: Any
    raw_data: pd.DataFrame | None
    working_data: pd.DataFrame | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExecutionArtifactState:
    """Small grouped view of exported paths and run diagnostics."""

    artifacts: dict[str, Path]
    warnings: list[str]
    events: list[str]
    debug_trace: list[dict[str, Any]]


@dataclass
class PipelineContext:
    """Holds the evolving state of a full model run."""

    config: FrameworkConfig
    run_id: str
    raw_input: Any
    large_data_handle: Any = None
    raw_data: pd.DataFrame | None = None
    working_data: pd.DataFrame | None = None
    target_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    split_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    model: Any = None
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None
    backtest_summary: pd.DataFrame | None = None
    model_summary: str | pd.DataFrame | None = None
    model_artifacts: dict[str, Any] = field(default_factory=dict)
    comparison_results: pd.DataFrame | None = None
    scenario_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    visualizations: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    debug_trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Adds a short breadcrumb that can be written into the final report."""

        self.events.append(message)

    def warn(self, message: str) -> None:
        """Stores non-fatal issues that the user should still review."""

        self.warnings.append(message)

    def add_debug_record(self, record: dict[str, Any]) -> None:
        """Stores structured timing and sizing data for performance debugging."""

        self.debug_trace.append(record)

    def set_metadata(self, key: PipelineMetadataKey | str, value: Any) -> None:
        """Stores metadata with validation for keys that have a declared schema."""

        set_pipeline_metadata(self, key, value)

    def get_metadata(self, key: PipelineMetadataKey | str, default: Any = None) -> Any:
        """Returns metadata by typed key while preserving dict-like compatibility."""

        return get_pipeline_metadata(self, key, default)

    def get_metadata_dict(self, key: PipelineMetadataKey | str) -> dict[str, Any]:
        """Returns a metadata mapping, or an empty mapping when the key is unset."""

        return get_pipeline_metadata_dict(self, key)

    def _known_metadata_key(self, key: PipelineMetadataKey | str) -> PipelineMetadataKey | None:
        return _known_metadata_key(key)

    @property
    def model_artifact_state(self) -> ModelArtifactState:
        """Returns a grouped view for newer code without changing stored fields."""

        return ModelArtifactState(
            model=self.model,
            model_summary=self.model_summary,
            feature_importance=self.feature_importance,
            model_artifacts=self.model_artifacts,
        )

    @property
    def diagnostic_result_state(self) -> DiagnosticResultState:
        """Returns a grouped diagnostics view for incremental context refactors."""

        return DiagnosticResultState(
            predictions=self.predictions,
            metrics=self.metrics,
            backtest_summary=self.backtest_summary,
            comparison_results=self.comparison_results,
            scenario_results=self.scenario_results,
            diagnostics_tables=self.diagnostics_tables,
            statistical_tests=self.statistical_tests,
            visualizations=self.visualizations,
        )

    @property
    def large_data_state(self) -> LargeDataState:
        """Returns a grouped large-data view for incremental context refactors."""

        return LargeDataState(
            large_data_handle=self.large_data_handle,
            raw_input=self.raw_input,
            raw_data=self.raw_data,
            working_data=self.working_data,
            metadata=self.metadata,
        )

    @property
    def execution_artifact_state(self) -> ExecutionArtifactState:
        """Returns a grouped export/debug view for incremental context refactors."""

        return ExecutionArtifactState(
            artifacts=self.artifacts,
            warnings=self.warnings,
            events=self.events,
            debug_trace=self.debug_trace,
        )


def _validate_metadata_value(metadata_key: PipelineMetadataKey, value: Any) -> None:
    expected_type = PIPELINE_METADATA_SCHEMA.get(metadata_key)
    if expected_type is not None and not isinstance(value, expected_type):
        raise TypeError(
            f"Pipeline metadata '{metadata_key.value}' must be "
            f"{_metadata_type_name(expected_type)}; received "
            f"{type(value).__name__}."
        )


def _known_metadata_key(key: PipelineMetadataKey | str) -> PipelineMetadataKey | None:
    if isinstance(key, PipelineMetadataKey):
        return key
    try:
        return PipelineMetadataKey(str(key))
    except ValueError:
        return None


def _metadata_type_name(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(sorted(type_.__name__ for type_ in expected_type))
    return expected_type.__name__
