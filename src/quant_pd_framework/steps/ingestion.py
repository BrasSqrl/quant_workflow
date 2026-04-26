"""Reads the starting dataframe from memory, CSV, Excel, or Parquet."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..base import BasePipelineStep
from ..config import ArtifactConfig, ColumnRole, PerformanceConfig
from ..context import PipelineContext
from ..export_layout import build_export_path_layout
from ..large_data import (
    DatasetHandle,
    build_memory_estimate_table,
    convert_csv_to_parquet,
    optimize_dataframe_dtypes,
    read_dataset_sample,
    read_tabular_path,
    stage_large_data_file,
)

INPUT_SOURCE_METADATA_ATTR = "quant_studio_input_source"


class IngestionStep(BasePipelineStep):
    """
    Pulls the source data into pandas so every downstream step sees a dataframe.

    This is the intake layer of the quant pipeline. It hides file-format details
    from the rest of the framework and normalizes the starting point.
    """

    name = "ingestion"

    def run(self, context: PipelineContext) -> PipelineContext:
        raw_input = context.raw_input
        if not hasattr(context, "diagnostics_tables"):
            context.diagnostics_tables = {}

        if isinstance(raw_input, DatasetHandle):
            dataframe = self._read_large_data_handle(context, raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            dataframe = raw_input.copy(deep=True)
            context.metadata["input_type"] = "dataframe"
            source_metadata = raw_input.attrs.get(INPUT_SOURCE_METADATA_ATTR)
            if isinstance(source_metadata, dict):
                context.metadata["input_source"] = dict(source_metadata)
        else:
            input_path = Path(raw_input)
            context.metadata["input_type"] = input_path.suffix.lower()
            context.metadata["input_source"] = self._describe_file_input(input_path)
            dataframe = self._read_file(context, input_path)

        if dataframe.empty:
            raise ValueError("The input dataframe is empty, so the pipeline cannot continue.")

        dataframe = self._apply_large_data_controls(context, dataframe)
        context.raw_data = dataframe
        context.working_data = dataframe.copy(deep=True)
        context.metadata["input_shape"] = {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
        }
        self._record_memory_estimate(context, dataframe)
        return context

    def _read_large_data_handle(
        self,
        context: PipelineContext,
        handle: DatasetHandle,
    ) -> pd.DataFrame:
        performance = self._performance_config(context)
        artifacts = self._artifact_config(context)
        active_handle = handle
        if performance.large_data_auto_stage_parquet or performance.convert_csv_to_parquet:
            active_handle = stage_large_data_file(
                handle,
                chunk_rows=performance.csv_conversion_chunk_rows,
                compression=artifacts.parquet_compression,
            )
        context.large_data_handle = active_handle
        context.metadata["input_type"] = "large_data_file"
        context.metadata["input_source"] = dict(handle.metadata)
        context.metadata["large_data_handle"] = active_handle.to_metadata()
        if active_handle.staging_metadata:
            context.metadata["csv_to_parquet_conversion"] = active_handle.staging_metadata
            context.diagnostics_tables["csv_to_parquet_conversion"] = pd.DataFrame(
                [active_handle.staging_metadata]
            )

        projected_columns = self._projected_source_columns(context, active_handle)
        sample_rows = int(performance.large_data_training_sample_rows)
        dataframe = read_dataset_sample(
            active_handle,
            rows=sample_rows,
            columns=projected_columns,
            random_state=context.config.split.random_state,
        )
        context.metadata["large_data_sample"] = {
            "sample_rows_requested": sample_rows,
            "sample_rows_loaded": int(len(dataframe)),
            "projected_columns": projected_columns or list(dataframe.columns),
            "sample_strategy": "duckdb_reservoir_or_head_fallback",
        }
        self._write_sample_development_artifacts(context, dataframe)
        return dataframe

    def _describe_file_input(self, path: Path) -> dict[str, str | int]:
        if not path.exists():
            return {
                "source_kind": "file_path",
                "display_label": str(path),
                "file_name": path.name,
                "relative_path": str(path),
                "suffix": path.suffix.lower(),
                "size_bytes": "",
                "modified_at_utc": "",
            }

        stat_result = path.stat()
        return {
            "source_kind": "file_path",
            "display_label": str(path),
            "file_name": path.name,
            "relative_path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": int(stat_result.st_size),
            "modified_at_utc": datetime.fromtimestamp(
                stat_result.st_mtime,
                tz=UTC,
            ).isoformat(),
        }

    def _read_file(self, context: PipelineContext, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        read_path = path
        performance = self._performance_config(context)
        artifacts = self._artifact_config(context)
        if path.suffix.lower() == ".csv" and performance.convert_csv_to_parquet:
            output_root = artifacts.output_root / getattr(context, "run_id", "standalone_ingestion")
            converted_dir = (
                build_export_path_layout(artifacts, output_root).data_input_dir / "converted_inputs"
            )
            converted_path = converted_dir / f"{path.stem}.parquet"
            conversion_metadata = convert_csv_to_parquet(
                path,
                converted_path,
                chunk_rows=performance.csv_conversion_chunk_rows,
                compression=artifacts.parquet_compression,
            )
            context.metadata["csv_to_parquet_conversion"] = conversion_metadata
            context.diagnostics_tables["csv_to_parquet_conversion"] = pd.DataFrame(
                [conversion_metadata]
            )
            read_path = converted_path

        return read_tabular_path(read_path)

    def _projected_source_columns(
        self,
        context: PipelineContext,
        handle: DatasetHandle,
    ) -> list[str] | None:
        if not context.config.performance.large_data_project_columns:
            return None
        preview_columns = set(
            read_dataset_sample(
                handle,
                rows=1,
                columns=None,
                random_state=context.config.split.random_state,
            ).columns
        )
        projected: list[str] = []
        for candidate in [
            context.config.target.source_column,
            context.config.split.date_column,
            context.config.split.entity_column,
        ]:
            if candidate and candidate in preview_columns and candidate not in projected:
                projected.append(candidate)
        configured_raw_columns = {
            candidate
            for spec in context.config.schema.column_specs
            for candidate in {spec.source_name or spec.name, spec.name}
        }
        for spec in context.config.schema.column_specs:
            if not spec.enabled or spec.role == ColumnRole.IGNORE:
                continue
            for candidate in [spec.source_name or spec.name, spec.name]:
                if candidate in preview_columns and candidate not in projected:
                    projected.append(candidate)
        if context.config.schema.pass_through_unconfigured_columns:
            for column_name in preview_columns:
                if column_name not in projected and column_name not in configured_raw_columns:
                    projected.append(column_name)
        return projected or None

    def _write_sample_development_artifacts(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
    ) -> None:
        output_root = context.config.artifacts.output_root / context.run_id
        sample_dir = (
            build_export_path_layout(
                context.config.artifacts,
                output_root,
            ).data_dir
            / "sample_development"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "training_sample.parquet"
        try:
            dataframe.to_parquet(
                sample_path,
                index=False,
                compression=context.config.artifacts.parquet_compression,
            )
            context.artifacts["sample_development_dir"] = sample_dir
            context.artifacts["large_data_training_sample"] = sample_path
        except Exception as exc:
            warn = getattr(context, "warn", None)
            if callable(warn):
                warn(f"Could not write large-data training sample artifact: {exc}")

    def _apply_large_data_controls(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        performance = self._performance_config(context)
        if not (performance.large_data_mode or performance.optimize_dtypes):
            return dataframe

        optimized, audit_table = optimize_dataframe_dtypes(dataframe, performance)
        if not audit_table.empty:
            context.diagnostics_tables["dtype_optimization"] = audit_table
            before_bytes = int(audit_table["old_memory_bytes"].sum())
            after_bytes = int(audit_table["new_memory_bytes"].sum())
            context.metadata["dtype_optimization"] = {
                "columns_changed": int(len(audit_table)),
                "memory_saved_bytes": int(audit_table["memory_saved_bytes"].sum()),
                "changed_column_memory_before_bytes": before_bytes,
                "changed_column_memory_after_bytes": after_bytes,
            }
        return optimized

    def _record_memory_estimate(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
    ) -> None:
        source_metadata = context.metadata.get("input_source", {})
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        estimate = build_memory_estimate_table(
            dataframe,
            source_metadata,
            self._performance_config(context),
        )
        context.diagnostics_tables["large_data_memory_estimate"] = estimate
        row = estimate.iloc[0].to_dict()
        context.metadata["large_data_memory_estimate"] = row
        if row.get("status") == "warn":
            warning = (
                "Estimated peak memory for this workflow exceeds the configured "
                "large-data memory limit. Consider Parquet input, sampled exports, "
                "or a larger instance."
            )
            warn = getattr(context, "warn", None)
            if callable(warn):
                warn(warning)

    def _performance_config(self, context: PipelineContext) -> PerformanceConfig:
        config = getattr(context, "config", None)
        return getattr(config, "performance", PerformanceConfig())

    def _artifact_config(self, context: PipelineContext) -> ArtifactConfig:
        config = getattr(context, "config", None)
        return getattr(config, "artifacts", ArtifactConfig())
