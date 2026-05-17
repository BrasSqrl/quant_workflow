"""Reads the starting dataframe from memory, CSV, Excel, or Parquet."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..base import BasePipelineStep
from ..config import ArtifactConfig, ColumnRole, PerformanceConfig
from ..context import (
    PipelineContext,
    PipelineMetadataKey,
    get_pipeline_metadata_dict,
    set_pipeline_metadata,
)
from ..export_layout import build_export_path_layout
from ..large_data import (
    DatasetHandle,
    build_memory_estimate_table,
    convert_csv_to_parquet,
    materialize_projected_parquet,
    optimize_dataframe_dtypes,
    profile_dataset_handle_cached,
    read_dataset_sample,
    read_tabular_path,
    stage_large_data_file,
)
from ..large_data_enterprise import (
    record_large_data_execution_plan,
    record_large_data_transformation_contract,
)
from ..large_data_policy import (
    build_large_data_override_audit,
    resolve_large_data_certification,
)
from ..large_data_runtime import DatasetProfile, PreparedDatasetManifest

INPUT_SOURCE_METADATA_ATTR = "quant_studio_input_source"
Meta = PipelineMetadataKey


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
            dataframe = raw_input.copy(deep=False)
            set_pipeline_metadata(context, Meta.INPUT_TYPE, "dataframe")
            source_metadata = raw_input.attrs.get(INPUT_SOURCE_METADATA_ATTR)
            if isinstance(source_metadata, dict):
                set_pipeline_metadata(context, Meta.INPUT_SOURCE, dict(source_metadata))
        else:
            input_path = Path(raw_input)
            set_pipeline_metadata(context, Meta.INPUT_TYPE, input_path.suffix.lower())
            set_pipeline_metadata(context, Meta.INPUT_SOURCE, self._describe_file_input(input_path))
            dataframe = self._read_file(context, input_path)

        if dataframe.empty:
            raise ValueError("The input dataframe is empty, so the pipeline cannot continue.")

        dataframe = self._apply_large_data_controls(context, dataframe)
        context.raw_data = dataframe
        context.working_data = dataframe
        context.raw_input = None
        set_pipeline_metadata(
            context,
            Meta.INPUT_SHAPE,
            {
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
            },
        )
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
                s3_cache_dir=Path(performance.s3_local_cache_dir),
            )
        context.large_data_handle = active_handle
        set_pipeline_metadata(context, Meta.INPUT_TYPE, "large_data_file")
        set_pipeline_metadata(context, Meta.INPUT_SOURCE, dict(handle.metadata))
        set_pipeline_metadata(context, Meta.LARGE_DATA_HANDLE, active_handle.to_metadata())
        self._record_large_data_profile(context, active_handle)
        record_large_data_transformation_contract(context)
        certification = self._record_large_data_certification(context, active_handle)
        record_large_data_execution_plan(
            context,
            certification,
            source_identifier=active_handle.source_identifier,
        )
        if active_handle.staging_metadata:
            set_pipeline_metadata(
                context,
                Meta.CSV_TO_PARQUET_CONVERSION,
                active_handle.staging_metadata,
            )
            context.diagnostics_tables["csv_to_parquet_conversion"] = pd.DataFrame(
                [active_handle.staging_metadata]
            )

        projected_columns = self._projected_source_columns(context, active_handle)
        active_handle = self._materialize_projected_dataset(
            context,
            active_handle,
            projected_columns,
        )
        context.large_data_handle = active_handle
        record_large_data_execution_plan(
            context,
            certification,
            source_identifier=active_handle.source_identifier,
        )
        sample_rows = int(performance.large_data_training_sample_rows)
        dataframe = read_dataset_sample(
            active_handle,
            rows=sample_rows,
            columns=projected_columns,
            random_state=context.config.split.random_state,
        )
        set_pipeline_metadata(
            context,
            Meta.LARGE_DATA_SAMPLE,
            {
                "sample_rows_requested": sample_rows,
                "sample_rows_loaded": int(len(dataframe)),
                "projected_columns": projected_columns or list(dataframe.columns),
                "sample_strategy": "duckdb_reservoir_or_head_fallback",
            },
        )
        sample_path = self._write_sample_development_artifacts(context, dataframe)
        self._record_prepared_dataset_manifest(
            context=context,
            active_handle=active_handle,
            sample_path=sample_path,
            projected_columns=projected_columns or list(dataframe.columns),
        )
        return dataframe

    def _record_large_data_profile(
        self,
        context: PipelineContext,
        handle: DatasetHandle,
    ) -> None:
        output_root = context.config.artifacts.output_root / context.run_id
        profile_dir = build_export_path_layout(
            context.config.artifacts,
            output_root,
        ).metadata_dir / "large_data"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = profile_dataset_handle_cached(
            handle,
            preview_rows=context.config.performance.ui_preview_rows,
            cache_enabled=context.config.performance.large_data_profile_cache_enabled,
        )
        profile_path = profile_dir / "dataset_profile.json"
        profile_path.write_text(
            json.dumps(profile, indent=2, default=str),
            encoding="utf-8",
        )
        set_pipeline_metadata(context, Meta.LARGE_DATA_PROFILE, profile)
        set_pipeline_metadata(context, Meta.LARGE_DATA_PROFILE_PATH, str(profile_path))
        context.artifacts["large_data_profile"] = profile_path
        context.artifacts["large_data_metadata_dir"] = profile_dir
        context.diagnostics_tables["large_data_source_profile"] = pd.DataFrame(
            [
                {
                    "source_kind": profile.get("source_kind"),
                    "source_suffix": profile.get("source_suffix"),
                    "active_suffix": profile.get("active_suffix"),
                    "row_count": profile.get("row_count"),
                    "column_count": profile.get("column_count"),
                    "preview_rows": profile.get("preview_rows"),
                    "profile_cache_key": profile.get("profile_cache_key"),
                    "profile_cache_hit": profile.get("profile_cache_hit"),
                }
            ]
        )
        set_pipeline_metadata(
            context,
            Meta.LARGE_DATA_DATASET_PROFILE,
            DatasetProfile.from_profile_dict(profile).to_dict(),
        )

    def _record_large_data_certification(
        self,
        context: PipelineContext,
        handle: DatasetHandle,
    ):
        certification = resolve_large_data_certification(
            context.config.model.model_type,
            context.config.performance,
        )
        certification_metadata = certification.to_metadata()
        certification_metadata["source_identifier"] = handle.source_identifier
        set_pipeline_metadata(context, Meta.LARGE_DATA_MODEL_CERTIFICATION, certification_metadata)
        context.diagnostics_tables["large_data_model_certification"] = pd.DataFrame(
            [certification_metadata]
        )
        if certification.status.value == "blocked":
            raise ValueError(
                "The selected model is blocked for this Large Data Mode policy. "
                f"{certification.recommendation}"
            )
        if certification.status.value == "sample_fit_full_score":
            context.warn(
                "Large Data Mode will fit this model on the governed sample and score "
                "the full file in chunks because the selected model is not certified "
                "for optimized full-data fitting."
            )
        if certification.status.value == "experimental_full_data_override":
            source_metadata = dict(handle.metadata)
            audit = build_large_data_override_audit(
                certification,
                source_metadata=source_metadata,
            )
            set_pipeline_metadata(context, Meta.LARGE_DATA_OVERRIDE_AUDIT, audit)
            context.diagnostics_tables["large_data_override_audit"] = pd.DataFrame([audit])
            context.warn(
                "A forced large-data model override is active. This path is audited "
                "and is not treated as certified."
            )
        return certification

    def _materialize_projected_dataset(
        self,
        context: PipelineContext,
        handle: DatasetHandle,
        projected_columns: list[str] | None,
    ) -> DatasetHandle:
        performance = self._performance_config(context)
        if not (
            performance.large_data_mode
            and performance.large_data_project_columns
            and projected_columns
        ):
            return handle
        output_root = context.config.artifacts.output_root / context.run_id
        layout = build_export_path_layout(context.config.artifacts, output_root)
        destination_path = layout.data_dir / "prepared_large_data" / "projected_dataset.parquet"
        projected_handle, metadata = materialize_projected_parquet(
            handle,
            columns=projected_columns,
            destination_path=destination_path,
            compression=context.config.artifacts.parquet_compression,
            duckdb_threads=performance.duckdb_threads,
            duckdb_memory_limit_gb=performance.duckdb_memory_limit_gb,
        )
        set_pipeline_metadata(context, Meta.LARGE_DATA_PROJECTED_DATASET, metadata)
        context.diagnostics_tables["large_data_projected_dataset"] = pd.DataFrame([metadata])
        if metadata.get("materialized"):
            context.artifacts["large_data_projected_dataset"] = destination_path
            set_pipeline_metadata(context, Meta.LARGE_DATA_HANDLE, projected_handle.to_metadata())
        return projected_handle

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
            set_pipeline_metadata(context, Meta.CSV_TO_PARQUET_CONVERSION, conversion_metadata)
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
    ) -> Path | None:
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
            return sample_path
        except Exception as exc:
            warn = getattr(context, "warn", None)
            if callable(warn):
                warn(f"Could not write large-data training sample artifact: {exc}")
            return None

    def _record_prepared_dataset_manifest(
        self,
        *,
        context: PipelineContext,
        active_handle: DatasetHandle,
        sample_path: Path | None,
        projected_columns: list[str],
    ) -> None:
        output_root = context.config.artifacts.output_root / context.run_id
        layout = build_export_path_layout(context.config.artifacts, output_root)
        manifest_dir = layout.metadata_dir / "large_data"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        prepared_manifest = PreparedDatasetManifest(
            run_id=context.run_id,
            source_identifier=active_handle.source_identifier,
            staged_path=str(active_handle.active_path),
            sample_path=str(sample_path) if sample_path is not None else "",
            projected_columns=list(projected_columns),
            transformation_contract_keys=[
                spec.output_feature
                for spec in context.config.transformations.transformations
                if spec.enabled
            ],
            target_column=context.config.target.output_column,
            row_count=get_pipeline_metadata_dict(context, Meta.LARGE_DATA_PROFILE).get(
                "row_count"
            ),
            cache_key=str(
                get_pipeline_metadata_dict(context, Meta.LARGE_DATA_DATASET_PROFILE).get(
                    "cache_key",
                    "",
                )
            ),
            profile_cache_key=str(
                get_pipeline_metadata_dict(context, Meta.LARGE_DATA_PROFILE).get(
                    "profile_cache_key",
                    "",
                )
            ),
            partition_columns=list(
                get_pipeline_metadata_dict(context, Meta.PARTITIONED_DATASET_MANIFEST).get(
                    "partition_columns",
                    [],
                )
            ),
            partition_paths=dict(
                get_pipeline_metadata_dict(context, Meta.PARTITIONED_DATASET_MANIFEST).get(
                    "partition_paths",
                    {},
                )
            ),
            artifact_size_estimates={
                "staged_path_bytes": self._path_size(active_handle.active_path),
                "sample_path_bytes": self._path_size(sample_path),
            },
        )
        manifest_path = manifest_dir / "prepared_dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(prepared_manifest.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        set_pipeline_metadata(context, Meta.PREPARED_DATASET_MANIFEST, prepared_manifest.to_dict())
        context.artifacts["prepared_dataset_manifest"] = manifest_path
        context.diagnostics_tables["prepared_dataset_manifest"] = pd.DataFrame(
            [prepared_manifest.to_dict()]
        )

    def _path_size(self, path: Path | None) -> int | None:
        if path is None:
            return None
        try:
            return int(path.stat().st_size)
        except OSError:
            return None

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
            set_pipeline_metadata(
                context,
                Meta.DTYPE_OPTIMIZATION,
                {
                    "columns_changed": int(len(audit_table)),
                    "memory_saved_bytes": int(audit_table["memory_saved_bytes"].sum()),
                    "changed_column_memory_before_bytes": before_bytes,
                    "changed_column_memory_after_bytes": after_bytes,
                },
            )
        return optimized

    def _record_memory_estimate(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
    ) -> None:
        source_metadata = get_pipeline_metadata_dict(context, Meta.INPUT_SOURCE)
        estimate = build_memory_estimate_table(
            dataframe,
            source_metadata,
            self._performance_config(context),
        )
        context.diagnostics_tables["large_data_memory_estimate"] = estimate
        row = estimate.iloc[0].to_dict()
        set_pipeline_metadata(context, Meta.LARGE_DATA_MEMORY_ESTIMATE, row)
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
