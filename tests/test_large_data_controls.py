"""Tests for large-data ingestion, conversion, and tabular export controls."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

import quant_pd_framework.large_data as large_data_module
from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    CreditRiskDiagnosticConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    LargeDataBackend,
    LargeDataExportPolicy,
    LargeDataModelPolicy,
    LargeDataPartitionStrategy,
    LargeDataWorkerMode,
    ModelType,
    PerformanceConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TabularOutputFormat,
    TargetConfig,
    TargetMode,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
    load_framework_config,
)
from quant_pd_framework.background_jobs import (
    BackgroundJobManifest,
    queue_background_workflow,
    read_background_manifest,
    request_background_cancel,
    write_background_manifest,
)
from quant_pd_framework.checkpointing import (
    CHECKPOINT_TABLE_REF_MARKER,
    load_context_checkpoint,
    save_context_checkpoint,
)
from quant_pd_framework.config_serialization import FRAMEWORK_CONFIG_SECTION_NAMES
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.large_data import (
    build_dataset_handle,
    build_s3_dataset_handle,
    convert_csv_to_parquet,
    parse_s3_uri,
    profile_dataset_handle_cached,
    read_dataset_preview,
    read_dataset_sample,
    stage_large_data_file,
)
from quant_pd_framework.large_data_enterprise import (
    record_large_data_feature_screening,
    record_large_data_transformation_contract,
)
from quant_pd_framework.large_data_policy import resolve_large_data_certification
from quant_pd_framework.large_data_runtime import (
    ResultTableRef,
    TableRef,
    count_table_rows,
    query_table_page,
)
from quant_pd_framework.run import _resolve_input_path
from quant_pd_framework.steps.large_data_scoring import _ChunkedParquetWriter
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def _large_data_test_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=build_common_schema("account_id"),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="default_status",
            mode=TargetMode.BINARY,
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.CROSS_SECTIONAL,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        diagnostics=DiagnosticConfig(
            data_quality=True,
            descriptive_statistics=True,
            missingness_analysis=True,
            correlation_analysis=False,
            vif_analysis=False,
            woe_iv_analysis=False,
            psi_analysis=False,
            adf_analysis=False,
            model_specification_tests=False,
            forecasting_statistical_tests=False,
            calibration_analysis=False,
            threshold_analysis=False,
            lift_gain_analysis=False,
            segment_analysis=False,
            residual_analysis=False,
            quantile_analysis=False,
            qq_analysis=False,
            interactive_visualizations=False,
            static_image_exports=False,
            export_excel_workbook=False,
            max_plot_rows=5000,
        ),
        credit_risk=CreditRiskDiagnosticConfig(enabled=False),
        performance=PerformanceConfig(
            large_data_mode=True,
            optimize_dtypes=True,
            diagnostic_sample_rows=2500,
            memory_limit_gb=256.0,
        ),
        artifacts=ArtifactConfig(
            output_root=output_root,
            export_code_snapshot=False,
            export_individual_figure_files=False,
        ),
    )


def test_parquet_input_records_dtype_and_memory_audits() -> None:
    dataframe = build_binary_dataframe(row_count=180)
    dataframe["low_cardinality_text"] = ["alpha" if row % 2 else "beta" for row in range(180)]
    dataframe["large_float"] = dataframe["balance"].astype("float64")

    with temporary_artifact_root("pytest_large_data_parquet") as artifact_root:
        parquet_path = artifact_root / "input.parquet"
        dataframe.to_parquet(parquet_path, index=False)

        config = _large_data_test_config(artifact_root)
        context = QuantModelOrchestrator(config=config).run(parquet_path)

    assert context.metadata["input_type"] == ".parquet"
    assert "dtype_optimization" in context.diagnostics_tables
    assert "large_data_memory_estimate" in context.diagnostics_tables
    assert context.metadata["large_data_memory_estimate"]["status"] == "pass"


def test_non_parquet_source_sampled_exports_are_csv_only() -> None:
    dataframe = build_binary_dataframe(row_count=180)

    with temporary_artifact_root("pytest_large_data_exports") as artifact_root:
        config = _large_data_test_config(artifact_root)
        config.artifacts = ArtifactConfig(
            output_root=artifact_root,
            export_code_snapshot=False,
            export_individual_figure_files=False,
            tabular_output_format=TabularOutputFormat.BOTH,
            large_data_export_policy=LargeDataExportPolicy.SAMPLED,
            large_data_sample_rows=25,
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)

        predictions_csv = Path(context.artifacts["predictions_csv"])
        manifest = json.loads(Path(context.artifacts["manifest"]).read_text(encoding="utf-8"))
        exported_sample = pd.read_csv(predictions_csv)

        assert len(exported_sample) == 25
        assert context.artifacts["predictions"] == predictions_csv
        assert manifest["core_artifacts"]["predictions_csv"] == str(predictions_csv)
        assert manifest["core_artifacts"]["predictions_parquet"] is None
        assert "tabular_export_policy" in context.diagnostics_tables


def test_chunked_csv_to_parquet_conversion_helper() -> None:
    with temporary_artifact_root("pytest_chunked_csv_to_parquet") as artifact_root:
        csv_path = artifact_root / "input.csv"
        parquet_path = artifact_root / "converted" / "input.parquet"
        csv_path.write_text("x,y\n1,a\n2,b\n3,c\n", encoding="utf-8")

        metadata = convert_csv_to_parquet(
            csv_path,
            parquet_path,
            chunk_rows=2,
            compression="snappy",
        )
        converted = pd.read_parquet(parquet_path)

    assert metadata["conversion_engine"] in {"duckdb_copy", "pyarrow_local_stream"}
    assert metadata["row_count"] == 3
    assert converted.to_dict(orient="list") == {"x": [1, 2, 3], "y": ["a", "b", "c"]}


def test_csv_to_parquet_rejects_mislabeled_excel_zip_with_actionable_message() -> None:
    with temporary_artifact_root("pytest_mislabeled_excel_csv") as artifact_root:
        csv_path = artifact_root / "input.csv"
        parquet_path = artifact_root / "converted" / "input.parquet"
        csv_path.write_bytes(b"PK\x03\x04\x14\x00fake-excel-zip,a\n\x00\x01\x02")

        with pytest.raises(ValueError, match="Excel or ZIP"):
            convert_csv_to_parquet(
                csv_path,
                parquet_path,
                chunk_rows=2,
                compression="snappy",
                progress_callback=lambda _event: None,
            )


def test_csv_to_parquet_wraps_inconsistent_columns_with_actionable_message() -> None:
    with temporary_artifact_root("pytest_inconsistent_csv_columns") as artifact_root:
        csv_path = artifact_root / "input.csv"
        parquet_path = artifact_root / "converted" / "input.parquet"
        csv_path.write_text("Report export\nx,y\n1,a\n", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            convert_csv_to_parquet(
                csv_path,
                parquet_path,
                chunk_rows=2,
                compression="snappy",
                progress_callback=lambda _event: None,
            )

    message = str(exc_info.value)
    assert "Could not parse" in message
    assert "one header row" in message
    assert "inconsistent column counts" in message


def test_s3_uri_handle_normalization_does_not_require_secrets() -> None:
    uri = "s3://example-bucket/path/to/input.csv"

    bucket, key = parse_s3_uri(uri)
    handle = build_s3_dataset_handle(uri, {"size_bytes": 123, "etag": "abc"})

    assert bucket == "example-bucket"
    assert key == "path/to/input.csv"
    assert handle.is_s3 is True
    assert handle.uri == uri
    assert handle.source_suffix == ".csv"
    assert handle.metadata["source_kind"] == "s3"
    assert handle.metadata["bucket"] == "example-bucket"
    assert handle.metadata["size_bytes"] == 123


def test_s3_excel_preview_reads_from_remote_object_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeS3Filesystem:
        def __init__(self, path: Path) -> None:
            self.path = path

        def open_input_file(self, _object_path: str):
            return self.path.open("rb")

    with temporary_artifact_root("pytest_s3_excel_preview") as artifact_root:
        excel_path = artifact_root / "input.xlsx"
        pd.DataFrame(
            {
                "balance": [100, 200, 300],
                "default_status": [0, 1, 0],
            }
        ).to_excel(excel_path, index=False)
        monkeypatch.setattr(
            large_data_module,
            "_s3_filesystem_and_path",
            lambda _uri: (FakeS3Filesystem(excel_path), "input.xlsx"),
        )
        handle = build_s3_dataset_handle(
            "s3://example-bucket/path/to/input.xlsx",
            {"size_bytes": excel_path.stat().st_size},
        )

        preview = read_dataset_preview(handle, rows=2)

    assert preview.to_dict(orient="list") == {
        "balance": [100, 200],
        "default_status": [0, 1],
    }


def test_s3_excel_stages_to_local_cache_with_excel_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeS3Filesystem:
        def __init__(self, path: Path) -> None:
            self.path = path

        def open_input_file(self, _object_path: str):
            return self.path.open("rb")

    with temporary_artifact_root("pytest_s3_excel_staging") as artifact_root:
        excel_path = artifact_root / "input.xlsx"
        pd.DataFrame(
            {
                "balance": [100, 200, 300],
                "default_status": [0, 1, 0],
            }
        ).to_excel(excel_path, index=False)
        monkeypatch.setattr(
            large_data_module,
            "_s3_filesystem_and_path",
            lambda _uri: (FakeS3Filesystem(excel_path), "input.xlsx"),
        )
        handle = build_s3_dataset_handle(
            "s3://example-bucket/path/to/input.xlsx",
            {"size_bytes": excel_path.stat().st_size},
        )

        staged = stage_large_data_file(
            handle,
            chunk_rows=2,
            compression="snappy",
            s3_cache_dir=artifact_root / "cache",
        )
        sample = read_dataset_sample(staged, rows=2, columns=None, random_state=42)

        assert staged.active_path.suffix == ".xlsx"
        assert staged.active_path.exists()
        assert staged.staging_metadata["conversion_engine"] == "s3_object_copy"
        assert sample.to_dict(orient="list") == {
            "balance": [100, 200],
            "default_status": [0, 1],
        }


def test_s3_excel_content_is_detected_even_when_key_ends_with_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeS3Filesystem:
        def __init__(self, path: Path) -> None:
            self.path = path

        def open_input_file(self, _object_path: str):
            return self.path.open("rb")

    with temporary_artifact_root("pytest_s3_excel_content_csv_key") as artifact_root:
        excel_path = artifact_root / "input.xlsx"
        pd.DataFrame(
            {
                "balance": [100, 200, 300],
                "default_status": [0, 1, 0],
            }
        ).to_excel(excel_path, index=False)
        monkeypatch.setattr(
            large_data_module,
            "_s3_filesystem_and_path",
            lambda _uri: (FakeS3Filesystem(excel_path), "input.csv"),
        )
        handle = build_s3_dataset_handle(
            "s3://example-bucket/path/to/input.csv",
            {"size_bytes": excel_path.stat().st_size},
        )

        preview = read_dataset_preview(handle, rows=2)
        staged = stage_large_data_file(
            handle,
            chunk_rows=2,
            compression="snappy",
            s3_cache_dir=artifact_root / "cache",
        )

        assert preview.to_dict(orient="list") == {
            "balance": [100, 200],
            "default_status": [0, 1],
        }
        assert staged.active_path.suffix == ".xlsx"
        assert staged.staging_metadata["detected_suffix"] == ".xlsx"
        assert staged.staging_metadata["source_suffix"] == ".csv"


def test_large_data_certification_policy_blocks_uncertified_models() -> None:
    certification = resolve_large_data_certification(
        ModelType.TOBIT_REGRESSION,
        PerformanceConfig(
            large_data_mode=True,
            large_data_model_policy=LargeDataModelPolicy.CERTIFIED_ONLY,
        ),
    )

    assert certification.status.value == "blocked"
    assert certification.execution_strategy == "blocked_uncertified_model"
    assert certification.fit_capability.status.value == "sample_fit_full_score"


def test_large_data_capability_matrix_classifies_common_model_families() -> None:
    logistic = resolve_large_data_certification(
        ModelType.LOGISTIC_REGRESSION,
        PerformanceConfig(large_data_mode=True),
    )
    xgboost = resolve_large_data_certification(
        ModelType.XGBOOST,
        PerformanceConfig(large_data_mode=True),
    )
    random_forest = resolve_large_data_certification(
        ModelType.RANDOM_FOREST,
        PerformanceConfig(large_data_mode=True),
    )

    assert logistic.fit_capability.status.value == "full_data_exact"
    assert xgboost.fit_capability.status.value == "full_data_incremental"
    assert random_forest.fit_capability.status.value == "in_memory_only"
    assert random_forest.status.value == "sample_fit_full_score"


def test_large_data_force_override_requires_confirmation_and_reason() -> None:
    blocked = resolve_large_data_certification(
        ModelType.TOBIT_REGRESSION,
        PerformanceConfig(
            large_data_mode=True,
            large_data_model_policy=LargeDataModelPolicy.FORCE_FULL_DATA_OVERRIDE,
            large_data_override_confirmed=True,
        ),
    )
    allowed = resolve_large_data_certification(
        ModelType.TOBIT_REGRESSION,
        PerformanceConfig(
            large_data_mode=True,
            large_data_model_policy=LargeDataModelPolicy.FORCE_FULL_DATA_OVERRIDE,
            large_data_override_confirmed=True,
            large_data_override_reason="Final validation requires this model on sized compute.",
        ),
    )

    assert blocked.status.value == "blocked"
    assert allowed.status.value == "experimental_full_data_override"


def test_background_job_manifest_records_cancel_requests() -> None:
    with temporary_artifact_root("pytest_background_job_manifest") as artifact_root:
        manifest_path = artifact_root / "job_manifest.json"
        manifest = BackgroundJobManifest(
            job_id="job_1",
            status="running",
            job_dir=str(artifact_root),
            config_path=str(artifact_root / "run_config.json"),
            input_kind="dataset_handle",
            input_identifier=str(artifact_root / "input.parquet"),
        )

        write_background_manifest(manifest_path, manifest)
        request_background_cancel(manifest_path)
        loaded = read_background_manifest(manifest_path)

    assert loaded.cancel_requested is True
    assert loaded.status == "running"
    assert "Cancel requested" in loaded.progress


def test_saved_run_resolves_parquet_snapshot_when_csv_is_absent() -> None:
    with temporary_artifact_root("pytest_parquet_snapshot_resolution") as artifact_root:
        config_path = artifact_root / "run_config.json"
        parquet_path = artifact_root / "input_snapshot.parquet"
        config_path.write_text("{}", encoding="utf-8")
        pd.DataFrame({"x": [1]}).to_parquet(parquet_path, index=False)

        resolved = _resolve_input_path(
            config_path,
            ["input_snapshot.csv", "input_snapshot.parquet"],
            None,
        )

    assert resolved == parquet_path


def test_large_data_config_round_trips_through_loader() -> None:
    config = _large_data_test_config(Path("artifacts"))
    config.artifacts = ArtifactConfig(
        output_root=Path("artifacts"),
        tabular_output_format=TabularOutputFormat.PARQUET,
        large_data_export_policy=LargeDataExportPolicy.SAMPLED,
        large_data_sample_rows=1234,
    )
    config.performance = PerformanceConfig(
        large_data_mode=True,
        optimize_dtypes=True,
        convert_csv_to_parquet=True,
        large_data_backend=LargeDataBackend.DISK_BACKED,
        large_data_model_policy=LargeDataModelPolicy.ALLOW_SAMPLE_FALLBACK,
        large_data_auto_detected=True,
        large_data_user_override_disabled=True,
        large_data_standard_execution_override_reason="Manual full-memory validation",
        large_data_effective_mode="standard_in_memory_forced",
        large_data_source_kind="data_load",
        large_data_partition_strategy=LargeDataPartitionStrategy.SPLIT,
        large_data_worker_mode=LargeDataWorkerMode.WORKER_SERVICE,
        large_data_profile_cache_enabled=False,
        large_data_prescreen_enabled=True,
        large_data_auto_apply_prescreen=True,
        large_data_certified_fit_enabled=False,
        csv_conversion_chunk_rows=25000,
        memory_limit_gb=64.0,
    )

    loaded = load_framework_config(config.to_dict())

    assert loaded.performance.large_data_mode is True
    assert loaded.performance.large_data_backend == LargeDataBackend.DISK_BACKED
    assert loaded.performance.large_data_model_policy == LargeDataModelPolicy.ALLOW_SAMPLE_FALLBACK
    assert loaded.performance.large_data_auto_detected is True
    assert loaded.performance.large_data_user_override_disabled is True
    assert loaded.performance.large_data_standard_execution_override_reason == (
        "Manual full-memory validation"
    )
    assert loaded.performance.large_data_effective_mode == "standard_in_memory_forced"
    assert loaded.performance.large_data_source_kind == "data_load"
    assert loaded.performance.large_data_partition_strategy == LargeDataPartitionStrategy.SPLIT
    assert loaded.performance.large_data_worker_mode == LargeDataWorkerMode.WORKER_SERVICE
    assert loaded.performance.large_data_profile_cache_enabled is False
    assert loaded.performance.large_data_prescreen_enabled is True
    assert loaded.performance.large_data_auto_apply_prescreen is True
    assert loaded.performance.large_data_certified_fit_enabled is False
    assert loaded.performance.convert_csv_to_parquet is True
    assert loaded.performance.csv_conversion_chunk_rows == 25000
    assert loaded.performance.large_data_training_sample_rows == 250000
    assert loaded.performance.large_data_score_chunk_rows == 100000
    assert loaded.performance.large_data_result_page_rows == 1000
    assert loaded.performance.large_data_max_in_memory_rows == 250000
    assert loaded.performance.duckdb_threads == 0
    assert loaded.performance.duckdb_memory_limit_gb is None
    assert loaded.performance.memory_limit_gb == 64.0
    assert loaded.artifacts.tabular_output_format == TabularOutputFormat.PARQUET
    assert loaded.artifacts.large_data_export_policy == LargeDataExportPolicy.SAMPLED
    assert loaded.artifacts.large_data_sample_rows == 1234
    assert set(FRAMEWORK_CONFIG_SECTION_NAMES).issubset(config.to_dict())


def test_file_backed_large_data_run_trains_on_sample_and_scores_full_file() -> None:
    dataframe = build_binary_dataframe(row_count=180)

    with temporary_artifact_root("pytest_file_backed_large_data_run") as artifact_root:
        parquet_path = artifact_root / "large_input.parquet"
        dataframe.to_parquet(parquet_path, index=False)
        handle = build_dataset_handle(
            parquet_path,
            {"source_kind": "data_load", "file_name": parquet_path.name, "size_bytes": 1},
        )
        config = _large_data_test_config(artifact_root)
        config.performance = PerformanceConfig(
            large_data_mode=True,
            optimize_dtypes=True,
            large_data_training_sample_rows=140,
            large_data_score_chunk_rows=50,
            memory_limit_gb=256.0,
        )

        context = QuantModelOrchestrator(config=config).run(handle)
        full_predictions = pd.read_parquet(context.artifacts["full_data_predictions"])

        assert len(context.raw_data) <= 140
        assert len(full_predictions) == len(dataframe)
        assert context.artifacts["sample_development_dir"].name == "sample_development"
        assert context.artifacts["full_data_scoring_dir"].name == "full_data_scoring"
        assert context.artifacts["large_data_metadata_dir"].name == "large_data"
        assert Path(context.artifacts["large_data_profile"]).exists()
        assert context.metadata["large_data_model_certification"]["status"] == (
            "full_data_certified"
        )
        assert "large_data_full_scoring_summary" in context.diagnostics_tables
        assert "large_data_source_profile" in context.diagnostics_tables
        assert "large_data_model_certification" in context.diagnostics_tables
        assert "diagnostic_registry" in context.diagnostics_tables
        assert "prepared_dataset_manifest" in context.diagnostics_tables
        assert "large_data_execution_plan" in context.diagnostics_tables
        assert "large_data_transformation_contract" in context.diagnostics_tables
        assert "large_data_feature_screening" in context.diagnostics_tables
        assert Path(context.artifacts["large_data_execution_plan"]).exists()
        assert Path(context.artifacts["large_data_transformation_contract"]).exists()
        assert Path(context.artifacts["large_data_feature_screening_manifest"]).exists()
        assert Path(context.artifacts["large_data_full_scoring_progress"]).exists()
        assert Path(context.artifacts["prepared_dataset_manifest"]).exists()


def test_large_data_profile_cache_reuses_unchanged_file_profile() -> None:
    with temporary_artifact_root("pytest_large_data_profile_cache") as artifact_root:
        parquet_path = artifact_root / "input.parquet"
        pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_parquet(
            parquet_path,
            index=False,
        )
        handle = build_dataset_handle(parquet_path, {"source_kind": "data_load"})
        cache_root = artifact_root / "profiles"

        first = profile_dataset_handle_cached(
            handle,
            preview_rows=2,
            cache_root=cache_root,
        )
        second = profile_dataset_handle_cached(
            handle,
            preview_rows=2,
            cache_root=cache_root,
        )

        assert first["profile_cache_hit"] is False
        assert second["profile_cache_hit"] is True
        assert first["profile_cache_key"] == second["profile_cache_key"]
        assert Path(second["profile_cache_path"]).exists()


def test_large_data_transformation_contract_classifies_supported_and_sample_only() -> None:
    with temporary_artifact_root("pytest_large_data_transform_contract") as artifact_root:
        config = _large_data_test_config(artifact_root)
        config.transformations = TransformationConfig(
            transformations=[
                TransformationSpec(
                    transform_type=TransformationType.SAFE_RATIO,
                    source_feature="balance",
                    secondary_feature="annual_income",
                    output_feature="balance_to_income",
                ),
                TransformationSpec(
                    transform_type=TransformationType.TARGET_ENCODING,
                    source_feature="region",
                    output_feature="region_target_encoded",
                ),
            ]
        )
        context = PipelineContext(config=config, run_id="run_contract", raw_input=None)

        payload = record_large_data_transformation_contract(context)
        statuses = {
            row["output_feature"]: row["large_data_status"]
            for row in payload["rows"]
        }

        assert statuses["balance_to_income"] == "compiled"
        assert statuses["region_target_encoded"] == "sample_only"
        assert Path(context.artifacts["large_data_transformation_contract"]).exists()


def test_large_data_feature_screening_is_advisory_unless_auto_apply_enabled() -> None:
    with temporary_artifact_root("pytest_large_data_feature_screen") as artifact_root:
        config = _large_data_test_config(artifact_root)
        frame = pd.DataFrame(
            {
                "constant_feature": [1, 1, 1, 1, 1, 1],
                "useful_feature": [0.1, 0.2, 0.9, 1.0, 1.1, 1.2],
                "default_status": [0, 0, 1, 1, 1, 0],
            }
        )
        context = PipelineContext(config=config, run_id="run_screen", raw_input=None)
        context.target_column = "default_status"
        context.feature_columns = ["constant_feature", "useful_feature"]
        context.numeric_features = list(context.feature_columns)
        context.split_frames = {"train": frame}

        record_large_data_feature_screening(context)

        assert context.feature_columns == ["constant_feature", "useful_feature"]
        assert "large_data_feature_screening" in context.diagnostics_tables

        config.performance.large_data_auto_apply_prescreen = True
        auto_context = PipelineContext(config=config, run_id="run_screen_auto", raw_input=None)
        auto_context.target_column = "default_status"
        auto_context.feature_columns = ["constant_feature", "useful_feature"]
        auto_context.numeric_features = list(auto_context.feature_columns)
        auto_context.split_frames = {"train": frame}

        record_large_data_feature_screening(auto_context)

    assert auto_context.feature_columns == ["useful_feature"]
    assert auto_context.metadata["large_data_prescreen_excluded_features"] == [
        "constant_feature"
    ]


def test_worker_service_queue_manifest_records_dispatch_mode() -> None:
    with temporary_artifact_root("pytest_large_data_worker_queue") as artifact_root:
        input_path = artifact_root / "input.parquet"
        pd.DataFrame({"x": [1], "default_status": [0]}).to_parquet(input_path, index=False)
        config = _large_data_test_config(artifact_root)

        manifest_path = queue_background_workflow(
            config=config,
            input_data=build_dataset_handle(input_path, {"source_kind": "data_load"}),
            queue_dir=artifact_root / "_job_queue",
        )
        manifest = read_background_manifest(manifest_path)

        assert manifest.status == "queued"
        assert manifest.dispatch_mode == "worker_service"
        assert Path(manifest.config_path).exists()


def test_table_ref_supports_paged_parquet_result_access() -> None:
    pytest.importorskip("duckdb")
    with temporary_artifact_root("pytest_table_ref_paged_results") as artifact_root:
        parquet_path = artifact_root / "predictions.parquet"
        pd.DataFrame(
            {
                "split": ["train", "train", "test", "test"],
                "segment": ["a", "b", "a", "b"],
                "predicted_probability": [0.2, 0.9, 0.4, 0.7],
            }
        ).to_parquet(parquet_path, index=False)

        table_ref = TableRef.from_path(parquet_path, name="full_data_predictions")
        result_ref = ResultTableRef.from_table_ref(
            table_ref,
            score_column="predicted_probability",
            split_column="split",
            segment_columns=["segment"],
        )
        page = query_table_page(
            result_ref,
            columns=["split", "segment", "predicted_probability"],
            filters=[{"column": "split", "op": "eq", "value": "test"}],
            sort_by="predicted_probability",
            descending=True,
            page=1,
            page_size=1,
        )
        row_count = count_table_rows(
            result_ref,
            filters=[{"column": "split", "op": "eq", "value": "test"}],
        )

    assert table_ref.row_count == 4
    assert row_count == 2
    assert page.to_dict(orient="records") == [
        {"split": "test", "segment": "b", "predicted_probability": 0.7}
    ]


def test_large_data_checkpoint_spills_large_dataframes_to_table_refs() -> None:
    with temporary_artifact_root("pytest_large_data_checkpoint_spill") as artifact_root:
        config = _large_data_test_config(artifact_root)
        config.performance = PerformanceConfig(
            large_data_mode=True,
            large_data_max_in_memory_rows=2,
        )
        context = PipelineContext(
            config=config,
            run_id="run_1",
            raw_input=None,
            working_data=pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}),
            predictions={"test": pd.DataFrame({"score": [0.1, 0.2, 0.3]})},
        )
        checkpoint_path = artifact_root / "checkpoints" / "01_test.joblib"

        save_context_checkpoint(context, checkpoint_path)
        raw_checkpoint = joblib.load(checkpoint_path)

        loaded = load_context_checkpoint(checkpoint_path)

    assert raw_checkpoint.working_data[CHECKPOINT_TABLE_REF_MARKER] is True
    assert raw_checkpoint.predictions["test"][CHECKPOINT_TABLE_REF_MARKER] is True
    assert loaded.working_data.equals(pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}))
    assert loaded.predictions["test"].equals(pd.DataFrame({"score": [0.1, 0.2, 0.3]}))


def test_csv_parquet_staging_reuses_unchanged_cache_file() -> None:
    with temporary_artifact_root("pytest_large_data_staging_cache") as artifact_root:
        csv_path = artifact_root / "input.csv"
        csv_path.write_text("x,y\n1,a\n2,b\n3,c\n", encoding="utf-8")
        handle = build_dataset_handle(csv_path, {"source_kind": "data_load"})

        first = stage_large_data_file(handle, chunk_rows=2, compression="snappy")
        second = stage_large_data_file(handle, chunk_rows=2, compression="snappy")

    assert first.active_path == second.active_path
    assert first.active_path.suffix == ".parquet"
    assert second.staging_metadata["reused_existing_staging_file"] is True


def test_chunked_parquet_writer_aligns_schema_across_chunks() -> None:
    with temporary_artifact_root("pytest_chunked_writer_schema_alignment") as artifact_root:
        output_path = artifact_root / "predictions.parquet"
        writer = _ChunkedParquetWriter(output_path, compression="snappy")
        writer.write(pd.DataFrame({"score": [0.1, 0.2], "segment": ["a", "b"]}))
        writer.write(pd.DataFrame({"score": ["0.3", "bad"], "extra": ["ignored", "ignored"]}))
        writer.close()

        output = pd.read_parquet(output_path)

    assert output.columns.tolist() == ["score", "segment"]
    assert len(output) == 4
    assert output["score"].isna().sum() == 1
    assert output["segment"].isna().sum() == 2
