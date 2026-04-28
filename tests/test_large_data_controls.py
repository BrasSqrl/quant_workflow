"""Tests for large-data ingestion, conversion, and tabular export controls."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    CreditRiskDiagnosticConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    LargeDataExportPolicy,
    PerformanceConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TabularOutputFormat,
    TargetConfig,
    TargetMode,
    load_framework_config,
)
from quant_pd_framework.config_serialization import FRAMEWORK_CONFIG_SECTION_NAMES
from quant_pd_framework.large_data import (
    build_dataset_handle,
    convert_csv_to_parquet,
    stage_large_data_file,
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

    assert metadata["chunk_count"] == 2
    assert metadata["row_count"] == 3
    assert converted.to_dict(orient="list") == {"x": [1, 2, 3], "y": ["a", "b", "c"]}


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
        csv_conversion_chunk_rows=25000,
        memory_limit_gb=64.0,
    )

    loaded = load_framework_config(config.to_dict())

    assert loaded.performance.large_data_mode is True
    assert loaded.performance.convert_csv_to_parquet is True
    assert loaded.performance.csv_conversion_chunk_rows == 25000
    assert loaded.performance.large_data_training_sample_rows == 250000
    assert loaded.performance.large_data_score_chunk_rows == 100000
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
        assert "large_data_full_scoring_summary" in context.diagnostics_tables
        assert "diagnostic_registry" in context.diagnostics_tables
        assert Path(context.artifacts["large_data_full_scoring_progress"]).exists()


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
