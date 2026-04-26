"""Tests for saved-run bundles that can be rerun outside the GUI."""

from __future__ import annotations

import json
from pathlib import Path

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    build_sample_pd_dataframe,
    load_framework_config,
)
from quant_pd_framework.run import main as run_cli_main
from tests.support import build_common_schema, temporary_artifact_root


def build_bundle_test_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=build_common_schema("loan_id", include_legacy_drop=True),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(source_column="default_status", positive_values=[1]),
        split=SplitConfig(
            data_structure=DataStructure.CROSS_SECTIONAL,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        diagnostics=DiagnosticConfig(
            interactive_visualizations=False,
            static_image_exports=False,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
    )


def test_saved_run_bundle_exports_rerun_assets() -> None:
    dataframe = build_sample_pd_dataframe(row_count=140, random_state=9)
    with temporary_artifact_root("pytest_bundle_assets") as artifact_root:
        config = build_bundle_test_config(artifact_root)

        context = QuantModelOrchestrator(config=config).run(dataframe)
        bundle_root = context.artifacts["output_root"]

        assert bundle_root is not None
        assert context.artifacts["config"] is not None
        assert context.artifacts["runner_script"] is not None
        assert context.artifacts["step_manifest"] is not None
        assert context.artifacts["rerun_readme"] is not None
        assert context.artifacts["input_snapshot"] is not None
        assert context.artifacts["code_snapshot_dir"] is not None
        assert context.artifacts["interactive_report"] is not None

        loaded_config = load_framework_config(context.artifacts["config"])
        assert loaded_config.split.data_structure == DataStructure.CROSS_SECTIONAL
        assert loaded_config.schema.column_specs[0].role == ColumnRole.DATE
        assert loaded_config.artifacts.export_code_snapshot is True

        step_manifest = json.loads(
            Path(context.artifacts["step_manifest"]).read_text(encoding="utf-8")
        )
        assert step_manifest["steps"][0]["name"] == "ingestion"
        assert step_manifest["steps"][-1]["name"] == "artifact_export"

        runner_text = Path(context.artifacts["runner_script"]).read_text(encoding="utf-8")
        assert "code_snapshot" in runner_text
        assert "run_config.json" in runner_text
        interactive_report_text = Path(context.artifacts["interactive_report"]).read_text(
            encoding="utf-8"
        )
        assert "Calibration / Thresholds" in interactive_report_text
        assert "Governance / Export Bundle" in interactive_report_text

        assert (
            bundle_root / "code" / "code_snapshot" / "src" / "quant_pd_framework" / "run.py"
        ).exists()
        assert (
            bundle_root / "code" / "code_snapshot" / "tests" / "test_pipeline_smoke.py"
        ).exists()
        assert (bundle_root / "code" / "code_snapshot" / "app" / "streamlit_app.py").exists()


def test_saved_run_bundle_exports_model_bundle_for_monitoring() -> None:
    dataframe = build_sample_pd_dataframe(row_count=140, random_state=21)
    with temporary_artifact_root("pytest_monitoring_bundle") as artifact_root:
        config = build_bundle_test_config(artifact_root)

        context = QuantModelOrchestrator(config=config).run(dataframe)
        bundle_root = context.artifacts["output_root"]
        monitoring_bundle_dir = context.artifacts["monitoring_bundle_dir"]
        monitoring_metadata_path = context.artifacts["monitoring_metadata"]

        assert bundle_root is not None
        assert monitoring_bundle_dir is not None
        assert monitoring_metadata_path is not None
        assert monitoring_bundle_dir == bundle_root / "model_bundle_for_monitoring"

        expected_bundle_paths = [
            monitoring_bundle_dir / "quant_model.joblib",
            monitoring_bundle_dir / "run_config.json",
            monitoring_bundle_dir / "generated_run.py",
            monitoring_bundle_dir / "monitoring_metadata.json",
            monitoring_bundle_dir / "artifact_manifest.json",
            monitoring_bundle_dir / "input_snapshot.csv",
            monitoring_bundle_dir / "predictions.csv",
            monitoring_bundle_dir / "code_snapshot",
        ]
        for expected_path in expected_bundle_paths:
            assert expected_path.exists()

        monitoring_metadata = json.loads(Path(monitoring_metadata_path).read_text(encoding="utf-8"))
        assert monitoring_metadata["bundle_version"] == "1.0"
        assert monitoring_metadata["created_by_run_id"] == context.run_id
        assert monitoring_metadata["score_column"] == "predicted_probability"
        assert monitoring_metadata["prediction_column"] == "predicted_class"
        assert (
            monitoring_metadata["bundled_artifacts"]["quant_model.joblib"] == "quant_model.joblib"
        )
        assert monitoring_metadata["bundled_artifacts"]["monitoring_metadata.json"] == (
            "monitoring_metadata.json"
        )
        assert monitoring_metadata["missing_optional_artifacts"] == []


def test_saved_run_bundle_monitoring_bundle_respects_optional_export_settings() -> None:
    dataframe = build_sample_pd_dataframe(row_count=140, random_state=25)
    with temporary_artifact_root("pytest_monitoring_bundle_optional") as artifact_root:
        config = build_bundle_test_config(artifact_root)
        config.artifacts = ArtifactConfig(
            output_root=artifact_root,
            export_input_snapshot=False,
            export_code_snapshot=False,
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)
        monitoring_bundle_dir = context.artifacts["monitoring_bundle_dir"]
        monitoring_metadata_path = context.artifacts["monitoring_metadata"]

        assert monitoring_bundle_dir is not None
        assert monitoring_metadata_path is not None
        assert not (monitoring_bundle_dir / "input_snapshot.csv").exists()
        assert not (monitoring_bundle_dir / "code_snapshot").exists()

        monitoring_metadata = json.loads(Path(monitoring_metadata_path).read_text(encoding="utf-8"))
        assert sorted(monitoring_metadata["missing_optional_artifacts"]) == [
            "code_snapshot",
            "input_snapshot.csv",
        ]
        assert monitoring_metadata["bundled_artifacts"]["input_snapshot.csv"] is None
        assert monitoring_metadata["bundled_artifacts"]["code_snapshot"] is None


def test_saved_run_bundle_cli_reruns_without_gui() -> None:
    dataframe = build_sample_pd_dataframe(row_count=140, random_state=17)
    with temporary_artifact_root("pytest_bundle_initial") as initial_output_root:
        with temporary_artifact_root("pytest_bundle_rerun") as rerun_output_root:
            config = build_bundle_test_config(initial_output_root)

            context = QuantModelOrchestrator(config=config).run(dataframe)
            config_path = context.artifacts["config"]

            assert config_path is not None
            return_code = run_cli_main(
                [
                    "--config",
                    str(config_path),
                    "--output-root",
                    str(rerun_output_root),
                ]
            )

            assert return_code == 0
            rerun_directories = [path for path in rerun_output_root.iterdir() if path.is_dir()]
            assert rerun_directories
            latest_rerun = max(rerun_directories, key=lambda path: path.name)
            assert (latest_rerun / "metadata" / "metrics.json").exists()
            assert (latest_rerun / "code" / "generated_run.py").exists()
            assert (latest_rerun / "metadata" / "step_manifest.json").exists()
