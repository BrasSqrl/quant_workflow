"""Regression tests for checkpointed staged workflow execution."""

from __future__ import annotations

import json

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    CreditRiskDiagnosticConfig,
    DiagnosticConfig,
    ExplainabilityConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    SchemaConfig,
    ScorecardWorkbenchConfig,
    SplitConfig,
    TargetConfig,
)
from quant_pd_framework.checkpointing import (
    CHECKPOINT_MANIFEST_FILE_NAME,
    load_context_checkpoint,
)
from quant_pd_framework.stage_runner import CheckpointedWorkflowRunner
from tests.support import temporary_artifact_root
from tests.test_pipeline_smoke import build_synthetic_dataframe


def build_checkpoint_test_config(
    output_root,
    *,
    keep_all_checkpoints: bool = False,
) -> FrameworkConfig:
    return FrameworkConfig(
        schema=SchemaConfig(
            column_specs=[
                ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ColumnSpec(name="legacy_unused_column", enabled=False),
            ]
        ),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(source_column="default_status", positive_values=[1]),
        split=SplitConfig(
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        diagnostics=DiagnosticConfig(
            data_quality=False,
            descriptive_statistics=False,
            missingness_analysis=False,
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
            export_excel_workbook=False,
        ),
        explainability=ExplainabilityConfig(enabled=False),
        scorecard_workbench=ScorecardWorkbenchConfig(enabled=False),
        credit_risk=CreditRiskDiagnosticConfig(enabled=False),
        artifacts=ArtifactConfig(
            output_root=output_root,
            keep_all_checkpoints=keep_all_checkpoints,
        ),
    )


def test_checkpointed_runner_writes_restartable_manifest_and_exports() -> None:
    dataframe = build_synthetic_dataframe(row_count=180)
    with temporary_artifact_root("pytest_checkpointed") as output_root:
        config = build_checkpoint_test_config(output_root)

        context = CheckpointedWorkflowRunner(
            config=config,
            use_subprocess=False,
        ).run_all(dataframe)

        checkpoint_manifest = (
            output_root / context.run_id / "checkpoints" / CHECKPOINT_MANIFEST_FILE_NAME
        )
        metadata_manifest = (
            output_root / context.run_id / "metadata" / CHECKPOINT_MANIFEST_FILE_NAME
        )
        manifest_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))

        assert manifest_payload["status"] == "completed"
        assert all(
            stage["status"] in {"completed", "failed_optional"}
            for stage in manifest_payload["stages"]
        )
        assert context.artifacts["model"].exists()
        assert context.artifacts["artifact_manifest"].exists()
        assert metadata_manifest.exists()
        assert context.metadata["checkpointed_execution"]["checkpoint_manifest"] == str(
            checkpoint_manifest
        )
        assert not list((output_root / context.run_id / "checkpoints").glob("*.joblib"))
        assert manifest_payload["checkpoint_retention"]["keep_all_checkpoints"] is False
        assert manifest_payload["checkpoint_retention"]["latest_context_retained"] == ""


def test_checkpointed_runner_can_execute_single_stage_in_subprocess() -> None:
    dataframe = build_synthetic_dataframe(row_count=80)
    with temporary_artifact_root("pytest_checkpointed_subprocess") as output_root:
        config = build_checkpoint_test_config(output_root)
        progress_events: list[dict] = []
        runner = CheckpointedWorkflowRunner(
            config=config,
            progress_callback=progress_events.append,
            use_subprocess=True,
        )
        manifest_path = runner.start(dataframe)

        runner.run_next(manifest_path)
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        context = load_context_checkpoint(manifest_payload["latest_context_path"])
        stage_started_event = next(
            event
            for event in progress_events
            if event.get("event_type") == "stage_started"
            and event.get("stage_id") == "prepare_data"
        )
        stage_completed_event = next(
            event
            for event in progress_events
            if event.get("event_type") == "stage_completed"
            and event.get("stage_id") == "prepare_data"
        )

        assert manifest_payload["stages"][0]["stage_id"] == "prepare_data"
        assert manifest_payload["stages"][0]["status"] == "completed"
        assert stage_started_event["stages"][0]["status"] == "running"
        assert stage_completed_event["stages"][0]["status"] == "completed"
        assert context.split_frames
        assert len(list(manifest_path.parent.glob("*.joblib"))) == 1


def test_checkpointed_runner_can_keep_all_checkpoint_contexts() -> None:
    dataframe = build_synthetic_dataframe(row_count=120)
    with temporary_artifact_root("pytest_checkpointed_keep_all") as output_root:
        config = build_checkpoint_test_config(output_root, keep_all_checkpoints=True)

        context = CheckpointedWorkflowRunner(
            config=config,
            use_subprocess=False,
        ).run_all(dataframe)

        checkpoints_dir = output_root / context.run_id / "checkpoints"
        manifest_payload = json.loads(
            (checkpoints_dir / CHECKPOINT_MANIFEST_FILE_NAME).read_text(encoding="utf-8")
        )

        assert manifest_payload["checkpoint_retention"]["keep_all_checkpoints"] is True
        assert manifest_payload["checkpoint_retention"]["policy"] == "keep_all"
        assert len(list(checkpoints_dir.glob("*.joblib"))) > 1
