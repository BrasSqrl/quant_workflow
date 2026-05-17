"""Tests for the file-backed run registry and audit event log."""

from __future__ import annotations

import json
from pathlib import Path

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.run_registry import (
    append_audit_event,
    audit_event_log_path,
    build_run_registry_entry_from_run_folder,
    load_audit_events,
    load_run_registry,
    per_run_audit_event_path,
    run_registry_path,
    sync_run_registry_from_artifacts,
    upsert_run_registry_entry,
    write_per_run_audit_log,
)
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def _registry_test_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=build_common_schema("account_id", include_legacy_drop=True),
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
            interactive_visualizations=False,
            static_image_exports=False,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
    )


def test_audit_event_log_redacts_sensitive_values_and_writes_run_copy() -> None:
    with temporary_artifact_root("pytest_run_registry_audit") as output_root:
        run_root = output_root / "run_2026-01-01_00-00-00_UTC"
        (run_root / "metadata").mkdir(parents=True)

        append_audit_event(
            output_root,
            "data_source_selected",
            source="streamlit",
            run_id=run_root.name,
            artifact_root=run_root,
            metadata={
                "display_label": "sample.csv",
                "aws_secret_access_key": "should-not-leak",  # pragma: allowlist secret
            },
        )

        events = load_audit_events(output_root, run_id=run_root.name)
        assert len(events) == 1
        assert events[0].metadata["aws_secret_access_key"] == "[redacted]"
        assert audit_event_log_path(output_root).exists()
        assert per_run_audit_event_path(run_root).exists()


def test_run_registry_upsert_recovers_from_corrupt_file() -> None:
    with temporary_artifact_root("pytest_run_registry_corrupt") as output_root:
        registry_path = run_registry_path(output_root)
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text("{not valid json", encoding="utf-8")

        entry = upsert_run_registry_entry(
            output_root,
            {
                "run_id": "run_2026-01-01_00-00-00_UTC",
                "status": "completed",
                "artifact_root": str(output_root / "run_2026-01-01_00-00-00_UTC"),
            },
        )

        assert entry.run_id == "run_2026-01-01_00-00-00_UTC"
        entries = load_run_registry(output_root)
        assert [candidate.run_id for candidate in entries] == [entry.run_id]


def test_registry_backfills_from_existing_run_folder() -> None:
    with temporary_artifact_root("pytest_run_registry_backfill") as output_root:
        run_root = output_root / "run_2026-01-01_00-00-00_UTC"
        (run_root / "metadata").mkdir(parents=True)
        (run_root / "config").mkdir()
        (run_root / "reports").mkdir()
        (run_root / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "core_artifacts": {
                        "output_root": str(run_root),
                        "decision_summary": str(run_root / "reports" / "decision_summary.md"),
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_root / "config" / "run_config.json").write_text(
            json.dumps(
                {
                    "execution": {"mode": "fit_new_model"},
                    "model": {"model_type": "logistic_regression"},
                    "target": {"mode": "binary"},
                    "performance": {"large_data_mode": False},
                }
            ),
            encoding="utf-8",
        )
        (run_root / "metadata" / "metrics.json").write_text(
            json.dumps({"test": {"auc": 0.72, "ks": 0.31}}),
            encoding="utf-8",
        )
        (run_root / "metadata" / "run_debug_trace.json").write_text(
            json.dumps(
                {
                    "run_id": run_root.name,
                    "run_started_at_utc": "2026-01-01T00:00:00+00:00",
                    "run_completed_at_utc": "2026-01-01T00:01:00+00:00",
                    "summary": {"total_run_seconds": 60, "warning_count": 1},
                }
            ),
            encoding="utf-8",
        )

        entry = build_run_registry_entry_from_run_folder(run_root)
        assert entry is not None
        assert entry.metrics_summary["test_auc"] == 0.72

        sync_run_registry_from_artifacts(output_root)
        entries = load_run_registry(output_root)
        assert len(entries) == 1
        assert entries[0].run_id == run_root.name


def test_completed_workflow_updates_registry_and_exports_audit_log() -> None:
    dataframe = build_binary_dataframe(row_count=120)
    with temporary_artifact_root("pytest_run_registry_workflow") as output_root:
        append_audit_event(
            output_root,
            "workflow_run_started",
            source="test",
            run_id="placeholder",
        )
        context = QuantModelOrchestrator(config=_registry_test_config(output_root)).run(dataframe)

        entries = load_run_registry(output_root)
        assert any(entry.run_id == context.run_id for entry in entries)
        run_root = output_root / context.run_id
        assert (run_root / "metadata" / "audit_events.jsonl").exists()

        append_audit_event(
            output_root,
            "review_record_saved",
            source="test",
            run_id=context.run_id,
            artifact_root=run_root,
        )
        audit_path = write_per_run_audit_log(output_root, context.run_id, run_root)
        assert "review_record_saved" in audit_path.read_text(encoding="utf-8")
