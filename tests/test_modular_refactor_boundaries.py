"""Compatibility checks for the modular Streamlit/export/large-data refactor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_pd_framework.config import ArtifactConfig
from quant_pd_framework.export_layout import build_export_path_layout
from quant_pd_framework.exporting.layout import (
    ensure_layout_directories,
    table_export_path,
)
from quant_pd_framework.exporting.manifest import write_manifest
from quant_pd_framework.large_data import (
    DatasetHandle,
    build_dataset_handle,
    is_s3_uri,
    parse_s3_uri,
)
from quant_pd_framework.large_data_support.handles import DatasetHandle as SupportDatasetHandle
from quant_pd_framework.streamlit_ui.run_execution import (
    build_background_checkpoint_flow_event,
    is_background_job_active,
    render_background_job_status,
)
from quant_pd_framework.streamlit_ui.steps.step1_data_schema import DatasetWorkspace
from quant_pd_framework.streamlit_ui.steps.step2_model_config import (
    render_model_configuration_intro,
)
from quant_pd_framework.streamlit_ui.steps.step3_readiness_run import (
    render_readiness_check_and_run,
)
from quant_pd_framework.streamlit_ui.steps.step4_results_artifacts import (
    render_results_artifacts_tab,
)
from quant_pd_framework.streamlit_ui.steps.step5_decision_summary import (
    render_decision_summary_tab,
)


def test_large_data_handle_reexports_remain_compatible(tmp_path: Path) -> None:
    data_path = tmp_path / "input.csv"
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(data_path, index=False)

    handle = build_dataset_handle(data_path, {"source_kind": "data_load"})

    assert isinstance(handle, DatasetHandle)
    assert isinstance(handle, SupportDatasetHandle)
    assert handle.active_path == data_path
    assert is_s3_uri("s3://bucket/path/file.csv")
    assert parse_s3_uri("s3://bucket/path/file.csv") == ("bucket", "path/file.csv")


def test_exporting_helpers_build_layout_and_manifest(tmp_path: Path) -> None:
    layout = build_export_path_layout(ArtifactConfig(output_root=tmp_path), tmp_path)
    ensure_layout_directories(layout)
    table_path = table_export_path(
        layout.tables_dir,
        "calibration_curve",
        sanitized_name="calibration_curve",
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("metric,value\nauc,0.7\n", encoding="utf-8")

    write_manifest(
        manifest_path,
        {"tables": {"calibration": str(table_path)}},
        output_root=tmp_path,
        monitoring_bundle_directory_name="model_bundle_for_monitoring",
    )

    assert manifest_path.exists()
    payload = manifest_path.read_text(encoding="utf-8")
    assert "artifact_index" in payload
    assert "calibration_curve.csv" in payload


def test_streamlit_step_modules_expose_refactor_boundaries() -> None:
    assert DatasetWorkspace.__name__ == "DatasetWorkspace"
    assert callable(render_model_configuration_intro)
    assert callable(render_readiness_check_and_run)
    assert callable(render_results_artifacts_tab)
    assert callable(render_decision_summary_tab)
    assert callable(render_background_job_status)
    assert callable(is_background_job_active)


def test_background_job_active_status_helper() -> None:
    assert is_background_job_active({"status": "running"}) is True
    assert is_background_job_active({"manifest": {"status": "queued"}}) is True
    assert is_background_job_active({"status": "completed"}) is False
    assert is_background_job_active({"manifest": {"status": "failed"}}) is False
    assert is_background_job_active(None) is False


def test_background_checkpoint_flow_event_reads_manifest_stages(tmp_path: Path) -> None:
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-05-19T00:00:00+00:00",
                "status": "running",
                "stages": [
                    {
                        "order": 1,
                        "label": "Prepare data",
                        "status": "completed",
                        "critical": True,
                    },
                    {
                        "order": 2,
                        "label": "Fit or load model",
                        "status": "running",
                        "critical": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    event = build_background_checkpoint_flow_event(
        {
            "status": "running",
            "current_stage": "Fit or load model",
            "manifest": {"checkpoint_manifest_path": str(manifest_path)},
        }
    )

    assert event is not None
    assert event["event_type"] == "stage_started"
    assert event["step_name"] == "Fit or load model"
    assert event["step_order"] == 1
    assert event["total_steps"] == 2
    assert event["stages"][1]["status"] == "running"


def test_background_checkpoint_flow_event_handles_missing_or_corrupt_manifest(
    tmp_path: Path,
) -> None:
    missing_event = build_background_checkpoint_flow_event(
        {
            "status": "running",
            "manifest": {"checkpoint_manifest_path": str(tmp_path / "missing.json")},
        }
    )
    corrupt_path = tmp_path / "checkpoint_manifest.json"
    corrupt_path.write_text("{not json", encoding="utf-8")
    corrupt_event = build_background_checkpoint_flow_event(
        {
            "status": "running",
            "manifest": {"checkpoint_manifest_path": str(corrupt_path)},
        }
    )

    assert missing_event is None
    assert corrupt_event is None
