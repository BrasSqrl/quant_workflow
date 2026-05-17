"""Operational hardening checks for timeouts, cancellation, and queue safety."""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from quant_pd_framework import (
    CleaningConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    SchemaConfig,
    TargetConfig,
)
from quant_pd_framework.background_jobs import BACKGROUND_JOB_MANIFEST
from quant_pd_framework.config import SplitConfig
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.orchestrator import (
    INHERITED_EXISTING_CONFIG_FIELDS,
    OVERRIDDEN_EXISTING_CONFIG_FIELDS,
)
from quant_pd_framework.safe_serialization import sha256_sidecar_path
from quant_pd_framework.stage_runner import CheckpointedWorkflowRunner
from quant_pd_framework.worker_service import _process_queued_jobs
from scripts.check_module_size import collect_module_size_failures
from tests.support import temporary_artifact_root


def test_existing_config_field_policy_covers_every_framework_config_field() -> None:
    field_names = {field.name for field in dataclass_fields(FrameworkConfig)}

    assert OVERRIDDEN_EXISTING_CONFIG_FIELDS | INHERITED_EXISTING_CONFIG_FIELDS == field_names
    assert OVERRIDDEN_EXISTING_CONFIG_FIELDS.isdisjoint(INHERITED_EXISTING_CONFIG_FIELDS)


def test_pipeline_context_grouped_state_views_preserve_existing_fields() -> None:
    config = FrameworkConfig(
        schema=SchemaConfig(),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(source_column="target"),
        split=SplitConfig(),
    )
    context = PipelineContext(config=config, run_id="run_1", raw_input=None)

    assert context.model_artifact_state.model is context.model
    assert context.diagnostic_result_state.metrics is context.metrics
    assert context.large_data_state.metadata is context.metadata
    assert context.execution_artifact_state.artifacts is context.artifacts


def test_stage_subprocess_timeout_marks_stage_failed(monkeypatch) -> None:
    from tests.test_checkpointed_workflow import build_checkpoint_test_config
    from tests.test_pipeline_smoke import build_synthetic_dataframe

    with temporary_artifact_root("pytest_stage_timeout") as output_root:
        config = build_checkpoint_test_config(output_root)
        config.performance.stage_subprocess_timeout_seconds = 1
        runner = CheckpointedWorkflowRunner(config=config, use_subprocess=True)
        manifest_path = runner.start(build_synthetic_dataframe(row_count=40))

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "cmd"), timeout=1)

        monkeypatch.setattr("quant_pd_framework.stage_runner.subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="exceeded subprocess timeout"):
            runner.run_next(manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["stages"][0]["status"] == "failed"


def test_worker_service_quarantines_unparseable_manifest(tmp_path) -> None:
    job_dir = tmp_path / "bad_job"
    job_dir.mkdir()
    manifest_path = job_dir / BACKGROUND_JOB_MANIFEST
    manifest_path.write_text("{not valid json", encoding="utf-8")

    processed = _process_queued_jobs(tmp_path)

    failed_manifest = next((tmp_path / "_failed_manifests").glob("*/manifest_parse_failure.json"))
    assert processed == 0
    assert failed_manifest.exists()
    assert not manifest_path.exists()


def test_module_size_checker_blocks_new_oversized_module(tmp_path) -> None:
    oversized = tmp_path / "src" / "quant_pd_framework" / "new_big_module.py"
    oversized.parent.mkdir(parents=True)
    oversized.write_text("\n".join("x = 1" for _ in range(1_501)), encoding="utf-8")

    failures = collect_module_size_failures(tmp_path)

    assert failures
    assert "new_big_module.py" in failures[0]


def test_governance_docs_exist_and_readme_is_concise() -> None:
    assert (Path.cwd() / "SECURITY.md").exists()
    assert (Path.cwd() / "CONTRIBUTING.md").exists()
    assert (Path.cwd() / ".github" / "CODEOWNERS").exists()
    assert len((Path.cwd() / "README.md").read_text(encoding="utf-8").splitlines()) < 500
    baseline = json.loads((Path.cwd() / ".secrets.baseline").read_text(encoding="utf-8"))
    assert baseline.get("results") == {}


def test_checkpoint_save_writes_hash_sidecar() -> None:
    from quant_pd_framework.checkpointing import save_context_checkpoint

    with temporary_artifact_root("pytest_checkpoint_hash") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="target"),
            split=SplitConfig(),
        )
        context = PipelineContext(config=config, run_id="run_1", raw_input=None)
        checkpoint_path = output_root / "context.joblib"

        save_context_checkpoint(context, checkpoint_path)

        assert sha256_sidecar_path(checkpoint_path).exists()
