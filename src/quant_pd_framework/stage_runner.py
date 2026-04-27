"""Checkpointed workflow runner for memory-isolated Quant Studio execution."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .base import BasePipelineStep
from .checkpointing import (
    build_checkpoint_paths,
    checkpoint_file_name,
    copy_manifest_to_metadata,
    find_next_pending_stage,
    initialize_checkpoint_manifest,
    load_context_checkpoint,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_started,
    read_checkpoint_manifest,
    save_context_checkpoint,
    write_checkpoint_manifest,
)
from .config import ExecutionMode, FrameworkConfig
from .context import PipelineContext
from .export_layout import build_export_path_layout
from .orchestrator import QuantModelOrchestrator
from .steps import (
    ArtifactExportStep,
    AssumptionCheckStep,
    BacktestStep,
    CleaningStep,
    CrossValidationStep,
    DiagnosticsStep,
    EvaluationStep,
    FeatureEngineeringStep,
    FeatureSubsetSearchStep,
    ImputationStep,
    IngestionStep,
    LargeDataFullScoringStep,
    ModelComparisonStep,
    ModelTrainingStep,
    SchemaManagementStep,
    SplitStep,
    TargetConstructionStep,
    TransformationStep,
    ValidationStep,
    VariableSelectionStep,
)

ProgressCallback = Callable[[dict[str, Any]], None]

DIAGNOSTIC_GROUP_LABELS: dict[str, str] = {
    "overview": "Diagnostics: overview",
    "governance": "Diagnostics: governance",
    "data_quality": "Diagnostics: data quality",
    "performance": "Diagnostics: performance",
    "stability_tests": "Diagnostics: stability and statistical tests",
    "comparison_explainability": "Diagnostics: comparison and explainability",
    "credit_risk": "Diagnostics: credit risk",
    "expanded_framework": "Diagnostics: expanded framework tests",
}


@dataclass(frozen=True, slots=True)
class WorkflowStageDefinition:
    """One restartable stage in a checkpointed workflow."""

    stage_id: str
    label: str
    description: str
    steps: tuple[BasePipelineStep, ...]
    critical: bool = True
    diagnostic_group: str | None = None

    def to_manifest_row(self, order: int) -> dict[str, Any]:
        return {
            "order": order,
            "stage_id": self.stage_id,
            "label": self.label,
            "description": self.description,
            "critical": self.critical,
            "diagnostic_group": self.diagnostic_group or "",
            "step_names": [step.name for step in self.steps],
        }


class CheckpointedWorkflowRunner:
    """Runs the framework through auditable disk-backed checkpoints."""

    def __init__(
        self,
        *,
        config: FrameworkConfig,
        progress_callback: ProgressCallback | None = None,
        use_subprocess: bool = True,
    ) -> None:
        self.orchestrator = QuantModelOrchestrator(config=config)
        self.config = self.orchestrator.config
        self.progress_callback = progress_callback
        self.use_subprocess = use_subprocess

    def run_all(self, data: pd.DataFrame | str | Path | Any) -> PipelineContext:
        """Runs every pending stage and returns the final context."""

        manifest_path = self.start(data)
        while True:
            manifest = read_checkpoint_manifest(manifest_path)
            if find_next_pending_stage(manifest) is None:
                break
            self.run_next(manifest_path)
        latest_context_path = Path(read_checkpoint_manifest(manifest_path)["latest_context_path"])
        context = load_context_checkpoint(latest_context_path)
        self._finalize_run_metadata(context, manifest_path)
        self._copy_checkpoint_manifest_to_metadata(context, manifest_path)
        latest_context_path = Path(read_checkpoint_manifest(manifest_path)["latest_context_path"])
        save_context_checkpoint(context, latest_context_path)
        completed_manifest = read_checkpoint_manifest(manifest_path)
        stages = self._stage_definitions()
        self._notify(
            {
                "event_type": "run_completed",
                "run_id": context.run_id,
                "completed_at_utc": context.metadata.get("run_completed_at_utc", ""),
                "elapsed_seconds": context.metadata.get("run_elapsed_seconds"),
                "step_order": len(stages),
                "total_steps": len(stages),
                "stages": self._manifest_stage_flow(completed_manifest),
            }
        )
        return context

    def start(self, data: pd.DataFrame | str | Path | Any) -> Path:
        """Creates the initial checkpoint and manifest for a new run."""

        context = PipelineContext(
            config=self.config,
            run_id=self.orchestrator._build_run_id(),
            raw_input=data,
        )
        run_started_at = datetime.now(UTC)
        context.metadata["execution_mode"] = self.config.execution.mode.value
        context.metadata["step_manifest"] = self.orchestrator.describe_steps()
        context.metadata["run_started_at_utc"] = run_started_at.isoformat()
        context.metadata["run_status"] = "running"
        context.metadata["checkpointed_execution"] = {
            "enabled": True,
            "use_subprocess": bool(self.use_subprocess),
        }
        checkpoint_paths = build_checkpoint_paths(self.config.artifacts.output_root, context.run_id)
        initial_context_path = checkpoint_paths.checkpoints_dir / "00_initial_context.joblib"
        save_context_checkpoint(context, initial_context_path)
        stages = self._stage_definitions()
        manifest = initialize_checkpoint_manifest(
            paths=checkpoint_paths,
            run_id=context.run_id,
            stages=[
                stage.to_manifest_row(order)
                for order, stage in enumerate(stages, start=1)
            ],
            initial_context_path=initial_context_path,
            execution_strategy=(
                "checkpointed_subprocess" if self.use_subprocess else "checkpointed_in_process"
            ),
        )
        self._notify(
            {
                "event_type": "run_started",
                "run_id": context.run_id,
                "started_at_utc": run_started_at.isoformat(),
                "elapsed_seconds": 0.0,
                "total_steps": len(stages),
                "checkpoint_manifest": str(checkpoint_paths.manifest_path),
                "stages": self._manifest_stage_flow(manifest),
            }
        )
        write_checkpoint_manifest(checkpoint_paths.manifest_path, manifest)
        return checkpoint_paths.manifest_path

    def run_next(self, manifest_path: str | Path) -> PipelineContext | None:
        """Runs the next pending stage from an existing checkpoint manifest."""

        manifest_path = Path(manifest_path)
        manifest = read_checkpoint_manifest(manifest_path)
        next_stage = find_next_pending_stage(manifest)
        if next_stage is None:
            return load_context_checkpoint(Path(manifest["latest_context_path"]))
        stage_id = str(next_stage["stage_id"])
        stage = self._stage_by_id(stage_id)
        if self.use_subprocess:
            return self._run_stage_subprocess(manifest_path, stage)
        return self.run_stage_from_manifest(manifest_path, stage_id)

    def run_stage_from_manifest(
        self,
        manifest_path: str | Path,
        stage_id: str,
    ) -> PipelineContext:
        """Runs a specific stage inside the current Python process."""

        manifest_path = Path(manifest_path)
        manifest = read_checkpoint_manifest(manifest_path)
        stage = self._stage_by_id(stage_id)
        context_path = Path(str(manifest["latest_context_path"]))
        context = load_context_checkpoint(context_path)
        total_steps = self._total_step_count(self._stage_definitions())
        step_offset = self._completed_step_count(manifest)
        started = perf_counter()
        mark_stage_started(manifest, stage.stage_id)
        write_checkpoint_manifest(manifest_path, manifest)
        self._notify_stage_manifest_event(manifest, stage, "stage_started")
        try:
            context = self._run_stage_steps(
                context=context,
                stage=stage,
                total_steps=total_steps,
                step_offset=step_offset,
            )
        except Exception as exc:
            elapsed = perf_counter() - started
            mark_stage_failed(
                manifest,
                stage.stage_id,
                error_message=str(exc),
                elapsed_seconds=elapsed,
                optional=not stage.critical,
            )
            write_checkpoint_manifest(manifest_path, manifest)
            self._notify_stage_manifest_event(
                manifest,
                stage,
                "stage_failed",
                error_message=str(exc),
            )
            if stage.critical:
                raise
            context.warn(f"Optional stage '{stage.label}' failed and was skipped: {exc}")
            stage_order = int(self._manifest_stage_row(manifest, stage.stage_id).get("order", 0))
            failure_context_path = (
                manifest_path.parent
                / checkpoint_file_name(stage_order, stage.stage_id)
            )
            save_context_checkpoint(context, failure_context_path)
            manifest["latest_context_path"] = str(failure_context_path)
            write_checkpoint_manifest(manifest_path, manifest)
            return context

        elapsed = perf_counter() - started
        stage_order = int(self._manifest_stage_row(manifest, stage.stage_id).get("order", 0))
        output_context_path = manifest_path.parent / checkpoint_file_name(
            stage_order,
            stage.stage_id,
        )
        context.metadata.setdefault("checkpointed_execution", {})["latest_stage"] = stage.stage_id
        context.metadata["checkpointed_execution"]["checkpoint_manifest"] = str(manifest_path)
        context.metadata["checkpointed_execution"]["checkpoint_dir"] = str(manifest_path.parent)
        save_context_checkpoint(context, output_context_path)
        mark_stage_completed(
            manifest,
            stage.stage_id,
            context_path=output_context_path,
            elapsed_seconds=elapsed,
        )
        write_checkpoint_manifest(manifest_path, manifest)
        self._notify_stage_manifest_event(manifest, stage, "stage_completed")
        del context
        gc.collect()
        return load_context_checkpoint(output_context_path)

    def _run_stage_subprocess(
        self,
        manifest_path: Path,
        stage: WorkflowStageDefinition,
    ) -> PipelineContext | None:
        manifest_before = read_checkpoint_manifest(manifest_path)
        self._notify_stage_manifest_event(manifest_before, stage, "stage_started")
        command = [
            sys.executable,
            "-m",
            "quant_pd_framework.run_stage",
            "--manifest",
            str(manifest_path),
            "--stage-id",
            stage.stage_id,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        manifest = read_checkpoint_manifest(manifest_path)
        if completed.returncode != 0:
            error_message = completed.stderr.strip() or completed.stdout.strip()
            self._notify_stage_manifest_event(
                manifest,
                stage,
                "stage_failed",
                error_message=error_message,
            )
            if stage.critical:
                raise RuntimeError(
                    f"Checkpoint stage '{stage.label}' failed in subprocess: {error_message}"
                )
            return None
        stage_row = self._manifest_stage_row(manifest, stage.stage_id)
        if stage_row.get("status") == "failed_optional":
            error_message = str(stage_row.get("error_message") or "")
            self._notify_stage_manifest_event(
                manifest,
                stage,
                "stage_failed",
                error_message=error_message,
            )
            return None
        self._notify_stage_manifest_event(manifest, stage, "stage_completed")
        return None

    def _run_stage_steps(
        self,
        *,
        context: PipelineContext,
        stage: WorkflowStageDefinition,
        total_steps: int,
        step_offset: int,
    ) -> PipelineContext:
        if stage.diagnostic_group:
            context.metadata["active_diagnostic_groups"] = [stage.diagnostic_group]
            context.metadata["diagnostics_replace_outputs"] = stage.diagnostic_group == "overview"
        try:
            context = self.orchestrator.run_context(
                context,
                steps=list(stage.steps),
                run_started=perf_counter(),
                step_offset=step_offset,
                total_steps=total_steps,
            )
        finally:
            context.metadata.pop("active_diagnostic_groups", None)
            context.metadata.pop("diagnostics_replace_outputs", None)
            gc.collect()
        return context

    def _stage_definitions(self) -> list[WorkflowStageDefinition]:
        if self.config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            return [
                WorkflowStageDefinition(
                    stage_id="prepare_data",
                    label="Prepare data",
                    description=(
                        "Load data, apply schema, create target, split, impute, "
                        "and transform."
                    ),
                    steps=(
                        IngestionStep(),
                        SchemaManagementStep(),
                        TargetConstructionStep(),
                        ValidationStep(),
                        CleaningStep(),
                        FeatureEngineeringStep(),
                        SplitStep(),
                        AssumptionCheckStep(),
                        ImputationStep(),
                        TransformationStep(),
                    ),
                ),
                WorkflowStageDefinition(
                    stage_id="feature_subset_search",
                    label="Run feature subset search",
                    description=(
                        "Fit and rank candidate feature subsets for the selected model type."
                    ),
                    steps=(FeatureSubsetSearchStep(),),
                ),
                WorkflowStageDefinition(
                    stage_id="export_package",
                    label="Export package",
                    description="Write subset-search comparison reports, tables, and manifest.",
                    steps=(ArtifactExportStep(),),
                ),
            ]
        return [
            WorkflowStageDefinition(
                stage_id="prepare_data",
                label="Prepare data",
                description=(
                    "Load data, apply schema, create target, split, impute, "
                    "transform, and select variables."
                ),
                steps=(
                    IngestionStep(),
                    SchemaManagementStep(),
                    TargetConstructionStep(),
                    ValidationStep(),
                    CleaningStep(),
                    FeatureEngineeringStep(),
                    SplitStep(),
                    AssumptionCheckStep(),
                    ImputationStep(),
                    TransformationStep(),
                    VariableSelectionStep(),
                ),
            ),
            WorkflowStageDefinition(
                stage_id="fit_model",
                label="Fit or load model",
                description="Fit the configured model or load the existing model artifact.",
                steps=(ModelTrainingStep(),),
            ),
            WorkflowStageDefinition(
                stage_id="score_evaluate",
                label="Score and evaluate",
                description=(
                    "Score train/validation/test splits, compare challengers, "
                    "and build backtesting bands."
                ),
                steps=(EvaluationStep(), ModelComparisonStep(), BacktestStep()),
            ),
            *self._diagnostic_stage_definitions(),
            WorkflowStageDefinition(
                stage_id="cross_validation",
                label="Cross-validation",
                description="Run optional k-fold or time-series fold diagnostics.",
                steps=(CrossValidationStep(),),
                critical=False,
            ),
            WorkflowStageDefinition(
                stage_id="large_data_full_scoring",
                label="Large-data full scoring",
                description="Score file-backed Large Data Mode inputs in chunks when enabled.",
                steps=(LargeDataFullScoringStep(),),
                critical=False,
            ),
            WorkflowStageDefinition(
                stage_id="export_package",
                label="Export package",
                description=(
                    "Write model, reports, tables, manifests, rerun bundle, "
                    "and monitoring handoff."
                ),
                steps=(ArtifactExportStep(),),
            ),
        ]

    def _diagnostic_stage_definitions(self) -> list[WorkflowStageDefinition]:
        return [
            WorkflowStageDefinition(
                stage_id=f"diagnostics_{group_name}",
                label=label,
                description=f"Run the {label.lower()} group.",
                steps=(DiagnosticsStep(),),
                critical=group_name == "overview",
                diagnostic_group=group_name,
            )
            for group_name, label in DIAGNOSTIC_GROUP_LABELS.items()
        ]

    def _stage_by_id(self, stage_id: str) -> WorkflowStageDefinition:
        for stage in self._stage_definitions():
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"Unknown workflow stage: {stage_id}")

    def _manifest_stage_row(self, manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
        for stage in manifest.get("stages", []):
            if stage.get("stage_id") == stage_id:
                return stage
        raise KeyError(f"Unknown manifest stage: {stage_id}")

    def _completed_step_count(self, manifest: dict[str, Any]) -> int:
        completed = 0
        for stage in manifest.get("stages", []):
            if stage.get("status") in {"completed", "skipped", "failed_optional"}:
                completed += len(stage.get("step_names", []))
        return completed

    def _total_step_count(self, stages: list[WorkflowStageDefinition]) -> int:
        return sum(len(stage.steps) for stage in stages)

    def _finalize_run_metadata(
        self,
        context: PipelineContext,
        manifest_path: Path,
    ) -> None:
        completed_at = datetime.now(UTC)
        started_at_text = context.metadata.get("run_started_at_utc")
        elapsed_seconds = None
        if isinstance(started_at_text, str) and started_at_text:
            try:
                started_at = datetime.fromisoformat(started_at_text)
                elapsed_seconds = (completed_at - started_at).total_seconds()
            except ValueError:
                elapsed_seconds = None
        context.metadata["run_completed_at_utc"] = completed_at.isoformat()
        context.metadata["run_elapsed_seconds"] = (
            round(float(elapsed_seconds), 6) if elapsed_seconds is not None else None
        )
        context.metadata["run_status"] = "completed"
        context.metadata.setdefault("checkpointed_execution", {})["checkpoint_manifest"] = str(
            manifest_path
        )
        self.orchestrator._refresh_exported_debug_trace(context)

    def _copy_checkpoint_manifest_to_metadata(
        self,
        context: PipelineContext,
        manifest_path: Path,
    ) -> None:
        output_root = context.config.artifacts.output_root / context.run_id
        layout = build_export_path_layout(context.config.artifacts, output_root)
        if not manifest_path.exists():
            return
        copied_path = copy_manifest_to_metadata(manifest_path, layout.metadata_dir)
        context.artifacts["checkpoint_manifest"] = copied_path
        artifact_manifest_path = layout.manifest_path
        if artifact_manifest_path.exists():
            try:
                payload = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
                payload["checkpoint_manifest"] = str(copied_path)
                payload.setdefault("core_artifacts", {})["checkpoint_manifest"] = str(copied_path)
                payload.setdefault("directories", {})["checkpoints"] = str(manifest_path.parent)
                artifact_manifest_path.write_text(
                    json.dumps(payload, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                return

    def _notify_stage_event(
        self,
        context: PipelineContext,
        stage: WorkflowStageDefinition,
        event_type: str,
        *,
        error_message: str = "",
    ) -> None:
        manifest = context.metadata.get("checkpointed_execution", {})
        stages = self._stage_definitions()
        stage_order = next(
            (
                index
                for index, candidate in enumerate(stages, start=1)
                if candidate.stage_id == stage.stage_id
            ),
            0,
        )
        self._notify(
            {
                "event_type": event_type,
                "run_id": context.run_id,
                "step_name": stage.label,
                "stage_id": stage.stage_id,
                "critical": stage.critical,
                "step_order": stage_order,
                "total_steps": len(stages),
                "elapsed_seconds": context.metadata.get("run_elapsed_seconds") or 0.0,
                "checkpoint_manifest": manifest.get("checkpoint_manifest", ""),
                "error_message": error_message,
                "stages": self._definition_stage_flow(stage, event_type),
            }
        )

    def _notify_stage_manifest_event(
        self,
        manifest: dict[str, Any],
        stage: WorkflowStageDefinition,
        event_type: str,
        *,
        error_message: str = "",
    ) -> None:
        stages = self._stage_definitions()
        stage_order = next(
            (
                index
                for index, candidate in enumerate(stages, start=1)
                if candidate.stage_id == stage.stage_id
            ),
            0,
        )
        self._notify(
            {
                "event_type": event_type,
                "run_id": manifest.get("run_id", ""),
                "step_name": stage.label,
                "stage_id": stage.stage_id,
                "critical": stage.critical,
                "step_order": stage_order,
                "total_steps": len(stages),
                "elapsed_seconds": self._manifest_elapsed_seconds(manifest),
                "checkpoint_manifest": str(
                    Path(str(manifest.get("checkpoints_dir", "")))
                    / "checkpoint_manifest.json"
                ),
                "error_message": error_message,
                "stages": self._manifest_stage_flow(
                    manifest,
                    active_stage_id=stage.stage_id,
                    active_event_type=event_type,
                ),
            }
        )

    def _notify(self, event: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            return

    def _manifest_elapsed_seconds(self, manifest: dict[str, Any]) -> float:
        created_at_text = str(manifest.get("created_at_utc") or "")
        if not created_at_text:
            return 0.0
        try:
            created_at = datetime.fromisoformat(created_at_text)
        except ValueError:
            return 0.0
        return round((datetime.now(UTC) - created_at).total_seconds(), 6)

    def _manifest_stage_flow(
        self,
        manifest: dict[str, Any],
        *,
        active_stage_id: str | None = None,
        active_event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Builds compact stage status rows for the live UI timeline."""

        rows: list[dict[str, Any]] = []
        for stage in manifest.get("stages", []):
            if not isinstance(stage, dict):
                continue
            status = str(stage.get("status") or "pending")
            stage_id = str(stage.get("stage_id") or "")
            if stage_id == active_stage_id and active_event_type == "stage_started":
                status = "running"
            elif stage_id == active_stage_id and active_event_type == "stage_completed":
                status = "completed"
            elif stage_id == active_stage_id and active_event_type == "stage_failed":
                status = "failed" if bool(stage.get("critical", True)) else "failed_optional"
            rows.append(
                {
                    "order": int(stage.get("order") or len(rows) + 1),
                    "stage_id": stage_id,
                    "label": str(stage.get("label") or stage_id.replace("_", " ").title()),
                    "status": status,
                    "critical": bool(stage.get("critical", True)),
                    "elapsed_seconds": stage.get("elapsed_seconds"),
                    "error_message": str(stage.get("error_message") or ""),
                }
            )
        return rows

    def _definition_stage_flow(
        self,
        active_stage: WorkflowStageDefinition,
        event_type: str,
    ) -> list[dict[str, Any]]:
        """Builds a best-effort flow when no manifest snapshot is available."""

        active_status = {
            "stage_started": "running",
            "stage_completed": "completed",
            "stage_failed": "failed_optional" if not active_stage.critical else "failed",
        }.get(event_type, "pending")
        rows: list[dict[str, Any]] = []
        for order, stage in enumerate(self._stage_definitions(), start=1):
            rows.append(
                {
                    "order": order,
                    "stage_id": stage.stage_id,
                    "label": stage.label,
                    "status": (
                        active_status
                        if stage.stage_id == active_stage.stage_id
                        else "pending"
                    ),
                    "critical": stage.critical,
                    "elapsed_seconds": None,
                    "error_message": "",
                }
            )
        return rows


def run_checkpoint_stage(manifest_path: str | Path, stage_id: str) -> int:
    """CLI entrypoint helper for running one checkpoint stage."""

    manifest_path = Path(manifest_path)
    manifest = read_checkpoint_manifest(manifest_path)
    context = load_context_checkpoint(Path(manifest["latest_context_path"]))
    runner = CheckpointedWorkflowRunner(
        config=context.config,
        progress_callback=None,
        use_subprocess=False,
    )
    try:
        runner.run_stage_from_manifest(manifest_path, stage_id)
    except Exception as exc:
        active_manifest = read_checkpoint_manifest(manifest_path)
        stage = runner._stage_by_id(stage_id)
        stage_row = runner._manifest_stage_row(active_manifest, stage_id)
        if stage_row.get("status") != "failed":
            mark_stage_failed(
                active_manifest,
                stage_id,
                error_message=str(exc),
                elapsed_seconds=0.0,
                optional=not stage.critical,
            )
            write_checkpoint_manifest(manifest_path, active_manifest)
        traceback.print_exc()
        return 0 if not stage.critical else 1
    return 0
