"""Top-level orchestration for the end-to-end quant modeling flow."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .base import BasePipelineStep
from .config import ExecutionMode, FrameworkConfig
from .config_io import load_framework_config
from .context import PipelineContext
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


class QuantModelOrchestrator:
    """
    Runs the full modeling lifecycle from raw dataframe/file to exported artifacts.

    The default orchestration is intentionally explicit so each primary quant step
    lives in its own class and can later be swapped for a richer implementation.
    """

    def __init__(
        self,
        config: FrameworkConfig,
        steps: list[BasePipelineStep] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = self._resolve_execution_config(config)
        self.config.validate()
        self.steps = steps or self._build_default_steps()
        self.progress_callback = progress_callback

    def _resolve_execution_config(self, config: FrameworkConfig) -> FrameworkConfig:
        resolved = deepcopy(config)
        execution = resolved.execution
        if (
            execution.mode != ExecutionMode.SCORE_EXISTING_MODEL
            or execution.existing_config_path is None
        ):
            return resolved

        base_config = load_framework_config(execution.existing_config_path)
        base_config.preset_name = resolved.preset_name or base_config.preset_name
        base_config.execution = execution
        base_config.comparison = resolved.comparison
        base_config.feature_policy = resolved.feature_policy
        base_config.feature_dictionary = resolved.feature_dictionary
        base_config.advanced_imputation = resolved.advanced_imputation
        base_config.transformations = resolved.transformations
        base_config.manual_review = resolved.manual_review
        base_config.suitability_checks = resolved.suitability_checks
        base_config.workflow_guardrails = resolved.workflow_guardrails
        base_config.explainability = resolved.explainability
        base_config.calibration = resolved.calibration
        base_config.scorecard = resolved.scorecard
        base_config.scorecard_workbench = resolved.scorecard_workbench
        base_config.imputation_sensitivity = resolved.imputation_sensitivity
        base_config.variable_selection = resolved.variable_selection
        base_config.documentation = resolved.documentation
        base_config.regulatory_reporting = resolved.regulatory_reporting
        base_config.scenario_testing = resolved.scenario_testing
        base_config.diagnostics = resolved.diagnostics
        base_config.distribution_diagnostics = resolved.distribution_diagnostics
        base_config.residual_diagnostics = resolved.residual_diagnostics
        base_config.outlier_diagnostics = resolved.outlier_diagnostics
        base_config.dependency_diagnostics = resolved.dependency_diagnostics
        base_config.time_series_diagnostics = resolved.time_series_diagnostics
        base_config.structural_breaks = resolved.structural_breaks
        base_config.feature_workbench = resolved.feature_workbench
        base_config.preset_recommendations = resolved.preset_recommendations
        base_config.credit_risk = resolved.credit_risk
        base_config.reproducibility = resolved.reproducibility
        base_config.artifacts = resolved.artifacts
        return base_config

    def _build_default_steps(self) -> list[BasePipelineStep]:
        if self.config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            return [
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
                FeatureSubsetSearchStep(),
                ArtifactExportStep(),
            ]
        return [
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
            ModelTrainingStep(),
            EvaluationStep(),
            ModelComparisonStep(),
            BacktestStep(),
            DiagnosticsStep(),
            CrossValidationStep(),
            LargeDataFullScoringStep(),
            ArtifactExportStep(),
        ]

    def describe_steps(self) -> list[dict[str, str | int]]:
        """Returns the exact ordered step stack used for the run."""

        return [
            {
                "order": index,
                "name": step.name,
                "class_name": step.__class__.__name__,
                "module": step.__class__.__module__,
            }
            for index, step in enumerate(self.steps, start=1)
        ]

    def run_context(
        self,
        context: PipelineContext,
        *,
        steps: list[BasePipelineStep] | None = None,
        run_started: float | None = None,
        step_offset: int = 0,
        total_steps: int | None = None,
    ) -> PipelineContext:
        """Executes a set of steps against an existing context.

        This is used by checkpointed execution, where a stage reloads the
        context from disk, runs only its assigned steps, then saves the result
        before the next stage starts in a fresh process.
        """

        active_steps = steps or self.steps
        total_step_count = total_steps or len(active_steps)
        run_started_at = context.metadata.get("run_started_at_utc", "")
        run_started = perf_counter() if run_started is None else run_started

        for local_step_index, step in enumerate(active_steps, start=1):
            step_index = step_offset + local_step_index
            before = self._debug_snapshot(context)
            started_at = datetime.now(UTC)
            started = perf_counter()
            status = "completed"
            error_message = ""
            error_traceback = ""
            self._notify_progress(
                {
                    "event_type": "step_started",
                    "run_id": context.run_id,
                    "started_at_utc": started_at.isoformat(),
                    "run_started_at_utc": run_started_at,
                    "step_order": step_index,
                    "total_steps": total_step_count,
                    "step_name": step.name,
                    "class_name": step.__class__.__name__,
                    "step_started_at_utc": started_at.isoformat(),
                    "elapsed_seconds": round(perf_counter() - run_started, 6),
                }
            )
            try:
                context = step(context)
            except Exception as exc:
                status = "failed"
                error_message = str(exc)
                error_traceback = traceback.format_exc()
                raise
            finally:
                elapsed_seconds = perf_counter() - started
                after = self._debug_snapshot(context)
                context.add_debug_record(
                    {
                        "order": len(context.debug_trace) + 1,
                        "step_name": step.name,
                        "class_name": step.__class__.__name__,
                        "module": step.__class__.__module__,
                        "status": status,
                        "started_at_utc": started_at.isoformat(),
                        "elapsed_seconds": round(elapsed_seconds, 6),
                        "before": before,
                        "after": after,
                        "error_message": error_message,
                        "error_traceback": error_traceback,
                    }
                )
                self._notify_progress(
                    {
                        "event_type": "step_completed" if status == "completed" else "step_failed",
                        "run_id": context.run_id,
                        "started_at_utc": started_at.isoformat(),
                        "run_started_at_utc": run_started_at,
                        "step_order": step_index,
                        "total_steps": total_step_count,
                        "step_name": step.name,
                        "class_name": step.__class__.__name__,
                        "status": status,
                        "step_elapsed_seconds": round(elapsed_seconds, 6),
                        "elapsed_seconds": round(perf_counter() - run_started, 6),
                        "error_message": error_message,
                    }
                )
                if step.name == "artifact_export":
                    self._refresh_exported_debug_trace(context)

        return context

    def run(self, data: pd.DataFrame | str | Path) -> PipelineContext:
        """Executes each step in sequence and returns the populated context."""

        context = PipelineContext(
            config=self.config,
            run_id=self._build_run_id(),
            raw_input=data,
        )
        run_started_at = datetime.now(UTC)
        run_started = perf_counter()
        context.metadata["execution_mode"] = self.config.execution.mode.value
        context.metadata["step_manifest"] = self.describe_steps()
        context.metadata["run_started_at_utc"] = run_started_at.isoformat()
        self._notify_progress(
            {
                "event_type": "run_started",
                "run_id": context.run_id,
                "started_at_utc": run_started_at.isoformat(),
                "elapsed_seconds": 0.0,
                "total_steps": len(self.steps),
            }
        )

        context = self.run_context(
            context,
            run_started=run_started,
            total_steps=len(self.steps),
        )

        run_completed_at = datetime.now(UTC)
        run_elapsed_seconds = round(perf_counter() - run_started, 6)
        context.metadata["run_completed_at_utc"] = run_completed_at.isoformat()
        context.metadata["run_elapsed_seconds"] = run_elapsed_seconds
        context.metadata["run_status"] = "completed"
        self._refresh_exported_debug_trace(context)
        self._notify_progress(
            {
                "event_type": "run_completed",
                "run_id": context.run_id,
                "started_at_utc": run_started_at.isoformat(),
                "completed_at_utc": run_completed_at.isoformat(),
                "elapsed_seconds": run_elapsed_seconds,
                "total_steps": len(self.steps),
            }
        )
        return context

    def _notify_progress(self, event: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            # Progress rendering must never change the modeling result.
            return

    def _build_run_id(self) -> str:
        """Builds a readable, filesystem-safe identifier for artifact folders."""

        return datetime.now(UTC).strftime("run_%Y-%m-%d_%H-%M-%S_UTC")

    def _debug_snapshot(self, context: PipelineContext) -> dict[str, Any]:
        working_shape = self._frame_shape(context.working_data)
        raw_shape = self._frame_shape(context.raw_data)
        split_rows = {
            split_name: int(frame.shape[0])
            for split_name, frame in context.split_frames.items()
        }
        prediction_rows = {
            split_name: int(frame.shape[0])
            for split_name, frame in context.predictions.items()
        }
        memory_profile = self._memory_profile(context)
        return {
            "raw_rows": raw_shape[0],
            "raw_columns": raw_shape[1],
            "working_rows": working_shape[0],
            "working_columns": working_shape[1],
            "feature_count": len(context.feature_columns),
            "numeric_feature_count": len(context.numeric_features),
            "categorical_feature_count": len(context.categorical_features),
            "split_rows": split_rows,
            "prediction_rows": prediction_rows,
            "diagnostic_table_count": len(context.diagnostics_tables),
            "visualization_count": len(context.visualizations),
            "warning_count": len(context.warnings),
            "event_count": len(context.events),
            "artifact_count": len(context.artifacts),
            "model_type": context.config.model.model_type.value,
            "execution_mode": context.config.execution.mode.value,
            **memory_profile,
        }

    def _frame_shape(self, dataframe: pd.DataFrame | None) -> tuple[int | None, int | None]:
        if dataframe is None:
            return None, None
        return int(dataframe.shape[0]), int(dataframe.shape[1])

    def _memory_profile(self, context: PipelineContext) -> dict[str, int | float | None]:
        performance = context.config.performance
        if not performance.capture_memory_profile:
            return {}
        deep = bool(performance.deep_memory_profile)
        raw_memory = self._frame_memory_bytes(context.raw_data, deep=deep)
        working_memory = self._frame_memory_bytes(context.working_data, deep=deep)
        split_memory = sum(
            self._frame_memory_bytes(frame, deep=deep) or 0
            for frame in context.split_frames.values()
        )
        prediction_memory = sum(
            self._frame_memory_bytes(frame, deep=deep) or 0
            for frame in context.predictions.values()
        )
        diagnostics_memory = sum(
            self._frame_memory_bytes(frame, deep=deep) or 0
            for frame in context.diagnostics_tables.values()
        )
        total_tracked = raw_memory + working_memory + split_memory + prediction_memory
        return {
            "raw_memory_bytes": raw_memory,
            "working_memory_bytes": working_memory,
            "split_memory_bytes": split_memory,
            "prediction_memory_bytes": prediction_memory,
            "diagnostics_table_memory_bytes": diagnostics_memory,
            "tracked_dataframe_memory_bytes": total_tracked,
            "tracked_dataframe_memory_gb": round(total_tracked / (1024**3), 6),
        }

    def _frame_memory_bytes(self, dataframe: pd.DataFrame | None, *, deep: bool) -> int:
        if dataframe is None:
            return 0
        try:
            return int(dataframe.memory_usage(deep=deep).sum())
        except Exception:
            return 0

    def _refresh_exported_debug_trace(self, context: PipelineContext) -> None:
        trace_path = context.artifacts.get("run_debug_trace")
        if trace_path is None:
            return
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
            payload["run_started_at_utc"] = context.metadata.get(
                "run_started_at_utc",
                payload.get("run_started_at_utc", ""),
            )
            payload["run_completed_at_utc"] = context.metadata.get(
                "run_completed_at_utc",
                payload.get("run_completed_at_utc", ""),
            )
            payload["step_count"] = len(context.debug_trace)
            payload["steps"] = context.debug_trace
            payload["summary"] = {
                **dict(payload.get("summary", {})),
                "total_step_seconds": round(
                    sum(float(row.get("elapsed_seconds", 0.0)) for row in context.debug_trace),
                    6,
                ),
                "total_run_seconds": context.metadata.get("run_elapsed_seconds"),
                "diagnostic_table_count": len(context.diagnostics_tables),
                "visualization_count": len(context.visualizations),
                "warning_count": len(context.warnings),
                "artifact_count": len(context.artifacts),
            }
            trace_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            context.warn(f"Could not refresh run debug trace after export timing: {exc}")
