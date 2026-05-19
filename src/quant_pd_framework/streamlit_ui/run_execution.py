"""Run execution helpers for the Streamlit controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import ExecutionMode, LargeDataWorkerMode, QuantModelOrchestrator
from quant_pd_framework.background_jobs import (
    load_background_snapshot,
    queue_background_workflow,
    read_background_manifest,
    request_background_cancel,
    start_background_workflow,
)
from quant_pd_framework.checkpointing import (
    find_next_pending_stage,
    load_context_checkpoint,
    read_checkpoint_manifest,
    save_context_checkpoint,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.stage_runner import CheckpointedWorkflowRunner
from quant_pd_framework.streamlit_ui.data import SelectedInputDataset
from quant_pd_framework.streamlit_ui.theme import render_html

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LargeDataExecutionOverride:
    """Resolved runtime choice for bypassing file-backed Large Data Mode."""

    detected_large_data_mode: bool
    force_standard_requested: bool
    force_standard_confirmed: bool
    reason: str
    source_kind: str
    effective_large_data_mode: bool
    user_override_disabled: bool
    blocked_reason: str = ""


def workflow_spinner_message(execution_mode: str) -> str:
    if execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS.value:
        return "Running feature subset search, comparison visuals, and export package..."
    return "Running the model, diagnostics, visualizations, and export package..."


def execute_workflow(
    *,
    preview_config: Any,
    dataframe: pd.DataFrame,
    selected_input: SelectedInputDataset,
    large_data_mode: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PipelineContext:
    """Runs the workflow through checkpointed stages with subprocess isolation."""

    runner = CheckpointedWorkflowRunner(
        config=preview_config,
        progress_callback=progress_callback,
        use_subprocess=True,
    )
    return runner.run_all(
        build_run_input(
            dataframe=dataframe,
            selected_input=selected_input,
            large_data_mode=large_data_mode,
        )
    )


def execute_workflow_in_process(
    *,
    preview_config: Any,
    dataframe: pd.DataFrame,
    selected_input: SelectedInputDataset,
    large_data_mode: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PipelineContext:
    """Runs the legacy in-process orchestrator for debugging and parity checks."""

    orchestrator = QuantModelOrchestrator(
        config=preview_config,
        progress_callback=progress_callback,
    )
    return orchestrator.run(
        build_run_input(
            dataframe=dataframe,
            selected_input=selected_input,
            large_data_mode=large_data_mode,
        )
    )


def start_checkpointed_workflow(
    *,
    preview_config: Any,
    dataframe: pd.DataFrame,
    selected_input: SelectedInputDataset,
    large_data_mode: bool,
) -> dict[str, Any]:
    """Creates a checkpointed workflow and returns its Streamlit session state."""

    runner = CheckpointedWorkflowRunner(config=preview_config, use_subprocess=True)
    manifest_path = runner.start(
        build_run_input(
            dataframe=dataframe,
            selected_input=selected_input,
            large_data_mode=large_data_mode,
        )
    )
    return {"manifest_path": str(manifest_path), "completed": False}


def start_large_data_background_workflow(
    *,
    preview_config: Any,
    dataframe: pd.DataFrame,
    selected_input: SelectedInputDataset,
    large_data_mode: bool,
) -> dict[str, Any]:
    """Starts a detached Large Data Mode workflow and returns UI state."""

    input_data = build_run_input(
        dataframe=dataframe,
        selected_input=selected_input,
        large_data_mode=large_data_mode,
    )
    worker_mode = _resolve_worker_mode(preview_config)
    if worker_mode == LargeDataWorkerMode.WORKER_SERVICE:
        manifest_path = queue_background_workflow(
            config=preview_config,
            input_data=input_data,
            queue_dir=preview_config.artifacts.output_root / "_job_queue",
        )
    else:
        manifest_path = start_background_workflow(config=preview_config, input_data=input_data)
    manifest = read_background_manifest(manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "status": manifest.status,
        "run_id": manifest.run_id,
        "completed": manifest.status == "completed",
        "dispatch_mode": manifest.dispatch_mode,
    }


def poll_large_data_background_workflow(
    background_state: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reads the background job manifest and returns a completed snapshot when available."""

    manifest_path = Path(str(background_state["manifest_path"]))
    manifest = read_background_manifest(manifest_path)
    snapshot = load_background_snapshot(manifest) if manifest.status == "completed" else None
    updated_state = {
        **background_state,
        "status": manifest.status,
        "run_id": manifest.run_id,
        "completed": manifest.status == "completed",
        "current_stage": manifest.current_stage,
        "progress": manifest.progress,
        "error_message": manifest.error_message,
        "manifest": manifest.to_dict(),
        "dispatch_mode": manifest.dispatch_mode,
    }
    return snapshot, updated_state


def cancel_large_data_background_workflow(background_state: dict[str, Any]) -> None:
    """Requests background cancellation at the next safe worker boundary."""

    request_background_cancel(Path(str(background_state["manifest_path"])))


def run_next_checkpoint_stage(
    *,
    preview_config: Any,
    checkpoint_state: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[PipelineContext | None, dict[str, Any]]:
    """Runs the next pending checkpoint stage."""

    manifest_path = Path(str(checkpoint_state["manifest_path"]))
    runner = CheckpointedWorkflowRunner(
        config=preview_config,
        progress_callback=progress_callback,
        use_subprocess=True,
    )
    context = runner.run_next(manifest_path)
    manifest = read_checkpoint_manifest(manifest_path)
    completed = find_next_pending_stage(manifest) is None
    updated_state = {
        **checkpoint_state,
        "completed": completed,
        "run_id": manifest.get("run_id", ""),
        "manifest_status": manifest.get("status", ""),
    }
    if completed and context is None:
        latest_context_path_text = str(manifest.get("latest_context_path") or "")
        latest_context_path = Path(latest_context_path_text) if latest_context_path_text else None
        if latest_context_path is not None and latest_context_path.exists():
            context = load_context_checkpoint(latest_context_path)
    if completed and context is not None:
        runner._finalize_run_metadata(context, manifest_path)
        latest_context_path = Path(read_checkpoint_manifest(manifest_path)["latest_context_path"])
        save_context_checkpoint(context, latest_context_path)
        runner._apply_checkpoint_retention(manifest_path, keep_latest=False)
        runner._copy_checkpoint_manifest_to_metadata(context, manifest_path)
    return context if completed else None, updated_state


def format_elapsed_seconds(seconds: float | int | None) -> str:
    total_seconds = max(0, int(round(float(seconds or 0))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def render_runtime_status(
    event: dict[str, Any],
    *,
    status_placeholder: st.delta_generator.DeltaGenerator,
    progress_placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    """Renders the current synchronous run state from orchestrator progress events."""

    event_type = str(event.get("event_type", "run_started"))
    total_steps = int(event.get("total_steps") or 0)
    step_order = int(event.get("step_order") or 0)
    elapsed_seconds = float(event.get("elapsed_seconds") or 0.0)
    step_name = str(event.get("step_name") or "Preparing workflow")
    status_label = _runtime_status_label(
        event_type,
        critical=bool(event.get("critical", True)),
    )
    progress_value = _runtime_progress_value(event_type, step_order, total_steps)
    stage_label = _format_step_name(step_name)

    with status_placeholder.container():
        st.markdown("#### Run Status")
        status_columns = st.columns(4)
        status_columns[0].metric("Status", status_label)
        status_columns[1].metric("Elapsed", format_elapsed_seconds(elapsed_seconds))
        status_columns[2].metric(
            "Current Stage",
            stage_label if event_type != "run_completed" else "Complete",
        )
        status_columns[3].metric(
            "Progress",
            f"{min(step_order, total_steps)}/{total_steps}" if total_steps else "Starting",
        )

        if event_type in {"step_failed", "stage_failed"} and event.get("error_message"):
            st.error(f"Failed during {stage_label}: {event['error_message']}")
        elif event_type == "run_completed":
            st.success(
                "Workflow completed in "
                f"{format_elapsed_seconds(event.get('elapsed_seconds'))}."
            )
        elif event_type == "stage_completed":
            st.info(f"Checkpoint completed: {stage_label}.")
        else:
            st.info("Workflow is running. The status updates as each major pipeline step finishes.")
        _render_stage_flow(event)

    progress_text = (
        f"{status_label}: {stage_label} ({format_elapsed_seconds(elapsed_seconds)} elapsed)"
        if event_type != "run_completed"
        else f"Completed in {format_elapsed_seconds(elapsed_seconds)}"
    )
    progress_placeholder.progress(progress_value, text=progress_text)


def render_background_job_status(background_state: dict[str, Any]) -> None:
    """Renders the latest detached large-data job state."""

    manifest_payload = background_state.get("manifest") or {}
    status = str(background_state.get("status") or manifest_payload.get("status") or "unknown")
    current_stage = str(
        background_state.get("current_stage") or manifest_payload.get("current_stage") or ""
    )
    progress = str(background_state.get("progress") or manifest_payload.get("progress") or "")
    with st.container():
        st.markdown("#### Background Large-Data Job")
        status_columns = st.columns(4)
        status_columns[0].metric("Status", status.replace("_", " ").title())
        status_columns[1].metric("Run ID", background_state.get("run_id") or "Pending")
        status_columns[2].metric("Current Stage", current_stage or "Starting")
        status_columns[3].metric("Progress", progress or "Queued")
        dispatch_mode = str(
            background_state.get("dispatch_mode") or manifest_payload.get("dispatch_mode") or ""
        )
        if dispatch_mode:
            st.caption(f"Execution dispatch: `{dispatch_mode}`")
        manifest_path = background_state.get("manifest_path")
        if manifest_path:
            st.caption(f"Job manifest: `{manifest_path}`")
        if status == "failed" and background_state.get("error_message"):
            st.error(str(background_state["error_message"]))


def _runtime_status_label(event_type: str, *, critical: bool = True) -> str:
    if event_type == "run_completed":
        return "Completed"
    if event_type == "stage_completed":
        return "Checkpointed"
    if event_type == "stage_failed":
        return "Failed" if critical else "Warning"
    if event_type == "step_failed":
        return "Failed"
    if event_type == "step_completed":
        return "Advancing"
    return "Running"


def _runtime_progress_value(event_type: str, step_order: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 0.0
    if event_type == "run_completed":
        return 1.0
    if event_type == "step_completed":
        return min(1.0, max(0.0, step_order / total_steps))
    if event_type == "step_started":
        return min(1.0, max(0.0, (step_order - 1) / total_steps))
    if event_type == "stage_started":
        return min(1.0, max(0.0, (step_order - 1) / total_steps))
    if event_type == "stage_completed":
        return min(1.0, max(0.0, step_order / total_steps)) if step_order else 0.0
    if event_type == "stage_failed":
        return min(1.0, max(0.0, (step_order - 1) / total_steps)) if step_order else 0.0
    return 0.0


def _format_step_name(step_name: str) -> str:
    return step_name.replace("_", " ").title()


def _render_stage_flow(event: dict[str, Any]) -> None:
    stages = event.get("stages")
    if not isinstance(stages, list) or not stages:
        return
    stage_cards = "".join(_stage_flow_card(stage) for stage in stages if isinstance(stage, dict))
    if not stage_cards:
        return
    render_html(
        '<section class="checkpoint-flow">'
        '<div class="checkpoint-flow__header">'
        '<span>Checkpoint Flow</span>'
        '<small>Major stages for the current workflow run</small>'
        '</div>'
        f'<div class="checkpoint-flow__grid">{stage_cards}</div>'
        '</section>'
    )


def _stage_flow_card(stage: dict[str, Any]) -> str:
    raw_status = str(stage.get("status") or "pending")
    status_class = _stage_status_class(raw_status)
    status_label = _stage_status_label(raw_status)
    order = escape(str(stage.get("order") or ""))
    label = escape(str(stage.get("label") or "Unnamed stage"))
    critical = bool(stage.get("critical", True))
    optional_badge = "" if critical else '<span class="checkpoint-flow__optional">Optional</span>'
    return (
        f'<article class="checkpoint-flow__card is-{status_class}">'
        f'<div class="checkpoint-flow__order">{order}</div>'
        '<div class="checkpoint-flow__body">'
        f'<strong>{label}</strong>'
        f'<span>{status_label}{optional_badge}</span>'
        '</div>'
        '</article>'
    )


def _stage_status_class(status: str) -> str:
    return {
        "completed": "completed",
        "running": "running",
        "failed": "failed",
        "failed_optional": "warning",
        "skipped": "warning",
        "pending": "pending",
        "initialized": "pending",
    }.get(status, "pending")


def _stage_status_label(status: str) -> str:
    return {
        "completed": "Completed",
        "running": "Running",
        "failed": "Failed",
        "failed_optional": "Optional failed",
        "skipped": "Skipped",
        "pending": "Pending",
        "initialized": "Pending",
    }.get(status, status.replace("_", " ").title())


def build_run_input(
    *,
    dataframe: pd.DataFrame,
    selected_input: SelectedInputDataset,
    large_data_mode: bool,
) -> Any:
    if large_data_mode and selected_input.dataset_handle is not None:
        return selected_input.dataset_handle
    file_input = _selected_file_input_path(selected_input)
    if file_input is not None:
        return file_input
    return dataframe


def resolve_large_data_execution_override(
    *,
    selected_input: SelectedInputDataset,
    detected_large_data_mode: bool,
    force_standard_requested: bool,
    force_standard_confirmed: bool,
    reason: str,
) -> LargeDataExecutionOverride:
    """Determines whether a Step 3 standard-execution override is safe to apply."""

    detected = bool(detected_large_data_mode)
    requested = bool(detected and force_standard_requested)
    confirmed = bool(requested and force_standard_confirmed)
    normalized_reason = str(reason or "").strip()
    source_kind = selected_source_kind(selected_input)
    blocked_reason = ""
    effective_large_data_mode = detected
    user_override_disabled = False

    if requested:
        if source_kind == "s3":
            blocked_reason = (
                "S3 inputs require Large Data Mode because standard in-memory execution "
                "does not safely load full S3 objects into pandas."
            )
        elif not confirmed:
            blocked_reason = (
                "Confirm the standard in-memory execution override before running."
            )
        elif selected_input.dataset_handle is not None and (
            _selected_file_input_path(selected_input) is None
        ):
            blocked_reason = (
                "This large-data source only has a preview dataframe available in the UI. "
                "Select a Data_Load or local-path file so the full file can be passed to "
                "standard execution."
            )
        else:
            effective_large_data_mode = False
            user_override_disabled = True

    return LargeDataExecutionOverride(
        detected_large_data_mode=detected,
        force_standard_requested=requested,
        force_standard_confirmed=confirmed,
        reason=normalized_reason,
        source_kind=source_kind,
        effective_large_data_mode=effective_large_data_mode,
        user_override_disabled=user_override_disabled,
        blocked_reason=blocked_reason,
    )


def build_config_for_large_data_execution_override(
    preview_config: Any,
    override: LargeDataExecutionOverride,
) -> Any:
    """Returns a run-only config with large-data override evidence embedded."""

    if preview_config is None:
        return None
    effective_mode = (
        "large_data"
        if override.effective_large_data_mode
        else "standard_in_memory_forced"
        if override.user_override_disabled
        else "standard_in_memory"
    )
    performance = dataclass_replace(
        preview_config.performance,
        large_data_mode=override.effective_large_data_mode,
        large_data_override_reason=(
            override.reason
            if override.user_override_disabled
            else preview_config.performance.large_data_override_reason
        ),
        large_data_auto_detected=override.detected_large_data_mode,
        large_data_user_override_disabled=override.user_override_disabled,
        large_data_standard_execution_override_reason=override.reason,
        large_data_effective_mode=effective_mode,
        large_data_source_kind=override.source_kind,
    )
    return dataclass_replace(preview_config, performance=performance)


def selected_source_kind(selected_input: SelectedInputDataset) -> str:
    metadata = selected_input.metadata or {}
    return str(metadata.get("source_kind") or "").strip().lower()


def _selected_file_input_path(selected_input: SelectedInputDataset) -> Path | None:
    metadata = selected_input.metadata or {}
    if metadata.get("source_kind") not in {"data_load", "local_path"}:
        return None
    relative_path = str(metadata.get("relative_path") or "")
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate if candidate.exists() else None


def build_execution_plan_cards(
    *,
    preview_config: Any,
    data_source_label: str,
    large_data_mode: bool,
    large_data_user_override_disabled: bool = False,
) -> list[dict[str, str]]:
    """Summarizes what the next run will do in user-facing terms."""

    if preview_config is None:
        return []

    fit_strategy = (
        "Sample fit / chunked full score"
        if large_data_mode
        else "Checkpointed staged workflow"
    )
    if preview_config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
        fit_strategy = "Feature subset comparison"
    elif preview_config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
        fit_strategy = "Existing model scoring"

    data_mode = (
        "Standard in-memory execution forced by user override"
        if large_data_user_override_disabled
        else "Large Data Mode"
        if large_data_mode
        else "Standard in-memory execution"
    )

    return [
        {
            "label": "Execution",
            "value": preview_config.execution.mode.value.replace("_", " ").title(),
        },
        {
            "label": "Model",
            "value": preview_config.model.model_type.value.replace("_", " ").title(),
        },
        {
            "label": "Data",
            "value": data_source_label or "No dataset selected",
        },
        {
            "label": "Fit Strategy",
            "value": fit_strategy,
        },
        {
            "label": "Data Mode",
            "value": data_mode,
        },
        {
            "label": "Export Profile",
            "value": preview_config.artifacts.export_profile.value.title(),
        },
    ]


def _resolve_worker_mode(preview_config: Any) -> LargeDataWorkerMode:
    configured = preview_config.performance.large_data_worker_mode
    mode = (
        configured
        if isinstance(configured, LargeDataWorkerMode)
        else LargeDataWorkerMode(configured)
    )
    if mode != LargeDataWorkerMode.AUTO:
        return mode
    queue_dir = preview_config.artifacts.output_root / "_job_queue"
    heartbeat_path = queue_dir / "worker_heartbeat.json"
    try:
        if heartbeat_path.exists():
            age_seconds = pd.Timestamp.utcnow().timestamp() - heartbeat_path.stat().st_mtime
            if age_seconds <= 120:
                return LargeDataWorkerMode.WORKER_SERVICE
    except OSError:
        pass
    return LargeDataWorkerMode.DETACHED_PROCESS
