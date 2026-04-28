"""Run execution helpers for the Streamlit controller."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import ExecutionMode, QuantModelOrchestrator
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


def _selected_file_input_path(selected_input: SelectedInputDataset) -> Path | None:
    metadata = selected_input.metadata or {}
    if metadata.get("source_kind") != "data_load":
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
            "label": "Export Profile",
            "value": preview_config.artifacts.export_profile.value.title(),
        },
    ]
