"""Run execution helpers for the Streamlit controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import ExecutionMode, QuantModelOrchestrator
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.streamlit_ui.data import SelectedInputDataset


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
    """Runs the orchestrator with the correct input object for the selected mode."""

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
    status_label = _runtime_status_label(event_type)
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

        if event_type == "step_failed" and event.get("error_message"):
            st.error(f"Failed during {stage_label}: {event['error_message']}")
        elif event_type == "run_completed":
            st.success(
                "Workflow completed in "
                f"{format_elapsed_seconds(event.get('elapsed_seconds'))}."
            )
        else:
            st.info("Workflow is running. The status updates as each major pipeline step finishes.")

    progress_text = (
        f"{status_label}: {stage_label} ({format_elapsed_seconds(elapsed_seconds)} elapsed)"
        if event_type != "run_completed"
        else f"Completed in {format_elapsed_seconds(elapsed_seconds)}"
    )
    progress_placeholder.progress(progress_value, text=progress_text)


def _runtime_status_label(event_type: str) -> str:
    if event_type == "run_completed":
        return "Completed"
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
    return 0.0


def _format_step_name(step_name: str) -> str:
    return step_name.replace("_", " ").title()


def build_run_input(
    *,
    dataframe: pd.DataFrame,
    selected_input: SelectedInputDataset,
    large_data_mode: bool,
) -> Any:
    if large_data_mode and selected_input.dataset_handle is not None:
        return selected_input.dataset_handle
    return dataframe


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
        else "In-memory pandas workflow"
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
