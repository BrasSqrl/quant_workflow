"""Run execution helpers for the Streamlit controller."""

from __future__ import annotations

from typing import Any

import pandas as pd

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
) -> PipelineContext:
    """Runs the orchestrator with the correct input object for the selected mode."""

    orchestrator = QuantModelOrchestrator(config=preview_config)
    return orchestrator.run(
        build_run_input(
            dataframe=dataframe,
            selected_input=selected_input,
            large_data_mode=large_data_mode,
        )
    )


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
