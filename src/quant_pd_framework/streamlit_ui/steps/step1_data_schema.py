"""Step 1 Dataset & Schema render helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework.gui_support import (
    build_column_editor_frame,
    build_feature_dictionary_editor_frame,
    build_feature_review_editor_frame,
    build_scorecard_override_editor_frame,
    build_subset_search_feature_options,
    build_transformation_editor_frame,
)
from quant_pd_framework.streamlit_ui.data import (
    build_editor_key,
    render_input_performance_notice,
    select_input_dataframe,
)
from quant_pd_framework.streamlit_ui.enterprise_workflow import (
    build_workflow_step_states,
    render_workflow_status_strip,
)
from quant_pd_framework.streamlit_ui.results import render_run_registry_panel
from quant_pd_framework.streamlit_ui.state import (
    WorkspaceState,
    WorkspaceStateKeys,
    get_last_run_snapshot,
    get_or_initialize_frame,
)
from quant_pd_framework.streamlit_ui.workspace import render_builder_workspace


@dataclass(slots=True)
class DatasetWorkspace:
    """Step 1 workspace frames and derived column options."""

    editor_key: str
    keys: WorkspaceStateKeys
    schema_frame: pd.DataFrame
    feature_dictionary_frame: pd.DataFrame
    transformation_frame: pd.DataFrame
    feature_review_frame: pd.DataFrame
    scorecard_override_frame: pd.DataFrame
    subset_search_feature_options: list[str]
    categorical_like_columns: list[str]


def select_dataset_input() -> Any:
    """Renders the Step 1 data-source selector."""

    return select_input_dataframe()


def render_no_dataset_placeholders(
    *,
    workflow_status_container: Any,
    data_tab: Any,
    configuration_tab: Any,
    readiness_tab: Any,
    results_tab: Any,
    decision_tab: Any,
) -> None:
    """Renders all tabs when no dataset has been selected."""

    with workflow_status_container:
        render_workflow_status_strip(
            build_workflow_step_states(
                dataframe_loaded=False,
                preview_config=None,
                preview_error=None,
                preview_findings=[],
                last_run_snapshot=get_last_run_snapshot(),
                current_config=None,
            )
        )
    with data_tab:
        st.info(
            "Select a Data_Load file, upload a CSV/Excel/Parquet file, or use the "
            "bundled sample dataset to begin."
        )
    with configuration_tab:
        st.info("Complete Step 1 before configuring the model workflow.")
    with readiness_tab:
        st.info("Readiness checks appear after a dataset and schema are available.")
    with results_tab:
        st.info("Run a valid workflow from Step 3 to populate results and artifacts.")
        render_run_registry_panel(Path("artifacts"))
    with decision_tab:
        st.info("Complete a run before reviewing the Step 5 decision summary.")


def render_input_notice(*, data_tab: Any, metadata: dict[str, Any]) -> None:
    """Renders source-specific performance guidance after a dataset is loaded."""

    with data_tab:
        render_input_performance_notice(metadata)


def initialize_dataset_workspace(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
) -> DatasetWorkspace:
    """Initializes Step 1 editor frames in session state."""

    editor_key = build_editor_key(dataframe, data_source_label)
    workspace_keys = WorkspaceStateKeys.from_editor_key(editor_key)
    schema_frame = get_or_initialize_frame(
        workspace_keys.schema_frame,
        lambda: build_column_editor_frame(
            dataframe,
            use_column_name_hints=data_source_label == "bundled_sample",
        ),
    )
    feature_dictionary_frame = get_or_initialize_frame(
        workspace_keys.feature_dictionary_frame,
        lambda: build_feature_dictionary_editor_frame(dataframe),
    )
    transformation_frame = get_or_initialize_frame(
        workspace_keys.transformation_frame,
        build_transformation_editor_frame,
    )
    feature_review_frame = get_or_initialize_frame(
        workspace_keys.feature_review_frame,
        build_feature_review_editor_frame,
    )
    scorecard_override_frame = get_or_initialize_frame(
        workspace_keys.scorecard_override_frame,
        build_scorecard_override_editor_frame,
    )
    subset_search_feature_options = build_subset_search_feature_options(
        schema_frame,
        transformation_frame,
    )
    categorical_like_columns = dataframe.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    return DatasetWorkspace(
        editor_key=editor_key,
        keys=workspace_keys,
        schema_frame=schema_frame,
        feature_dictionary_frame=feature_dictionary_frame,
        transformation_frame=transformation_frame,
        feature_review_frame=feature_review_frame,
        scorecard_override_frame=scorecard_override_frame,
        subset_search_feature_options=subset_search_feature_options,
        categorical_like_columns=categorical_like_columns,
    )


def render_data_schema_workspace(
    *,
    data_tab: Any,
    data_source_label: str,
    dataframe: pd.DataFrame,
    workspace: DatasetWorkspace,
    preset_transformations: Any,
    advanced_workspace: bool,
    target_mode: str,
    model_type: str,
    data_structure: str,
) -> dict[str, Any]:
    """Renders the Step 1 builder workspace and returns edited frames."""

    with data_tab:
        return render_builder_workspace(
            data_source_label=data_source_label,
            dataframe=dataframe,
            workspace_state=WorkspaceState(
                keys=workspace.keys,
                schema_frame=workspace.schema_frame,
                feature_dictionary_frame=workspace.feature_dictionary_frame,
                transformation_frame=workspace.transformation_frame,
                feature_review_frame=workspace.feature_review_frame,
                scorecard_override_frame=workspace.scorecard_override_frame,
            ),
            preset_transformations=preset_transformations,
            advanced_workspace=advanced_workspace,
            target_mode=target_mode,
            model_type=model_type,
            data_structure=data_structure,
        )

