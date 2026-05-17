"""Step 3 Readiness Check & Run render helpers."""

from __future__ import annotations

import traceback
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import ExecutionMode
from quant_pd_framework.streamlit_ui.audit import record_gui_audit_event
from quant_pd_framework.streamlit_ui.enterprise_workflow import (
    build_preflight_summary,
    build_resource_readiness_check,
    render_issue_center,
    render_preflight_summary,
    render_resource_readiness_check,
)
from quant_pd_framework.streamlit_ui.error_guidance import classify_workflow_exception
from quant_pd_framework.streamlit_ui.results import render_workflow_readiness
from quant_pd_framework.streamlit_ui.run_execution import (
    build_execution_plan_cards,
    cancel_large_data_background_workflow,
    execute_workflow,
    poll_large_data_background_workflow,
    render_background_job_status,
    render_runtime_status,
    run_next_checkpoint_stage,
    start_checkpointed_workflow,
    start_large_data_background_workflow,
    workflow_spinner_message,
)
from quant_pd_framework.streamlit_ui.state import (
    build_run_snapshot,
    set_last_run_snapshot,
)
from quant_pd_framework.streamlit_ui.workflow_feedback import (
    render_execution_plan,
    render_run_failure,
    render_run_success,
)


def render_readiness_check_and_run(
    *,
    readiness_tab: Any,
    preview_config: Any,
    preview_findings: list[Any],
    preview_error: str | None,
    readiness_issues: list[Any],
    dataframe: pd.DataFrame,
    data_source_label: str,
    edited_schema: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    selected_input: Any,
    large_data_mode: bool,
    execution_mode: str,
    checkpoint_state_key: str,
    background_state_key: str,
) -> None:
    """Renders Step 3 and executes the selected workflow style when clicked."""

    run_button_label = (
        "Run Feature Subset Search"
        if execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS.value
        else "Run Quant Model Workflow"
    )
    with readiness_tab:
        render_workflow_readiness(
            preview_config=preview_config,
            preview_findings=preview_findings,
            preview_error=preview_error,
        )
        render_issue_center(readiness_issues)
        if preview_config is not None:
            preflight_cards, preflight_details = build_preflight_summary(
                dataframe=dataframe,
                data_source_label=data_source_label,
                preview_config=preview_config,
                edited_schema=edited_schema,
                transformation_frame=transformation_frame,
            )
            render_preflight_summary(cards=preflight_cards, details=preflight_details)
            resource_cards, resource_details = build_resource_readiness_check(
                dataframe=dataframe,
                preview_config=preview_config,
                large_data_mode=large_data_mode,
            )
            render_resource_readiness_check(
                cards=resource_cards,
                details=resource_details,
            )
        render_execution_plan(
            build_execution_plan_cards(
                preview_config=preview_config,
                data_source_label=data_source_label,
                large_data_mode=large_data_mode,
            ),
            large_data_mode=large_data_mode,
        )
        workflow_run_style = st.radio(
            "Workflow run style",
            options=["full", "step_by_step"],
            index=0,
            horizontal=True,
            format_func={
                "full": "Run full workflow",
                "step_by_step": "Run checkpointed step-by-step",
            }.get,
            help=(
                "Full workflow uses the same checkpoint engine automatically. "
                "Step-by-step mode runs one checkpoint stage per click so a failed "
                "diagnostic group can be reviewed without refitting the model."
            ),
        )
        checkpoint_state = st.session_state.get(checkpoint_state_key)
        background_state = st.session_state.get(background_state_key)
        background_large_data_run = (
            large_data_mode
            and workflow_run_style == "full"
            and selected_input.dataset_handle is not None
        )
        if background_large_data_run and background_state:
            snapshot, background_state = poll_large_data_background_workflow(background_state)
            st.session_state[background_state_key] = background_state
            render_background_job_status(background_state)
            if snapshot is not None:
                set_last_run_snapshot(snapshot)
                st.success("Background large-data workflow completed. Results are available.")
            background_actions = st.columns(2)
            if background_actions[0].button(
                "Cancel background job",
                width="stretch",
                key="readiness_cancel_background_job_button",
                disabled=background_state.get("status") in {"completed", "failed"},
            ):
                cancel_large_data_background_workflow(background_state)
                st.warning("Cancel requested. The worker will stop at the next safe boundary.")
            if background_actions[1].button(
                "Clear background job state",
                width="stretch",
                key="readiness_clear_background_job_button",
            ):
                st.session_state.pop(background_state_key, None)
                background_state = None
        if workflow_run_style == "step_by_step" and checkpoint_state:
            st.caption(
                "Active checkpointed run: "
                f"`{checkpoint_state.get('run_id', 'initialized')}` | "
                f"{checkpoint_state.get('manifest_status', 'initialized')}"
            )
            reset_checkpoint_run = st.button(
                "Reset checkpointed run",
                width="stretch",
                key="readiness_reset_checkpointed_run_button",
            )
            if reset_checkpoint_run:
                st.session_state.pop(checkpoint_state_key, None)
                checkpoint_state = None
        run_clicked = st.button(
            "Run Next Checkpoint Stage"
            if workflow_run_style == "step_by_step" and checkpoint_state
            else "Start Checkpointed Run"
            if workflow_run_style == "step_by_step"
            else "Refresh Background Large-Data Job"
            if background_large_data_run and background_state
            else "Start Background Large-Data Run"
            if background_large_data_run
            else run_button_label,
            type="primary",
            width="stretch",
            key="readiness_run_workflow_button",
        )

    if not run_clicked:
        return

    with readiness_tab:
        if preview_error or preview_config is None:
            st.error(preview_error or "Resolve the readiness issues before running the workflow.")
            set_last_run_snapshot(None)
            return
        try:
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            def render_progress(event: dict[str, Any]) -> None:
                render_runtime_status(
                    event,
                    status_placeholder=status_placeholder,
                    progress_placeholder=progress_placeholder,
                )
                event_type = str(event.get("event_type") or "")
                if event_type in {
                    "run_started",
                    "run_completed",
                    "step_failed",
                    "stage_failed",
                }:
                    audit_type = (
                        "workflow_run_completed"
                        if event_type == "run_completed"
                        else "workflow_run_failed"
                        if event_type in {"step_failed", "stage_failed"}
                        else "workflow_run_started"
                    )
                    record_gui_audit_event(
                        preview_config.artifacts.output_root,
                        audit_type,
                        run_id=str(event.get("run_id") or ""),
                        artifact_root=preview_config.artifacts.output_root
                        / str(event.get("run_id") or ""),
                        metadata={
                            "event_type": event_type,
                            "stage": event.get("step_name") or event.get("stage_id"),
                            "step_order": event.get("step_order"),
                            "total_steps": event.get("total_steps"),
                            "elapsed_seconds": event.get("elapsed_seconds"),
                            "error_message": event.get("error_message", ""),
                        },
                        debounce_key=f"{audit_type}_{event.get('run_id', '')}_{event_type}",
                    )

            if workflow_run_style == "step_by_step":
                checkpoint_state = st.session_state.get(checkpoint_state_key)
                if checkpoint_state is None:
                    checkpoint_state = start_checkpointed_workflow(
                        preview_config=preview_config,
                        dataframe=dataframe,
                        selected_input=selected_input,
                        large_data_mode=large_data_mode,
                    )
                    st.session_state[checkpoint_state_key] = checkpoint_state
                with st.spinner("Running the next checkpoint stage..."):
                    context, checkpoint_state = run_next_checkpoint_stage(
                        preview_config=preview_config,
                        checkpoint_state=checkpoint_state,
                        progress_callback=render_progress,
                    )
                st.session_state[checkpoint_state_key] = checkpoint_state
                if context is None:
                    st.info(
                        "Checkpoint stage completed. Continue running stages until "
                        "the export package is complete."
                    )
                    set_last_run_snapshot(None)
                else:
                    snapshot = build_run_snapshot(context, preview_config.to_dict())
                    set_last_run_snapshot(snapshot)
                    render_run_success(snapshot)
            elif background_large_data_run:
                background_state = st.session_state.get(background_state_key)
                if background_state is None or background_state.get("status") in {
                    "completed",
                    "failed",
                }:
                    background_state = start_large_data_background_workflow(
                        preview_config=preview_config,
                        dataframe=dataframe,
                        selected_input=selected_input,
                        large_data_mode=large_data_mode,
                    )
                    st.session_state[background_state_key] = background_state
                    render_background_job_status(background_state)
                    st.info(
                        "Large-data workflow is running in the background. "
                        "Use Refresh Background Large-Data Job to update status."
                    )
                else:
                    snapshot, background_state = poll_large_data_background_workflow(
                        background_state
                    )
                    st.session_state[background_state_key] = background_state
                    render_background_job_status(background_state)
                    if snapshot is not None:
                        set_last_run_snapshot(snapshot)
                        render_run_success(snapshot)
                    elif background_state.get("status") == "failed":
                        st.error(background_state.get("error_message", "Run failed."))
                    else:
                        st.info("Background workflow is still running.")
            else:
                with st.spinner(workflow_spinner_message(execution_mode)):
                    context = execute_workflow(
                        preview_config=preview_config,
                        dataframe=dataframe,
                        selected_input=selected_input,
                        large_data_mode=large_data_mode,
                        progress_callback=render_progress,
                    )
                snapshot = build_run_snapshot(context, preview_config.to_dict())
                set_last_run_snapshot(snapshot)
                render_run_success(snapshot)
        except Exception as exc:
            record_gui_audit_event(
                preview_config.artifacts.output_root,
                "workflow_run_failed",
                metadata={
                    "execution_mode": preview_config.execution.mode.value,
                    "model_type": preview_config.model.model_type.value,
                    "target_mode": preview_config.target.mode.value,
                    "error_message": str(exc),
                },
            )
            render_run_failure(
                classify_workflow_exception(
                    exc,
                    technical_details=traceback.format_exc(),
                )
            )
            set_last_run_snapshot(None)

