"""Step 3 Readiness Check & Run render helpers."""

from __future__ import annotations

import traceback
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import ExecutionMode
from quant_pd_framework.segmented_model import build_segment_key_series
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
    build_config_for_large_data_execution_override,
    build_execution_plan_cards,
    cancel_large_data_background_workflow,
    execute_workflow,
    is_background_job_active,
    poll_large_data_background_workflow,
    render_background_job_status,
    render_runtime_status,
    resolve_large_data_execution_override,
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
        detected_large_data_mode = bool(large_data_mode)
        force_standard_requested = False
        force_standard_confirmed = False
        force_standard_reason = ""
        workflow_columns = st.columns([1.3, 1.0])
        with workflow_columns[0]:
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
        with workflow_columns[1]:
            if detected_large_data_mode:
                force_standard_requested = st.toggle(
                    "Force standard in-memory execution",
                    value=False,
                    help=(
                        "Bypasses auto-enabled Large Data Mode for this run only. "
                        "This can load the full dataset into memory and may fail or "
                        "make the app unresponsive."
                    ),
                )
        if force_standard_requested:
            confirmation_label = (
                "I understand this may load the full dataset into memory and can fail "
                "or make the app unresponsive."
            )
            force_standard_confirmed = st.checkbox(
                confirmation_label,
                value=False,
                key="readiness_force_standard_execution_confirmed",
            )
            force_standard_reason = st.text_area(
                "Override reason",
                value="",
                placeholder=(
                    "Example: machine has sufficient RAM and final development requires "
                    "standard full in-memory model fitting."
                ),
                help="Written to the run audit metadata when the override is used.",
                key="readiness_force_standard_execution_reason",
            ).strip()
        large_data_override = resolve_large_data_execution_override(
            selected_input=selected_input,
            detected_large_data_mode=detected_large_data_mode,
            force_standard_requested=force_standard_requested,
            force_standard_confirmed=force_standard_confirmed,
            reason=force_standard_reason,
        )
        runtime_config = build_config_for_large_data_execution_override(
            preview_config,
            large_data_override,
        )
        effective_large_data_mode = large_data_override.effective_large_data_mode
        if large_data_override.blocked_reason:
            st.error(large_data_override.blocked_reason)
        elif large_data_override.user_override_disabled:
            st.warning(
                "Large Data Mode will be bypassed for this run. The standard workflow "
                "will attempt to load and process the full file in memory."
            )
        if preview_config is not None:
            _render_segmented_model_execution_plan(
                dataframe=dataframe,
                preview_config=runtime_config,
            )
            preflight_cards, preflight_details = build_preflight_summary(
                dataframe=dataframe,
                data_source_label=data_source_label,
                preview_config=runtime_config,
                edited_schema=edited_schema,
                transformation_frame=transformation_frame,
            )
            render_preflight_summary(cards=preflight_cards, details=preflight_details)
            resource_cards, resource_details = build_resource_readiness_check(
                dataframe=dataframe,
                preview_config=runtime_config,
                large_data_mode=effective_large_data_mode,
            )
            render_resource_readiness_check(
                cards=resource_cards,
                details=resource_details,
            )
        render_execution_plan(
            build_execution_plan_cards(
                preview_config=runtime_config,
                data_source_label=data_source_label,
                large_data_mode=effective_large_data_mode,
                large_data_user_override_disabled=(
                    large_data_override.user_override_disabled
                ),
            ),
            large_data_mode=effective_large_data_mode,
        )
        checkpoint_state = st.session_state.get(checkpoint_state_key)
        background_state = st.session_state.get(background_state_key)
        background_large_data_run = (
            effective_large_data_mode
            and workflow_run_style == "full"
            and selected_input.dataset_handle is not None
        )
        background_active = False
        background_refresh_clicked = False
        if background_large_data_run and background_state:
            snapshot, background_state = poll_large_data_background_workflow(background_state)
            st.session_state[background_state_key] = background_state
            render_background_job_status(background_state)
            background_active = is_background_job_active(background_state)
            if snapshot is not None:
                set_last_run_snapshot(snapshot)
                st.success("Background large-data workflow completed. Results are available.")
            if background_active:
                manifest_payload = background_state.get("manifest") or {}
                cancel_requested = bool(
                    manifest_payload.get("cancel_requested")
                    or background_state.get("cancel_requested")
                )
                background_actions = st.columns(2)
                background_refresh_clicked = background_actions[0].button(
                    "Refresh background status",
                    width="stretch",
                    key="readiness_refresh_background_job_button",
                )
                if background_actions[1].button(
                    "Stop background job",
                    width="stretch",
                    key="readiness_stop_background_job_button",
                    disabled=cancel_requested,
                ):
                    cancel_large_data_background_workflow(background_state)
                    manifest_payload["cancel_requested"] = True
                    background_state["manifest"] = manifest_payload
                    background_state["cancel_requested"] = True
                    background_state["progress"] = (
                        "Cancel requested. The worker will stop at the next safe boundary."
                    )
                    st.session_state[background_state_key] = background_state
                    st.warning(
                        "Stop requested. The worker will stop at the next safe boundary."
                    )
                    st.rerun()
                if cancel_requested:
                    st.warning("Stop has already been requested for this background job.")
            else:
                background_actions = st.columns(2)
                if background_actions[0].button(
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
        if background_active:
            run_clicked = background_refresh_clicked
        else:
            run_clicked = st.button(
                "Run Next Checkpoint Stage"
                if workflow_run_style == "step_by_step" and checkpoint_state
                else "Start Checkpointed Run"
                if workflow_run_style == "step_by_step"
                else "Start New Background Large-Data Run"
                if background_large_data_run and background_state
                else "Start Background Large-Data Run"
                if background_large_data_run
                else run_button_label,
                type="primary",
                width="stretch",
                key="readiness_run_workflow_button",
                disabled=bool(large_data_override.blocked_reason),
            )

    if not run_clicked:
        return

    with readiness_tab:
        if preview_error or runtime_config is None:
            st.error(preview_error or "Resolve the readiness issues before running the workflow.")
            set_last_run_snapshot(None)
            return
        if large_data_override.blocked_reason:
            st.error(large_data_override.blocked_reason)
            set_last_run_snapshot(None)
            return
        try:
            if large_data_override.user_override_disabled:
                record_gui_audit_event(
                    runtime_config.artifacts.output_root,
                    "large_data_standard_execution_override",
                    metadata={
                        "source_kind": large_data_override.source_kind,
                        "reason": large_data_override.reason,
                        "large_data_override_reason": large_data_override.reason,
                        "large_data_auto_detected": (
                            large_data_override.detected_large_data_mode
                        ),
                        "large_data_effective_mode": (
                            runtime_config.performance.large_data_effective_mode
                        ),
                    },
                    debounce_key="large_data_standard_execution_override",
                    debounce_payload={
                        "source_kind": large_data_override.source_kind,
                        "reason": large_data_override.reason,
                    },
                )
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
                        runtime_config.artifacts.output_root,
                        audit_type,
                        run_id=str(event.get("run_id") or ""),
                        artifact_root=runtime_config.artifacts.output_root
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
                        preview_config=runtime_config,
                        dataframe=dataframe,
                        selected_input=selected_input,
                        large_data_mode=effective_large_data_mode,
                    )
                    st.session_state[checkpoint_state_key] = checkpoint_state
                with st.spinner("Running the next checkpoint stage..."):
                    context, checkpoint_state = run_next_checkpoint_stage(
                        preview_config=runtime_config,
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
                    snapshot = build_run_snapshot(context, runtime_config.to_dict())
                    set_last_run_snapshot(snapshot)
                    render_run_success(snapshot)
            elif background_large_data_run:
                background_state = st.session_state.get(background_state_key)
                if background_state is None or background_state.get("status") in {
                    "completed",
                    "failed",
                }:
                    background_state = start_large_data_background_workflow(
                        preview_config=runtime_config,
                        dataframe=dataframe,
                        selected_input=selected_input,
                        large_data_mode=effective_large_data_mode,
                    )
                    st.session_state[background_state_key] = background_state
                    st.rerun()
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
                        preview_config=runtime_config,
                        dataframe=dataframe,
                        selected_input=selected_input,
                        large_data_mode=effective_large_data_mode,
                        progress_callback=render_progress,
                    )
                snapshot = build_run_snapshot(context, runtime_config.to_dict())
                set_last_run_snapshot(snapshot)
                render_run_success(snapshot)
        except Exception as exc:
            record_gui_audit_event(
                runtime_config.artifacts.output_root,
                "workflow_run_failed",
                metadata={
                    "execution_mode": runtime_config.execution.mode.value,
                    "model_type": runtime_config.model.model_type.value,
                    "target_mode": runtime_config.target.mode.value,
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


def _render_segmented_model_execution_plan(
    *,
    dataframe: pd.DataFrame,
    preview_config: Any,
) -> None:
    segmented_config = getattr(preview_config, "segmented_model", None)
    if segmented_config is None or not segmented_config.enabled:
        return

    st.markdown("#### Segmented Model Execution Plan")
    segment_columns = list(segmented_config.segment_columns)
    missing_columns = [column for column in segment_columns if column not in dataframe.columns]
    if missing_columns:
        st.error(
            "Segmented modeling is enabled, but these segment columns are missing from "
            f"the current dataset preview: {', '.join(missing_columns)}."
        )
        return
    if not segment_columns:
        st.error("Segmented modeling is enabled but no segment columns are selected.")
        return

    keys = build_segment_key_series(dataframe, segment_columns)
    counts = keys.value_counts(dropna=False)
    target_column = (
        getattr(preview_config.target, "source_column", None)
        or getattr(preview_config.target, "output_column", None)
    )
    target_values = (
        pd.to_numeric(dataframe[target_column], errors="coerce")
        if target_column in dataframe.columns
        else pd.Series(index=dataframe.index, dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for segment_key, row_count in counts.items():
        status = "eligible"
        reason = ""
        mask = keys == segment_key
        event_count = None
        non_event_count = None
        if int(row_count) < int(segmented_config.min_segment_rows):
            status = "fallback_global"
            reason = (
                f"Rows {int(row_count):,} below minimum "
                f"{int(segmented_config.min_segment_rows):,}."
            )
        elif getattr(getattr(preview_config.target, "mode", None), "value", "") == "binary":
            event_count = int(target_values.loc[mask].fillna(0).sum())
            non_event_count = int(row_count) - event_count
            if event_count < int(segmented_config.min_segment_events):
                status = "fallback_global"
                reason = (
                    f"Events {event_count:,} below minimum "
                    f"{int(segmented_config.min_segment_events):,}."
                )
            elif non_event_count < int(segmented_config.min_segment_events):
                status = "fallback_global"
                reason = (
                    f"Non-events {non_event_count:,} below minimum "
                    f"{int(segmented_config.min_segment_events):,}."
                )
        rows.append(
            {
                "segment_key": str(segment_key),
                "row_count_preview": int(row_count),
                "event_count_preview": event_count,
                "non_event_count_preview": non_event_count,
                "planned_status": status,
                "reason": reason,
            }
        )

    plan = pd.DataFrame(rows)
    if len(plan) > int(segmented_config.max_segments):
        st.error(
            "Segmented modeling resolves "
            f"{len(plan):,} segment combinations, above the configured maximum of "
            f"{int(segmented_config.max_segments):,}. Reduce segment columns or increase "
            "the maximum after governance review."
        )
    else:
        st.info(
            "The global model will always be fit first. Eligible segments receive their "
            "own model; ineligible, missing, or unseen segments route to the global fallback."
        )
    st.dataframe(plan, width="stretch", hide_index=True)
