"""Streamlit renderers for workflow execution status and recovery guidance."""

from __future__ import annotations

import streamlit as st

from quant_pd_framework.streamlit_ui.artifact_summary import (
    build_artifact_summary_frame,
    build_primary_artifact_cards,
)
from quant_pd_framework.streamlit_ui.error_guidance import WorkflowErrorGuidance
from quant_pd_framework.streamlit_ui.run_execution import format_elapsed_seconds
from quant_pd_framework.streamlit_ui.state import prepare_table_for_display
from quant_pd_framework.streamlit_ui.theme import render_metric_strip


def render_execution_plan(cards: list[dict[str, str]], *, large_data_mode: bool) -> None:
    if not cards:
        return

    with st.expander("Execution plan and output expectations", expanded=large_data_mode):
        render_metric_strip(cards, compact=True)
        if large_data_mode:
            st.info(
                "Large Data Mode will fit the model on the configured governed sample, "
                "then score the full file in chunks and write separate large-data artifacts."
            )
        else:
            st.caption(
                "Standard mode keeps the existing pandas workflow and writes the normal "
                "model-development artifact package."
            )


def render_run_success(snapshot: dict[str, object]) -> None:
    timing = snapshot.get("run_timing", {})
    elapsed_text = ""
    if isinstance(timing, dict) and timing.get("elapsed_seconds") is not None:
        elapsed_text = f" in {format_elapsed_seconds(timing.get('elapsed_seconds'))}"
    st.success(f"Completed run `{snapshot['run_id']}`{elapsed_text}.")
    artifacts = snapshot.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return

    cards = build_primary_artifact_cards(artifacts)
    if cards:
        with st.expander("Run output locations", expanded=True):
            render_metric_strip(cards[:5], compact=True)
            st.dataframe(
                prepare_table_for_display(build_artifact_summary_frame(artifacts).head(12)),
                width="stretch",
                hide_index=True,
            )


def render_run_failure(guidance: WorkflowErrorGuidance) -> None:
    st.error(guidance.title)
    st.markdown(f"**Likely cause:** {guidance.likely_cause}")
    st.markdown(f"**Recommended action:** {guidance.recommended_action}")
    with st.expander("Technical details", expanded=False):
        st.code(guidance.technical_details or guidance.technical_summary)
