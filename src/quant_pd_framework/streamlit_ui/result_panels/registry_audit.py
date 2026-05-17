"""Run registry and audit-trail result panels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework.run_registry import (
    AuditEvent,
    RunRegistryEntry,
    load_audit_events,
    load_run_registry,
    sync_run_registry_from_artifacts,
)
from quant_pd_framework.streamlit_ui.state import prepare_table_for_display
from quant_pd_framework.streamlit_ui.theme import render_metric_strip


def render_run_registry_panel(output_root: str | Path) -> None:
    """Renders the searchable completed-run registry and audit event table."""

    output_root = Path(output_root)
    st.markdown("### Run Registry")
    st.caption(
        "Browse previously completed or known failed runs without opening artifact "
        "folders manually. The registry is an index; the run folders remain the "
        "source of truth."
    )
    actions = st.columns([0.35, 0.65])
    if actions[0].button(
        "Refresh registry from artifacts",
        key=f"refresh_run_registry_{output_root}",
        width="stretch",
    ):
        sync_run_registry_from_artifacts(output_root)
        st.success("Registry refreshed from artifact folders.")

    entries = load_run_registry(output_root, refresh_from_artifacts=True)
    if not entries:
        st.info(f"No run registry entries were found under `{output_root}`.")
        _render_audit_trail(output_root, run_id="")
        return

    registry_frame = _run_registry_frame(entries)
    filter_columns = st.columns(4)
    search_text = filter_columns[0].text_input(
        "Search runs",
        value="",
        key=f"run_registry_search_{output_root}",
        help="Search run ID, source, model type, execution mode, or artifact path.",
    ).strip().lower()
    status_options = ["All", *sorted(registry_frame["status"].dropna().unique().tolist())]
    selected_status = filter_columns[1].selectbox(
        "Status",
        options=status_options,
        key=f"run_registry_status_{output_root}",
    )
    model_options = ["All", *sorted(registry_frame["model_type"].dropna().unique().tolist())]
    selected_model = filter_columns[2].selectbox(
        "Model type",
        options=model_options,
        key=f"run_registry_model_{output_root}",
    )
    reviewer_options = [
        "All",
        *sorted(registry_frame["reviewer_status"].dropna().unique().tolist()),
    ]
    selected_reviewer = filter_columns[3].selectbox(
        "Reviewer status",
        options=reviewer_options,
        key=f"run_registry_reviewer_{output_root}",
    )
    filtered = registry_frame.copy()
    if search_text:
        searchable = filtered.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered.loc[searchable.str.contains(search_text)]
    if selected_status != "All":
        filtered = filtered.loc[filtered["status"] == selected_status]
    if selected_model != "All":
        filtered = filtered.loc[filtered["model_type"] == selected_model]
    if selected_reviewer != "All":
        filtered = filtered.loc[filtered["reviewer_status"] == selected_reviewer]

    st.dataframe(
        prepare_table_for_display(filtered),
        width="stretch",
        hide_index=True,
    )
    if filtered.empty:
        st.info("No runs match the current filters.")
        _render_audit_trail(output_root, run_id="")
        return

    selected_run_id = st.selectbox(
        "Inspect run",
        options=filtered["run_id"].tolist(),
        key=f"run_registry_selected_run_{output_root}",
    )
    selected_entry = next(entry for entry in entries if entry.run_id == selected_run_id)
    _render_run_registry_entry_detail(selected_entry)
    _render_audit_trail(output_root, run_id=selected_run_id)


def _run_registry_frame(entries: list[RunRegistryEntry]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(
            {
                "run_id": entry.run_id,
                "status": entry.status,
                "completed_at_utc": entry.completed_at_utc,
                "elapsed_seconds": entry.elapsed_seconds,
                "execution_mode": entry.execution_mode,
                "model_type": entry.model_type,
                "target_mode": entry.target_mode,
                "dataset_source": entry.dataset_source_label,
                "large_data": entry.large_data_mode,
                "reviewer_status": entry.reviewer_status,
                "warning_count": entry.warning_count,
                "primary_metric": _primary_metric_text(entry.metrics_summary),
                "artifact_root": entry.artifact_root,
            }
        )
    return pd.DataFrame(rows)


def _render_run_registry_entry_detail(entry: RunRegistryEntry) -> None:
    st.markdown("#### Selected Run Detail")
    render_metric_strip(
        [
            {"label": "Run ID", "value": entry.run_id},
            {"label": "Status", "value": entry.status.replace("_", " ").title()},
            {"label": "Model", "value": entry.model_type.replace("_", " ").title()},
            {"label": "Reviewer", "value": entry.reviewer_status or "Not reviewed"},
        ],
        compact=True,
    )
    detail_rows = [
        ("Artifact root", entry.artifact_root),
        ("Started", entry.started_at_utc),
        ("Completed", entry.completed_at_utc),
        ("Elapsed seconds", entry.elapsed_seconds),
        ("Execution mode", entry.execution_mode),
        ("Target mode", entry.target_mode),
        ("Dataset source", entry.dataset_source_label),
        ("Dataset source kind", entry.dataset_source_kind),
        ("Large Data Mode", entry.large_data_mode),
        ("Warnings", entry.warning_count),
        ("Reviewer name", entry.reviewer_name),
        ("Review updated", entry.review_updated_at_utc),
        ("Error", entry.error_message),
    ]
    st.dataframe(
        prepare_table_for_display(pd.DataFrame(detail_rows, columns=["field", "value"])),
        width="stretch",
        hide_index=True,
    )
    if entry.metrics_summary:
        st.markdown("##### Metrics Summary")
        st.dataframe(
            prepare_table_for_display(
                pd.DataFrame(
                    [
                        {"metric": metric, "value": value}
                        for metric, value in entry.metrics_summary.items()
                    ]
                )
            ),
            width="stretch",
            hide_index=True,
        )
    if entry.artifact_paths:
        st.markdown("##### Primary Artifacts")
        st.dataframe(
            prepare_table_for_display(
                pd.DataFrame(
                    [
                        {"artifact": key, "path": value}
                        for key, value in entry.artifact_paths.items()
                    ]
                )
            ),
            width="stretch",
            hide_index=True,
        )


def _render_audit_trail(output_root: Path, *, run_id: str) -> None:
    st.markdown("### Audit Trail")
    event_type_filter = st.text_input(
        "Audit event contains",
        value="",
        key=f"audit_event_filter_{output_root}_{run_id or 'all'}",
        help="Optional text filter across event type, source, run ID, and metadata.",
    ).strip().lower()
    events = load_audit_events(output_root, run_id=run_id or None, limit=500)
    frame = _audit_event_frame(events)
    if frame.empty:
        st.info("No audit events have been recorded for the current filter.")
        return
    if event_type_filter:
        searchable = frame.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        frame = frame.loc[searchable.str.contains(event_type_filter)]
    st.dataframe(
        prepare_table_for_display(frame),
        width="stretch",
        hide_index=True,
    )


def _audit_event_frame(events: list[AuditEvent]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "timestamp_utc": event.timestamp_utc,
                "event_type": event.event_type,
                "source": event.source,
                "run_id": event.run_id,
                "artifact_root": event.artifact_root,
                "metadata": json.dumps(event.metadata, sort_keys=True, default=str),
            }
        )
    return pd.DataFrame(rows)


def _primary_metric_text(metrics: dict[str, Any]) -> str:
    for key, value in metrics.items():
        if value not in {"", None}:
            try:
                return f"{key}: {float(value):.4g}"
            except (TypeError, ValueError):
                return f"{key}: {value}"
    return ""
