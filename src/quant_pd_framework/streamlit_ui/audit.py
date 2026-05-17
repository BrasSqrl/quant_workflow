"""Streamlit-facing audit helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from quant_pd_framework.run_registry import append_audit_event, fingerprint_payload

SESSION_ID_KEY = "quant_studio_audit_session_id"


def get_audit_session_id() -> str:
    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = uuid4().hex
    return str(st.session_state[SESSION_ID_KEY])


def record_gui_audit_event(
    output_root: str | Path,
    event_type: str,
    *,
    run_id: str = "",
    artifact_root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    debounce_key: str = "",
    debounce_payload: Any = None,
) -> None:
    """Records a GUI audit event, with optional Streamlit-session debouncing."""

    if debounce_key:
        fingerprint = fingerprint_payload(
            debounce_payload if debounce_payload is not None else metadata or {}
        )
        state_key = f"audit_event_fingerprint_{debounce_key}"
        if st.session_state.get(state_key) == fingerprint:
            return
        st.session_state[state_key] = fingerprint
    append_audit_event(
        output_root,
        event_type,
        source="streamlit",
        run_id=run_id,
        session_id=get_audit_session_id(),
        artifact_root=artifact_root,
        metadata=metadata or {},
    )
