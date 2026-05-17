"""Step 5 Decision Summary render helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from quant_pd_framework.streamlit_ui.results import render_decision_summary
from quant_pd_framework.streamlit_ui.state import get_last_run_snapshot


def render_decision_summary_tab(
    *,
    decision_tab: Any,
    preview_config: Any,
) -> None:
    """Renders Step 5 decision summary and package downloads."""

    with decision_tab:
        last_run_snapshot = get_last_run_snapshot()
        if last_run_snapshot:
            current_config = preview_config.to_dict() if preview_config is not None else None
            if current_config is not None and last_run_snapshot.get("config") != current_config:
                st.warning(
                    "Decision summary is from the last completed run and may be stale because "
                    "the current workflow configuration has changed."
                )
            render_decision_summary(last_run_snapshot)
        else:
            st.info("Run a valid workflow from Step 3 to populate the decision summary.")
