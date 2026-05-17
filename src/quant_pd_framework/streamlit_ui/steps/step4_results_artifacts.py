"""Step 4 Results & Artifacts render helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from quant_pd_framework.streamlit_ui.results import (
    render_run_registry_panel,
    render_run_results,
)
from quant_pd_framework.streamlit_ui.state import get_last_run_snapshot


def render_results_artifacts_tab(
    *,
    results_tab: Any,
    preview_config: Any,
    output_root: str,
) -> None:
    """Renders Step 4 live results plus the historical run registry."""

    with results_tab:
        last_run_snapshot = get_last_run_snapshot()
        if last_run_snapshot:
            current_config = preview_config.to_dict() if preview_config is not None else None
            if current_config is not None and last_run_snapshot.get("config") != current_config:
                st.warning(
                    "Results are from the last completed run and may be stale because "
                    "the current workflow configuration has changed."
                )
            render_run_results(last_run_snapshot)
            st.divider()
            if preview_config is not None:
                registry_root = preview_config.artifacts.output_root
            else:
                snapshot_root = last_run_snapshot.get("artifacts", {}).get("output_root")
                registry_root = (
                    Path(str(snapshot_root)).parent if snapshot_root else Path(output_root)
                )
            render_run_registry_panel(registry_root)
        else:
            st.info("Run a valid workflow from Step 3 to populate results and artifacts.")
            registry_root = (
                preview_config.artifacts.output_root
                if preview_config is not None
                else Path(output_root)
            )
            render_run_registry_panel(registry_root)
