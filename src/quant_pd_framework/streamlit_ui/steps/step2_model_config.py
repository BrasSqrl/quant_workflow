"""Step 2 Model Configuration render helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework.streamlit_ui.enterprise_workflow import (
    build_configuration_risk_score,
    build_model_suitability_explainer,
    build_runtime_artifact_estimate,
    render_configuration_risk_score,
    render_guidance_center,
    render_model_suitability_explainer,
    render_runtime_artifact_estimate,
)


def render_model_configuration_intro() -> None:
    """Renders the Step 2 section header and guidance center."""

    st.markdown(
        """
        <div class="step-panel-intro">
          <span class="step-panel-kicker">Model Configuration</span>
          <h3 class="step-panel-title">Configure the workflow</h3>
          <p class="step-panel-copy">
            Use the grouped panels to keep core setup visible while hiding
            lower-priority tuning until you need it.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_guidance_center()


def render_configuration_quality_panels(
    *,
    configuration_tab: Any,
    profile_manager_container: Any,
    render_profile_manager: Callable[..., None],
    render_configuration_diff_viewer: Callable[..., None],
    dataframe: pd.DataFrame,
    data_source_label: str,
    source_metadata: dict[str, Any],
    workspace_keys: Any,
    preview_config: Any,
    preview_error: str | None,
    edited_schema: pd.DataFrame,
    feature_dictionary_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    feature_review_frame: pd.DataFrame,
    scorecard_override_frame: pd.DataFrame,
    current_config: dict[str, Any] | None,
    active_profile_payload: dict[str, Any] | None,
    last_run_snapshot: dict[str, Any] | None,
) -> None:
    """Renders Step 2 profile, diff, suitability, risk, and estimate panels."""

    with configuration_tab:
        with profile_manager_container:
            render_profile_manager(
                dataframe=dataframe,
                data_source_label=data_source_label,
                source_metadata=source_metadata,
                workspace_keys=workspace_keys,
                preview_config=preview_config,
                preview_error=preview_error,
                edited_schema=edited_schema,
                feature_dictionary_frame=feature_dictionary_frame,
                transformation_frame=transformation_frame,
                feature_review_frame=feature_review_frame,
                scorecard_override_frame=scorecard_override_frame,
            )
        render_configuration_diff_viewer(
            current_config=current_config,
            active_profile=active_profile_payload,
            last_run_snapshot=last_run_snapshot,
        )
        if preview_config is not None:
            suitability_cards, suitability_details = build_model_suitability_explainer(
                dataframe=dataframe,
                preview_config=preview_config,
                edited_schema=edited_schema,
                transformation_frame=transformation_frame,
            )
            render_model_suitability_explainer(
                cards=suitability_cards,
                details=suitability_details,
            )
            risk_cards, risk_details = build_configuration_risk_score(
                dataframe=dataframe,
                preview_config=preview_config,
                edited_schema=edited_schema,
                transformation_frame=transformation_frame,
            )
            render_configuration_risk_score(cards=risk_cards, details=risk_details)
            estimate_cards, estimate_details = build_runtime_artifact_estimate(
                dataframe=dataframe,
                preview_config=preview_config,
                transformation_frame=transformation_frame,
            )
            render_runtime_artifact_estimate(
                cards=estimate_cards,
                details=estimate_details,
            )
        else:
            st.info("Resolve the configuration before reviewing Step 2 suitability guidance.")

