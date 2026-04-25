"""Workspace renderers for dataset preparation and schema design."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st

from quant_pd_framework.gui_support import (
    SUPPORTED_DTYPES,
    SUPPORTED_MISSING_VALUE_POLICIES,
    build_template_workbook_bytes,
    frames_equivalent,
    load_template_workbook,
)
from quant_pd_framework.streamlit_ui.data import DEFAULT_PERFORMANCE_CONFIG, render_dataset_overview
from quant_pd_framework.streamlit_ui.state import (
    WorkspaceState,
    store_workspace_frame,
)
from quant_pd_framework.streamlit_ui.theme import render_html


def schema_editor_column_config() -> dict[str, object]:
    return {
        "enabled": st.column_config.CheckboxColumn("Enabled"),
        "name": st.column_config.TextColumn("Column name", disabled=True),
        "rename_to": st.column_config.TextColumn("Rename to"),
        "role": st.column_config.SelectboxColumn(
            "Role",
            options=[
                "ignore",
                "feature",
                "target_source",
                "date",
                "identifier",
                "group",
            ],
        ),
        "dtype": st.column_config.SelectboxColumn(
            "Data type",
            options=SUPPORTED_DTYPES,
        ),
        "create_if_missing": st.column_config.CheckboxColumn("Create if missing"),
        "missing_value_policy": st.column_config.SelectboxColumn(
            "Missing-value policy",
            options=SUPPORTED_MISSING_VALUE_POLICIES,
        ),
        "fill_value": st.column_config.TextColumn("Fill value"),
        "imputation_group": st.column_config.CheckboxColumn("Imputation group"),
        "add_missing_indicator": st.column_config.CheckboxColumn("Missing flag"),
        "notes": st.column_config.TextColumn("Notes"),
    }


def render_schema_guidance() -> None:
    with st.expander("Schema Guidance", expanded=False):
        st.markdown(
            """
            - Mark one enabled row as `target_source`.
            - Use `date` and `identifier` roles when the workflow is time-series or panel.
            - `missing_value_policy` is fit on the training split and reused downstream.
            - `imputation_group` participates in grouped scalar imputation.
            - `add_missing_indicator` creates a feature flag that records missingness.
            """
        )


def render_schema_editor_panel(schema_frame: pd.DataFrame, *, editor_key: str) -> pd.DataFrame:
    st.caption(
        "Mark one enabled row as target_source. Add date and identifier "
        "roles when using time-series or panel workflows. Missing-value "
        "policies are fit on the training split and reused downstream. "
        "Group columns and missing flags enable the advanced imputation layer."
    )
    edited_schema = st.data_editor(
        schema_frame,
        key=editor_key,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config=schema_editor_column_config(),
    )
    render_schema_guidance()
    return edited_schema


def render_builder_workspace(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
    workspace_state: WorkspaceState,
) -> dict[str, pd.DataFrame]:
    section_options = [
        "Dataset Preview",
        "Column Designer",
        "Feature Dictionary",
        "Transformations",
        "Template Workbook",
    ]

    edited_schema = workspace_state.schema_frame
    edited_feature_dictionary = workspace_state.feature_dictionary_frame
    edited_transformations = workspace_state.transformation_frame
    feature_review_frame = workspace_state.feature_review_frame
    scorecard_override_frame = workspace_state.scorecard_override_frame

    render_html(
        '<div class="workflow-stage">'
        '<div class="workflow-stage__index">1</div>'
        '<div class="workflow-stage__body">'
        '<span class="workflow-stage__kicker">Dataset & Schema</span>'
        "<h2>Prepare data and schema</h2>"
        "<p>Inspect the input, define governed schema rules, document features, "
        "stage transformations, and exchange the review workbook offline.</p>"
        "</div>"
        "</div>"
    )

    selected_section = st.radio(
        "Workspace section",
        options=section_options,
        horizontal=True,
        key=f"{workspace_state.keys.editor_key}_workspace_section",
        label_visibility="collapsed",
    )

    if selected_section == "Dataset Preview":
        render_dataset_overview(dataframe, data_source_label)
        preview_rows = DEFAULT_PERFORMANCE_CONFIG.ui_preview_rows
        st.caption(f"Showing the first {preview_rows} rows of the raw input dataframe.")
        st.dataframe(dataframe.head(preview_rows), width="stretch", hide_index=True)

    elif selected_section == "Column Designer":
        edited_schema = render_schema_editor_panel(
            workspace_state.schema_frame,
            editor_key=workspace_state.keys.editor_key,
        )

    elif selected_section == "Feature Dictionary":
        st.caption(
            "Document the modeled feature set with business definitions, source lineage, "
            "expected signs, and inclusion rationale."
        )
        edited_feature_dictionary = st.data_editor(
            workspace_state.feature_dictionary_frame,
            key=workspace_state.keys.feature_dictionary_widget,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "feature_name": st.column_config.TextColumn("Feature"),
                "business_name": st.column_config.TextColumn("Business name"),
                "definition": st.column_config.TextColumn("Definition"),
                "source_system": st.column_config.TextColumn("Source system"),
                "unit": st.column_config.TextColumn("Unit"),
                "allowed_range": st.column_config.TextColumn("Allowed range"),
                "missingness_meaning": st.column_config.TextColumn("Missingness meaning"),
                "expected_sign": st.column_config.TextColumn("Expected sign"),
                "inclusion_rationale": st.column_config.TextColumn("Inclusion rationale"),
                "notes": st.column_config.TextColumn("Notes"),
            },
        )

    elif selected_section == "Transformations":
        st.caption(
            "Governed transformations are fit on the training split and then replayed on "
            "validation, test, and scored data."
        )
        edited_transformations = st.data_editor(
            workspace_state.transformation_frame,
            key=workspace_state.keys.transformation_widget,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "transform_type": st.column_config.TextColumn("Type"),
                "source_feature": st.column_config.TextColumn("Source feature"),
                "secondary_feature": st.column_config.TextColumn("Secondary feature"),
                "categorical_value": st.column_config.TextColumn("Categorical value"),
                "output_feature": st.column_config.TextColumn("Output feature"),
                "lower_quantile": st.column_config.NumberColumn(
                    "Lower q",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                ),
                "upper_quantile": st.column_config.NumberColumn(
                    "Upper q",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                ),
                "parameter_value": st.column_config.NumberColumn("Parameter"),
                "window_size": st.column_config.NumberColumn("Window", min_value=1, step=1),
                "lag_periods": st.column_config.NumberColumn(
                    "Lag periods",
                    min_value=1,
                    step=1,
                ),
                "bin_edges": st.column_config.TextColumn("Bin edges"),
                "generated_automatically": st.column_config.CheckboxColumn("Generated"),
                "notes": st.column_config.TextColumn("Notes"),
            },
        )
        with st.expander("Transformation Guidance", expanded=False):
            st.markdown(
                """
                - `winsorize` clips a numeric feature using train-fit quantiles.
                - `log1p` applies a log transform to numeric values greater than `-1`.
                - `box_cox` applies a train-fit power transform for strictly positive values.
                - `natural_spline` expands a numeric feature into a train-fit
                  natural cubic spline basis.
                - `yeo_johnson` fits a train-based power transform that can
                  handle zero and negative values.
                - `capped_zscore` standardizes a numeric feature and clips it
                  at the configured z-cap.
                - `piecewise_linear` creates a positive hinge term above the
                  configured cut point.
                - `ratio` creates `source / secondary`.
                - `interaction` creates `source * secondary`.
                - `lag`, `difference`, `ewma`, and rolling transforms add time-aware features.
                - `manual_bins` creates an ordered categorical feature using
                  your internal edges.
                """
            )

    else:
        st.caption(
            "Download the editable workbook for offline review, then upload a completed "
            "version to repopulate the workspace tables."
        )
        template_payload = build_template_workbook_bytes(
            schema_frame=workspace_state.schema_frame,
            feature_dictionary_frame=workspace_state.feature_dictionary_frame,
            transformation_frame=workspace_state.transformation_frame,
            feature_review_frame=feature_review_frame,
            scorecard_override_frame=scorecard_override_frame,
        )
        st.download_button(
            "Download Review Workbook",
            data=template_payload,
            file_name="quant_studio_review_workbook.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        uploaded_template = st.file_uploader(
            "Upload completed workbook",
            type=["xlsx"],
            key=f"{workspace_state.keys.editor_key}_template_workbook_upload",
        )
        if uploaded_template is not None:
            template_bytes = uploaded_template.getvalue()
            upload_hash = hashlib.sha256(template_bytes).hexdigest()
            upload_state_key = f"{workspace_state.keys.editor_key}_template_workbook_hash"
            if st.session_state.get(upload_state_key) != upload_hash:
                workbook_frames = load_template_workbook(BytesIO(template_bytes))
                st.session_state[workspace_state.keys.schema_frame] = workbook_frames["schema"]
                st.session_state[workspace_state.keys.feature_dictionary_frame] = (
                    workbook_frames["feature_dictionary"]
                )
                st.session_state[workspace_state.keys.transformation_frame] = (
                    workbook_frames["transformations"]
                )
                st.session_state[workspace_state.keys.feature_review_frame] = workbook_frames[
                    "feature_review"
                ]
                st.session_state[workspace_state.keys.scorecard_override_frame] = (
                    workbook_frames["scorecard_overrides"]
                )
                st.session_state[upload_state_key] = upload_hash
                st.rerun()

    schema_changed = not frames_equivalent(workspace_state.schema_frame, edited_schema)
    store_workspace_frame(workspace_state.keys.schema_frame, edited_schema)
    store_workspace_frame(
        workspace_state.keys.feature_dictionary_frame,
        edited_feature_dictionary,
    )
    store_workspace_frame(
        workspace_state.keys.transformation_frame,
        edited_transformations,
    )
    store_workspace_frame(
        workspace_state.keys.feature_review_frame,
        feature_review_frame,
    )
    store_workspace_frame(
        workspace_state.keys.scorecard_override_frame,
        scorecard_override_frame,
    )

    if schema_changed:
        st.rerun()

    return {
        "schema": edited_schema,
        "feature_dictionary": edited_feature_dictionary,
        "transformations": edited_transformations,
        "feature_review": feature_review_frame,
        "scorecard_overrides": scorecard_override_frame,
    }
