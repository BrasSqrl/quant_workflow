"""Workspace renderers for dataset preparation and schema design."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from quant_pd_framework.gui_support import (
    SUPPORTED_DTYPES,
    SUPPORTED_MISSING_VALUE_POLICIES,
    SUPPORTED_TRANSFORMATION_TYPES,
    build_template_workbook_bytes,
    build_transformation_recipe_catalog_frame,
    build_transformation_recommendations,
    build_transformation_row,
    build_transformation_summary_cards,
    build_transformation_validation_frame,
    frames_equivalent,
    load_template_workbook,
    normalize_transformation_frame,
    transformation_recommendation_to_row,
)
from quant_pd_framework.streamlit_ui.data import DEFAULT_PERFORMANCE_CONFIG, render_dataset_overview
from quant_pd_framework.streamlit_ui.state import (
    WorkspaceState,
    store_workspace_frame,
)
from quant_pd_framework.streamlit_ui.theme import render_html

DATA_REVIEW_PROFILE_ROWS = 100_000
LEAKAGE_NAME_PATTERNS = {
    "default": "Column name references default outcomes.",
    "chargeoff": "Column name references charge-off outcomes.",
    "charge_off": "Column name references charge-off outcomes.",
    "loss": "Column name references loss after performance is observed.",
    "recovery": "Column name references recovery after default.",
    "post_": "Column name appears to reference post-event information.",
    "status": "Column name may encode downstream account status.",
    "workout": "Column name may encode workout or collection activity.",
    "collection": "Column name may encode collection activity.",
    "bankruptcy": "Column name may encode an outcome or legal event.",
    "writeoff": "Column name references write-off outcomes.",
    "write_off": "Column name references write-off outcomes.",
}


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


def build_data_contract_scorecard(
    dataframe: pd.DataFrame,
    schema_frame: pd.DataFrame,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    """Builds a compact dataset/schema readiness scorecard for Step 1."""

    profiled = dataframe.head(DATA_REVIEW_PROFILE_ROWS)
    active_schema = _active_schema_rows(schema_frame)
    feature_count = int(
        active_schema["role"].astype(str).str.lower().eq("feature").sum()
        if "role" in active_schema.columns
        else 0
    )
    target_columns = _role_columns(active_schema, "target_source")
    date_columns = _role_columns(active_schema, "date")
    identifier_columns = _role_columns(active_schema, "identifier")
    missing_cells = int(profiled.isna().sum().sum()) if not profiled.empty else 0
    total_profile_cells = max(int(profiled.shape[0] * profiled.shape[1]), 1)
    duplicate_rows = int(profiled.duplicated().sum()) if not profiled.empty else 0
    high_cardinality_columns = _high_cardinality_columns(profiled)
    date_coverage = _date_coverage_text(profiled, date_columns)
    target_distribution = _target_distribution_text(profiled, target_columns)

    cards = [
        {"label": "Rows", "value": f"{len(dataframe):,}"},
        {"label": "Columns", "value": f"{len(dataframe.columns):,}"},
        {"label": "Profiled rows", "value": f"{len(profiled):,}"},
        {"label": "Enabled features", "value": f"{feature_count:,}"},
        {"label": "Target sources", "value": f"{len(target_columns):,}"},
        {"label": "Missing cells", "value": f"{missing_cells:,}"},
        {"label": "Duplicate rows", "value": f"{duplicate_rows:,}"},
        {"label": "High-cardinality fields", "value": f"{len(high_cardinality_columns):,}"},
    ]
    rows = [
        _contract_row(
            "Target role",
            len(target_columns) == 1,
            f"{', '.join(target_columns) or 'No target_source role selected.'}",
            "Mark exactly one enabled target_source row in Column Designer.",
        ),
        _contract_row(
            "Date role",
            bool(date_columns),
            date_coverage,
            "Assign a date role for time-series or panel workflows.",
            warning_only=True,
        ),
        _contract_row(
            "Identifier role",
            bool(identifier_columns),
            ", ".join(identifier_columns) or "No identifier role selected.",
            "Assign an identifier role for panel workflows.",
            warning_only=True,
        ),
        _contract_row(
            "Missingness",
            missing_cells == 0,
            f"{missing_cells:,} missing cells in profiled rows "
            f"({missing_cells / total_profile_cells:.1%}).",
            "Review missing-value policies and missingness indicators.",
            warning_only=True,
        ),
        _contract_row(
            "Duplicate rows",
            duplicate_rows == 0,
            f"{duplicate_rows:,} duplicates in profiled rows.",
            "Leave duplicate-row cleaning on unless duplicates are expected.",
            warning_only=True,
        ),
        _contract_row(
            "Target distribution",
            target_distribution != "Unavailable",
            target_distribution,
            "Confirm the target source and positive target mapping.",
            warning_only=True,
        ),
        _contract_row(
            "High-cardinality fields",
            not high_cardinality_columns,
            ", ".join(high_cardinality_columns[:8]) or "No high-cardinality text fields found.",
            "Group, encode, ignore, or document high-cardinality categorical fields.",
            warning_only=True,
        ),
    ]
    return cards, pd.DataFrame(rows)


def build_potential_leakage_flags(
    dataframe: pd.DataFrame,
    schema_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Flags column names that often indicate target leakage or post-outcome data."""

    active_schema = _active_schema_rows(schema_frame)
    target_columns = set(_role_columns(active_schema, "target_source"))
    rows: list[dict[str, str]] = []
    for column_name in dataframe.columns:
        if str(column_name) in target_columns:
            continue
        lowered = str(column_name).lower()
        matched_reasons = [
            reason for pattern, reason in LEAKAGE_NAME_PATTERNS.items() if pattern in lowered
        ]
        if not matched_reasons:
            continue
        rows.append(
            {
                "column": str(column_name),
                "severity": "review",
                "reason": " ".join(dict.fromkeys(matched_reasons)),
                "recommended_action": (
                    "Confirm this field is known before the prediction date; otherwise "
                    "mark it ignore in Column Designer."
                ),
            }
        )
    return pd.DataFrame(rows, columns=["column", "severity", "reason", "recommended_action"])


def build_schema_fingerprint(
    dataframe: pd.DataFrame,
    *,
    data_source_label: str,
) -> pd.DataFrame:
    """Builds deterministic source/schema identifiers for audit review."""

    column_signature = "|".join(
        f"{column}:{dtype}" for column, dtype in dataframe.dtypes.astype(str).items()
    )
    sample = dataframe.head(min(len(dataframe), 1_000))
    if sample.empty:
        sample_hash = "empty"
    else:
        sample_hash = hashlib.sha256(
            pd.util.hash_pandas_object(sample, index=True).to_numpy().tobytes()
        ).hexdigest()
    rows = [
        ("Data source", data_source_label or "Selected input"),
        ("Rows", f"{len(dataframe):,}"),
        ("Columns", f"{len(dataframe.columns):,}"),
        ("Column signature hash", hashlib.sha256(column_signature.encode()).hexdigest()),
        ("Sample content hash", sample_hash),
        ("Profiled sample rows", f"{len(sample):,}"),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def render_data_review_panel(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
    schema_frame: pd.DataFrame,
) -> None:
    """Renders Step 1 data-contract, leakage, and fingerprint guidance."""

    st.caption(
        "Use this review before configuring the model. Full shape is shown, while "
        f"quality checks use the first {DATA_REVIEW_PROFILE_ROWS:,} rows for responsiveness."
    )
    cards, contract_table = build_data_contract_scorecard(dataframe, schema_frame)
    _render_local_metric_cards(cards)
    with st.expander("Data Contract Scorecard", expanded=True):
        st.dataframe(contract_table, width="stretch", hide_index=True)
    with st.expander("Potential Leakage Flags", expanded=True):
        leakage_flags = build_potential_leakage_flags(dataframe, schema_frame)
        if leakage_flags.empty:
            st.success("No obvious leakage-like column names were detected.")
        else:
            st.warning(
                "These are name-based flags only. Review timing and business meaning "
                "before excluding a field."
            )
            st.dataframe(leakage_flags, width="stretch", hide_index=True)
    with st.expander("Schema Fingerprint", expanded=False):
        st.dataframe(
            build_schema_fingerprint(dataframe, data_source_label=data_source_label),
            width="stretch",
            hide_index=True,
        )


def render_transformation_studio(
    *,
    dataframe: pd.DataFrame,
    schema_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    workspace_state: WorkspaceState,
    preset_transformations: Any | None,
    advanced_workspace: bool,
    target_mode: str,
    model_type: str,
    data_structure: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Renders the guided Step 1 Transformation Studio."""

    working_frame = normalize_transformation_frame(transformation_frame)
    validation_frame = build_transformation_validation_frame(
        working_frame,
        dataframe,
        schema_frame,
        target_mode=target_mode,
        data_structure=data_structure,
    )
    _render_local_metric_cards(
        build_transformation_summary_cards(working_frame, validation_frame)
    )
    st.caption(
        "Build governed transformations as recommendations, reusable recipes, or custom "
        "rows. Accepted rows still write to the same workbook/config schema."
    )

    recommendation_tab, recipe_tab, builder_tab, pipeline_tab = st.tabs(
        ["Recommendations", "Recipe Library", "Custom Builder", "Pipeline Review"]
    )
    with recommendation_tab:
        _render_transformation_recommendations_tab(
            dataframe=dataframe,
            schema_frame=schema_frame,
            transformation_frame=working_frame,
            workspace_state=workspace_state,
            target_mode=target_mode,
            model_type=model_type,
            data_structure=data_structure,
        )
    with recipe_tab:
        _render_recipe_library_tab(
            dataframe=dataframe,
            transformation_frame=working_frame,
            workspace_state=workspace_state,
        )
    with builder_tab:
        _render_custom_builder_tab(
            dataframe=dataframe,
            transformation_frame=working_frame,
            workspace_state=workspace_state,
        )
    with pipeline_tab:
        edited_frame = _render_pipeline_review_tab(
            dataframe=dataframe,
            transformation_frame=working_frame,
            validation_frame=validation_frame,
            workspace_state=workspace_state,
        )
        working_frame = edited_frame

    transformation_controls = _render_advanced_generation_controls(
        workspace_state=workspace_state,
        preset_transformations=preset_transformations,
        advanced_workspace=advanced_workspace,
    )
    return working_frame, transformation_controls


def _render_transformation_recommendations_tab(
    *,
    dataframe: pd.DataFrame,
    schema_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    workspace_state: WorkspaceState,
    target_mode: str,
    model_type: str,
    data_structure: str,
) -> None:
    recommendations = build_transformation_recommendations(
        dataframe,
        schema_frame,
        target_mode=target_mode,
        model_type=model_type,
        data_structure=data_structure,
    )
    rejected_key = f"{workspace_state.keys.editor_key}_rejected_transformation_recs"
    rejected = set(st.session_state.get(rejected_key, []))
    if not recommendations.empty:
        recommendations = recommendations.loc[
            ~recommendations["recommendation_id"].isin(rejected)
        ].reset_index(drop=True)
    if recommendations.empty:
        st.info(
            "No new profile-driven recommendations are available. Use Recipe Library "
            "or Custom Builder to add governed transformations manually."
        )
        if rejected and st.button(
            "Reset dismissed recommendations",
            key=f"{workspace_state.keys.editor_key}_reset_rejected_transformations",
        ):
            st.session_state[rejected_key] = []
            st.rerun()
        return

    st.caption(
        "Recommendations are advisory. Accept only the transformations with a defensible "
        "business or statistical rationale."
    )
    st.dataframe(
        recommendations[
            [
                "recipe_group",
                "transform_type",
                "source_feature",
                "output_feature",
                "reason",
                "large_data_status",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    labels = [
        f"{row.transform_type} on {row.source_feature} - {row.reason}"
        for row in recommendations.itertuples(index=False)
    ]
    selected_label = st.selectbox(
        "Recommendation to review",
        options=labels,
        key=f"{workspace_state.keys.editor_key}_selected_recommendation",
    )
    selected_index = labels.index(selected_label)
    selected = recommendations.iloc[selected_index].to_dict()
    with st.expander("Edit before accepting", expanded=True):
        edited_output = st.text_input(
            "Output feature",
            value=str(selected.get("output_feature", "")),
            key=f"{workspace_state.keys.editor_key}_recommendation_output",
        )
        edited_parameter = st.text_input(
            "Parameter value",
            value=str(selected.get("parameter_value", "")),
            key=f"{workspace_state.keys.editor_key}_recommendation_parameter",
        )
        edited_notes = st.text_area(
            "Rationale / notes",
            value=str(selected.get("notes", "")),
            key=f"{workspace_state.keys.editor_key}_recommendation_notes",
            height=80,
        )
    accept_column, reject_column = st.columns(2)
    with accept_column:
        if st.button(
            "Accept Recommendation",
            key=f"{workspace_state.keys.editor_key}_accept_recommendation",
            width="stretch",
        ):
            selected["output_feature"] = edited_output
            selected["parameter_value"] = edited_parameter
            selected["notes"] = edited_notes
            _append_transformation_and_rerun(
                workspace_state=workspace_state,
                transformation_frame=transformation_frame,
                row=transformation_recommendation_to_row(selected),
            )
    with reject_column:
        if st.button(
            "Dismiss Recommendation",
            key=f"{workspace_state.keys.editor_key}_reject_recommendation",
            width="stretch",
        ):
            rejected.add(str(selected.get("recommendation_id", "")))
            st.session_state[rejected_key] = sorted(rejected)
            st.rerun()


def _render_recipe_library_tab(
    *,
    dataframe: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    workspace_state: WorkspaceState,
) -> None:
    catalog = build_transformation_recipe_catalog_frame()
    st.caption(
        "Recipe cards group available transformations by use case. Add a recipe, then "
        "finalize ordering and validation in Pipeline Review."
    )
    group_options = catalog["recipe_group"].drop_duplicates().tolist()
    selected_group = st.selectbox(
        "Recipe group",
        options=group_options,
        key=f"{workspace_state.keys.editor_key}_recipe_group",
    )
    group_catalog = catalog.loc[catalog["recipe_group"].eq(selected_group)].reset_index(
        drop=True
    )
    selected_type = st.selectbox(
        "Recipe",
        options=group_catalog["transform_type"].tolist(),
        format_func=lambda value: str(value).replace("_", " ").title(),
        key=f"{workspace_state.keys.editor_key}_recipe_type",
    )
    recipe = group_catalog.loc[group_catalog["transform_type"].eq(selected_type)].iloc[0]
    st.markdown(f"**What it does:** {recipe['what_it_does']}")
    st.caption(f"When to use: {recipe['when_to_use']}")
    st.caption(f"Parameters: {recipe['key_parameters']}")
    source_feature, secondary_feature, categorical_value = _render_source_feature_controls(
        dataframe=dataframe,
        transform_type=selected_type,
        key_prefix=f"{workspace_state.keys.editor_key}_recipe",
    )
    parameters = _render_transformation_parameter_controls(
        selected_type,
        key_prefix=f"{workspace_state.keys.editor_key}_recipe",
    )
    suggested_row = _build_ui_transformation_row(
        transform_type=selected_type,
        source_feature=source_feature,
        secondary_feature=secondary_feature,
        categorical_value=categorical_value,
        parameters=parameters,
        notes=f"Added from {selected_group} recipe.",
    )
    output_feature = st.text_input(
        "Output feature",
        value=suggested_row["output_feature"],
        key=f"{workspace_state.keys.editor_key}_recipe_output",
    )
    suggested_row["output_feature"] = output_feature
    if st.button(
        "Add Recipe To Pipeline",
        key=f"{workspace_state.keys.editor_key}_add_recipe",
        width="stretch",
        disabled=not source_feature,
    ):
        _append_transformation_and_rerun(
            workspace_state=workspace_state,
            transformation_frame=transformation_frame,
            row=suggested_row,
        )
    with st.expander("Browse all recipes", expanded=False):
        st.dataframe(catalog, width="stretch", hide_index=True)


def _render_custom_builder_tab(
    *,
    dataframe: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    workspace_state: WorkspaceState,
) -> None:
    catalog = build_transformation_recipe_catalog_frame()
    st.caption(
        "Use the builder for one governed transformation at a time. Parameter fields "
        "appear only when relevant to the selected transform."
    )
    selected_group = st.selectbox(
        "Purpose",
        options=catalog["recipe_group"].drop_duplicates().tolist(),
        key=f"{workspace_state.keys.editor_key}_builder_group",
    )
    builder_catalog = catalog.loc[catalog["recipe_group"].eq(selected_group)]
    transform_type = st.selectbox(
        "Transform type",
        options=builder_catalog["transform_type"].tolist(),
        format_func=lambda value: str(value).replace("_", " ").title(),
        key=f"{workspace_state.keys.editor_key}_builder_transform",
    )
    source_feature, secondary_feature, categorical_value = _render_source_feature_controls(
        dataframe=dataframe,
        transform_type=transform_type,
        key_prefix=f"{workspace_state.keys.editor_key}_builder",
    )
    parameters = _render_transformation_parameter_controls(
        transform_type,
        key_prefix=f"{workspace_state.keys.editor_key}_builder",
    )
    draft_row = _build_ui_transformation_row(
        transform_type=transform_type,
        source_feature=source_feature,
        secondary_feature=secondary_feature,
        categorical_value=categorical_value,
        parameters=parameters,
        notes="Added from Transformation Studio custom builder.",
    )
    output_feature = st.text_input(
        "Output feature",
        value=draft_row["output_feature"],
        key=f"{workspace_state.keys.editor_key}_builder_output",
    )
    notes = st.text_area(
        "Rationale / notes",
        value=draft_row["notes"],
        key=f"{workspace_state.keys.editor_key}_builder_notes",
        height=80,
    )
    draft_row["output_feature"] = output_feature
    draft_row["notes"] = notes
    preview = build_transformation_preview(
        dataframe=dataframe,
        source_feature=source_feature,
        transform_type=transform_type,
        parameter_value=parameters.get("parameter_value", ""),
        bin_edges=parameters.get("bin_edges", ""),
    )
    if preview["error"]:
        st.warning(preview["error"])
    else:
        st.dataframe(preview["summary"], width="stretch", hide_index=True)
        st.dataframe(preview["examples"], width="stretch", hide_index=True)
    st.code(preview.get("snippet", ""), language="python")
    if st.button(
        "Add Custom Transformation",
        key=f"{workspace_state.keys.editor_key}_add_custom_transform",
        width="stretch",
        disabled=not source_feature,
    ):
        _append_transformation_and_rerun(
            workspace_state=workspace_state,
            transformation_frame=transformation_frame,
            row=draft_row,
        )


def _render_pipeline_review_tab(
    *,
    dataframe: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    workspace_state: WorkspaceState,
) -> pd.DataFrame:
    st.caption(
        "Review the final ordered pipeline. Power users can open the advanced table "
        "editor for direct workbook-compatible edits."
    )
    if validation_frame.empty:
        st.info("No enabled transformations are configured yet.")
    else:
        st.dataframe(
            validation_frame.drop(columns=["generated_python"], errors="ignore"),
            width="stretch",
            hide_index=True,
        )
        selected_row = st.selectbox(
            "Pipeline row action",
            options=validation_frame["row"].astype(int).tolist(),
            key=f"{workspace_state.keys.editor_key}_pipeline_action_row",
        )
        duplicate_column, delete_column = st.columns(2)
        with duplicate_column:
            if st.button(
                "Duplicate Row",
                key=f"{workspace_state.keys.editor_key}_duplicate_transform_row",
                width="stretch",
            ):
                _duplicate_transformation_and_rerun(
                    workspace_state=workspace_state,
                    transformation_frame=transformation_frame,
                    row_number=int(selected_row),
                )
        with delete_column:
            if st.button(
                "Delete Row",
                key=f"{workspace_state.keys.editor_key}_delete_transform_row",
                width="stretch",
            ):
                _delete_transformation_and_rerun(
                    workspace_state=workspace_state,
                    transformation_frame=transformation_frame,
                    row_number=int(selected_row),
                )
        with st.expander("Generated Python/config snippets", expanded=False):
            st.dataframe(
                validation_frame[["row", "transform_type", "generated_python"]],
                width="stretch",
                hide_index=True,
            )
    with st.expander("Advanced table editor", expanded=False):
        edited_frame = st.data_editor(
            transformation_frame,
            key=workspace_state.keys.transformation_widget,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config=_transformation_editor_column_config(),
        )
    render_transformation_preview_panel(
        dataframe,
        edited_frame,
        editor_key=workspace_state.keys.editor_key,
    )
    return edited_frame


def _render_advanced_generation_controls(
    *,
    workspace_state: WorkspaceState,
    preset_transformations: Any | None,
    advanced_workspace: bool,
) -> dict[str, Any]:
    default_auto = bool(getattr(preset_transformations, "auto_interactions_enabled", False))
    default_numeric = bool(
        getattr(preset_transformations, "include_numeric_numeric_interactions", True)
    )
    default_categorical = bool(
        getattr(preset_transformations, "include_categorical_numeric_interactions", False)
    )
    default_max = int(getattr(preset_transformations, "max_auto_interactions", 5))
    default_levels = int(getattr(preset_transformations, "max_categorical_levels", 3))
    default_min_score = float(getattr(preset_transformations, "min_interaction_score", 0.0))
    with st.expander("Advanced Generation", expanded=False):
        if not advanced_workspace:
            st.caption(
                "Guided mode keeps generated interaction controls on preset defaults. "
                "Switch to Advanced to edit them."
            )
        auto_enabled = st.checkbox(
            "Auto-screen interaction terms",
            value=default_auto,
            help=(
                "Screens train-split interaction candidates and persists selected "
                "interaction features into the saved run config."
            ),
            disabled=not advanced_workspace,
            key=f"{workspace_state.keys.editor_key}_studio_auto_interactions",
        )
        numeric_interactions = st.checkbox(
            "Numeric-numeric interactions",
            value=default_numeric,
            disabled=not advanced_workspace or not auto_enabled,
            key=f"{workspace_state.keys.editor_key}_studio_numeric_interactions",
        )
        categorical_interactions = st.checkbox(
            "Categorical-numeric interactions",
            value=default_categorical,
            disabled=not advanced_workspace or not auto_enabled,
            key=f"{workspace_state.keys.editor_key}_studio_categorical_interactions",
        )
        max_auto = int(
            st.number_input(
                "Max auto interactions",
                min_value=1,
                max_value=20,
                value=default_max,
                step=1,
                disabled=not advanced_workspace or not auto_enabled,
                key=f"{workspace_state.keys.editor_key}_studio_max_interactions",
            )
        )
        max_levels = int(
            st.number_input(
                "Max categorical levels per feature",
                min_value=1,
                max_value=10,
                value=default_levels,
                step=1,
                disabled=not advanced_workspace or not auto_enabled,
                key=f"{workspace_state.keys.editor_key}_studio_max_levels",
            )
        )
        min_score = st.number_input(
            "Min interaction score",
            min_value=0.0,
            max_value=1.0,
            value=default_min_score,
            step=0.01,
            format="%.2f",
            disabled=not advanced_workspace or not auto_enabled,
            key=f"{workspace_state.keys.editor_key}_studio_min_interaction_score",
        )
    return {
        "auto_interactions_enabled": auto_enabled,
        "include_numeric_numeric_interactions": numeric_interactions,
        "include_categorical_numeric_interactions": categorical_interactions,
        "max_auto_interactions": max_auto,
        "max_categorical_levels": max_levels,
        "min_interaction_score": float(min_score),
    }


def _transformation_editor_column_config() -> dict[str, object]:
    return {
        "enabled": st.column_config.CheckboxColumn("Enabled"),
        "transform_type": st.column_config.SelectboxColumn(
            "Type",
            options=SUPPORTED_TRANSFORMATION_TYPES,
        ),
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
    }


def _render_source_feature_controls(
    *,
    dataframe: pd.DataFrame,
    transform_type: str,
    key_prefix: str,
) -> tuple[str, str, str]:
    columns = [str(column) for column in dataframe.columns]
    if not columns:
        return "", "", ""
    categorical_value = ""
    if transform_type in {
        "row_missing_count",
        "row_missing_share",
        "any_missing_flag",
    }:
        selected_sources = st.multiselect(
            "Source features",
            options=columns,
            default=columns[: min(3, len(columns))],
            key=f"{key_prefix}_source_features",
        )
        source_feature = ", ".join(selected_sources)
    else:
        source_feature = st.selectbox(
            "Source feature",
            options=columns,
            key=f"{key_prefix}_source_feature",
        )
    secondary_feature = ""
    if transform_type in {
        "ratio",
        "safe_ratio",
        "margin_ratio",
        "debt_service_ratio",
        "add",
        "subtract",
        "product",
        "interaction",
        "date_age_days",
        "date_age_months",
    }:
        secondary_options = [column for column in columns if column != source_feature]
        if secondary_options:
            secondary_feature = st.selectbox(
                "Secondary feature",
                options=secondary_options,
                key=f"{key_prefix}_secondary_feature",
            )
    if transform_type == "interaction":
        categorical_value = st.text_input(
            "Categorical value filter",
            value="",
            key=f"{key_prefix}_categorical_value_note",
            help=(
                "Optional. Use when the interaction should apply to one category level."
            ),
        )
    return source_feature, secondary_feature, categorical_value


def _render_transformation_parameter_controls(
    transform_type: str,
    *,
    key_prefix: str,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    if transform_type == "winsorize":
        left, right = st.columns(2)
        with left:
            parameters["lower_quantile"] = st.number_input(
                "Lower quantile",
                min_value=0.0,
                max_value=0.49,
                value=0.01,
                step=0.01,
                key=f"{key_prefix}_lower_quantile",
            )
        with right:
            parameters["upper_quantile"] = st.number_input(
                "Upper quantile",
                min_value=0.51,
                max_value=1.0,
                value=0.99,
                step=0.01,
                key=f"{key_prefix}_upper_quantile",
            )
    if transform_type in {
        "capped_zscore",
        "piecewise_linear",
        "power",
        "natural_spline",
        "quantile_bins",
        "equal_width_bins",
        "monotonic_bins",
        "woe_encoding",
        "bad_rate_encoding",
        "rare_category_collapse",
        "target_encoding",
        "date_fiscal_quarter",
    }:
        default_parameter = _default_transformation_parameter(transform_type)
        parameters["parameter_value"] = st.number_input(
            "Parameter value",
            value=default_parameter,
            step=1.0 if default_parameter >= 1 else 0.01,
            key=f"{key_prefix}_parameter_value",
        )
    if transform_type in {"manual_bins", "woe_encoding", "bad_rate_encoding"}:
        parameters["bin_edges"] = st.text_input(
            "Bin edges",
            value="" if transform_type != "manual_bins" else "0.2, 0.5, 0.8",
            help="Comma-separated internal numeric edges. Omit -inf and inf.",
            key=f"{key_prefix}_bin_edges",
        )
    if transform_type in {"lag", "difference", "pct_change"}:
        parameters["lag_periods"] = int(
            st.number_input(
                "Lag periods",
                min_value=1,
                value=1,
                step=1,
                key=f"{key_prefix}_lag_periods",
            )
        )
    if transform_type == "ewma" or transform_type.startswith("rolling_"):
        parameters["window_size"] = int(
            st.number_input(
                "Window size",
                min_value=2,
                value=3,
                step=1,
                key=f"{key_prefix}_window_size",
            )
        )
    return parameters


def _default_transformation_parameter(transform_type: str) -> float:
    defaults = {
        "capped_zscore": 3.0,
        "piecewise_linear": 0.0,
        "power": 2.0,
        "natural_spline": 4.0,
        "quantile_bins": 5.0,
        "equal_width_bins": 5.0,
        "monotonic_bins": 5.0,
        "woe_encoding": 5.0,
        "bad_rate_encoding": 5.0,
        "rare_category_collapse": 0.01,
        "target_encoding": 20.0,
        "date_fiscal_quarter": 1.0,
    }
    return defaults.get(transform_type, 0.0)


def _build_ui_transformation_row(
    *,
    transform_type: str,
    source_feature: str,
    secondary_feature: str,
    categorical_value: str,
    parameters: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    return build_transformation_row(
        transform_type=transform_type,
        source_feature=source_feature,
        secondary_feature=secondary_feature,
        categorical_value=categorical_value,
        lower_quantile=parameters.get("lower_quantile", ""),
        upper_quantile=parameters.get("upper_quantile", ""),
        parameter_value=parameters.get("parameter_value", ""),
        window_size=parameters.get("window_size", ""),
        lag_periods=parameters.get("lag_periods", ""),
        bin_edges=parameters.get("bin_edges", ""),
        notes=notes,
    )


def _append_transformation_and_rerun(
    *,
    workspace_state: WorkspaceState,
    transformation_frame: pd.DataFrame,
    row: dict[str, Any],
) -> None:
    updated = pd.concat(
        [normalize_transformation_frame(transformation_frame), pd.DataFrame([row])],
        ignore_index=True,
    )
    st.session_state.pop(workspace_state.keys.transformation_widget, None)
    store_workspace_frame(
        workspace_state.keys.transformation_frame,
        normalize_transformation_frame(updated),
    )
    st.rerun()


def _duplicate_transformation_and_rerun(
    *,
    workspace_state: WorkspaceState,
    transformation_frame: pd.DataFrame,
    row_number: int,
) -> None:
    normalized = normalize_transformation_frame(transformation_frame)
    row_index = row_number - 1
    if row_index < 0 or row_index >= len(normalized):
        return
    duplicated = normalized.iloc[[row_index]].copy(deep=True)
    duplicated.loc[:, "output_feature"] = (
        duplicated["output_feature"].astype(str).str.strip() + "_copy"
    )
    updated = pd.concat([normalized, duplicated], ignore_index=True)
    st.session_state.pop(workspace_state.keys.transformation_widget, None)
    store_workspace_frame(workspace_state.keys.transformation_frame, updated)
    st.rerun()


def _delete_transformation_and_rerun(
    *,
    workspace_state: WorkspaceState,
    transformation_frame: pd.DataFrame,
    row_number: int,
) -> None:
    normalized = normalize_transformation_frame(transformation_frame)
    row_index = row_number - 1
    if row_index < 0 or row_index >= len(normalized):
        return
    updated = normalized.drop(index=normalized.index[row_index]).reset_index(drop=True)
    st.session_state.pop(workspace_state.keys.transformation_widget, None)
    store_workspace_frame(workspace_state.keys.transformation_frame, updated)
    st.rerun()


def render_transformation_preview_panel(
    dataframe: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    *,
    editor_key: str,
) -> None:
    """Renders a lightweight before/after preview for one transformation."""

    with st.expander("Transformation Preview", expanded=False):
        st.caption(
            "Preview one transformation on a capped sample before running the full workflow. "
            "This is exploratory only; governed transforms are still fit inside the pipeline."
        )
        configured_rows = _configured_transformation_rows(transformation_frame, dataframe)
        if configured_rows:
            labels = [row["label"] for row in configured_rows]
            selected_label = st.selectbox(
                "Configured transformation",
                options=labels,
                key=f"{editor_key}_configured_transform_preview",
            )
            selected_row = configured_rows[labels.index(selected_label)]
            transform_type = selected_row["transform_type"]
            source_feature = selected_row["source_feature"]
            parameter_value = selected_row.get("parameter_value")
            bin_edges = selected_row.get("bin_edges", "")
        else:
            st.info(
                "No enabled configured transformation is available yet. Use the controls "
                "below for an ad hoc preview."
            )
            numeric_columns = [
                column
                for column in dataframe.columns
                if pd.api.types.is_numeric_dtype(dataframe[column])
            ]
            if not numeric_columns:
                st.warning("No numeric columns are available for transformation preview.")
                return
            source_feature = st.selectbox(
                "Feature",
                options=numeric_columns,
                key=f"{editor_key}_adhoc_transform_feature",
            )
            transform_type = st.selectbox(
                "Preview transform",
                options=[
                    "winsorize",
                    "log1p",
                    "signed_log1p",
                    "standard_scale",
                    "robust_scale",
                    "min_max_scale",
                    "quantile_bins",
                    "equal_width_bins",
                ],
                key=f"{editor_key}_adhoc_transform_type",
            )
            parameter_value = 5
            bin_edges = ""
        preview = build_transformation_preview(
            dataframe=dataframe,
            source_feature=str(source_feature),
            transform_type=str(transform_type),
            parameter_value=parameter_value,
            bin_edges=bin_edges,
        )
        if preview["error"]:
            st.warning(preview["error"])
            return
        st.caption("Summary statistics before and after the preview transform.")
        st.dataframe(preview["summary"], width="stretch", hide_index=True)
        if not preview["examples"].empty:
            st.caption("Example input/output rows from the preview sample.")
            st.dataframe(preview["examples"], width="stretch", hide_index=True)
        chart_frame = preview["chart"]
        if not chart_frame.empty:
            st.caption("Before/after distribution preview.")
            st.bar_chart(chart_frame.set_index("bucket"))
        st.code(preview.get("snippet", ""), language="python")


def build_transformation_preview(
    *,
    dataframe: pd.DataFrame,
    source_feature: str,
    transform_type: str,
    parameter_value: Any = None,
    bin_edges: Any = None,
) -> dict[str, Any]:
    """Returns summary and chart data for a lightweight transformation preview."""

    if source_feature not in dataframe.columns:
        return {
            "error": f"`{source_feature}` is not in the dataset.",
            "summary": pd.DataFrame(),
            "chart": pd.DataFrame(),
            "examples": pd.DataFrame(),
            "snippet": "",
        }
    source = dataframe[source_feature].head(DATA_REVIEW_PROFILE_ROWS)
    values = pd.to_numeric(source, errors="coerce")
    if values.dropna().empty:
        return {
            "error": f"`{source_feature}` has no numeric values to preview.",
            "summary": pd.DataFrame(),
            "chart": pd.DataFrame(),
            "examples": pd.DataFrame(),
            "snippet": "",
        }
    try:
        transformed = _apply_preview_transform(values, transform_type, parameter_value, bin_edges)
    except Exception as exc:
        return {
            "error": f"Could not preview `{transform_type}`: {exc}",
            "summary": pd.DataFrame(),
            "chart": pd.DataFrame(),
            "examples": pd.DataFrame(),
            "snippet": "",
        }
    summary = _preview_summary(source_feature, transform_type, values, transformed)
    chart = _preview_chart(values, transformed)
    examples = _preview_examples(source, transformed)
    snippet = (
        "TransformationSpec("
        f"transform_type=TransformationType('{transform_type}'), "
        f"source_feature='{source_feature}', "
        f"parameter_value={parameter_value!r}, "
        f"bin_edges={bin_edges!r})"
    )
    return {
        "error": "",
        "summary": summary,
        "chart": chart,
        "examples": examples,
        "snippet": snippet,
    }


def _active_schema_rows(schema_frame: pd.DataFrame) -> pd.DataFrame:
    if schema_frame.empty:
        return pd.DataFrame(columns=schema_frame.columns)
    working = schema_frame.copy(deep=False)
    if "enabled" not in working.columns:
        working["enabled"] = True
    return working.loc[working["enabled"].map(_is_enabled_value)].copy(deep=False)


def _role_columns(schema_frame: pd.DataFrame, role: str) -> list[str]:
    if schema_frame.empty or "role" not in schema_frame.columns:
        return []
    name_column = "name" if "name" in schema_frame.columns else schema_frame.columns[0]
    return (
        schema_frame.loc[schema_frame["role"].astype(str).str.lower().eq(role), name_column]
        .dropna()
        .astype(str)
        .tolist()
    )


def _high_cardinality_columns(dataframe: pd.DataFrame) -> list[str]:
    if dataframe.empty:
        return []
    threshold = max(50, int(len(dataframe) * 0.2))
    candidates = dataframe.select_dtypes(include=["object", "string", "category"]).columns
    return [
        str(column)
        for column in candidates
        if int(dataframe[column].nunique(dropna=True)) > threshold
    ]


def _date_coverage_text(dataframe: pd.DataFrame, date_columns: list[str]) -> str:
    if not date_columns:
        return "No date role selected."
    parts: list[str] = []
    for column in date_columns:
        if column not in dataframe.columns:
            parts.append(f"{column}: missing from dataframe")
            continue
        values = pd.to_datetime(dataframe[column], errors="coerce").dropna()
        if values.empty:
            parts.append(f"{column}: no valid dates")
        else:
            parts.append(f"{column}: {values.min().date()} to {values.max().date()}")
    return "; ".join(parts)


def _target_distribution_text(dataframe: pd.DataFrame, target_columns: list[str]) -> str:
    if len(target_columns) != 1 or target_columns[0] not in dataframe.columns:
        return "Unavailable"
    counts = dataframe[target_columns[0]].value_counts(dropna=False).head(6)
    return ", ".join(f"{value}: {count:,}" for value, count in counts.items())


def _contract_row(
    area: str,
    passed: bool,
    detail: str,
    recommended_action: str,
    *,
    warning_only: bool = False,
) -> dict[str, str]:
    if passed:
        status = "pass"
    else:
        status = "warning" if warning_only else "blocker"
    return {
        "area": area,
        "status": status,
        "detail": detail,
        "recommended_action": "No action needed." if passed else recommended_action,
    }


def _render_local_metric_cards(cards: list[dict[str, str]]) -> None:
    columns = st.columns(min(4, max(len(cards), 1)))
    for index, card in enumerate(cards):
        with columns[index % len(columns)]:
            st.metric(card["label"], card["value"])


def _configured_transformation_rows(
    transformation_frame: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    if transformation_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row_index, row in transformation_frame.fillna("").iterrows():
        if "enabled" in row and not _is_enabled_value(row["enabled"]):
            continue
        transform_type = str(row.get("transform_type", "")).strip()
        source_feature = str(row.get("source_feature", "")).strip()
        if not transform_type or source_feature not in dataframe.columns:
            continue
        rows.append(
            {
                "label": f"{row_index + 1}: {transform_type}({source_feature})",
                "transform_type": transform_type,
                "source_feature": source_feature,
                "parameter_value": row.get("parameter_value", ""),
                "bin_edges": row.get("bin_edges", ""),
            }
        )
    return rows


def _is_enabled_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _apply_preview_transform(
    values: pd.Series,
    transform_type: str,
    parameter_value: Any,
    bin_edges: Any,
) -> pd.Series:
    transform_type = transform_type.strip().lower()
    if transform_type == "winsorize":
        return values.clip(values.quantile(0.01), values.quantile(0.99))
    if transform_type == "log1p":
        return pd.Series(np.where(values > -1, np.log1p(values), np.nan), index=values.index)
    if transform_type == "signed_log1p":
        return np.sign(values) * np.log1p(values.abs())
    if transform_type == "standard_scale":
        std_value = values.std(ddof=0)
        return (values - values.mean()) / std_value if std_value else values * 0
    if transform_type == "robust_scale":
        iqr = values.quantile(0.75) - values.quantile(0.25)
        return (values - values.median()) / iqr if iqr else values * 0
    if transform_type == "min_max_scale":
        value_range = values.max() - values.min()
        return (values - values.min()) / value_range if value_range else values * 0
    if transform_type == "quantile_bins":
        bin_count = _coerce_preview_bin_count(parameter_value)
        return pd.qcut(values, q=bin_count, duplicates="drop").astype("string")
    if transform_type == "equal_width_bins":
        bin_count = _coerce_preview_bin_count(parameter_value)
        return pd.cut(values, bins=bin_count, duplicates="drop").astype("string")
    if transform_type == "manual_bins":
        parsed_edges = _parse_preview_edges(bin_edges)
        return pd.cut(
            values,
            bins=[-np.inf, *parsed_edges, np.inf],
            include_lowest=True,
            duplicates="drop",
        ).astype("string")
    raise ValueError(f"{transform_type} is not supported in the lightweight preview.")


def _coerce_preview_bin_count(value: Any) -> int:
    try:
        return max(2, min(20, int(float(value))))
    except (TypeError, ValueError):
        return 5


def _parse_preview_edges(value: Any) -> list[float]:
    if isinstance(value, list):
        return sorted(float(item) for item in value)
    text = str(value or "").strip()
    if not text:
        raise ValueError("manual_bins preview requires bin_edges.")
    return sorted(float(item.strip()) for item in text.split(",") if item.strip())


def _preview_summary(
    source_feature: str,
    transform_type: str,
    before: pd.Series,
    after: pd.Series,
) -> pd.DataFrame:
    before_numeric = pd.to_numeric(before, errors="coerce")
    after_numeric = pd.to_numeric(after, errors="coerce")
    rows = [
        {
            "field": source_feature,
            "transform": transform_type,
            "version": "before",
            "non_missing": int(before.notna().sum()),
            "missing": int(before.isna().sum()),
            "mean": _format_optional_float(before_numeric.mean()),
            "std": _format_optional_float(before_numeric.std(ddof=0)),
            "min": _format_optional_float(before_numeric.min()),
            "max": _format_optional_float(before_numeric.max()),
            "unique": int(before.nunique(dropna=True)),
        },
        {
            "field": source_feature,
            "transform": transform_type,
            "version": "after",
            "non_missing": int(after.notna().sum()),
            "missing": int(after.isna().sum()),
            "mean": _format_optional_float(after_numeric.mean()),
            "std": _format_optional_float(after_numeric.std(ddof=0)),
            "min": _format_optional_float(after_numeric.min()),
            "max": _format_optional_float(after_numeric.max()),
            "unique": int(after.nunique(dropna=True)),
        },
    ]
    return pd.DataFrame(rows)


def _preview_examples(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    example_count = min(10, len(before))
    return pd.DataFrame(
        {
            "input_value": before.head(example_count).reset_index(drop=True),
            "preview_output": after.head(example_count).reset_index(drop=True),
        }
    )


def _preview_chart(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    after_numeric = pd.to_numeric(after, errors="coerce")
    if after_numeric.dropna().empty:
        counts = after.astype("string").fillna("Missing").value_counts().head(12)
        return pd.DataFrame({"bucket": counts.index.astype(str), "count": counts.values})
    before_bucket = pd.cut(pd.to_numeric(before, errors="coerce"), bins=10, duplicates="drop")
    after_bucket = pd.cut(after_numeric, bins=10, duplicates="drop")
    before_counts = before_bucket.astype("string").value_counts().sort_index()
    after_counts = after_bucket.astype("string").value_counts().sort_index()
    rows = []
    for bucket, count in before_counts.items():
        rows.append({"bucket": f"Before {bucket}", "count": int(count)})
    for bucket, count in after_counts.items():
        rows.append({"bucket": f"After {bucket}", "count": int(count)})
    return pd.DataFrame(rows)


def _format_optional_float(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return ""


def render_builder_workspace(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
    workspace_state: WorkspaceState,
    preset_transformations: Any | None = None,
    advanced_workspace: bool = False,
    target_mode: str = "binary",
    model_type: str = "logistic_regression",
    data_structure: str = "cross_sectional",
) -> dict[str, Any]:
    section_options = [
        "Dataset Preview",
        "Data Review",
        "Column Designer",
        "Feature Dictionary",
        "Transformation Studio",
        "Template Workbook",
    ]

    edited_schema = workspace_state.schema_frame
    edited_feature_dictionary = workspace_state.feature_dictionary_frame
    edited_transformations = workspace_state.transformation_frame
    feature_review_frame = workspace_state.feature_review_frame
    scorecard_override_frame = workspace_state.scorecard_override_frame
    control_prefix = workspace_state.keys.editor_key
    default_auto_interactions = bool(
        getattr(preset_transformations, "auto_interactions_enabled", False)
    )
    default_numeric_interactions = bool(
        getattr(preset_transformations, "include_numeric_numeric_interactions", True)
    )
    default_categorical_interactions = bool(
        getattr(preset_transformations, "include_categorical_numeric_interactions", False)
    )
    default_max_auto_interactions = int(
        getattr(preset_transformations, "max_auto_interactions", 5)
    )
    default_max_categorical_levels = int(
        getattr(preset_transformations, "max_categorical_levels", 3)
    )
    default_min_interaction_score = float(
        getattr(preset_transformations, "min_interaction_score", 0.0)
    )
    transformation_controls = {
        "auto_interactions_enabled": bool(
            st.session_state.get(
                f"{control_prefix}_studio_auto_interactions",
                default_auto_interactions,
            )
        ),
        "include_numeric_numeric_interactions": bool(
            st.session_state.get(
                f"{control_prefix}_studio_numeric_interactions",
                default_numeric_interactions,
            )
        ),
        "include_categorical_numeric_interactions": bool(
            st.session_state.get(
                f"{control_prefix}_studio_categorical_interactions",
                default_categorical_interactions,
            )
        ),
        "max_auto_interactions": int(
            st.session_state.get(
                f"{control_prefix}_studio_max_interactions",
                default_max_auto_interactions,
            )
        ),
        "max_categorical_levels": int(
            st.session_state.get(
                f"{control_prefix}_studio_max_levels",
                default_max_categorical_levels,
            )
        ),
        "min_interaction_score": float(
            st.session_state.get(
                f"{control_prefix}_studio_min_interaction_score",
                default_min_interaction_score,
            )
        ),
    }

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

    elif selected_section == "Data Review":
        render_data_review_panel(
            dataframe=dataframe,
            data_source_label=data_source_label,
            schema_frame=workspace_state.schema_frame,
        )

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

    elif selected_section == "Transformation Studio":
        edited_transformations, transformation_controls = render_transformation_studio(
            dataframe=dataframe,
            schema_frame=edited_schema,
            transformation_frame=workspace_state.transformation_frame,
            workspace_state=workspace_state,
            preset_transformations=preset_transformations,
            advanced_workspace=advanced_workspace,
            target_mode=target_mode,
            model_type=model_type,
            data_structure=data_structure,
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
        "transformation_controls": transformation_controls,
    }
