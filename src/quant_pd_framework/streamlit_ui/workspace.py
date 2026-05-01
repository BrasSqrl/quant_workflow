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
    frames_equivalent,
    load_template_workbook,
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
        st.dataframe(preview["summary"], width="stretch", hide_index=True)
        chart_frame = preview["chart"]
        if not chart_frame.empty:
            st.bar_chart(chart_frame.set_index("bucket"))


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
        }
    source = dataframe[source_feature].head(DATA_REVIEW_PROFILE_ROWS)
    values = pd.to_numeric(source, errors="coerce")
    if values.dropna().empty:
        return {
            "error": f"`{source_feature}` has no numeric values to preview.",
            "summary": pd.DataFrame(),
            "chart": pd.DataFrame(),
        }
    try:
        transformed = _apply_preview_transform(values, transform_type, parameter_value, bin_edges)
    except Exception as exc:
        return {
            "error": f"Could not preview `{transform_type}`: {exc}",
            "summary": pd.DataFrame(),
            "chart": pd.DataFrame(),
        }
    summary = _preview_summary(source_feature, transform_type, values, transformed)
    chart = _preview_chart(values, transformed)
    return {"error": "", "summary": summary, "chart": chart}


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
) -> dict[str, pd.DataFrame]:
    section_options = [
        "Dataset Preview",
        "Data Review",
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
            },
        )
        with st.expander("Transformation Guidance", expanded=False):
            st.markdown(
                """
                - **Numeric shape:** `winsorize`, `log1p`, `signed_log1p`, `box_cox`,
                  `yeo_johnson`, `sqrt`, `reciprocal`, `square`, `power`, and
                  `piecewise_linear` reshape skew, tails, and nonlinear effects.
                - **Scaling and rank:** `standard_scale`, `robust_scale`,
                  `min_max_scale`, `percentile_rank`, `normal_score`,
                  `center_mean`, and `center_median` normalize magnitude while
                  fitting parameters on train only.
                - **Combinations:** `ratio`, `safe_ratio`, `margin_ratio`,
                  `debt_service_ratio`, `add`, `subtract`, `product`, and
                  `interaction` combine two fields using the secondary feature.
                - **Time-aware:** `lag`, `difference`, `pct_change`, `ewma`,
                  rolling, expanding, cumulative, baseline, and event-distance
                  transforms respect configured date/entity ordering.
                - **Binning and encoding:** manual, quantile, equal-width,
                  monotonic, WOE, bad-rate, rare-category, frequency, ordinal,
                  and target encodings support scorecards and categorical signal.
                - **Date and missingness:** date-part, fiscal-quarter, age, row
                  missing count/share, and any-missing flags document structural
                  information without editing source data.
                """
            )
        render_transformation_preview_panel(
            dataframe,
            edited_transformations,
            editor_key=workspace_state.keys.editor_key,
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
