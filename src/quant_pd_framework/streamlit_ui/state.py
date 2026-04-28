"""Typed Streamlit session helpers and shared UI utility functions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_pd_framework import TargetMode
from quant_pd_framework.presentation import plotly_display_config, prepare_display_table

LAST_RUN_SNAPSHOT_KEY = "last_run_snapshot"


def get_or_initialize_frame(
    state_key: str,
    builder: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    if state_key not in st.session_state:
        st.session_state[state_key] = builder()
    return st.session_state[state_key].copy(deep=True)


def store_workspace_frame(state_key: str, frame: pd.DataFrame) -> None:
    st.session_state[state_key] = frame.copy(deep=True)


def set_last_run_snapshot(snapshot: dict[str, Any] | None) -> None:
    st.session_state[LAST_RUN_SNAPSHOT_KEY] = snapshot


def get_last_run_snapshot() -> dict[str, Any] | None:
    return st.session_state.get(LAST_RUN_SNAPSHOT_KEY)


@dataclass(frozen=True)
class WorkspaceStateKeys:
    editor_key: str
    schema_frame: str
    feature_dictionary_widget: str
    feature_dictionary_frame: str
    transformation_widget: str
    transformation_frame: str
    feature_review_widget: str
    feature_review_frame: str
    scorecard_override_widget: str
    scorecard_override_frame: str

    @classmethod
    def from_editor_key(cls, editor_key: str) -> WorkspaceStateKeys:
        return cls(
            editor_key=editor_key,
            schema_frame=f"{editor_key}_schema_frame",
            feature_dictionary_widget=f"{editor_key}_feature_dictionary_widget",
            feature_dictionary_frame=f"{editor_key}_feature_dictionary_frame",
            transformation_widget=f"{editor_key}_transformation_widget",
            transformation_frame=f"{editor_key}_transformation_frame",
            feature_review_widget=f"{editor_key}_feature_review_widget",
            feature_review_frame=f"{editor_key}_feature_review_frame",
            scorecard_override_widget=f"{editor_key}_scorecard_override_widget",
            scorecard_override_frame=f"{editor_key}_scorecard_override_frame",
        )


@dataclass
class WorkspaceState:
    keys: WorkspaceStateKeys
    schema_frame: pd.DataFrame
    feature_dictionary_frame: pd.DataFrame
    transformation_frame: pd.DataFrame
    feature_review_frame: pd.DataFrame
    scorecard_override_frame: pd.DataFrame


def build_plotly_key(*parts: Any) -> str:
    signature = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"plotly_{digest}"


def render_plotly_figure(figure: go.Figure, *, key: str) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        config=plotly_display_config(),
        key=key,
    )


@st.cache_data(show_spinner=False)
def read_text_artifact(path_value: str) -> str:
    return Path(path_value).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def read_binary_artifact(path_value: str) -> bytes:
    return Path(path_value).read_bytes()


def build_run_snapshot(context, config_dict: dict[str, Any]) -> dict[str, Any]:
    artifact_paths = {
        key: (str(value) if value is not None else "") for key, value in context.artifacts.items()
    }
    score_column = infer_snapshot_score_column(context)
    lazy_results = bool(context.config.performance.lazy_streamlit_results)
    prediction_row_cap = int(context.config.diagnostics.max_plot_rows)
    table_row_cap = max(
        int(context.config.diagnostics.max_plot_rows),
        int(context.config.performance.ui_preview_rows),
    )
    prediction_snapshots = {
        key: _snapshot_frame_for_session(
            _compact_prediction_frame_for_session(context, value),
            lazy=lazy_results,
            row_cap=prediction_row_cap,
            random_state=context.config.split.random_state,
        )
        for key, value in context.predictions.items()
    }
    diagnostics_table_snapshots = {
        key: _snapshot_frame_for_session(
            value,
            lazy=lazy_results,
            row_cap=table_row_cap,
            random_state=context.config.split.random_state,
        )
        for key, value in context.diagnostics_tables.items()
    }
    return {
        "run_id": context.run_id,
        "metrics": context.metrics,
        "feature_importance": (
            context.feature_importance.copy(deep=True)
            if context.feature_importance is not None
            else pd.DataFrame()
        ),
        "backtest_summary": (
            context.backtest_summary.copy(deep=True)
            if context.backtest_summary is not None
            else pd.DataFrame()
        ),
        "predictions": prediction_snapshots,
        "warnings": list(context.warnings),
        "events": list(context.events),
        "artifacts": artifact_paths,
        "config": config_dict,
        "report_path": artifact_paths.get("report", ""),
        "diagnostics_tables": diagnostics_table_snapshots,
        "statistical_tests": context.statistical_tests,
        "visualizations": context.visualizations,
        "streamlit_snapshot": {
            "lazy_results": lazy_results,
            "prediction_row_cap": prediction_row_cap,
            "diagnostics_table_row_cap": table_row_cap,
            "prediction_row_counts": {
                key: int(value.shape[0]) for key, value in context.predictions.items()
            },
            "prediction_rows_stored": {
                key: int(value.shape[0]) for key, value in prediction_snapshots.items()
            },
            "diagnostics_table_row_counts": {
                key: int(value.shape[0]) for key, value in context.diagnostics_tables.items()
            },
            "diagnostics_table_rows_stored": {
                key: int(value.shape[0]) for key, value in diagnostics_table_snapshots.items()
            },
        },
        "feature_columns": list(context.feature_columns),
        "numeric_features": list(context.numeric_features),
        "categorical_features": list(context.categorical_features),
        "target_column": context.target_column,
        "target_mode": context.config.target.mode.value,
        "execution_mode": context.config.execution.mode.value,
        "model_type": context.config.model.model_type.value,
        "run_timing": {
            "started_at_utc": context.metadata.get("run_started_at_utc", ""),
            "completed_at_utc": context.metadata.get("run_completed_at_utc", ""),
            "elapsed_seconds": context.metadata.get("run_elapsed_seconds"),
            "backtest_split": context.metadata.get("backtest_split", ""),
        },
        "run_diagnostics": build_run_diagnostics(context),
        "labels_available": bool(context.metadata.get("labels_available", False)),
        "input_shape": dict(context.metadata.get("input_shape", {})),
        "feature_summary": dict(context.metadata.get("feature_summary", {})),
        "split_summary": dict(context.metadata.get("split_summary", {})),
        "subset_search_best_candidate": dict(
            context.metadata.get("subset_search_best_candidate", {})
        ),
        "threshold": context.config.model.threshold,
        "score_column": score_column,
        "prediction_column": str(context.metadata.get("prediction_column", "predicted_class")),
        "date_column": context.config.split.date_column,
        "default_segment_column": context.config.diagnostics.default_segment_column,
        "include_enhanced_report_visuals": bool(
            context.config.artifacts.include_enhanced_report_visuals
        ),
        "include_advanced_visual_analytics": bool(
            context.config.artifacts.include_advanced_visual_analytics
        ),
        "keep_all_checkpoints": bool(context.config.artifacts.keep_all_checkpoints),
    }


def build_run_diagnostics(context) -> dict[str, Any]:
    """Summarizes run-level timing and tracked dataframe memory for the UI."""

    memory_records: list[dict[str, Any]] = []
    for step_record in getattr(context, "debug_trace", []):
        if not isinstance(step_record, dict):
            continue
        for snapshot_key in ("before", "after"):
            snapshot = step_record.get(snapshot_key)
            if isinstance(snapshot, dict):
                memory_records.append(snapshot)
    peak_tracked_bytes = _max_debug_memory_value(
        memory_records,
        "tracked_dataframe_memory_bytes",
    )
    return {
        "elapsed_seconds": context.metadata.get("run_elapsed_seconds"),
        "peak_tracked_dataframe_memory_bytes": peak_tracked_bytes,
        "peak_tracked_dataframe_memory_gb": (
            round(peak_tracked_bytes / (1024**3), 6)
            if peak_tracked_bytes is not None
            else None
        ),
        "memory_profile_available": peak_tracked_bytes is not None,
    }


def _max_debug_memory_value(
    memory_records: list[dict[str, Any]],
    field_name: str,
) -> int | None:
    values: list[int] = []
    for record in memory_records:
        value = record.get(field_name)
        if value is None:
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _snapshot_frame_for_session(
    frame: pd.DataFrame,
    *,
    lazy: bool,
    row_cap: int,
    random_state: int,
) -> pd.DataFrame:
    if not lazy or len(frame) <= row_cap:
        return frame.copy(deep=True)
    return frame.sample(row_cap, random_state=random_state).sort_index().copy(deep=True)


def _compact_prediction_frame_for_session(context, frame: pd.DataFrame) -> pd.DataFrame:
    """Keeps GUI session state focused on scores and audit identifiers."""

    if not context.config.artifacts.compact_prediction_exports:
        return frame
    preferred_columns = _prediction_display_columns(context, frame)
    if not preferred_columns:
        return frame
    return frame.loc[:, preferred_columns].copy(deep=False)


def _prediction_display_columns(context, frame: pd.DataFrame) -> list[str]:
    schema_columns = []
    for spec in context.config.schema.column_specs:
        if spec.role.value in {"identifier", "date"} and spec.name in frame.columns:
            schema_columns.append(spec.name)
    configured_columns = [
        context.config.split.date_column,
        context.config.split.entity_column,
        context.target_column,
        context.config.diagnostics.default_segment_column,
        *context.metadata.get("hazard_time_features", []),
        *_low_cardinality_segment_columns(context, frame),
        "split",
        "predicted_probability",
        "predicted_probability_recommended",
        "predicted_class",
        "predicted_value",
        "residual",
        "prediction_score",
        "scorecard_score",
        "scorecard_points",
    ]
    columns = [*schema_columns, *configured_columns]
    return list(dict.fromkeys(column for column in columns if column and column in frame.columns))


def _low_cardinality_segment_columns(context, frame: pd.DataFrame) -> list[str]:
    return [
        feature_name
        for feature_name in context.categorical_features
        if feature_name in frame.columns
        and frame[feature_name].nunique(dropna=True)
        <= context.config.performance.max_categorical_cardinality
    ]


def infer_snapshot_score_column(context) -> str:
    available_columns = {
        str(column_name)
        for prediction_frame in context.predictions.values()
        for column_name in prediction_frame.columns
    }
    candidates: list[str] = []
    configured_score_column = context.metadata.get("score_column")
    if configured_score_column:
        candidates.append(str(configured_score_column))

    target_mode = context.config.target.mode.value
    if target_mode == TargetMode.BINARY.value:
        recommended_score_column = context.metadata.get("recommended_calibration_score_column")
        if recommended_score_column:
            candidates.append(str(recommended_score_column))
        candidates.extend(
            [
                "predicted_probability_recommended",
                "predicted_probability",
                "prediction_score",
            ]
        )
        fallback = "predicted_probability"
    else:
        candidates.extend(["predicted_value", "prediction_score"])
        fallback = "predicted_value"

    for candidate in dict.fromkeys(candidates):
        if candidate in available_columns:
            return candidate
    return fallback


def render_download_button(label: str, payload: Any, file_name: str, mime: str) -> None:
    if mime == "application/json":
        data = json.dumps(payload, indent=2)
    else:
        data = payload
    st.download_button(label, data=data, file_name=file_name, mime=mime)


def prepare_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    display_table = prepare_display_table(table).copy()
    for column_name in display_table.columns:
        series = display_table[column_name]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            display_table[column_name] = series.map(_coerce_streamlit_cell_value)
    return display_table


def _coerce_streamlit_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, default=str)
    return str(value)
