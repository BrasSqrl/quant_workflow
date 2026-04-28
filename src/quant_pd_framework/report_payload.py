"""Controls embedded Plotly payload size for standalone HTML reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPORT_PAYLOAD_AUDIT_COLUMNS = [
    "figure_name",
    "action",
    "original_points",
    "report_points",
    "original_payload_mb",
    "report_payload_mb",
    "reason",
]


def optimize_report_visualizations(
    visualizations: Mapping[str, Any],
    *,
    max_points_per_figure: int,
    max_figure_payload_mb: float,
    max_total_figure_payload_mb: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Returns report-safe figures plus an audit trail of size decisions."""

    optimized: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    max_figure_bytes = int(max_figure_payload_mb * 1024 * 1024)
    max_total_bytes = int(max_total_figure_payload_mb * 1024 * 1024)
    cumulative_report_bytes = 0
    total_original_bytes = 0
    total_report_bytes = 0
    total_original_points = 0
    total_report_points = 0

    for figure_name, figure in visualizations.items():
        if not isinstance(figure, go.Figure):
            optimized[figure_name] = figure
            audit_rows.append(
                _audit_row(
                    figure_name=figure_name,
                    action="kept",
                    original_points=0,
                    report_points=0,
                    original_payload_bytes=0,
                    report_payload_bytes=0,
                    reason="Non-Plotly object passed through unchanged.",
                )
            )
            continue

        original_figure = go.Figure(figure)
        original_points = count_figure_points(original_figure)
        original_payload_bytes = serialized_figure_size_bytes(original_figure)
        report_figure = downsample_figure(
            original_figure,
            max_points_per_figure=max_points_per_figure,
        )
        report_points = count_figure_points(report_figure)
        report_payload_bytes = serialized_figure_size_bytes(report_figure)
        action = "kept" if report_points == original_points else "downsampled"
        reason = (
            "Within configured report limits."
            if action == "kept"
            else f"Downsampled to at most {max_points_per_figure:,} points for the report."
        )

        if report_payload_bytes > max_figure_bytes:
            action = "skipped"
            reason = (
                f"Skipped because the serialized chart payload "
                f"({report_payload_bytes / 1024 / 1024:.2f} MB) exceeds the "
                f"per-chart cap ({max_figure_payload_mb:.2f} MB)."
            )
            report_points = 0
            report_payload_bytes = 0
        elif cumulative_report_bytes + report_payload_bytes > max_total_bytes:
            action = "skipped"
            reason = (
                f"Skipped because adding this chart would exceed the total embedded "
                f"chart payload cap ({max_total_figure_payload_mb:.2f} MB)."
            )
            report_points = 0
            report_payload_bytes = 0
        else:
            optimized[figure_name] = report_figure
            cumulative_report_bytes += report_payload_bytes

        total_original_points += original_points
        total_report_points += report_points
        total_original_bytes += original_payload_bytes
        total_report_bytes += report_payload_bytes
        audit_rows.append(
            _audit_row(
                figure_name=figure_name,
                action=action,
                original_points=original_points,
                report_points=report_points,
                original_payload_bytes=original_payload_bytes,
                report_payload_bytes=report_payload_bytes,
                reason=reason,
            )
        )

    if audit_rows:
        audit_rows.append(
            _audit_row(
                figure_name="__total__",
                action="summary",
                original_points=total_original_points,
                report_points=total_report_points,
                original_payload_bytes=total_original_bytes,
                report_payload_bytes=total_report_bytes,
                reason=(
                    f"Embedded {len(optimized):,} of {len(visualizations):,} report figures "
                    "after size controls."
                ),
            )
        )
    return optimized, pd.DataFrame(audit_rows, columns=REPORT_PAYLOAD_AUDIT_COLUMNS)


def downsample_figure(figure: go.Figure, *, max_points_per_figure: int) -> go.Figure:
    """Downsamples trace arrays while preserving layout and trace metadata."""

    if count_figure_points(figure) <= max_points_per_figure:
        return go.Figure(figure)

    trace_count = max(len(figure.data), 1)
    per_trace_cap = max(2, math.ceil(max_points_per_figure / trace_count))
    downsampled_traces: list[dict[str, Any]] = []
    for trace in figure.data:
        trace_payload = trace.to_plotly_json()
        trace_points = _trace_point_count(trace_payload)
        if trace_points <= per_trace_cap:
            downsampled_traces.append(trace_payload)
            continue
        indices = np.unique(
            np.linspace(0, trace_points - 1, num=per_trace_cap, dtype=int)
        ).tolist()
        downsampled_traces.append(
            _downsample_payload_value(
                trace_payload,
                indices=indices,
                expected_length=trace_points,
            )
        )

    return go.Figure(
        data=downsampled_traces,
        layout=figure.layout.to_plotly_json(),
    )


def count_figure_points(figure: go.Figure) -> int:
    """Counts plottable rows across traces using the longest array per trace."""

    return sum(_trace_point_count(trace.to_plotly_json()) for trace in figure.data)


def serialized_figure_size_bytes(figure: go.Figure) -> int:
    """Returns the UTF-8 byte size of the Plotly JSON payload."""

    payload = json.dumps(
        figure.to_plotly_json(),
        default=_json_default,
        separators=(",", ":"),
    )
    return len(payload.encode("utf-8"))


def _trace_point_count(trace_payload: Mapping[str, Any]) -> int:
    candidate_lengths = [
        length
        for field_name in (
            "x",
            "y",
            "z",
            "r",
            "theta",
            "lat",
            "lon",
            "locations",
            "values",
            "open",
            "high",
            "low",
            "close",
            "text",
            "customdata",
            "ids",
        )
        if (length := _sequence_length(trace_payload.get(field_name))) is not None
    ]
    return max(candidate_lengths, default=0)


def _sequence_length(value: Any) -> int | None:
    if value is None or isinstance(value, (str, bytes, bytearray, dict)):
        return None
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim > 0 else None
    if isinstance(value, (pd.Series, pd.Index)):
        return int(len(value))
    if isinstance(value, Sequence):
        return len(value)
    return None


def _downsample_payload_value(
    value: Any,
    *,
    indices: list[int],
    expected_length: int,
) -> Any:
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == expected_length:
        return value[indices].tolist()
    if isinstance(value, (pd.Series, pd.Index)) and len(value) == expected_length:
        return value.take(indices).tolist()
    if isinstance(value, (list, tuple)) and len(value) == expected_length:
        return [value[index] for index in indices]
    if isinstance(value, dict):
        return {
            key: _downsample_payload_value(
                child_value,
                indices=indices,
                expected_length=expected_length,
            )
            for key, child_value in value.items()
        }
    return value


def _audit_row(
    *,
    figure_name: str,
    action: str,
    original_points: int,
    report_points: int,
    original_payload_bytes: int,
    report_payload_bytes: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "figure_name": figure_name,
        "action": action,
        "original_points": int(original_points),
        "report_points": int(report_points),
        "original_payload_mb": round(original_payload_bytes / 1024 / 1024, 4),
        "report_payload_mb": round(report_payload_bytes / 1024 / 1024, 4),
        "reason": reason,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, pd.Interval):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
