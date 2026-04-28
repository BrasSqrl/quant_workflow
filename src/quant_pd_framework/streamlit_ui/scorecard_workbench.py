"""Scorecard binning review helpers for the Binning Theater UI."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework.presentation import format_metric_value
from quant_pd_framework.streamlit_ui.state import prepare_table_for_display
from quant_pd_framework.streamlit_ui.theme import render_metric_strip


def build_manual_bin_override_string(woe_table: pd.DataFrame) -> str:
    """Builds a comma-separated manual-bin override string from scorecard buckets."""

    if woe_table.empty or "bucket_label" not in woe_table.columns:
        return ""
    edges: list[float] = []
    for label in woe_table["bucket_label"].astype(str).tolist():
        for candidate in _extract_numeric_bounds(label):
            if math.isfinite(candidate):
                edges.append(candidate)
    edge_counts = Counter(edges)
    internal_edges = sorted(edge for edge, count in edge_counts.items() if count > 1)
    if not internal_edges:
        unique_edges = sorted(edge_counts)
        internal_edges = unique_edges[1:-1] if len(unique_edges) > 2 else []
    return ", ".join(format_metric_value(edge) for edge in internal_edges)


def build_binning_quality_messages(
    *,
    summary_row: pd.Series | None,
    woe_table: pd.DataFrame,
) -> list[dict[str, str]]:
    """Returns plain-English quality messages for one scorecard feature."""

    messages: list[dict[str, str]] = []
    if summary_row is None:
        return [{"level": "info", "message": "No feature-level scorecard summary is available."}]

    iv_value = _to_float(summary_row.get("information_value"))
    if iv_value is not None:
        if iv_value >= 0.5:
            messages.append(
                {
                    "level": "watch",
                    "message": "Very high IV can be useful but should be checked for leakage.",
                }
            )
        elif iv_value >= 0.1:
            messages.append({"level": "good", "message": "IV is in a useful review range."})
        elif iv_value >= 0.02:
            messages.append({"level": "info", "message": "IV is modest but may still add value."})
        else:
            messages.append({"level": "watch", "message": "IV is weak for this feature."})

    largest_share = _to_float(summary_row.get("largest_bin_share"))
    if largest_share is not None and largest_share >= 0.5:
        messages.append(
            {
                "level": "watch",
                "message": "One bucket contains at least half the observations.",
            }
        )

    trend = str(summary_row.get("bad_rate_trend", "")).lower()
    if "mixed" in trend or "flat" in trend:
        messages.append(
            {
                "level": "watch",
                "message": "Bad-rate movement is not cleanly monotonic.",
            }
        )
    elif trend:
        messages.append({"level": "good", "message": "Bad-rate movement is directionally stable."})

    if not woe_table.empty and "total" in woe_table.columns:
        totals = pd.to_numeric(woe_table["total"], errors="coerce")
        if (totals <= 0).any():
            messages.append({"level": "watch", "message": "At least one bucket has no rows."})
        elif len(totals) < 2:
            messages.append({"level": "watch", "message": "Only one bucket is available."})

    if not messages:
        messages.append({"level": "info", "message": "No material binning concerns detected."})
    return messages


def render_binning_theater(
    *,
    selected_feature: str,
    feature_summary: pd.DataFrame,
    woe_table: pd.DataFrame,
    points_table: pd.DataFrame,
) -> None:
    """Renders the scorecard Binning Theater review panel."""

    selected_summary = (
        feature_summary.loc[feature_summary["feature_name"] == selected_feature]
        .head(1)
        .reset_index(drop=True)
    )
    selected_woe = _sort_bucket_table(
        woe_table.loc[woe_table["feature_name"] == selected_feature].copy()
    )
    selected_points = _sort_bucket_table(
        points_table.loc[points_table["feature_name"] == selected_feature].copy()
    )
    summary_row = selected_summary.iloc[0] if not selected_summary.empty else None
    override_string = build_manual_bin_override_string(selected_woe)
    quality_messages = build_binning_quality_messages(
        summary_row=summary_row,
        woe_table=selected_woe,
    )

    with st.expander("Binning Theater", expanded=True):
        st.caption(
            "Review-only scorecard binning workspace. Use this to inspect bin quality "
            "and copy manual override edges for a future rerun."
        )
        if summary_row is not None:
            render_metric_strip(
                [
                    {
                        "label": "IV",
                        "value": format_metric_value(summary_row.get("information_value")),
                    },
                    {
                        "label": "Bins",
                        "value": format_metric_value(summary_row.get("bin_count")),
                    },
                    {
                        "label": "Largest Bin",
                        "value": format_metric_value(summary_row.get("largest_bin_share")),
                    },
                    {
                        "label": "WoE Span",
                        "value": format_metric_value(summary_row.get("woe_span")),
                    },
                ],
                compact=True,
            )
        for message in quality_messages:
            if message["level"] == "good":
                st.success(message["message"])
            elif message["level"] == "watch":
                st.warning(message["message"])
            else:
                st.info(message["message"])

        if override_string:
            st.code(override_string, language="text")
            st.caption(
                "Copy this string into Step 2 -> Governance & Review -> scorecard "
                "bin overrides if you want to reuse these internal edges."
            )
        else:
            st.caption("Manual override edges are not available for this feature.")

        table_columns = st.columns(2)
        with table_columns[0]:
            st.markdown("**Selected WoE buckets**")
            st.dataframe(
                prepare_table_for_display(selected_woe),
                width="stretch",
                hide_index=True,
            )
        with table_columns[1]:
            st.markdown("**Selected points buckets**")
            st.dataframe(
                prepare_table_for_display(selected_points),
                width="stretch",
                hide_index=True,
            )


def _extract_numeric_bounds(label: str) -> list[float]:
    candidates = re.findall(r"[-+]?(?:inf|\d+(?:\.\d+)?(?:e[-+]?\d+)?)", label.lower())
    values: list[float] = []
    for candidate in candidates:
        if candidate in {"inf", "+inf"}:
            values.append(math.inf)
        elif candidate == "-inf":
            values.append(-math.inf)
        else:
            try:
                values.append(float(candidate))
            except ValueError:
                continue
    return values


def _sort_bucket_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "bucket_rank" not in table.columns:
        return table
    return table.sort_values("bucket_rank")


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric
