"""Reusable low-level helpers for diagnostic table and figure generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quant_pd_framework.context import PipelineContext
from quant_pd_framework.presentation import apply_fintech_figure_theme, friendly_asset_title


def sample_rows_for_diagnostics(
    dataframe: pd.DataFrame,
    max_rows: int,
    context: PipelineContext,
) -> pd.DataFrame:
    """Samples rows consistently across expensive diagnostic calculations."""

    if context.config.performance.large_data_mode:
        max_rows = min(max_rows, context.config.performance.diagnostic_sample_rows)
    if len(dataframe) <= max_rows:
        return dataframe
    return dataframe.sample(max_rows, random_state=context.config.split.random_state)


def sample_frame_for_plotting(dataframe: pd.DataFrame, context: PipelineContext) -> pd.DataFrame:
    """Applies the configured plot row cap with Large Data Mode safeguards."""

    max_rows = context.config.diagnostics.max_plot_rows
    return sample_rows_for_diagnostics(dataframe, max_rows, context)


def sanitize_asset_name(name: str) -> str:
    """Creates filesystem- and key-safe diagnostic asset identifiers."""

    return "".join(character if character.isalnum() else "_" for character in name).strip("_")


def bucket_numeric_series(series: pd.Series, bucket_count: int) -> pd.Series:
    """Ranks before qcut so equal numeric values do not collapse all buckets."""

    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=min(bucket_count, ranked.nunique()), duplicates="drop")


def compute_population_stability_index(expected: pd.Series, actual: pd.Series) -> float:
    """Computes PSI for numeric or categorical series with stable small-count guards."""

    expected_series = pd.Series(expected).dropna()
    actual_series = pd.Series(actual).dropna()
    if expected_series.empty or actual_series.empty:
        return float("nan")

    if pd.api.types.is_numeric_dtype(expected_series):
        bucket_edges = np.unique(
            np.quantile(
                expected_series,
                np.linspace(0, 1, min(11, max(3, expected_series.nunique()))),
            )
        )
        if len(bucket_edges) < 2:
            return 0.0
        bucket_edges = bucket_edges.astype(float)
        bucket_edges[0] = -np.inf
        bucket_edges[-1] = np.inf
        if len(np.unique(bucket_edges)) < 2:
            return 0.0
        expected_bucket = pd.cut(
            expected_series,
            bins=bucket_edges,
            include_lowest=True,
            duplicates="drop",
        )
        actual_bucket = pd.cut(
            actual_series,
            bins=bucket_edges,
            include_lowest=True,
            duplicates="drop",
        )
        expected_dist = expected_bucket.value_counts(normalize=True, sort=False)
        actual_dist = actual_bucket.value_counts(normalize=True, sort=False)
    else:
        expected_dist = expected_series.astype(str).value_counts(normalize=True)
        actual_dist = actual_series.astype(str).value_counts(normalize=True)

    all_buckets = expected_dist.index.union(actual_dist.index)
    psi_value = 0.0
    for bucket in all_buckets:
        expected_pct = max(float(expected_dist.get(bucket, 0.0)), 1e-6)
        actual_pct = max(float(actual_dist.get(bucket, 0.0)), 1e-6)
        psi_value += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
    return float(psi_value)


def apply_visual_theme_to_context(context: PipelineContext) -> None:
    """Themes every diagnostic figure through the shared report/UI style."""

    themed: dict[str, go.Figure] = {}
    for figure_name, figure in context.visualizations.items():
        themed[figure_name] = apply_fintech_figure_theme(
            figure,
            title=friendly_asset_title(figure_name, kind="figure"),
        )
    context.visualizations = themed


def coerce_jsonlike_cell(value: Any) -> Any:
    """Normalizes diagnostic cells that can break table display or Parquet export."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    return value
