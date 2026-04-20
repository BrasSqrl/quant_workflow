"""Shared test helpers for framework integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from uuid import uuid4

import numpy as np
import pandas as pd

from quant_pd_framework import ColumnRole, ColumnSpec, SchemaConfig
from quant_pd_framework.reference_workflows import (
    build_reference_lgd_dataframe,
    build_reference_lifetime_pd_dataframe,
    build_reference_pd_dataframe,
)


@contextmanager
def temporary_artifact_root(prefix: str) -> Iterator[Path]:
    """Creates an isolated temporary workspace outside the repo and cleans it up afterward."""

    directory = Path(gettempdir()) / f"quant_pd_{prefix}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        rmtree(directory, ignore_errors=True)


def build_binary_dataframe(row_count: int = 160) -> pd.DataFrame:
    return build_reference_pd_dataframe(row_count=row_count)


def build_continuous_dataframe(row_count: int = 160) -> pd.DataFrame:
    return build_reference_lgd_dataframe(row_count=row_count)


def build_panel_forecast_dataframe(
    entity_count: int = 12,
    periods_per_entity: int = 18,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=31)
    rows: list[dict[str, object]] = []
    for entity_index in range(entity_count):
        entity_name = f"SEG_{entity_index:02d}"
        base_utilization = rng.uniform(0.2, 0.75)
        for period_index, as_of_date in enumerate(
            pd.date_range("2023-01-01", periods=periods_per_entity, freq="QE")
        ):
            unemployment_rate = 3.4 + 0.12 * period_index + rng.normal(0, 0.15)
            gdp_gap = -0.5 + 0.08 * period_index + rng.normal(0, 0.1)
            utilization = base_utilization + rng.normal(0, 0.04)
            forecast_value = (
                0.015
                + 0.08 * utilization
                + 0.012 * unemployment_rate
                - 0.01 * gdp_gap
                + 0.002 * entity_index
                + rng.normal(0, 0.01)
            )
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "segment_id": entity_name,
                    "utilization": utilization,
                    "unemployment_rate": unemployment_rate,
                    "gdp_gap": gdp_gap,
                    "forecast_value": forecast_value,
                }
            )
    return pd.DataFrame(rows)


def build_lifetime_pd_dataframe(
    entity_count: int = 20,
    periods_per_entity: int = 10,
) -> pd.DataFrame:
    return build_reference_lifetime_pd_dataframe(
        entity_count=entity_count,
        periods_per_entity=periods_per_entity,
    )


def build_common_schema(identifier_name: str, *, include_legacy_drop: bool = False) -> SchemaConfig:
    column_specs = [
        ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
        ColumnSpec(name=identifier_name, dtype="string", role=ColumnRole.IDENTIFIER),
    ]
    if include_legacy_drop:
        column_specs.append(ColumnSpec(name="legacy_text_field", enabled=False))
    return SchemaConfig(column_specs=column_specs)
