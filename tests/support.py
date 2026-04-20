"""Shared test helpers for framework integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from uuid import uuid4

import pandas as pd

from quant_pd_framework import ColumnRole, ColumnSpec, SchemaConfig
from quant_pd_framework.reference_workflows import (
    build_reference_ccar_dataframe,
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
    return build_reference_ccar_dataframe(
        entity_count=entity_count,
        periods_per_entity=periods_per_entity,
    )


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
