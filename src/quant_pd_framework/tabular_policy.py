"""Tabular artifact format decisions based on the original input source."""

from __future__ import annotations

from typing import Any

from .config import TabularOutputFormat

PARQUET_INPUT_SUFFIXES = {".parquet", ".pq"}


def original_input_is_parquet(metadata: dict[str, Any] | None) -> bool:
    """Returns whether the user's original selected input was a Parquet file."""

    return _input_source_suffix(metadata) in PARQUET_INPUT_SUFFIXES


def resolve_tabular_output_format(metadata: dict[str, Any] | None) -> TabularOutputFormat:
    """
    Resolves the artifact table format from the original input file type.

    Quant Studio keeps tabular outputs as Parquet only when the Step 1 source
    was Parquet. CSV, Excel, bundled-sample, dataframe, and unknown sources
    export tabular artifacts as CSV for broader review compatibility.
    """

    if original_input_is_parquet(metadata):
        return TabularOutputFormat.PARQUET
    return TabularOutputFormat.CSV


def _input_source_suffix(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    source_metadata = metadata.get("input_source")
    if isinstance(source_metadata, dict):
        suffix = _suffix_from_mapping(source_metadata)
        if suffix:
            return suffix
    large_data_handle_metadata = metadata.get("large_data_handle")
    if isinstance(large_data_handle_metadata, dict):
        suffix = _suffix_from_mapping(large_data_handle_metadata)
        if suffix:
            return suffix
    return _suffix_from_mapping(metadata)


def _suffix_from_mapping(metadata: dict[str, Any]) -> str:
    for key in ("suffix", "source_suffix", "input_type"):
        value = metadata.get(key)
        if value:
            return str(value).strip().lower()
    return ""
