"""Large-dataset helpers for intake, memory estimation, and tabular exports."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PerformanceConfig
from .large_data_support.constants import (
    GIB,
    PARQUET_SUFFIXES,
)
from .large_data_support.constants import (
    SUPPORTED_TABULAR_SUFFIXES as SUPPORTED_TABULAR_SUFFIXES,
)
from .large_data_support.handles import (
    DatasetHandle,
    is_s3_uri,
    normalize_s3_metadata,
)
from .large_data_support.handles import (
    build_dataset_handle as build_dataset_handle,
)
from .large_data_support.handles import (
    build_s3_dataset_handle as build_s3_dataset_handle,
)
from .large_data_support.handles import (
    parse_s3_uri as parse_s3_uri,
)
from .large_data_support.handles import (
    read_tabular_path as read_tabular_path,
)

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}

__all__ = [
    "DatasetHandle",
    "GIB",
    "PARQUET_SUFFIXES",
    "SUPPORTED_TABULAR_SUFFIXES",
    "build_dataset_handle",
    "build_memory_estimate_table",
    "build_profile_cache_key",
    "build_s3_dataset_handle",
    "convert_csv_to_parquet",
    "convert_s3_csv_to_parquet",
    "copy_s3_object_to_local",
    "describe_s3_uri",
    "is_s3_uri",
    "iter_dataset_batches",
    "materialize_projected_parquet",
    "normalize_s3_metadata",
    "optimize_dataframe_dtypes",
    "parse_s3_uri",
    "profile_dataset_handle",
    "profile_dataset_handle_cached",
    "read_dataset_preview",
    "read_dataset_sample",
    "read_tabular_path",
    "stage_large_data_file",
]


def describe_s3_uri(uri: str) -> dict[str, Any]:
    """Reads lightweight object metadata for an S3 URI using environment credentials."""

    metadata = normalize_s3_metadata(uri)
    try:
        filesystem, object_path = _s3_filesystem_and_path(uri)
        info = filesystem.get_file_info(object_path)
    except Exception:
        return metadata
    if getattr(info, "size", None) is not None and int(info.size) >= 0:
        metadata["size_bytes"] = int(info.size)
    mtime = getattr(info, "mtime", None)
    if mtime is not None:
        metadata["modified_at_utc"] = str(mtime)
    return metadata


def stage_large_data_file(
    handle: DatasetHandle,
    *,
    chunk_rows: int,
    compression: str,
    s3_cache_dir: Path | None = None,
) -> DatasetHandle:
    """
    Returns a handle whose active path is a reusable Parquet staging file.

    Existing Parquet inputs are reused directly. CSV inputs are converted into a
    source-adjacent cache keyed by file size and modified timestamp so repeated
    runs do not reconvert unchanged data.
    """

    if handle.is_s3:
        return _stage_s3_data_file(
            handle,
            chunk_rows=chunk_rows,
            compression=compression,
            s3_cache_dir=s3_cache_dir or Path(".quant_studio_cache") / "s3",
        )

    if handle.path is None:
        raise ValueError("Local large-data staging requires a local file path.")

    source_path = handle.path
    suffix = source_path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        return handle.with_staging(
            source_path,
            {
                "staging_required": False,
                "reused_existing_parquet": True,
                "source_path": str(source_path),
                "staged_path": str(source_path),
            },
        )
    if suffix != ".csv":
        return handle

    stat_result = source_path.stat()
    cache_dir = source_path.parent / ".quant_studio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{source_path.stem}_{stat_result.st_size}_{stat_result.st_mtime_ns}"
    staged_path = cache_dir / f"{cache_key}.parquet"
    metadata_path = cache_dir / f"{cache_key}.json"
    reused = staged_path.exists()
    if not reused:
        conversion_metadata = convert_csv_to_parquet(
            source_path,
            staged_path,
            chunk_rows=chunk_rows,
            compression=compression,
        )
        metadata_path.write_text(
            json.dumps(conversion_metadata, indent=2),
            encoding="utf-8",
        )
    else:
        conversion_metadata = {
            "source_path": str(source_path),
            "destination_path": str(staged_path),
            "chunk_rows": int(chunk_rows),
            "compression": compression,
            "reused_existing_staging_file": True,
        }

    return handle.with_staging(
        staged_path,
        {
            **conversion_metadata,
            "staging_required": True,
            "reused_existing_staging_file": reused,
            "metadata_path": str(metadata_path),
        },
    )


def materialize_projected_parquet(
    handle: DatasetHandle,
    *,
    columns: list[str],
    destination_path: Path,
    compression: str,
    duckdb_threads: int = 0,
    duckdb_memory_limit_gb: float | None = None,
) -> tuple[DatasetHandle, dict[str, Any]]:
    """Writes a projected Parquet dataset once so downstream chunks read fewer columns."""

    selected_columns = list(dict.fromkeys(column for column in columns if column))
    if not selected_columns:
        return handle, {
            "materialized": False,
            "reason": "no_projected_columns",
            "source_identifier": handle.source_identifier,
        }
    if handle.is_s3 and handle.staged_path is None:
        return handle, {
            "materialized": False,
            "reason": "s3_source_not_staged",
            "source_identifier": handle.source_identifier,
        }
    try:
        import duckdb
    except ImportError:
        return handle, {
            "materialized": False,
            "reason": "duckdb_not_available",
            "source_identifier": handle.source_identifier,
        }

    source_path = handle.active_path
    if source_path.resolve() == destination_path.resolve():
        return handle, {
            "materialized": False,
            "reason": "source_already_projected_destination",
            "source_identifier": handle.source_identifier,
            "destination_path": str(destination_path),
        }

    suffix = source_path.suffix.lower()
    if suffix not in PARQUET_SUFFIXES and suffix != ".csv":
        return handle, {
            "materialized": False,
            "reason": f"unsupported_suffix:{suffix}",
            "source_identifier": handle.source_identifier,
        }

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    reader = "read_parquet" if suffix in PARQUET_SUFFIXES else "read_csv_auto"
    select_sql = ", ".join(_quote_identifier(column) for column in selected_columns)
    compression_sql = compression.replace("'", "").upper()
    reused = destination_path.exists()
    metadata = {
        "materialized": True,
        "reused_existing_projected_file": reused,
        "source_identifier": handle.source_identifier,
        "source_path": str(source_path),
        "destination_path": str(destination_path),
        "projected_columns": selected_columns,
        "compression": compression,
        "engine": "duckdb_projected_copy",
    }
    if not reused:
        with duckdb.connect(database=":memory:") as connection:
            if duckdb_threads > 0:
                connection.execute(f"PRAGMA threads={int(duckdb_threads)}")
            if duckdb_memory_limit_gb is not None:
                connection.execute(f"PRAGMA memory_limit='{float(duckdb_memory_limit_gb)}GB'")
            source_sql = str(source_path).replace("'", "''")
            destination_sql = str(destination_path).replace("'", "''")
            connection.execute(
                f"COPY (SELECT {select_sql} FROM {reader}('{source_sql}')) "
                f"TO '{destination_sql}' "
                f"(FORMAT PARQUET, COMPRESSION '{compression_sql}')"
            )
    metadata["row_count"] = _parquet_row_count(destination_path)
    staged_metadata = dict(handle.staging_metadata)
    staged_metadata["projected_dataset"] = metadata
    return handle.with_staging(destination_path, staged_metadata), metadata


def _stage_s3_data_file(
    handle: DatasetHandle,
    *,
    chunk_rows: int,
    compression: str,
    s3_cache_dir: Path,
) -> DatasetHandle:
    """Stages S3 CSV, Excel, and Parquet sources into a local reusable cache."""

    source_uri = handle.uri or handle.source_identifier
    suffix = handle.source_suffix
    effective_suffix = _resolve_s3_effective_suffix(source_uri, suffix)
    if effective_suffix not in {".csv", *EXCEL_SUFFIXES, *PARQUET_SUFFIXES}:
        raise ValueError(
            "S3 large-data intake supports CSV, Excel, and Parquet files. "
            f"Received suffix: {suffix or 'unknown'}."
        )
    s3_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _source_cache_key(source_uri, handle.metadata)
    staged_suffix = ".parquet" if effective_suffix == ".csv" else effective_suffix
    staged_path = s3_cache_dir / f"{cache_key}{staged_suffix}"
    metadata_path = s3_cache_dir / f"{cache_key}.json"
    reused = staged_path.exists()
    if reused:
        staging_metadata = {
            "source_uri": source_uri,
            "destination_path": str(staged_path),
            "reused_existing_staging_file": True,
            "staging_required": True,
            "s3_local_cache_dir": str(s3_cache_dir),
            "metadata_path": str(metadata_path),
        }
    elif effective_suffix == ".csv":
        staging_metadata = convert_s3_csv_to_parquet(
            source_uri,
            staged_path,
            chunk_rows=chunk_rows,
            compression=compression,
        )
    else:
        staging_metadata = copy_s3_object_to_local(
            source_uri,
            staged_path,
        )
        staging_metadata["compression"] = compression
    staging_metadata.update(
        {
            "staging_required": True,
            "source_kind": "s3",
            "source_uri": source_uri,
            "source_suffix": suffix,
            "detected_suffix": effective_suffix,
            "reused_existing_staging_file": reused,
            "metadata_path": str(metadata_path),
            "s3_local_cache_dir": str(s3_cache_dir),
        }
    )
    if not reused:
        metadata_path.write_text(json.dumps(staging_metadata, indent=2), encoding="utf-8")
    return handle.with_staging(staged_path, staging_metadata)


def read_dataset_preview(
    handle: DatasetHandle,
    *,
    rows: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Reads a small preview from a file-backed dataset."""

    if handle.is_s3 and handle.staged_path is None:
        return _read_s3_rows(handle, rows=rows, columns=columns)

    path = handle.active_path
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=rows, usecols=columns)
    if suffix in EXCEL_SUFFIXES:
        preview = pd.read_excel(path, nrows=rows)
        if columns:
            return preview.loc[:, [column for column in columns if column in preview.columns]]
        return preview
    if suffix in PARQUET_SUFFIXES:
        return _read_parquet_rows(path, rows=rows, columns=columns)
    raise ValueError(f"Unsupported large-data preview format: {path.suffix}")


def read_dataset_sample(
    handle: DatasetHandle,
    *,
    rows: int,
    columns: list[str] | None,
    random_state: int,
) -> pd.DataFrame:
    """Reads a deterministic sample for model development."""

    if handle.is_s3 and handle.staged_path is None:
        return read_dataset_preview(handle, rows=rows, columns=columns)

    sampled = _read_duckdb_sample(
        handle.active_path,
        rows=rows,
        columns=columns,
        random_state=random_state,
    )
    if sampled is not None:
        return sampled
    return read_dataset_preview(handle, rows=rows, columns=columns)


def iter_dataset_batches(
    handle: DatasetHandle,
    *,
    batch_rows: int,
    columns: list[str] | None = None,
):
    """Yields dataframe batches from a file-backed dataset."""

    if handle.is_s3 and handle.staged_path is None:
        raise ValueError("S3 datasets must be staged locally before chunked scoring.")

    path = handle.active_path
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=batch_rows, usecols=columns)
        return
    if suffix in EXCEL_SUFFIXES:
        yield from _iter_excel_batches(path, batch_rows=batch_rows, columns=columns)
        return
    if suffix in PARQUET_SUFFIXES:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading Parquet batches requires `pyarrow`.") from exc
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=columns):
            yield batch.to_pandas()
        return
    raise ValueError(f"Unsupported batch-read format: {path.suffix}")


def profile_dataset_handle(
    handle: DatasetHandle,
    *,
    preview_rows: int = 50,
    selected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Builds a lightweight profile without loading the full file into memory."""

    preview = read_dataset_preview(handle, rows=preview_rows, columns=selected_columns)
    row_count: int | None = None
    if handle.suffix in PARQUET_SUFFIXES and not handle.is_s3:
        try:
            import pyarrow.parquet as pq

            row_count = int(pq.ParquetFile(handle.active_path).metadata.num_rows)
        except ImportError:
            row_count = None
    elif handle.suffix in PARQUET_SUFFIXES and handle.is_s3 and handle.staged_path is not None:
        try:
            import pyarrow.parquet as pq

            row_count = int(pq.ParquetFile(handle.active_path).metadata.num_rows)
        except ImportError:
            row_count = None
    elif handle.suffix == ".csv":
        row_count = None

    approximate_quantiles = _preview_quantiles(preview)
    approximate_cardinality = {
        column: int(preview[column].nunique(dropna=True)) for column in preview.columns
    }
    null_counts = {column: int(preview[column].isna().sum()) for column in preview.columns}

    return {
        "source_kind": "s3" if handle.is_s3 else "local_file",
        "source_uri": handle.uri or "",
        "source_path": str(handle.path) if handle.path is not None else "",
        "active_path": (
            str(handle.active_path)
            if handle.staged_path is not None or handle.path is not None
            else ""
        ),
        "source_suffix": handle.source_suffix,
        "active_suffix": handle.suffix,
        "source_metadata": dict(handle.metadata),
        "row_count": row_count,
        "column_count": int(preview.shape[1]),
        "columns": list(preview.columns),
        "dtypes": {column: str(dtype) for column, dtype in preview.dtypes.items()},
        "null_counts": null_counts,
        "null_counts_preview": null_counts,
        "approximate_cardinality": approximate_cardinality,
        "unique_counts_preview": approximate_cardinality,
        "approximate_quantiles": approximate_quantiles,
        "profile_basis": "file_metadata_plus_preview_rows",
        "selected_columns": list(selected_columns or []),
        "preview_rows": int(len(preview)),
    }


def profile_dataset_handle_cached(
    handle: DatasetHandle,
    *,
    preview_rows: int = 50,
    selected_columns: list[str] | None = None,
    cache_enabled: bool = True,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Builds or reuses a persistent lightweight profile for file-backed datasets."""

    cache_key = build_profile_cache_key(handle, selected_columns=selected_columns)
    profile_dir = cache_root or Path(".quant_studio_cache") / "profiles"
    profile_path = profile_dir / f"{cache_key}.json"
    if cache_enabled and profile_path.exists():
        try:
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            cached["profile_cache_key"] = cache_key
            cached["profile_cache_path"] = str(profile_path)
            cached["profile_cache_hit"] = True
            cached["profile_cache_enabled"] = True
            return cached
        except Exception:
            pass

    profile = profile_dataset_handle(
        handle,
        preview_rows=preview_rows,
        selected_columns=selected_columns,
    )
    profile["profile_cache_key"] = cache_key
    profile["profile_cache_path"] = str(profile_path)
    profile["profile_cache_hit"] = False
    profile["profile_cache_enabled"] = bool(cache_enabled)
    if cache_enabled:
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    return profile


def build_profile_cache_key(
    handle: DatasetHandle,
    *,
    selected_columns: list[str] | None = None,
) -> str:
    """Returns the persistent profile-cache fingerprint for one dataset view."""

    metadata = dict(handle.metadata)
    active_path = None
    active_stat: dict[str, Any] = {}
    if handle.staged_path is not None or handle.path is not None:
        try:
            active_path = handle.active_path
            stat_result = active_path.stat()
            active_stat = {
                "active_path": str(active_path),
                "active_size_bytes": int(stat_result.st_size),
                "active_modified_ns": int(stat_result.st_mtime_ns),
            }
        except OSError:
            active_stat = {"active_path": str(handle.source_identifier)}
    fingerprint = {
        "source_identifier": handle.source_identifier,
        "source_suffix": handle.source_suffix,
        "active_suffix": handle.suffix,
        "selected_columns": list(selected_columns or []),
        "source_size_bytes": metadata.get("size_bytes", ""),
        "source_modified_at_utc": metadata.get("modified_at_utc", ""),
        "source_modified_ns": metadata.get("modified_ns", ""),
        "source_etag": metadata.get("etag", ""),
        **active_stat,
    }
    encoded = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _preview_quantiles(preview: pd.DataFrame) -> dict[str, dict[str, float]]:
    quantiles: dict[str, dict[str, float]] = {}
    for column in preview.columns:
        numeric = pd.to_numeric(preview[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        values = numeric.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        quantiles[column] = {
            str(index): float(value) for index, value in values.items() if pd.notna(value)
        }
    return quantiles


def optimize_dataframe_dtypes(
    dataframe: pd.DataFrame,
    performance: PerformanceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies memory-oriented dtype reductions and returns an audit table."""

    working = dataframe.copy(deep=False)
    audit_rows: list[dict[str, Any]] = []

    for column_name in working.columns:
        series = working[column_name]
        old_dtype = str(series.dtype)
        old_memory = int(series.memory_usage(deep=True))
        action = ""
        converted = series

        if performance.downcast_numeric and pd.api.types.is_integer_dtype(series):
            converted = pd.to_numeric(series, downcast="integer")
            action = "downcast_integer"
        elif performance.downcast_numeric and pd.api.types.is_float_dtype(series):
            converted = pd.to_numeric(series, downcast="float")
            action = "downcast_float"
        elif (
            performance.convert_low_cardinality_strings
            and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
            and _should_convert_to_category(series, performance)
        ):
            converted = series.astype("category")
            action = "convert_string_to_category"

        if action:
            working[column_name] = converted
            new_memory = int(working[column_name].memory_usage(deep=True))
            new_dtype = str(working[column_name].dtype)
            if new_dtype != old_dtype or new_memory != old_memory:
                audit_rows.append(
                    {
                        "column_name": column_name,
                        "action": action,
                        "old_dtype": old_dtype,
                        "new_dtype": new_dtype,
                        "old_memory_bytes": old_memory,
                        "new_memory_bytes": new_memory,
                        "memory_saved_bytes": old_memory - new_memory,
                    }
                )

    return working, pd.DataFrame(audit_rows)


def _should_convert_to_category(series: pd.Series, performance: PerformanceConfig) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = unique_count / max(len(non_null), 1)
    return (
        unique_count <= performance.category_max_unique_values
        and unique_ratio <= performance.category_max_unique_ratio
    )


def build_memory_estimate_table(
    dataframe: pd.DataFrame,
    source_metadata: dict[str, Any],
    performance: PerformanceConfig,
) -> pd.DataFrame:
    """Builds a lightweight RAM estimate for the loaded dataset and workflow."""

    dataframe_memory_bytes = int(dataframe.memory_usage(deep=True).sum())
    file_size_bytes = _coerce_int(source_metadata.get("size_bytes"), default=0)
    estimated_peak_bytes = int(
        max(
            file_size_bytes * performance.memory_estimate_file_multiplier,
            dataframe_memory_bytes * performance.memory_estimate_dataframe_multiplier,
        )
    )
    configured_limit_gb = performance.memory_limit_gb
    status = "not_evaluated"
    if configured_limit_gb is not None:
        status = "pass" if estimated_peak_bytes <= configured_limit_gb * GIB else "warn"

    return pd.DataFrame(
        [
            {
                "input_rows": int(dataframe.shape[0]),
                "input_columns": int(dataframe.shape[1]),
                "source_file_size_gb": round(file_size_bytes / GIB, 4),
                "dataframe_memory_gb": round(dataframe_memory_bytes / GIB, 4),
                "estimated_peak_memory_gb": round(estimated_peak_bytes / GIB, 4),
                "configured_memory_limit_gb": configured_limit_gb,
                "file_size_multiplier": performance.memory_estimate_file_multiplier,
                "dataframe_multiplier": performance.memory_estimate_dataframe_multiplier,
                "status": status,
            }
        ]
    )


def convert_csv_to_parquet(
    source_path: Path,
    destination_path: Path,
    *,
    chunk_rows: int,
    compression: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Converts a CSV file to Parquet without loading the full file into pandas."""

    if progress_callback is None:
        duckdb_metadata = _convert_csv_to_parquet_with_duckdb(
            source_path,
            destination_path,
            chunk_rows=chunk_rows,
            compression=compression,
        )
        if duckdb_metadata is not None:
            return duckdb_metadata
    return _convert_csv_to_parquet_with_pyarrow_stream(
        source_path,
        destination_path,
        chunk_rows=chunk_rows,
        compression=compression,
        progress_callback=progress_callback,
    )


def convert_s3_csv_to_parquet(
    source_uri: str,
    destination_path: Path,
    *,
    chunk_rows: int,
    compression: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Streams an S3 CSV object into a local Parquet cache."""

    filesystem, object_path = _s3_filesystem_and_path(source_uri)
    return _convert_csv_stream_to_parquet(
        source_display=source_uri,
        source_opener=lambda: filesystem.open_input_file(object_path),
        destination_path=destination_path,
        chunk_rows=chunk_rows,
        compression=compression,
        engine="pyarrow_s3_stream",
        progress_callback=progress_callback,
    )


def copy_s3_object_to_local(source_uri: str, destination_path: Path) -> dict[str, Any]:
    """Copies an S3 object into the local staging cache without storing credentials."""

    filesystem, object_path = _s3_filesystem_and_path(source_uri)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with filesystem.open_input_file(object_path) as source, destination_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024 * 16)
    return {
        "source_uri": source_uri,
        "destination_path": str(destination_path),
        "conversion_engine": "s3_object_copy",
        "row_count": _parquet_row_count(destination_path),
    }


def _convert_csv_to_parquet_with_duckdb(
    source_path: Path,
    destination_path: Path,
    *,
    chunk_rows: int,
    compression: str,
) -> dict[str, Any] | None:
    try:
        import duckdb
    except ImportError:
        return None

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    compression_sql = compression.replace("'", "").upper()
    try:
        with duckdb.connect(database=":memory:") as connection:
            connection.execute(
                (
                    "COPY (SELECT * FROM read_csv_auto(?)) TO ? "
                    f"(FORMAT PARQUET, COMPRESSION '{compression_sql}')"
                ),
                [str(source_path), str(destination_path)],
            )
    except Exception:
        return None

    return {
        "source_path": str(source_path),
        "destination_path": str(destination_path),
        "chunk_rows": int(chunk_rows),
        "chunk_count": None,
        "row_count": _parquet_row_count(destination_path),
        "compression": compression,
        "conversion_engine": "duckdb_copy",
    }


def _convert_csv_to_parquet_with_pyarrow_stream(
    source_path: Path,
    destination_path: Path,
    *,
    chunk_rows: int,
    compression: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return _convert_csv_stream_to_parquet(
        source_display=str(source_path),
        source_opener=lambda: source_path.open("rb"),
        destination_path=destination_path,
        chunk_rows=chunk_rows,
        compression=compression,
        engine="pyarrow_local_stream",
        source_size_bytes=source_path.stat().st_size,
        progress_callback=progress_callback,
    )


def _convert_csv_stream_to_parquet(
    *,
    source_display: str,
    source_opener,
    destination_path: Path,
    chunk_rows: int,
    compression: str,
    engine: str,
    source_size_bytes: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.csv as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Streaming CSV-to-Parquet conversion requires `pyarrow`.") from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    chunk_count = 0
    row_count = 0
    _notify_csv_conversion_progress(
        progress_callback,
        phase="starting",
        progress=0.0,
        chunk_count=0,
        row_count=0,
        source_size_bytes=source_size_bytes,
        source_bytes_read=0,
    )
    try:
        with source_opener() as source:
            _validate_csv_stream_signature(source_display, source)
            try:
                reader = pc.open_csv(
                    source,
                    read_options=pc.ReadOptions(block_size=max(1, int(chunk_rows)) * 1024),
                )
                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(
                            destination_path,
                            table.schema,
                            compression=compression,
                        )
                    else:
                        table = table.cast(writer.schema, safe=False)
                    writer.write_table(table)
                    chunk_count += 1
                    row_count += int(table.num_rows)
                    source_bytes_read = _safe_stream_position(source)
                    progress = None
                    if source_size_bytes:
                        progress = min(
                            0.99,
                            max(0.0, source_bytes_read / max(1, source_size_bytes)),
                        )
                    _notify_csv_conversion_progress(
                        progress_callback,
                        phase="converting",
                        progress=progress,
                        chunk_count=chunk_count,
                        row_count=row_count,
                        source_size_bytes=source_size_bytes,
                        source_bytes_read=source_bytes_read,
                    )
            except Exception as exc:
                raise ValueError(_csv_parse_failure_message(source_display, exc)) from exc
    finally:
        if writer is not None:
            writer.close()

    _notify_csv_conversion_progress(
        progress_callback,
        phase="complete",
        progress=1.0,
        chunk_count=chunk_count,
        row_count=row_count,
        source_size_bytes=source_size_bytes,
        source_bytes_read=source_size_bytes,
    )
    return {
        "source_path": source_display if not is_s3_uri(source_display) else "",
        "source_uri": source_display if is_s3_uri(source_display) else "",
        "destination_path": str(destination_path),
        "chunk_rows": int(chunk_rows),
        "chunk_count": int(chunk_count),
        "row_count": int(row_count),
        "compression": compression,
        "conversion_engine": engine,
    }


def _notify_csv_conversion_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    progress_callback({"event": "csv_to_parquet_progress", **payload})


def _safe_stream_position(source: Any) -> int:
    try:
        return int(source.tell())
    except (AttributeError, OSError, ValueError, TypeError):
        return 0


def _read_parquet_rows(
    path: Path,
    *,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading Parquet previews requires `pyarrow`.") from exc

    parquet_file = pq.ParquetFile(path)
    frames: list[pd.DataFrame] = []
    remaining = rows
    for batch in parquet_file.iter_batches(batch_size=min(rows, 100_000), columns=columns):
        frame = batch.to_pandas()
        frames.append(frame.iloc[:remaining].copy(deep=True))
        remaining -= len(frames[-1])
        if remaining <= 0:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_duckdb_sample(
    path: Path,
    *,
    rows: int,
    columns: list[str] | None,
    random_state: int,
) -> pd.DataFrame | None:
    try:
        import duckdb
    except ImportError:
        return None

    suffix = path.suffix.lower()
    if suffix not in PARQUET_SUFFIXES and suffix != ".csv":
        return None
    reader = "read_parquet" if suffix in PARQUET_SUFFIXES else "read_csv_auto"
    quoted_columns = (
        "*"
        if not columns
        else ", ".join(_quote_identifier(column_name) for column_name in columns)
    )
    query = (
        f"SELECT {quoted_columns} FROM {reader}(?) "
        f"USING SAMPLE reservoir({int(rows)} ROWS) REPEATABLE ({int(random_state)})"
    )
    try:
        with duckdb.connect(database=":memory:") as connection:
            return connection.execute(query, [str(path)]).df()
    except Exception:
        return None


def _read_s3_rows(
    handle: DatasetHandle,
    *,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    uri = handle.uri or handle.source_identifier
    suffix = _resolve_s3_effective_suffix(uri, handle.source_suffix)
    if suffix == ".csv":
        return _read_s3_csv_rows(uri, rows=rows, columns=columns)
    if suffix in EXCEL_SUFFIXES:
        return _read_s3_excel_rows(uri, rows=rows, columns=columns)
    if suffix in PARQUET_SUFFIXES:
        return _read_s3_parquet_rows(uri, rows=rows, columns=columns)
    raise ValueError(f"Unsupported S3 preview format: {suffix}")


def _read_s3_csv_rows(
    uri: str,
    *,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    filesystem, object_path = _s3_filesystem_and_path(uri)
    with filesystem.open_input_file(object_path) as source:
        if _looks_like_excel_prefix(_read_stream_prefix(source)):
            return _read_excel_rows_from_stream(
                source,
                source_display=uri,
                rows=rows,
                columns=columns,
            )
        return _read_csv_rows_from_stream(
            source,
            source_display=uri,
            rows=rows,
            columns=columns,
        )


def _read_s3_excel_rows(
    uri: str,
    *,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    filesystem, object_path = _s3_filesystem_and_path(uri)
    with filesystem.open_input_file(object_path) as source:
        return _read_excel_rows_from_stream(
            source,
            source_display=uri,
            rows=rows,
            columns=columns,
        )


def _read_excel_rows_from_stream(
    source: Any,
    *,
    source_display: str,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    payload = _read_stream_payload(source)
    buffer = BytesIO(bytes(payload or b""))
    try:
        preview = pd.read_excel(buffer, nrows=rows)
    except ImportError as exc:
        raise ImportError("Reading S3 Excel previews requires `openpyxl`.") from exc
    except Exception as exc:
        raise ValueError(
            f"Could not parse `{source_display}` as an Excel workbook. "
            "If this is a ZIP archive rather than an Excel file, extract it first. "
            f"Parser detail: {_sanitize_error_detail(str(exc))}"
        ) from exc
    if columns:
        return preview.loc[:, [column for column in columns if column in preview.columns]]
    return preview


def _iter_excel_batches(
    path: Path,
    *,
    batch_rows: int,
    columns: list[str] | None,
):
    start_row = 0
    while True:
        skiprows = range(1, start_row + 1) if start_row else None
        frame = pd.read_excel(
            path,
            nrows=batch_rows,
            skiprows=skiprows,
            usecols=columns,
        )
        if frame.empty:
            break
        yield frame
        if len(frame) < batch_rows:
            break
        start_row += len(frame)


def _read_csv_rows_from_stream(
    source: Any,
    *,
    source_display: str,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    try:
        import pyarrow.csv as pc
    except ImportError as exc:
        raise ImportError("Reading CSV previews requires `pyarrow`.") from exc

    _validate_csv_stream_signature(source_display, source)
    frames: list[pd.DataFrame] = []
    remaining = rows
    try:
        reader = pc.open_csv(source)
        for batch in reader:
            table = batch.to_pandas()
            if columns:
                table = table.loc[:, [column for column in columns if column in table.columns]]
            frames.append(table.iloc[:remaining].copy(deep=True))
            remaining -= len(frames[-1])
            if remaining <= 0:
                break
    except Exception as exc:
        raise ValueError(_csv_parse_failure_message(source_display, exc)) from exc
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_s3_parquet_rows(
    uri: str,
    *,
    rows: int,
    columns: list[str] | None,
) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading S3 Parquet previews requires `pyarrow`.") from exc

    filesystem, object_path = _s3_filesystem_and_path(uri)
    with filesystem.open_input_file(object_path) as source:
        parquet_file = pq.ParquetFile(source)
        frames: list[pd.DataFrame] = []
        remaining = rows
        for batch in parquet_file.iter_batches(batch_size=min(rows, 100_000), columns=columns):
            frame = batch.to_pandas()
            frames.append(frame.iloc[:remaining].copy(deep=True))
            remaining -= len(frames[-1])
            if remaining <= 0:
                break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _s3_filesystem_and_path(uri: str):
    try:
        import pyarrow.fs as pafs
    except ImportError as exc:
        raise ImportError("S3 data access requires `pyarrow` with filesystem support.") from exc

    filesystem, object_path = pafs.FileSystem.from_uri(uri)
    return filesystem, object_path


def _resolve_s3_effective_suffix(source_uri: str, suffix: str) -> str:
    normalized_suffix = str(suffix or "").lower()
    prefix = _read_s3_object_prefix(source_uri)
    if _looks_like_excel_prefix(prefix):
        return ".xlsx"
    if _looks_like_parquet_prefix(prefix):
        return ".parquet"
    if normalized_suffix:
        return normalized_suffix
    return ".csv" if _looks_text_like(prefix) else normalized_suffix


def _read_s3_object_prefix(source_uri: str) -> bytes:
    try:
        filesystem, object_path = _s3_filesystem_and_path(source_uri)
        with filesystem.open_input_file(object_path) as source:
            return _read_stream_prefix(source)
    except Exception:
        return b""


def _validate_csv_stream_signature(source_display: str, source: Any) -> None:
    prefix = _read_stream_prefix(source)
    if not prefix:
        return
    stripped = prefix.lstrip()
    lower_name = source_display.lower()
    if _looks_like_parquet_prefix(prefix) or _looks_like_parquet_prefix(stripped):
        raise ValueError(
            "The object appears to be a Parquet file, but it is being read as CSV. "
            "Use an S3 path ending in `.parquet` or `.pq`."
        )
    if _looks_like_excel_prefix(prefix):
        raise ValueError(
            "The object appears to be an Excel or ZIP file, but it is being read "
            "as CSV. Use an S3 path ending in `.xlsx`, `.xlsm`, or `.xls` for Excel "
            "files, or export it as UTF-8 CSV or Parquet."
        )
    if prefix.startswith(b"\x1f\x8b"):
        raise ValueError(
            "The object appears to be gzip-compressed. S3 CSV preview expects an "
            "uncompressed `.csv` object. Decompress it first or convert it to Parquet."
        )
    if prefix.startswith(b"BZh") or prefix.startswith(b"\xfd7zXZ\x00"):
        raise ValueError(
            "The object appears to be compressed. S3 CSV preview expects an "
            "uncompressed `.csv` object. Decompress it first or convert it to Parquet."
        )
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) or _nul_byte_ratio(prefix) > 0.10:
        raise ValueError(
            "The object looks like UTF-16 or binary data, not a plain UTF-8 CSV. "
            "Export it as UTF-8 CSV or convert it to Parquet before using S3 intake."
        )
    if lower_name.endswith((".xlsx", ".xls", ".xlsm")):
        raise ValueError(
            "The object has an Excel extension but is being read as CSV. Use the "
            "Excel S3 path directly instead of a `.csv` suffix."
        )


def _looks_like_excel_prefix(prefix: bytes) -> bool:
    return prefix.startswith(b"PK\x03\x04") or prefix.startswith(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    )


def _looks_like_parquet_prefix(prefix: bytes) -> bool:
    return prefix.startswith(b"PAR1")


def _looks_text_like(payload: bytes) -> bool:
    if not payload:
        return False
    sample = payload[:1024]
    if _nul_byte_ratio(sample) > 0.01:
        return False
    text_bytes = sum(
        1
        for byte in sample
        if byte in {9, 10, 13} or 32 <= byte <= 126 or byte >= 128
    )
    return text_bytes / len(sample) > 0.90


def _read_stream_prefix(source: Any, *, byte_count: int = 4096) -> bytes:
    position = None
    try:
        position = source.tell()
    except Exception:
        position = None
    try:
        prefix = source.read(byte_count)
    except Exception:
        return b""
    finally:
        if position is not None:
            try:
                source.seek(position)
            except Exception:
                pass
    if isinstance(prefix, str):
        return prefix.encode("utf-8", errors="replace")
    return bytes(prefix or b"")


def _read_stream_payload(source: Any) -> bytes:
    try:
        source.seek(0)
    except Exception:
        pass
    payload = source.read()
    if isinstance(payload, str):
        return payload.encode("utf-8", errors="replace")
    return bytes(payload or b"")


def _nul_byte_ratio(payload: bytes) -> float:
    if not payload:
        return 0.0
    return payload.count(b"\x00") / len(payload)


def _csv_parse_failure_message(source_display: str, exc: Exception) -> str:
    detail = _sanitize_error_detail(str(exc))
    extra = ""
    if "expected 1 columns" in detail.lower() or "got" in detail.lower():
        extra = (
            " The first rows may have inconsistent column counts, a preamble before "
            "the header, a non-comma delimiter, or a malformed quoted field."
        )
    return (
        f"Could not parse `{source_display}` as a plain CSV.{extra} "
        "Confirm the object is an uncompressed UTF-8 comma-delimited CSV with one "
        "header row, or convert it to Parquet. Parser detail: "
        f"{detail}"
    )


def _sanitize_error_detail(detail: str, *, max_length: int = 240) -> str:
    cleaned = "".join(
        character if character.isprintable() or character in {"\t", " "} else "?"
        for character in detail.replace("\r", " ").replace("\n", " ")
    )
    cleaned = cleaned.encode("ascii", errors="replace").decode("ascii")
    if len(cleaned) > max_length:
        return cleaned[: max_length - 3].rstrip() + "..."
    return cleaned


def _source_cache_key(source_identifier: str, metadata: dict[str, Any]) -> str:
    fingerprint = {
        "source_identifier": source_identifier,
        "size_bytes": metadata.get("size_bytes", ""),
        "modified_at_utc": metadata.get("modified_at_utc", ""),
        "modified_ns": metadata.get("modified_ns", ""),
        "etag": metadata.get("etag", ""),
    }
    encoded = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _parquet_row_count(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        return max(0, line_count - 1)
    except OSError:
        return None


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
