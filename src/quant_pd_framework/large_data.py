"""Large-dataset helpers for intake, memory estimation, and tabular exports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PerformanceConfig

SUPPORTED_TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".xlsm", ".parquet", ".pq"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
GIB = 1024**3


@dataclass(slots=True)
class DatasetHandle:
    """A file-backed dataset reference used when large data mode avoids eager loading."""

    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    staged_path: Path | None = None
    staging_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_path(self) -> Path:
        return self.staged_path or self.path

    @property
    def suffix(self) -> str:
        return self.active_path.suffix.lower()

    def with_staging(
        self,
        staged_path: Path,
        staging_metadata: dict[str, Any],
    ) -> DatasetHandle:
        return replace(
            self,
            staged_path=staged_path,
            staging_metadata=staging_metadata,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_path": str(self.path),
            "active_path": str(self.active_path),
            "source_suffix": self.path.suffix.lower(),
            "active_suffix": self.suffix,
            "metadata": dict(self.metadata),
            "staging": dict(self.staging_metadata),
        }


def read_tabular_path(path: Path) -> pd.DataFrame:
    """Reads a supported tabular file into pandas."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError("Reading Excel files requires `openpyxl`.") from exc
    if suffix in PARQUET_SUFFIXES:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError("Reading Parquet files requires `pyarrow` or `fastparquet`.") from exc
    raise ValueError(
        "Unsupported input format. Provide a pandas dataframe, CSV, Excel, or Parquet file."
    )


def build_dataset_handle(path: Path, metadata: dict[str, Any] | None = None) -> DatasetHandle:
    """Builds a file-backed dataset handle with lightweight metadata."""

    return DatasetHandle(path=path, metadata=dict(metadata or {}))


def stage_large_data_file(
    handle: DatasetHandle,
    *,
    chunk_rows: int,
    compression: str,
) -> DatasetHandle:
    """
    Returns a handle whose active path is a reusable Parquet staging file.

    Existing Parquet inputs are reused directly. CSV inputs are converted into a
    source-adjacent cache keyed by file size and modified timestamp so repeated
    runs do not reconvert unchanged data.
    """

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


def read_dataset_preview(
    handle: DatasetHandle,
    *,
    rows: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Reads a small preview from a file-backed dataset."""

    path = handle.active_path
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=rows, usecols=columns)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
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

    path = handle.active_path
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=batch_rows, usecols=columns)
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


def profile_dataset_handle(handle: DatasetHandle, *, preview_rows: int = 50) -> dict[str, Any]:
    """Builds a lightweight profile without loading the full file into memory."""

    preview = read_dataset_preview(handle, rows=preview_rows)
    row_count: int | None = None
    if handle.suffix in PARQUET_SUFFIXES:
        try:
            import pyarrow.parquet as pq

            row_count = int(pq.ParquetFile(handle.active_path).metadata.num_rows)
        except ImportError:
            row_count = None
    elif handle.suffix == ".csv":
        row_count = None

    return {
        "active_path": str(handle.active_path),
        "row_count": row_count,
        "column_count": int(preview.shape[1]),
        "columns": list(preview.columns),
        "dtypes": {column: str(dtype) for column, dtype in preview.dtypes.items()},
        "preview_rows": int(len(preview)),
    }


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
) -> dict[str, Any]:
    """Converts a CSV file to Parquet in chunks using pyarrow."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Chunked CSV-to-Parquet conversion requires `pyarrow`.") from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    chunk_count = 0
    row_count = 0
    try:
        for chunk in pd.read_csv(source_path, chunksize=chunk_rows):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
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
            row_count += int(len(chunk))
    finally:
        if writer is not None:
            writer.close()

    return {
        "source_path": str(source_path),
        "destination_path": str(destination_path),
        "chunk_rows": int(chunk_rows),
        "chunk_count": int(chunk_count),
        "row_count": int(row_count),
        "compression": compression,
    }


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
