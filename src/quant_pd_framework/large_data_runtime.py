"""File-backed runtime primitives for large-data UI and execution paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import LargeDataExecutionStageStatus
from .large_data import PARQUET_SUFFIXES


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """UI-safe dataset metadata that avoids storing full dataframes."""

    source_kind: str
    source_identifier: str
    active_path: str
    source_suffix: str
    active_suffix: str
    row_count: int | None
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    preview_rows: int
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: str = ""

    @classmethod
    def from_profile_dict(cls, profile: dict[str, Any]) -> DatasetProfile:
        source_identifier = (
            str(profile.get("source_uri") or "")
            or str(profile.get("source_path") or "")
            or str(profile.get("active_path") or "")
        )
        cache_key = str(profile.get("profile_cache_key") or "") or _stable_hash(
            {
                "source_identifier": source_identifier,
                "active_path": profile.get("active_path", ""),
                "source_metadata": profile.get("source_metadata", {}),
            }
        )
        row_count = profile.get("row_count")
        return cls(
            source_kind=str(profile.get("source_kind") or ""),
            source_identifier=source_identifier,
            active_path=str(profile.get("active_path") or ""),
            source_suffix=str(profile.get("source_suffix") or ""),
            active_suffix=str(profile.get("active_suffix") or ""),
            row_count=int(row_count) if row_count is not None else None,
            column_count=int(profile.get("column_count") or 0),
            columns=[str(column) for column in profile.get("columns", [])],
            dtypes={str(key): str(value) for key, value in profile.get("dtypes", {}).items()},
            preview_rows=int(profile.get("preview_rows") or 0),
            metadata=dict(profile.get("source_metadata", {})),
            cache_key=cache_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_identifier": self.source_identifier,
            "active_path": self.active_path,
            "source_suffix": self.source_suffix,
            "active_suffix": self.active_suffix,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "preview_rows": self.preview_rows,
            "metadata": dict(self.metadata),
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True, slots=True)
class DatasetCatalogEntry:
    """One selectable large-data source in the UI catalog."""

    label: str
    profile: DatasetProfile
    profile_path: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "profile": self.profile.to_dict(),
            "profile_path": self.profile_path,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True, slots=True)
class TableRef:
    """Reference to a file-backed table that can be queried by page."""

    name: str
    path: str
    format: str
    row_count: int | None = None
    column_count: int | None = None
    columns: list[str] = field(default_factory=list)
    source_role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        name: str,
        source_role: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TableRef:
        resolved_path = Path(path)
        suffix = resolved_path.suffix.lower()
        table_format = "parquet" if suffix in PARQUET_SUFFIXES else "csv"
        row_count, columns = inspect_table_path(resolved_path)
        return cls(
            name=name,
            path=str(resolved_path),
            format=table_format,
            row_count=row_count,
            column_count=len(columns) if columns else None,
            columns=columns,
            source_role=source_role,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "format": self.format,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "source_role": self.source_role,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TableRef:
        return cls(
            name=str(payload.get("name") or ""),
            path=str(payload.get("path") or ""),
            format=str(payload.get("format") or ""),
            row_count=payload.get("row_count"),
            column_count=payload.get("column_count"),
            columns=[str(column) for column in payload.get("columns", [])],
            source_role=str(payload.get("source_role") or ""),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ResultTableRef(TableRef):
    """Specialized table reference for prediction/result browsing."""

    score_column: str = ""
    target_column: str = ""
    split_column: str = "split"
    date_columns: list[str] = field(default_factory=list)
    segment_columns: list[str] = field(default_factory=list)

    @classmethod
    def from_table_ref(
        cls,
        table_ref: TableRef,
        *,
        score_column: str = "",
        target_column: str = "",
        split_column: str = "split",
        date_columns: list[str] | None = None,
        segment_columns: list[str] | None = None,
    ) -> ResultTableRef:
        return cls(
            name=table_ref.name,
            path=table_ref.path,
            format=table_ref.format,
            row_count=table_ref.row_count,
            column_count=table_ref.column_count,
            columns=list(table_ref.columns),
            source_role=table_ref.source_role,
            metadata=dict(table_ref.metadata),
            score_column=score_column,
            target_column=target_column,
            split_column=split_column,
            date_columns=list(date_columns or []),
            segment_columns=list(segment_columns or []),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "score_column": self.score_column,
                "target_column": self.target_column,
                "split_column": self.split_column,
                "date_columns": list(self.date_columns),
                "segment_columns": list(self.segment_columns),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResultTableRef:
        table_ref = TableRef.from_dict(payload)
        return cls.from_table_ref(
            table_ref,
            score_column=str(payload.get("score_column") or ""),
            target_column=str(payload.get("target_column") or ""),
            split_column=str(payload.get("split_column") or "split"),
            date_columns=[str(column) for column in payload.get("date_columns", [])],
            segment_columns=[str(column) for column in payload.get("segment_columns", [])],
        )


@dataclass(frozen=True, slots=True)
class PreparedDatasetManifest:
    """Manifest for a staged large-data modeling dataset."""

    run_id: str
    source_identifier: str
    staged_path: str
    sample_path: str = ""
    projected_columns: list[str] = field(default_factory=list)
    split_paths: dict[str, str] = field(default_factory=dict)
    transformation_contract_keys: list[str] = field(default_factory=list)
    target_column: str = ""
    row_count: int | None = None
    cache_key: str = ""
    profile_cache_key: str = ""
    partition_columns: list[str] = field(default_factory=list)
    partition_paths: dict[str, str] = field(default_factory=dict)
    artifact_size_estimates: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_identifier": self.source_identifier,
            "staged_path": self.staged_path,
            "sample_path": self.sample_path,
            "projected_columns": list(self.projected_columns),
            "split_paths": dict(self.split_paths),
            "transformation_contract_keys": list(self.transformation_contract_keys),
            "target_column": self.target_column,
            "row_count": self.row_count,
            "cache_key": self.cache_key,
            "profile_cache_key": self.profile_cache_key or self.cache_key,
            "partition_columns": list(self.partition_columns),
            "partition_paths": dict(self.partition_paths),
            "artifact_size_estimates": dict(self.artifact_size_estimates),
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True, slots=True)
class LargeDataExecutionPlanStage:
    """One planned large-data execution stage and its capability status."""

    stage_name: str
    status: LargeDataExecutionStageStatus
    basis: str
    description: str
    blocking: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "basis": self.basis,
            "description": self.description,
            "blocking": self.blocking,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class LargeDataExecutionPlan:
    """Auditable plan for how a large-data run will execute."""

    run_id: str
    source_identifier: str
    model_type: str
    backend: str
    policy: str
    certification_status: str
    fit_capability_status: str
    stages: list[LargeDataExecutionPlanStage]
    profile_cache_key: str = ""
    worker_mode: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_identifier": self.source_identifier,
            "model_type": self.model_type,
            "backend": self.backend,
            "policy": self.policy,
            "certification_status": self.certification_status,
            "fit_capability_status": self.fit_capability_status,
            "profile_cache_key": self.profile_cache_key,
            "worker_mode": self.worker_mode,
            "stages": [stage.to_dict() for stage in self.stages],
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True, slots=True)
class LargeDataTransformationContract:
    """Records how each governed transformation can execute in Large Data Mode."""

    run_id: str
    rows: list[dict[str, Any]]
    supported_count: int
    fallback_count: int
    unsupported_count: int
    blocked_count: int
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rows": [dict(row) for row in self.rows],
            "supported_count": self.supported_count,
            "fallback_count": self.fallback_count,
            "unsupported_count": self.unsupported_count,
            "blocked_count": self.blocked_count,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True, slots=True)
class LargeDataFeatureScreenManifest:
    """Records advisory feature pre-screening evidence for large-data runs."""

    run_id: str
    basis: str
    row_count: int
    feature_count: int
    auto_apply: bool
    excluded_features: list[str]
    table_path: str = ""
    parquet_path: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "basis": self.basis,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "auto_apply": self.auto_apply,
            "excluded_features": list(self.excluded_features),
            "table_path": self.table_path,
            "parquet_path": self.parquet_path,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True, slots=True)
class PartitionedDatasetManifest:
    """Manifest for sample/prepared data written as split-aware Parquet partitions."""

    run_id: str
    dataset_role: str
    partition_strategy: str
    partition_columns: list[str]
    partition_paths: dict[str, str]
    row_counts: dict[str, int]
    base_path: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_role": self.dataset_role,
            "partition_strategy": self.partition_strategy,
            "partition_columns": list(self.partition_columns),
            "partition_paths": dict(self.partition_paths),
            "row_counts": dict(self.row_counts),
            "base_path": self.base_path,
            "created_at_utc": self.created_at_utc,
        }


def inspect_table_path(path: Path) -> tuple[int | None, list[str]]:
    """Returns row count and column names without materializing the whole table."""

    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            return int(parquet_file.metadata.num_rows), list(parquet_file.schema.names)
        except Exception:
            return None, []
    if suffix == ".csv":
        try:
            import pyarrow.csv as pc

            reader = pc.open_csv(path, read_options=pc.ReadOptions(block_size=1 << 20))
            return None, list(reader.schema.names)
        except Exception:
            try:
                preview = pd.read_csv(path, nrows=0)
                return None, list(preview.columns)
            except Exception:
                return None, []
    return None, []


def query_table_page(
    table_ref: TableRef | dict[str, Any],
    *,
    columns: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    sort_by: str | None = None,
    descending: bool = False,
    page: int = 1,
    page_size: int = 1_000,
    duckdb_threads: int = 0,
    duckdb_memory_limit_gb: float | None = None,
) -> pd.DataFrame:
    """Returns one page from a file-backed table using DuckDB pushdown."""

    ref = table_ref if isinstance(table_ref, TableRef) else TableRef.from_dict(table_ref)
    select_sql = _select_sql(columns, ref.columns)
    source_sql = _source_sql(ref)
    where_sql, parameters = _where_sql(filters or [], ref.columns)
    order_sql = _order_sql(sort_by, descending, ref.columns)
    limit = max(1, int(page_size))
    offset = max(0, (max(1, int(page)) - 1) * limit)
    query = (
        f"SELECT {select_sql} FROM {source_sql} "
        f"{where_sql} {order_sql} LIMIT {limit} OFFSET {offset}"
    )
    with _duckdb_connection(
        threads=duckdb_threads,
        memory_limit_gb=duckdb_memory_limit_gb,
    ) as connection:
        return connection.execute(query, parameters).df()


def count_table_rows(
    table_ref: TableRef | dict[str, Any],
    *,
    filters: list[dict[str, Any]] | None = None,
    duckdb_threads: int = 0,
    duckdb_memory_limit_gb: float | None = None,
) -> int:
    """Counts rows after optional filters without loading the table."""

    ref = table_ref if isinstance(table_ref, TableRef) else TableRef.from_dict(table_ref)
    source_sql = _source_sql(ref)
    where_sql, parameters = _where_sql(filters or [], ref.columns)
    with _duckdb_connection(
        threads=duckdb_threads,
        memory_limit_gb=duckdb_memory_limit_gb,
    ) as connection:
        value = connection.execute(
            f"SELECT COUNT(*) AS row_count FROM {source_sql} {where_sql}",
            parameters,
        ).fetchone()[0]
    return int(value)


def distinct_column_values(
    table_ref: TableRef | dict[str, Any],
    column_name: str,
    *,
    limit: int = 1_000,
    duckdb_threads: int = 0,
    duckdb_memory_limit_gb: float | None = None,
) -> list[str]:
    """Returns distinct values for one column with a hard cap for UI filters."""

    ref = table_ref if isinstance(table_ref, TableRef) else TableRef.from_dict(table_ref)
    if column_name not in ref.columns:
        return []
    query = (
        f"SELECT DISTINCT CAST({_quote_identifier(column_name)} AS VARCHAR) AS value "
        f"FROM {_source_sql(ref)} WHERE {_quote_identifier(column_name)} IS NOT NULL "
        f"ORDER BY value LIMIT {max(1, int(limit))}"
    )
    with _duckdb_connection(
        threads=duckdb_threads,
        memory_limit_gb=duckdb_memory_limit_gb,
    ) as connection:
        rows = connection.execute(query).fetchall()
    return [str(row[0]) for row in rows]


def _duckdb_connection(*, threads: int, memory_limit_gb: float | None):
    import duckdb

    connection = duckdb.connect(database=":memory:")
    if threads > 0:
        connection.execute(f"PRAGMA threads={int(threads)}")
    if memory_limit_gb is not None:
        connection.execute(f"PRAGMA memory_limit='{float(memory_limit_gb)}GB'")
    return connection


def _source_sql(table_ref: TableRef) -> str:
    escaped_path = str(Path(table_ref.path)).replace("'", "''")
    if table_ref.format == "parquet" or Path(table_ref.path).suffix.lower() in PARQUET_SUFFIXES:
        return f"read_parquet('{escaped_path}')"
    return f"read_csv_auto('{escaped_path}')"


def _select_sql(columns: list[str] | None, available_columns: list[str]) -> str:
    if not columns:
        return "*"
    selected = [
        column for column in columns if not available_columns or column in available_columns
    ]
    return ", ".join(_quote_identifier(column) for column in selected) if selected else "*"


def _where_sql(
    filters: list[dict[str, Any]],
    available_columns: list[str],
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    for filter_spec in filters:
        column_name = str(filter_spec.get("column") or "")
        if available_columns and column_name not in available_columns:
            continue
        operator = str(filter_spec.get("op") or "eq").lower()
        value = filter_spec.get("value")
        quoted = _quote_identifier(column_name)
        if operator in {"eq", "="}:
            clauses.append(f"{quoted} = ?")
            parameters.append(value)
        elif operator in {"contains", "like"}:
            clauses.append(f"CAST({quoted} AS VARCHAR) ILIKE ?")
            parameters.append(f"%{value}%")
        elif operator == "in" and isinstance(value, list) and value:
            placeholders = ", ".join("?" for _ in value)
            clauses.append(f"{quoted} IN ({placeholders})")
            parameters.extend(value)
        elif operator == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
            clauses.append(f"{quoted} BETWEEN ? AND ?")
            parameters.extend([value[0], value[1]])
    if not clauses:
        return "", parameters
    return "WHERE " + " AND ".join(clauses), parameters


def _order_sql(sort_by: str | None, descending: bool, available_columns: list[str]) -> str:
    if not sort_by or (available_columns and sort_by not in available_columns):
        return ""
    direction = "DESC" if descending else "ASC"
    return f"ORDER BY {_quote_identifier(sort_by)} {direction}"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
