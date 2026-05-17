"""Dataset handles and source metadata helpers for large-data intake."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from quant_pd_framework.large_data_support.constants import PARQUET_SUFFIXES


@dataclass(slots=True)
class DatasetHandle:
    """A file-backed dataset reference used when large data mode avoids eager loading."""

    path: Path | None = None
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    staged_path: Path | None = None
    staging_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_path(self) -> Path:
        if self.staged_path is not None:
            return self.staged_path
        if self.path is not None:
            return self.path
        raise ValueError(
            "This dataset handle does not have a local active path. Stage the "
            "remote source before requesting active_path."
        )

    @property
    def source_identifier(self) -> str:
        if self.uri:
            return self.uri
        if self.path is not None:
            return str(self.path)
        return ""

    @property
    def is_s3(self) -> bool:
        return is_s3_uri(self.source_identifier)

    @property
    def source_suffix(self) -> str:
        metadata_suffix = str(self.metadata.get("suffix") or "").lower()
        if metadata_suffix:
            return metadata_suffix
        parsed_path = urlparse(self.source_identifier).path if self.uri else self.source_identifier
        return Path(parsed_path).suffix.lower()

    @property
    def suffix(self) -> str:
        if self.staged_path is not None or self.path is not None:
            return self.active_path.suffix.lower()
        return self.source_suffix

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
        active_path = (
            str(self.active_path)
            if self.staged_path is not None or self.path is not None
            else ""
        )
        return {
            "source_path": str(self.path) if self.path is not None else "",
            "source_uri": self.uri or "",
            "active_path": active_path,
            "source_suffix": self.source_suffix,
            "active_suffix": self.suffix,
            "source_kind": "s3" if self.is_s3 else "local_file",
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


def build_s3_dataset_handle(uri: str, metadata: dict[str, Any] | None = None) -> DatasetHandle:
    """Builds a remote S3 dataset handle with lightweight metadata."""

    if not is_s3_uri(uri):
        raise ValueError(f"Expected an s3:// URI. Received: {uri}")
    normalized_metadata = normalize_s3_metadata(uri, metadata or {})
    return DatasetHandle(uri=uri, metadata=normalized_metadata)


def is_s3_uri(value: str | Path | None) -> bool:
    """Returns whether a value is an S3 URI."""

    return str(value or "").strip().lower().startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parses an S3 URI into bucket and key."""

    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("S3 paths must use the form s3://bucket/key.")
    return parsed.netloc, parsed.path.lstrip("/")


def normalize_s3_metadata(uri: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalizes S3 metadata without storing credentials."""

    bucket, key = parse_s3_uri(uri)
    suffix = Path(key).suffix.lower()
    payload = {
        "source_kind": "s3",
        "display_label": uri,
        "source_uri": uri,
        "bucket": bucket,
        "key": key,
        "file_name": Path(key).name,
        "relative_path": uri,
        "suffix": suffix,
        "size_bytes": "",
        "modified_at_utc": "",
        "etag": "",
    }
    payload.update(dict(metadata or {}))
    payload["source_kind"] = "s3"
    payload["source_uri"] = uri
    payload["display_label"] = uri
    payload["bucket"] = bucket
    payload["key"] = key
    payload["suffix"] = suffix
    return payload
