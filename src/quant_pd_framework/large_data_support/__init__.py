"""Large-data helper modules for file-backed execution."""

from quant_pd_framework.large_data_support.constants import (
    GIB,
    PARQUET_SUFFIXES,
    SUPPORTED_TABULAR_SUFFIXES,
)
from quant_pd_framework.large_data_support.handles import (
    DatasetHandle,
    build_dataset_handle,
    build_s3_dataset_handle,
    is_s3_uri,
    normalize_s3_metadata,
    parse_s3_uri,
    read_tabular_path,
)

__all__ = [
    "DatasetHandle",
    "GIB",
    "PARQUET_SUFFIXES",
    "SUPPORTED_TABULAR_SUFFIXES",
    "build_dataset_handle",
    "build_s3_dataset_handle",
    "is_s3_uri",
    "normalize_s3_metadata",
    "parse_s3_uri",
    "read_tabular_path",
]

