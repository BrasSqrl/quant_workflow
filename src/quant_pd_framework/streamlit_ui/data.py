"""Data source and dataset preview helpers for the Streamlit app."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework import PerformanceConfig, build_sample_pd_dataframe
from quant_pd_framework.large_data import (
    PARQUET_SUFFIXES,
    SUPPORTED_TABULAR_SUFFIXES,
    DatasetHandle,
    build_dataset_handle,
    convert_csv_to_parquet,
    profile_dataset_handle,
    read_dataset_preview,
    read_tabular_path,
)
from quant_pd_framework.streamlit_ui.theme import render_html

MAX_UPLOAD_SIZE_MB = 51_200
DEFAULT_PERFORMANCE_CONFIG = PerformanceConfig()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_LOAD_DIR = PROJECT_ROOT / "Data_Load"
SUPPORTED_DATA_FILE_SUFFIXES = SUPPORTED_TABULAR_SUFFIXES
INPUT_SOURCE_METADATA_ATTR = "quant_studio_input_source"


@dataclass(frozen=True, slots=True)
class SelectedInputDataset:
    """Holds the dataframe selected in the workflow plus audit metadata."""

    dataframe: pd.DataFrame | None
    label: str
    metadata: dict[str, Any]
    dataset_handle: DatasetHandle | None = None
    large_data_mode: bool = False


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe_bytes(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(buffer)
    if suffix in PARQUET_SUFFIXES:
        return pd.read_parquet(buffer)
    raise ValueError("Unsupported file type. Provide CSV, Excel, or Parquet.")


@st.cache_data(show_spinner=False)
def load_data_load_dataframe(
    path_text: str,
    suffix: str,
    modified_ns: int,
    size_bytes: int,
) -> pd.DataFrame:
    """Loads a Data_Load file with cache invalidation tied to file metadata."""

    _ = (modified_ns, size_bytes)
    path = Path(path_text)
    return read_tabular_path(path)


def convert_data_load_csv_to_parquet(path: Path) -> Path:
    destination = path.with_suffix(".parquet")
    convert_csv_to_parquet(
        path,
        destination,
        chunk_rows=DEFAULT_PERFORMANCE_CONFIG.csv_conversion_chunk_rows,
        compression="snappy",
    )
    return destination


@st.cache_data(show_spinner=False)
def load_bundled_sample_dataframe() -> pd.DataFrame:
    return build_sample_pd_dataframe()


def list_data_load_files(data_load_dir: Path = DATA_LOAD_DIR) -> list[Path]:
    """Returns supported, direct-child data files from the landing directory."""

    if not data_load_dir.exists():
        return []
    return sorted(
        [
            path
            for path in data_load_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_DATA_FILE_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )


def describe_data_file(path: Path) -> dict[str, Any]:
    """Builds lightweight file metadata for display and run auditability."""

    stat_result = path.stat()
    modified_at_utc = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_path = path.name
    return {
        "source_kind": "data_load",
        "display_label": f"Data_Load/{path.name}",
        "file_name": path.name,
        "relative_path": relative_path,
        "suffix": path.suffix.lower(),
        "size_bytes": int(stat_result.st_size),
        "modified_at_utc": modified_at_utc.isoformat(),
        "modified_ns": int(stat_result.st_mtime_ns),
    }


def format_file_size(size_bytes: int) -> str:
    """Formats byte counts for compact sidebar display."""

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def format_data_load_file_option(path: Path) -> str:
    metadata = describe_data_file(path)
    modified_at = datetime.fromisoformat(metadata["modified_at_utc"]).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    return f"{path.name} ({format_file_size(metadata['size_bytes'])}, {modified_at})"


def attach_input_source_metadata(
    dataframe: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    dataframe.attrs[INPUT_SOURCE_METADATA_ATTR] = metadata
    return dataframe


def select_input_dataframe() -> SelectedInputDataset:
    with st.container():
        render_html(
            '<div class="step-panel-intro">'
            '<span class="step-panel-kicker">Data Source</span>'
            '<h3 class="step-panel-title">Choose the input dataset</h3>'
            '<p class="step-panel-copy">'
            "Use the bundled sample, select a governed landing-zone file, "
            "or upload a CSV, Excel, or Parquet file for a full workflow run."
            "</p>"
            "</div>"
        )
        with st.expander("Data Source", expanded=True):
            large_data_mode = st.toggle(
                "Large Data Mode",
                value=False,
                help=(
                    "Use Data_Load file-backed intake for large files. The app previews "
                    "a small sample, then the run trains on a governed sample and scores "
                    "the full file in chunks."
                ),
            )
            source_mode = st.radio(
                "Input method",
                options=["sample", "data_load", "upload"],
                format_func={
                    "sample": "Bundled sample data",
                    "data_load": "Select from Data_Load",
                    "upload": "Upload file",
                }.get,
                horizontal=False,
            )
            if large_data_mode and source_mode == "upload":
                st.warning(
                    "Large Data Mode supports Data_Load or CLI file paths only. "
                    "Browser upload still buffers the full file through Streamlit."
                )
                return SelectedInputDataset(None, "", {}, large_data_mode=True)
            if source_mode == "sample":
                st.caption("Uses the bundled synthetic PD dataset for a quick start.")
            elif source_mode == "data_load":
                DATA_LOAD_DIR.mkdir(exist_ok=True)
                st.caption(
                    "Drop CSV, Excel, or Parquet files into `Data_Load/`, then refresh this list."
                )
                refresh_clicked = st.button("Refresh available files", width="stretch")
                if refresh_clicked:
                    load_data_load_dataframe.clear()
                available_files = list_data_load_files()
                if not available_files:
                    st.info("No supported CSV, Excel, or Parquet files were found in Data_Load.")
                    return SelectedInputDataset(None, "", {})
                selected_path = st.selectbox(
                    "Available Data_Load files",
                    options=available_files,
                    format_func=format_data_load_file_option,
                )
                file_metadata = describe_data_file(selected_path)
                if selected_path.suffix.lower() == ".csv":
                    if st.button("Convert selected CSV to Parquet", width="stretch"):
                        try:
                            converted_path = convert_data_load_csv_to_parquet(selected_path)
                            load_data_load_dataframe.clear()
                            st.success(f"Created `{converted_path.name}`. Refresh the file list.")
                        except Exception as exc:
                            st.error(f"CSV-to-Parquet conversion failed: {exc}")
                st.caption(
                    "Selected file: "
                    f"`{file_metadata['relative_path']}` | "
                    f"{format_file_size(file_metadata['size_bytes'])}"
                )
                if large_data_mode:
                    dataset_handle = build_dataset_handle(selected_path, file_metadata)
                    preview_frame = read_dataset_preview(
                        dataset_handle,
                        rows=DEFAULT_PERFORMANCE_CONFIG.ui_preview_rows,
                    )
                    try:
                        profile = profile_dataset_handle(dataset_handle)
                        file_metadata["large_data_profile"] = profile
                        if profile.get("row_count") is not None:
                            st.caption(
                                f"File-backed profile: {int(profile['row_count']):,} rows, "
                                f"{int(profile['column_count']):,} columns."
                            )
                    except Exception as exc:
                        st.caption(f"File-backed profile unavailable: {exc}")
                    attach_input_source_metadata(preview_frame, file_metadata)
                    return SelectedInputDataset(
                        preview_frame,
                        file_metadata["display_label"],
                        file_metadata,
                        dataset_handle=dataset_handle,
                        large_data_mode=True,
                    )
                dataframe = load_data_load_dataframe(
                    str(selected_path),
                    file_metadata["suffix"],
                    file_metadata["modified_ns"],
                    file_metadata["size_bytes"],
                )
                attach_input_source_metadata(dataframe, file_metadata)
                return SelectedInputDataset(
                    dataframe,
                    file_metadata["display_label"],
                    file_metadata,
                    large_data_mode=False,
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload CSV, Excel, or Parquet",
                    type=["csv", "xlsx", "xls", "xlsm", "parquet", "pq"],
                    max_upload_size=MAX_UPLOAD_SIZE_MB,
                    help=(
                        "Configured upload limit: 50 GB per file. Practical limits still "
                        "depend on available memory and the size expansion that happens "
                        "when pandas parses large CSV, Excel, or Parquet files."
                    ),
                )
                if (
                    uploaded_file is not None
                    and getattr(uploaded_file, "size", 0)
                    > DEFAULT_PERFORMANCE_CONFIG.upload_warning_mb * 1024 * 1024
                ):
                    st.warning(
                        "Large upload detected. The file is within the configured limit, but "
                        "practical runtime still depends on local memory and pandas parsing cost."
                    )

    if source_mode == "upload" and uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        uploaded_file.seek(0)
        dataframe = load_uploaded_dataframe_bytes(uploaded_file.read(), suffix)
        metadata = {
            "source_kind": "upload",
            "display_label": uploaded_file.name,
            "file_name": uploaded_file.name,
            "relative_path": "",
            "suffix": suffix,
            "size_bytes": int(getattr(uploaded_file, "size", 0) or 0),
            "modified_at_utc": "",
        }
        attach_input_source_metadata(dataframe, metadata)
        return SelectedInputDataset(dataframe, uploaded_file.name, metadata)
    if source_mode == "sample":
        dataframe = load_bundled_sample_dataframe()
        metadata = {
            "source_kind": "bundled_sample",
            "display_label": "Bundled Sample",
            "file_name": "",
            "relative_path": "",
            "suffix": ".csv",
            "size_bytes": "",
            "modified_at_utc": "",
        }
        attach_input_source_metadata(dataframe, metadata)
        return SelectedInputDataset(
            dataframe,
            "bundled_sample",
            metadata,
            large_data_mode=large_data_mode,
        )
    return SelectedInputDataset(None, "", {}, large_data_mode=large_data_mode)


def build_editor_key(dataframe: pd.DataFrame, data_source_label: str) -> str:
    signature = "|".join(
        [
            data_source_label,
            str(dataframe.shape[0]),
            str(dataframe.shape[1]),
            ",".join(map(str, dataframe.columns)),
        ]
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"schema_editor_{digest}"


def render_dataset_overview(dataframe: pd.DataFrame, data_source_label: str) -> None:
    file_label = "Bundled Sample" if data_source_label == "bundled_sample" else data_source_label
    left_column, right_column, third_column = st.columns(3)
    left_column.metric("Rows", f"{len(dataframe):,}")
    right_column.metric("Columns", f"{dataframe.shape[1]:,}")
    third_column.metric("Source", file_label)


def render_input_performance_notice(dataframe: pd.DataFrame) -> None:
    performance = DEFAULT_PERFORMANCE_CONFIG
    if (
        dataframe.shape[0] >= performance.dataframe_warning_rows
        or dataframe.shape[1] >= performance.dataframe_warning_columns
    ):
        st.warning(
            "Large input detected. Quant Studio will sample heavy diagnostics and report "
            "previews where needed, but full runtime still depends on local memory."
        )


def sample_frame(dataframe: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(dataframe) <= max_rows:
        return dataframe
    return dataframe.sample(max_rows, random_state=42)
