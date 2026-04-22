"""Data source and dataset preview helpers for the Streamlit app."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from quant_pd_framework import PerformanceConfig, build_sample_pd_dataframe

MAX_UPLOAD_SIZE_MB = 51_200
DEFAULT_PERFORMANCE_CONFIG = PerformanceConfig()


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe_bytes(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)
    raise ValueError("Unsupported file type. Upload CSV or Excel.")


@st.cache_data(show_spinner=False)
def load_bundled_sample_dataframe() -> pd.DataFrame:
    return build_sample_pd_dataframe()


def select_input_dataframe() -> tuple[pd.DataFrame | None, str]:
    with st.sidebar:
        with st.expander("Data Source", expanded=True):
            use_sample_data = st.toggle("Use bundled sample data", value=True)
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx", "xls"],
                max_upload_size=MAX_UPLOAD_SIZE_MB,
                help=(
                    "Configured upload limit: 50 GB per file. Practical limits still depend "
                    "on available memory and the size expansion that happens when pandas "
                    "parses large CSV or Excel files."
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

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        uploaded_file.seek(0)
        dataframe = load_uploaded_dataframe_bytes(uploaded_file.read(), suffix)
        return dataframe, uploaded_file.name
    if use_sample_data:
        return load_bundled_sample_dataframe(), "bundled_sample"
    return None, ""


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
