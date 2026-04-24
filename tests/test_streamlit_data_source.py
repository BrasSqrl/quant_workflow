"""Tests for Streamlit data-source selection helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant_pd_framework.steps.export import ArtifactExportStep
from quant_pd_framework.steps.ingestion import IngestionStep
from quant_pd_framework.streamlit_ui.data import (
    INPUT_SOURCE_METADATA_ATTR,
    attach_input_source_metadata,
    describe_data_file,
    list_data_load_files,
    load_data_load_dataframe,
)
from tests.support import temporary_artifact_root


def test_list_data_load_files_returns_supported_direct_children() -> None:
    with temporary_artifact_root("data_load_listing") as data_load_dir:
        (data_load_dir / "b.csv").write_text("x\n1\n", encoding="utf-8")
        (data_load_dir / "a.xlsx").write_bytes(b"placeholder")
        (data_load_dir / ".hidden.csv").write_text("x\n1\n", encoding="utf-8")
        (data_load_dir / "notes.txt").write_text(
            "not a supported data file",
            encoding="utf-8",
        )
        nested_dir = data_load_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "nested.csv").write_text("x\n1\n", encoding="utf-8")

        files = list_data_load_files(data_load_dir)

    assert [path.name for path in files] == ["a.xlsx", "b.csv"]


def test_data_load_csv_metadata_flows_into_ingestion_context() -> None:
    with temporary_artifact_root("data_load_metadata") as data_load_dir:
        path = data_load_dir / "loans.csv"
        path.write_text("balance,default_status\n100,0\n200,1\n", encoding="utf-8")
        metadata = describe_data_file(path)

        dataframe = load_data_load_dataframe(
            str(path),
            metadata["suffix"],
            metadata["modified_ns"],
            metadata["size_bytes"],
        )
        attach_input_source_metadata(dataframe, metadata)

        assert dataframe.attrs[INPUT_SOURCE_METADATA_ATTR]["file_name"] == "loans.csv"
        assert dataframe.to_dict(orient="list") == {
            "balance": [100, 200],
            "default_status": [0, 1],
        }

        context = SimpleNamespace(raw_input=dataframe, metadata={})
        result = IngestionStep().run(context)

    assert isinstance(result.raw_data, pd.DataFrame)
    assert result.metadata["input_type"] == "dataframe"
    assert result.metadata["input_source"]["source_kind"] == "data_load"
    assert result.metadata["input_source"]["file_name"] == "loans.csv"
    assert result.metadata["input_shape"] == {"rows": 2, "columns": 2}


def test_reproducibility_manifest_rows_include_input_source_metadata() -> None:
    context = SimpleNamespace(
        metadata={
            "input_source": {
                "source_kind": "data_load",
                "display_label": "Data_Load/loans.csv",
                "file_name": "loans.csv",
                "relative_path": "Data_Load/loans.csv",
                "suffix": ".csv",
                "size_bytes": 42,
                "modified_at_utc": "2026-04-24T12:00:00+00:00",
            }
        }
    )

    rows = ArtifactExportStep()._build_input_source_manifest_rows(context)

    assert {"field": "input_source::source_kind", "value": "data_load"} in rows
    assert {"field": "input_source::file_name", "value": "loans.csv"} in rows
    assert {"field": "input_source::size_bytes", "value": 42} in rows
