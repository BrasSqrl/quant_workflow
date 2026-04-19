"""Reads the starting dataframe from memory, CSV, or Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import BasePipelineStep
from ..context import PipelineContext


class IngestionStep(BasePipelineStep):
    """
    Pulls the source data into pandas so every downstream step sees a dataframe.

    This is the intake layer of the quant pipeline. It hides file-format details
    from the rest of the framework and normalizes the starting point.
    """

    name = "ingestion"

    def run(self, context: PipelineContext) -> PipelineContext:
        raw_input = context.raw_input

        if isinstance(raw_input, pd.DataFrame):
            dataframe = raw_input.copy(deep=True)
            context.metadata["input_type"] = "dataframe"
        else:
            input_path = Path(raw_input)
            context.metadata["input_type"] = input_path.suffix.lower()
            dataframe = self._read_file(input_path)

        if dataframe.empty:
            raise ValueError("The input dataframe is empty, so the pipeline cannot continue.")

        context.raw_data = dataframe
        context.working_data = dataframe.copy(deep=True)
        context.metadata["input_shape"] = {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
        }
        return context

    def _read_file(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            try:
                return pd.read_excel(path)
            except ImportError as exc:
                raise ImportError(
                    "Reading Excel files requires the optional dependency `openpyxl`."
                ) from exc

        raise ValueError(
            "Unsupported input format. Provide a pandas dataframe, CSV file, or Excel file."
        )
