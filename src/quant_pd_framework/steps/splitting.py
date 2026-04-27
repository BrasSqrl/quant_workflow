"""Partitions the prepared dataset into train, validation, and test sets."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from ..base import BasePipelineStep
from ..config import DataStructure, TargetMode
from ..context import PipelineContext


class SplitStep(BasePipelineStep):
    """
    Applies the split strategy that matches the declared data structure.

    Cross-sectional datasets use randomized stratified sampling, while time-aware
    datasets use chronological splits to reduce leakage risk.
    """

    name = "splitting"

    def run(self, context: PipelineContext) -> PipelineContext:
        dataframe = context.working_data
        target_column = context.target_column
        if dataframe is None or target_column is None:
            raise ValueError("Splitting requires a dataframe and target column.")
        labels_available = (
            bool(context.metadata.get("labels_available", False))
            and target_column in dataframe.columns
        )

        split_config = context.config.split
        if split_config.data_structure == DataStructure.CROSS_SECTIONAL:
            split_frames = self._split_cross_sectional(
                dataframe, target_column, labels_available, context
            )
        else:
            split_frames = self._split_time_aware(
                dataframe, target_column, labels_available, context
            )

        context.split_frames = split_frames
        context.metadata["split_summary"] = {
            split_name: {
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
            }
            for split_name, frame in split_frames.items()
        }
        return context

    def _split_cross_sectional(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        labels_available: bool,
        context: PipelineContext,
    ) -> dict[str, pd.DataFrame]:
        split_config = context.config.split
        holdout_size = split_config.validation_size + split_config.test_size
        if holdout_size == 0:
            return {"train": dataframe.reset_index(drop=True)}

        stratify = (
            dataframe[target_column]
            if labels_available
            and split_config.stratify
            and context.config.target.mode == TargetMode.BINARY
            else None
        )

        train_frame, temp_frame = train_test_split(
            dataframe,
            test_size=holdout_size,
            random_state=split_config.random_state,
            stratify=stratify,
        )
        split_frames = {"train": train_frame.reset_index(drop=True)}
        if split_config.validation_size == 0:
            split_frames["test"] = temp_frame.reset_index(drop=True)
            return split_frames
        if split_config.test_size == 0:
            split_frames["validation"] = temp_frame.reset_index(drop=True)
            return split_frames

        temp_stratify = (
            temp_frame[target_column]
            if labels_available
            and split_config.stratify
            and context.config.target.mode == TargetMode.BINARY
            else None
        )
        validation_share_of_temp = split_config.validation_size / holdout_size

        validation_frame, test_frame = train_test_split(
            temp_frame,
            test_size=1 - validation_share_of_temp,
            random_state=split_config.random_state,
            stratify=temp_stratify,
        )

        split_frames["validation"] = validation_frame.reset_index(drop=True)
        split_frames["test"] = test_frame.reset_index(drop=True)
        return split_frames

    def _split_time_aware(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        labels_available: bool,
        context: PipelineContext,
    ) -> dict[str, pd.DataFrame]:
        split_config = context.config.split
        date_column = split_config.date_column
        if date_column is None:
            raise ValueError("Time-aware splits require SplitConfig.date_column.")

        working = dataframe.copy(deep=False)
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        if working[date_column].isna().all():
            raise ValueError(
                f"Date column '{date_column}' could not be parsed into valid timestamps."
            )

        if split_config.data_structure == DataStructure.PANEL and split_config.entity_column:
            # Sorting by entity and then date keeps each panel member ordered in time,
            # while the final split boundary is still chronological over the full sample.
            working = working.sort_values([date_column, split_config.entity_column])
        else:
            working = working.sort_values(date_column)

        total_rows = len(working)
        train_end = math.floor(total_rows * split_config.train_size)

        split_frames = {
            "train": working.iloc[:train_end].reset_index(drop=True),
        }
        if split_config.validation_size > 0 and split_config.test_size > 0:
            validation_end = train_end + math.floor(total_rows * split_config.validation_size)
            split_frames["validation"] = working.iloc[train_end:validation_end].reset_index(
                drop=True
            )
            split_frames["test"] = working.iloc[validation_end:].reset_index(drop=True)
        elif split_config.validation_size > 0:
            split_frames["validation"] = working.iloc[train_end:].reset_index(drop=True)
        elif split_config.test_size > 0:
            split_frames["test"] = working.iloc[train_end:].reset_index(drop=True)

        for split_name, split_frame in split_frames.items():
            if split_frame.empty:
                raise ValueError(
                    f"The {split_name} split is empty. Adjust the split "
                    "percentages or add more rows."
                )
            if (
                labels_available
                and split_frame[target_column].nunique() < 2
                and split_name != "test"
            ):
                context.warn(
                    f"The {split_name} split only contains one target class. "
                    "This can happen with strongly time-ordered defaults."
                )

        return split_frames
