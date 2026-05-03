"""Partitions the prepared dataset into train, validation, and test sets."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ..base import BasePipelineStep
from ..config import DataStructure, SplitConfig, SplitStrategy, TargetMode
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
        split_strategy = self._resolve_strategy(split_config)
        if split_strategy == SplitStrategy.RANDOM:
            split_frames = self._split_cross_sectional(
                dataframe, target_column, labels_available, context
            )
        elif split_strategy == SplitStrategy.CHRONOLOGICAL_PERCENTAGE:
            split_frames = self._split_time_aware_percentage(
                dataframe, target_column, labels_available, context
            )
        elif split_strategy == SplitStrategy.DATE_CUTOFF:
            split_frames = self._split_date_cutoff(dataframe, labels_available, context)
        elif split_strategy == SplitStrategy.EXPLICIT_DATE_WINDOWS:
            split_frames = self._split_explicit_date_windows(dataframe, labels_available, context)
        elif split_strategy == SplitStrategy.CUSTOM_COLUMN:
            split_frames = self._split_custom_column(dataframe, labels_available, context)
        else:
            raise ValueError(f"Unsupported split strategy: {split_strategy.value}")

        context.split_frames = split_frames
        context.metadata["split_summary"] = {
            split_name: {
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                **self._date_range_summary(frame, split_config.date_column),
            }
            for split_name, frame in split_frames.items()
        }
        context.metadata["split_assignment"] = {
            "strategy": split_strategy.value,
            "data_structure": split_config.data_structure.value,
            "date_column": split_config.date_column,
            "entity_column": split_config.entity_column,
            "custom_split_column": split_config.custom_split_column,
        }
        return context

    def _resolve_strategy(self, split_config: SplitConfig) -> SplitStrategy:
        strategy = SplitStrategy(split_config.split_strategy)
        if strategy != SplitStrategy.AUTO:
            return strategy
        if split_config.data_structure == DataStructure.CROSS_SECTIONAL:
            return SplitStrategy.RANDOM
        return SplitStrategy.CHRONOLOGICAL_PERCENTAGE

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
            and context.config.target.mode in {TargetMode.BINARY, TargetMode.MULTICLASS}
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
            and context.config.target.mode in {TargetMode.BINARY, TargetMode.MULTICLASS}
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

    def _split_time_aware_percentage(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        labels_available: bool,
        context: PipelineContext,
    ) -> dict[str, pd.DataFrame]:
        split_config = context.config.split
        working = self._prepare_time_ordered_frame(dataframe, split_config)

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

        self._validate_split_frames(split_frames, target_column, labels_available, context)
        return split_frames

    def _split_date_cutoff(
        self,
        dataframe: pd.DataFrame,
        labels_available: bool,
        context: PipelineContext,
    ) -> dict[str, pd.DataFrame]:
        split_config = context.config.split
        working = self._prepare_time_ordered_frame(dataframe, split_config, require_all_dates=True)
        date_column = str(split_config.date_column)
        validation_start = self._parse_optional_date(
            split_config.validation_start_date,
            field_name="validation_start_date",
        )
        test_start = self._parse_optional_date(
            split_config.test_start_date,
            field_name="test_start_date",
        )
        if (
            validation_start is not None
            and test_start is not None
            and validation_start >= test_start
        ):
            raise ValueError("validation_start_date must be before test_start_date.")

        date_values = working[date_column]
        if validation_start is not None and test_start is not None:
            split_frames = {
                "train": working.loc[date_values < validation_start],
                "validation": working.loc[
                    (date_values >= validation_start) & (date_values < test_start)
                ],
                "test": working.loc[date_values >= test_start],
            }
        elif validation_start is not None:
            split_frames = {
                "train": working.loc[date_values < validation_start],
                "validation": working.loc[date_values >= validation_start],
            }
        elif test_start is not None:
            split_frames = {
                "train": working.loc[date_values < test_start],
                "test": working.loc[date_values >= test_start],
            }
        else:
            raise ValueError(
                "date_cutoff split strategy requires validation_start_date or test_start_date."
            )
        split_frames = self._reset_split_indexes(split_frames)
        self._validate_split_frames(
            split_frames,
            str(context.target_column),
            labels_available,
            context,
        )
        return split_frames

    def _split_explicit_date_windows(
        self,
        dataframe: pd.DataFrame,
        labels_available: bool,
        context: PipelineContext,
    ) -> dict[str, pd.DataFrame]:
        split_config = context.config.split
        working = self._prepare_time_ordered_frame(dataframe, split_config, require_all_dates=True)
        date_column = str(split_config.date_column)
        date_values = working[date_column]
        split_masks = {
            split_name: self._window_mask(date_values, split_config, split_name)
            for split_name in ("train", "validation", "test")
            if split_config._has_date_window(split_name)
        }
        if "train" not in split_masks:
            raise ValueError("explicit_date_windows requires a train date window.")
        if not (set(split_masks) & {"validation", "test"}):
            raise ValueError(
                "explicit_date_windows requires at least one validation or test date window."
            )
        assignment_counts = sum(mask.astype(int) for mask in split_masks.values())
        overlap_count = int((assignment_counts > 1).sum())
        if overlap_count:
            raise ValueError(
                "Explicit date windows overlap; "
                f"{overlap_count:,} rows match more than one split."
            )
        unassigned_count = int((assignment_counts == 0).sum())
        if unassigned_count:
            raise ValueError(
                "Explicit date windows do not cover all rows; "
                f"{unassigned_count:,} rows are outside configured windows."
            )
        split_frames = {
            split_name: working.loc[mask]
            for split_name, mask in split_masks.items()
        }
        split_frames = self._reset_split_indexes(split_frames)
        self._validate_split_frames(
            split_frames,
            str(context.target_column),
            labels_available,
            context,
        )
        return split_frames

    def _split_custom_column(
        self,
        dataframe: pd.DataFrame,
        labels_available: bool,
        context: PipelineContext,
    ) -> dict[str, pd.DataFrame]:
        split_config = context.config.split
        split_column = split_config.custom_split_column
        if split_column is None or split_column not in dataframe.columns:
            raise ValueError(
                f"Custom split column '{split_column}' is missing from the dataframe."
            )
        labels = dataframe[split_column].map(self._normalize_custom_split_label)
        missing_labels = labels.isna()
        if missing_labels.any():
            examples = (
                dataframe.loc[missing_labels, split_column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            raise ValueError(
                "Custom split column must contain train, validation/val, or test labels. "
                f"Unrecognized examples: {', '.join(examples) or 'missing values'}."
            )
        split_frames = {
            split_name: dataframe.loc[labels == split_name]
            for split_name in ("train", "validation", "test")
            if bool((labels == split_name).any())
        }
        split_frames = self._reset_split_indexes(split_frames)
        self._validate_split_frames(
            split_frames,
            str(context.target_column),
            labels_available,
            context,
        )
        return split_frames

    def _prepare_time_ordered_frame(
        self,
        dataframe: pd.DataFrame,
        split_config: SplitConfig,
        *,
        require_all_dates: bool = False,
    ) -> pd.DataFrame:
        date_column = split_config.date_column
        if date_column is None:
            raise ValueError("Time-aware splits require SplitConfig.date_column.")
        if date_column not in dataframe.columns:
            raise ValueError(f"Date column '{date_column}' is missing from the dataframe.")

        working = dataframe.copy(deep=False)
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        if working[date_column].isna().all():
            raise ValueError(
                f"Date column '{date_column}' could not be parsed into valid timestamps."
            )
        invalid_dates = int(working[date_column].isna().sum())
        if require_all_dates and invalid_dates:
            raise ValueError(
                f"Date column '{date_column}' has {invalid_dates:,} unparseable values. "
                "Date cutoff and explicit-window splits require every row to have a valid date."
            )

        if split_config.data_structure == DataStructure.PANEL and split_config.entity_column:
            return working.sort_values([date_column, split_config.entity_column])
        return working.sort_values(date_column)

    def _window_mask(
        self,
        date_values: pd.Series,
        split_config: SplitConfig,
        split_name: str,
    ) -> pd.Series:
        start = self._parse_optional_date(
            getattr(split_config, f"{split_name}_start_date"),
            field_name=f"{split_name}_start_date",
        )
        end = self._parse_optional_date(
            getattr(split_config, f"{split_name}_end_date"),
            field_name=f"{split_name}_end_date",
        )
        if start is None and end is None:
            return pd.Series(False, index=date_values.index)
        if start is not None and end is not None and start > end:
            raise ValueError(f"{split_name}_start_date must be on or before {split_name}_end_date.")
        mask = pd.Series(True, index=date_values.index)
        if start is not None:
            mask &= date_values >= start
        if end is not None:
            mask &= date_values <= end
        return mask

    def _parse_optional_date(self, value: str | None, *, field_name: str) -> pd.Timestamp | None:
        if value is None or not str(value).strip():
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{field_name} could not be parsed as a date: {value!r}.")
        return pd.Timestamp(parsed)

    def _normalize_custom_split_label(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"train", "training", "development", "dev"}:
            return "train"
        if normalized in {"validation", "valid", "val"}:
            return "validation"
        if normalized in {"test", "testing", "holdout", "out_of_time", "oot"}:
            return "test"
        return None

    def _reset_split_indexes(
        self,
        split_frames: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        return {
            split_name: split_frame.reset_index(drop=True)
            for split_name, split_frame in split_frames.items()
        }

    def _validate_split_frames(
        self,
        split_frames: dict[str, pd.DataFrame],
        target_column: str,
        labels_available: bool,
        context: PipelineContext,
    ) -> None:
        if "train" not in split_frames:
            raise ValueError("The split strategy must create a train split.")
        for split_name, split_frame in split_frames.items():
            if split_frame.empty:
                raise ValueError(
                    f"The {split_name} split is empty. Adjust the split settings "
                    "or add more rows."
                )
            if (
                labels_available
                and target_column in split_frame.columns
                and split_frame[target_column].nunique() < 2
                and split_name != "test"
            ):
                context.warn(
                    f"The {split_name} split only contains one target class. "
                    "This can happen with strongly time-ordered defaults."
                )

    def _date_range_summary(
        self,
        frame: pd.DataFrame,
        date_column: str | None,
    ) -> dict[str, str]:
        if not date_column or date_column not in frame.columns or frame.empty:
            return {}
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if dates.empty:
            return {}
        return {
            "min_date": dates.min().date().isoformat(),
            "max_date": dates.max().date().isoformat(),
        }
