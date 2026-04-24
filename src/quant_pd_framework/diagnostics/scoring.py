"""Batch scoring helpers for model explainability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def predict_modified_frames(
    *,
    model: Any,
    base_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    modifications: Sequence[Mapping[str, Any]],
) -> list[np.ndarray]:
    """Scores many modified copies of a feature frame in one model-adapter call."""

    if not modifications:
        return []
    lengths: list[int] = []
    frames: list[pd.DataFrame] = []
    selected_columns = list(feature_columns)
    for changes in modifications:
        modified = base_frame.copy(deep=False)
        for feature_name, value in changes.items():
            modified[feature_name] = value
        frames.append(modified.loc[:, selected_columns])
        lengths.append(len(modified))

    if not frames or sum(lengths) == 0:
        return [np.asarray([], dtype=float) for _ in modifications]

    combined = pd.concat(frames, ignore_index=True)
    scores = np.asarray(model.predict_score(combined), dtype=float).reshape(-1)
    scored_frames: list[np.ndarray] = []
    start_index = 0
    for frame_length in lengths:
        stop_index = start_index + frame_length
        scored_frames.append(scores[start_index:stop_index])
        start_index = stop_index
    return scored_frames
