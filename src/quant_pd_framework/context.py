"""Shared state container passed between pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FrameworkConfig


@dataclass
class PipelineContext:
    """Holds the evolving state of a full model run."""

    config: FrameworkConfig
    run_id: str
    raw_input: pd.DataFrame | str | Path
    raw_data: pd.DataFrame | None = None
    working_data: pd.DataFrame | None = None
    target_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    split_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    model: Any = None
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None
    backtest_summary: pd.DataFrame | None = None
    model_summary: str | pd.DataFrame | None = None
    model_artifacts: dict[str, Any] = field(default_factory=dict)
    comparison_results: pd.DataFrame | None = None
    scenario_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    visualizations: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Adds a short breadcrumb that can be written into the final report."""

        self.events.append(message)

    def warn(self, message: str) -> None:
        """Stores non-fatal issues that the user should still review."""

        self.warnings.append(message)
