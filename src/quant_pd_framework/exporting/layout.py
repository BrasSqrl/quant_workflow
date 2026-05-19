"""Artifact directory layout helpers."""

from __future__ import annotations

from pathlib import Path

from quant_pd_framework.export_layout import ExportPathLayout
from quant_pd_framework.presentation import infer_asset_section

TABLE_SECTION_DIRECTORIES = {
    "model_performance": "model_performance",
    "calibration_thresholds": "calibration",
    "stability_drift": "stability",
    "sample_segmentation": "segmentation",
    "segmented_model_build": "segmented_model",
    "feature_effects": "explainability",
    "statistical_tests": "statistical_tests",
    "feature_subset_search": "feature_subset_search",
    "scorecard_workbench": "scorecard",
    "credit_risk_development": "credit_risk",
    "data_quality": "diagnostics",
    "backtesting_time": "backtesting",
    "governance_export": "governance",
}


def ensure_layout_directories(paths: ExportPathLayout) -> None:
    """Creates the standard artifact directory tree."""

    directories = [
        paths.reports_dir,
        paths.model_dir,
        paths.data_input_dir,
        paths.data_predictions_dir,
        paths.tables_dir,
        paths.config_dir,
        paths.metadata_dir,
        paths.workbooks_dir,
        paths.code_dir,
    ]
    directories.extend(paths.tables_dir / name for name in TABLE_SECTION_DIRECTORIES.values())
    for directory in dict.fromkeys(directories):
        directory.mkdir(parents=True, exist_ok=True)


def table_export_path(tables_dir: Path, table_name: str, *, sanitized_name: str) -> Path:
    """Returns the grouped CSV export path for a diagnostic table."""

    section = infer_asset_section(table_name, kind="table")
    directory_name = TABLE_SECTION_DIRECTORIES.get(section, "other")
    return tables_dir / directory_name / f"{sanitized_name}.csv"


def table_group_directories(tables_dir: Path) -> dict[str, str | None]:
    """Returns the existing grouped table directories for the artifact manifest."""

    return {
        group_name: str(tables_dir / group_name) if (tables_dir / group_name).exists() else None
        for group_name in sorted(set(TABLE_SECTION_DIRECTORIES.values()))
    }
