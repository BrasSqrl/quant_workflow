"""Focused helpers used by the diagnostics step."""

from .assets import (
    apply_visual_theme_to_context,
    bucket_numeric_series,
    compute_population_stability_index,
    sample_frame_for_plotting,
    sample_rows_for_diagnostics,
    sanitize_asset_name,
)
from .registry import DIAGNOSTIC_REGISTRY, build_diagnostic_registry_table
from .scoring import predict_modified_frames

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "apply_visual_theme_to_context",
    "bucket_numeric_series",
    "build_diagnostic_registry_table",
    "compute_population_stability_index",
    "predict_modified_frames",
    "sample_frame_for_plotting",
    "sample_rows_for_diagnostics",
    "sanitize_asset_name",
]
