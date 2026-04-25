"""Shared JSON-friendly serialization helpers for framework config objects."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

FRAMEWORK_CONFIG_SECTION_NAMES: tuple[str, ...] = (
    "schema",
    "feature_engineering",
    "target",
    "split",
    "execution",
    "model",
    "comparison",
    "subset_search",
    "feature_policy",
    "feature_dictionary",
    "advanced_imputation",
    "transformations",
    "manual_review",
    "suitability_checks",
    "workflow_guardrails",
    "explainability",
    "calibration",
    "scorecard",
    "scorecard_workbench",
    "imputation_sensitivity",
    "variable_selection",
    "documentation",
    "regulatory_reporting",
    "scenario_testing",
    "diagnostics",
    "distribution_diagnostics",
    "residual_diagnostics",
    "outlier_diagnostics",
    "dependency_diagnostics",
    "time_series_diagnostics",
    "structural_breaks",
    "feature_workbench",
    "preset_recommendations",
    "credit_risk",
    "robustness",
    "cross_validation",
    "reproducibility",
    "performance",
    "artifacts",
)


def serialize_config_value(value: Any) -> Any:
    """Converts config values into JSON-friendly primitives."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_config_value(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): serialize_config_value(item)
            for key, item in value.items()
        }
    if is_dataclass(value):
        return {
            key: serialize_config_value(item)
            for key, item in asdict(value).items()
        }
    return value


def framework_config_to_dict(config: Any) -> dict[str, Any]:
    """Serializes a `FrameworkConfig` while validating expected section names."""

    payload = serialize_config_value(config)
    if not isinstance(payload, dict):
        raise TypeError("FrameworkConfig serialization did not produce a dictionary.")
    missing = [section for section in FRAMEWORK_CONFIG_SECTION_NAMES if section not in payload]
    if missing:
        raise ValueError(
            "FrameworkConfig serialization is missing expected sections: "
            + ", ".join(missing)
        )
    return payload
