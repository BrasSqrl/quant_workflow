"""Configuration profile persistence for the Streamlit workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_pd_framework.config import FrameworkConfig
from quant_pd_framework.config_io import load_framework_config
from quant_pd_framework.gui_support import GUIBuildInputs

PROFILE_SCHEMA_VERSION = "1.0"
SAVED_PROFILE_DIR = Path("configs") / "saved_profiles"
WORKSPACE_TABLE_NAMES = (
    "schema",
    "feature_dictionary",
    "transformations",
    "feature_review",
    "scorecard_overrides",
)


def build_configuration_profile(
    *,
    profile_name: str,
    notes: str,
    tags: str = "",
    model_purpose: str = "",
    dataframe: pd.DataFrame,
    data_source_label: str,
    source_metadata: dict[str, Any] | None,
    framework_config: FrameworkConfig,
    schema_frame: pd.DataFrame,
    feature_dictionary_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    feature_review_frame: pd.DataFrame,
    scorecard_override_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Builds a versioned profile payload without storing raw source data."""

    resolved_name = profile_name.strip() or "Quant Studio Configuration"
    created_at_utc = _utc_timestamp()
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "metadata": {
            "profile_name": resolved_name,
            "notes": notes.strip(),
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "model_purpose": model_purpose.strip(),
            "target_mode": framework_config.target.mode.value,
            "model_type": framework_config.model.model_type.value,
            "execution_mode": framework_config.execution.mode.value,
            "created_at_utc": created_at_utc,
            "application": "Quant Studio",
        },
        "dataset_fingerprint": build_dataset_fingerprint(
            dataframe=dataframe,
            data_source_label=data_source_label,
            source_metadata=source_metadata,
        ),
        "framework_config": framework_config.to_dict(),
        "workspace_tables": {
            "schema": dataframe_to_profile_table(schema_frame),
            "feature_dictionary": dataframe_to_profile_table(feature_dictionary_frame),
            "transformations": dataframe_to_profile_table(transformation_frame),
            "feature_review": dataframe_to_profile_table(feature_review_frame),
            "scorecard_overrides": dataframe_to_profile_table(scorecard_override_frame),
        },
    }


def build_dataset_fingerprint(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
    source_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Records reusable dataset shape metadata without copying data rows."""

    columns = [str(column) for column in dataframe.columns]
    dtypes = {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}
    schema_payload = {
        "columns": columns,
        "dtypes": dtypes,
    }
    schema_hash = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "data_source_label": data_source_label,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": columns,
        "dtypes": dtypes,
        "schema_hash": schema_hash,
        "source_metadata": _json_safe(source_metadata or {}),
    }


def dataframe_to_profile_table(frame: pd.DataFrame) -> dict[str, Any]:
    """Serializes an editor dataframe to JSON-safe records."""

    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        records.append({str(key): _json_safe(value) for key, value in row.items()})
    return {
        "columns": [str(column) for column in frame.columns],
        "records": records,
    }


def profile_table_to_frame(profile_payload: dict[str, Any], table_name: str) -> pd.DataFrame | None:
    """Returns a workspace editor table from a profile payload when available."""

    table_payload = profile_payload.get("workspace_tables", {}).get(table_name)
    if not isinstance(table_payload, dict):
        return None
    columns = [str(column) for column in table_payload.get("columns", [])]
    records = table_payload.get("records", [])
    frame = pd.DataFrame(records)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.loc[:, columns]
    return frame


def save_configuration_profile(
    profile_payload: dict[str, Any],
    *,
    directory: Path = SAVED_PROFILE_DIR,
) -> Path:
    """Persists a profile payload to the local saved-profile directory."""

    validate_configuration_profile(profile_payload)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / profile_file_name(profile_payload)
    output_path.write_bytes(profile_to_download_bytes(profile_payload))
    return output_path


def list_configuration_profiles(directory: Path = SAVED_PROFILE_DIR) -> list[Path]:
    """Lists locally saved profile JSON files newest first."""

    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def build_profile_library_frame(directory: Path = SAVED_PROFILE_DIR) -> pd.DataFrame:
    """Summarizes local profiles for searchable display in the GUI."""

    rows: list[dict[str, Any]] = []
    for path in list_configuration_profiles(directory):
        try:
            profile_payload = load_configuration_profile(path)
        except Exception as exc:
            rows.append(
                {
                    "profile_name": path.stem,
                    "created_at_utc": "",
                    "target_mode": "",
                    "model_type": "",
                    "execution_mode": "",
                    "tags": "",
                    "model_purpose": "",
                    "path": str(path),
                    "status": f"Invalid: {exc}",
                }
            )
            continue
        metadata = profile_payload.get("metadata", {})
        tags_value = metadata.get("tags", [])
        tags_text = tags_value if isinstance(tags_value, str) else ", ".join(tags_value)
        rows.append(
            {
                "profile_name": str(metadata.get("profile_name", path.stem)),
                "created_at_utc": str(metadata.get("created_at_utc", "")),
                "target_mode": str(metadata.get("target_mode", "")),
                "model_type": str(metadata.get("model_type", "")),
                "execution_mode": str(metadata.get("execution_mode", "")),
                "tags": tags_text,
                "model_purpose": str(metadata.get("model_purpose", "")),
                "path": str(path),
                "status": "Ready",
            }
        )
    return pd.DataFrame(rows)


def duplicate_configuration_profile(
    source_path: str | Path,
    *,
    new_profile_name: str | None = None,
) -> Path:
    """Copies an existing profile with updated metadata."""

    profile_payload = load_configuration_profile(source_path)
    metadata = profile_payload.setdefault("metadata", {})
    original_name = str(metadata.get("profile_name") or "Quant Studio Configuration")
    metadata["profile_name"] = new_profile_name or f"{original_name} Copy"
    metadata["created_at_utc"] = _utc_timestamp()
    return save_configuration_profile(profile_payload, directory=Path(source_path).parent)


def delete_configuration_profile(source_path: str | Path) -> None:
    """Deletes a local saved profile file."""

    Path(source_path).unlink(missing_ok=True)


def load_configuration_profile(source: str | Path | bytes | bytearray) -> dict[str, Any]:
    """Loads and validates a profile from a path or uploaded JSON bytes."""

    if isinstance(source, bytes | bytearray):
        payload = json.loads(bytes(source).decode("utf-8"))
    else:
        source_path = Path(source)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    validate_configuration_profile(payload)
    return payload


def validate_configuration_profile(profile_payload: dict[str, Any]) -> None:
    """Validates the profile wrapper and embedded framework config."""

    if not isinstance(profile_payload, dict):
        raise ValueError("Configuration profile must be a JSON object.")
    if profile_payload.get("profile_schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported configuration profile version: "
            f"{profile_payload.get('profile_schema_version')!r}."
        )
    if "framework_config" not in profile_payload:
        raise ValueError("Configuration profile is missing framework_config.")
    if "workspace_tables" not in profile_payload:
        raise ValueError("Configuration profile is missing workspace_tables.")
    load_framework_config(profile_payload["framework_config"])


def framework_config_from_profile(profile_payload: dict[str, Any]) -> FrameworkConfig:
    """Returns the typed framework config embedded in a profile."""

    validate_configuration_profile(profile_payload)
    return load_framework_config(profile_payload["framework_config"])


def gui_inputs_from_framework_config(config: FrameworkConfig) -> GUIBuildInputs:
    """Maps a resolved config back into GUI default inputs."""

    positive_values_text = ""
    if config.target.positive_values:
        positive_values_text = ",".join(str(value) for value in config.target.positive_values)
    return GUIBuildInputs(
        preset_name=config.preset_name,
        model=copy.deepcopy(config.model),
        cleaning=copy.deepcopy(config.cleaning),
        feature_engineering=copy.deepcopy(config.feature_engineering),
        comparison=copy.deepcopy(config.comparison),
        subset_search=copy.deepcopy(config.subset_search),
        feature_policy=copy.deepcopy(config.feature_policy),
        feature_dictionary=copy.deepcopy(config.feature_dictionary),
        advanced_imputation=copy.deepcopy(config.advanced_imputation),
        transformations=copy.deepcopy(config.transformations),
        manual_review=copy.deepcopy(config.manual_review),
        suitability_checks=copy.deepcopy(config.suitability_checks),
        workflow_guardrails=copy.deepcopy(config.workflow_guardrails),
        explainability=copy.deepcopy(config.explainability),
        calibration=copy.deepcopy(config.calibration),
        scorecard=copy.deepcopy(config.scorecard),
        scorecard_workbench=copy.deepcopy(config.scorecard_workbench),
        imputation_sensitivity=copy.deepcopy(config.imputation_sensitivity),
        variable_selection=copy.deepcopy(config.variable_selection),
        documentation=copy.deepcopy(config.documentation),
        regulatory_reporting=copy.deepcopy(config.regulatory_reporting),
        scenario_testing=copy.deepcopy(config.scenario_testing),
        diagnostics=copy.deepcopy(config.diagnostics),
        credit_risk=copy.deepcopy(config.credit_risk),
        robustness=copy.deepcopy(config.robustness),
        cross_validation=copy.deepcopy(config.cross_validation),
        reproducibility=copy.deepcopy(config.reproducibility),
        performance=copy.deepcopy(config.performance),
        data_structure=config.split.data_structure,
        train_size=config.split.train_size,
        validation_size=config.split.validation_size,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=config.split.stratify,
        execution_mode=config.execution.mode,
        existing_model_path=config.execution.existing_model_path,
        existing_config_path=config.execution.existing_config_path,
        target_mode=config.target.mode,
        target_output_column=config.target.output_column,
        positive_values_text=positive_values_text,
        drop_target_source_column=config.target.drop_source_column,
        pass_through_unconfigured_columns=config.schema.pass_through_unconfigured_columns,
        output_root=config.artifacts.output_root,
        artifacts=copy.deepcopy(config.artifacts),
    )


def compare_profile_to_dataset(
    profile_payload: dict[str, Any],
    dataframe: pd.DataFrame,
) -> list[str]:
    """Returns non-blocking warnings when a profile is loaded onto different data."""

    fingerprint = profile_payload.get("dataset_fingerprint", {})
    profile_columns = [str(column) for column in fingerprint.get("columns", [])]
    current_columns = [str(column) for column in dataframe.columns]
    warnings: list[str] = []
    missing_columns = [column for column in profile_columns if column not in current_columns]
    new_columns = [column for column in current_columns if column not in profile_columns]
    if missing_columns:
        warnings.append(
            "Current dataset is missing profile columns: " + ", ".join(missing_columns[:10])
        )
    if new_columns:
        warnings.append(
            "Current dataset has columns not present in the profile: "
            + ", ".join(new_columns[:10])
        )
    profile_row_count = fingerprint.get("row_count")
    if isinstance(profile_row_count, int) and profile_row_count != int(dataframe.shape[0]):
        warnings.append(
            "Profile was saved from "
            f"{profile_row_count:,} rows; current dataset has {int(dataframe.shape[0]):,} rows."
        )
    return warnings


def profile_to_download_bytes(profile_payload: dict[str, Any]) -> bytes:
    """Returns stable pretty-printed JSON bytes for save/download."""

    return json.dumps(
        profile_payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def profile_file_name(profile_payload: dict[str, Any]) -> str:
    """Builds a safe deterministic-ish filename for a profile payload."""

    metadata = profile_payload.get("metadata", {})
    profile_name = str(metadata.get("profile_name") or "quant_studio_configuration")
    created_at = str(metadata.get("created_at_utc") or _utc_timestamp())
    timestamp = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace("T", "_")
        .replace("+0000", "")
        .replace("Z", "")
    )
    return f"{_slugify(profile_name)}_{timestamp}.json"


def profile_display_name(profile_payload: dict[str, Any]) -> str:
    """Returns a compact label for the active profile."""

    metadata = profile_payload.get("metadata", {})
    profile_name = str(metadata.get("profile_name") or "Unnamed profile")
    created_at = str(metadata.get("created_at_utc") or "")
    return f"{profile_name} ({created_at})" if created_at else profile_name


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_").lower()
    return slug or "quant_studio_configuration"
