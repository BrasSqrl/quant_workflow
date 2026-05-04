"""On-demand ongoing-monitoring package builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

PACKAGE_ROOT = "model_bundle_for_monitoring"
PACKAGE_VERSION = "1.0"


def build_monitoring_package_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Builds a JSON-safe package contract from a completed run snapshot."""

    artifacts = {
        key: str(value)
        for key, value in (snapshot.get("artifacts") or {}).items()
        if value
    }
    target_mode = str(snapshot.get("target_mode", ""))
    score_column = str(snapshot.get("score_column") or _default_score_column(target_mode))
    prediction_column = str(
        snapshot.get("prediction_column") or _default_prediction_column(target_mode)
    )
    return {
        "created_by_run_id": str(snapshot.get("run_id", "")),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_type": str(snapshot.get("model_type", "")),
        "execution_mode": str(snapshot.get("execution_mode", "")),
        "target_mode": target_mode,
        "target_column": str(snapshot.get("target_column", "")),
        "score_column": score_column,
        "prediction_column": prediction_column,
        "threshold": snapshot.get("threshold"),
        "selected_features": [str(feature) for feature in snapshot.get("feature_columns", [])],
        "date_column": str(snapshot.get("date_column") or ""),
        "entity_column": str(snapshot.get("entity_column") or ""),
        "segment_columns": [
            str(column)
            for column in [snapshot.get("default_segment_column")]
            if column
        ],
        "artifacts": artifacts,
    }


def build_monitoring_package_from_payload(payload: dict[str, Any]) -> bytes:
    """Builds the ongoing-monitoring zip package from exported run artifacts."""

    output = BytesIO()
    artifacts = payload.get("artifacts") or {}
    bundled_paths: dict[str, str | None] = {}
    missing_optional_artifacts: list[str] = []
    missing_required_artifacts: list[str] = []

    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        _add_required_file(
            archive,
            artifacts.get("model"),
            "quant_model.joblib",
            bundled_paths,
            missing_required_artifacts,
        )
        _add_required_file(
            archive,
            artifacts.get("config"),
            "run_config.json",
            bundled_paths,
            missing_required_artifacts,
        )
        _add_required_file(
            archive,
            artifacts.get("runner_script"),
            "generated_run.py",
            bundled_paths,
            missing_required_artifacts,
        )
        _add_required_file(
            archive,
            artifacts.get("manifest") or artifacts.get("artifact_manifest"),
            "artifact_manifest.json",
            bundled_paths,
            missing_required_artifacts,
        )
        _add_optional_tabular_as_csv(
            archive,
            artifacts.get("predictions"),
            "predictions.csv",
            bundled_paths,
            missing_optional_artifacts,
        )
        _add_optional_tabular_as_csv(
            archive,
            artifacts.get("input_snapshot"),
            "input_snapshot.csv",
            bundled_paths,
            missing_optional_artifacts,
        )
        _add_optional_directory(
            archive,
            artifacts.get("code_snapshot_dir"),
            "code_snapshot",
            bundled_paths,
            missing_optional_artifacts,
        )

        bundled_paths["monitoring_metadata.json"] = "monitoring_metadata.json"
        metadata = _build_monitoring_metadata(
            payload=payload,
            bundled_paths=bundled_paths,
            missing_optional_artifacts=missing_optional_artifacts,
            missing_required_artifacts=missing_required_artifacts,
        )
        metadata_text = json.dumps(metadata, indent=2, default=str)
        archive.writestr(f"{PACKAGE_ROOT}/monitoring_metadata.json", metadata_text)

    return output.getvalue()


def _add_required_file(
    archive: ZipFile,
    source_value: Any,
    bundle_name: str,
    bundled_paths: dict[str, str | None],
    missing_required_artifacts: list[str],
) -> None:
    source_path = _path(source_value)
    if source_path is None or not source_path.exists() or not source_path.is_file():
        bundled_paths[bundle_name] = None
        missing_required_artifacts.append(bundle_name)
        return
    archive.write(source_path, f"{PACKAGE_ROOT}/{bundle_name}")
    bundled_paths[bundle_name] = bundle_name


def _add_optional_tabular_as_csv(
    archive: ZipFile,
    source_value: Any,
    bundle_name: str,
    bundled_paths: dict[str, str | None],
    missing_optional_artifacts: list[str],
) -> None:
    source_path = _path(source_value)
    if source_path is None or not source_path.exists() or not source_path.is_file():
        bundled_paths[bundle_name] = None
        missing_optional_artifacts.append(bundle_name)
        return
    if source_path.suffix.lower() == ".csv":
        archive.write(source_path, f"{PACKAGE_ROOT}/{bundle_name}")
    elif source_path.suffix.lower() in {".parquet", ".pq"}:
        dataframe = pd.read_parquet(source_path)
        archive.writestr(f"{PACKAGE_ROOT}/{bundle_name}", dataframe.to_csv(index=False))
    else:
        archive.write(source_path, f"{PACKAGE_ROOT}/{bundle_name}")
    bundled_paths[bundle_name] = bundle_name


def _add_optional_directory(
    archive: ZipFile,
    source_value: Any,
    bundle_name: str,
    bundled_paths: dict[str, str | None],
    missing_optional_artifacts: list[str],
) -> None:
    source_path = _path(source_value)
    if source_path is None or not source_path.exists() or not source_path.is_dir():
        bundled_paths[bundle_name] = None
        missing_optional_artifacts.append(bundle_name)
        return
    for file_path in sorted(source_path.rglob("*")):
        if file_path.is_file():
            relative_path = file_path.relative_to(source_path).as_posix()
            archive.write(file_path, f"{PACKAGE_ROOT}/{bundle_name}/{relative_path}")
    bundled_paths[bundle_name] = bundle_name


def _build_monitoring_metadata(
    *,
    payload: dict[str, Any],
    bundled_paths: dict[str, str | None],
    missing_optional_artifacts: list[str],
    missing_required_artifacts: list[str],
) -> dict[str, Any]:
    return {
        "bundle_type": "quant_studio_model_bundle_for_monitoring",
        "bundle_version": PACKAGE_VERSION,
        "created_by_run_id": payload.get("created_by_run_id", ""),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_type": payload.get("model_type", ""),
        "target_mode": payload.get("target_mode", ""),
        "target_column": payload.get("target_column", ""),
        "score_column": payload.get("score_column", ""),
        "prediction_column": payload.get("prediction_column", ""),
        "threshold": payload.get("threshold"),
        "selected_features": list(payload.get("selected_features", [])),
        "date_column": payload.get("date_column", ""),
        "entity_column": payload.get("entity_column", ""),
        "segment_columns": list(payload.get("segment_columns", [])),
        "bundled_artifacts": bundled_paths,
        "missing_optional_artifacts": missing_optional_artifacts,
        "missing_required_artifacts": missing_required_artifacts,
        "creation_policy": "on_demand_step_5_download",
    }


def _default_score_column(target_mode: str) -> str:
    return "predicted_probability" if target_mode == "binary" else "predicted_value"


def _default_prediction_column(target_mode: str) -> str:
    return "predicted_class" if target_mode == "binary" else "predicted_value"


def _path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return Path(text)
