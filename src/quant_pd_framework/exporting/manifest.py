"""Artifact manifest and index helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT_LAYOUT_VERSION = "2.0"


def write_json(path: Path, payload: Any) -> None:
    """Writes JSON with the export step's existing numpy-scalar handling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    """Converts common non-JSON scalar values used in artifact metadata."""

    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    output_root: Path,
    monitoring_bundle_directory_name: str,
) -> None:
    """Writes the artifact manifest with a generated artifact index."""

    manifest["artifact_layout_version"] = ARTIFACT_LAYOUT_VERSION
    manifest.pop("artifact_index", None)
    manifest["artifact_index"] = build_artifact_index(
        manifest,
        output_root,
        monitoring_bundle_directory_name=monitoring_bundle_directory_name,
    )
    write_json(path, manifest)


def build_artifact_index(
    manifest: dict[str, Any],
    output_root: Path,
    *,
    monitoring_bundle_directory_name: str,
) -> list[dict[str, str]]:
    """Builds a reviewer-friendly index of paths recorded in a manifest."""

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for key_path, path in iter_manifest_file_paths(manifest):
        try:
            resolved = path.resolve()
            relative_path = resolved.relative_to(output_root.resolve()).as_posix()
        except ValueError:
            relative_path = str(path)
        if relative_path in seen:
            continue
        seen.add(relative_path)
        category = artifact_category(
            relative_path,
            monitoring_bundle_directory_name=monitoring_bundle_directory_name,
        )
        rows.append(
            {
                "key": key_path,
                "category": category,
                "purpose": artifact_purpose(
                    key_path,
                    relative_path,
                    monitoring_bundle_directory_name=monitoring_bundle_directory_name,
                ),
                "relative_path": relative_path,
                "send_to": artifact_audience(category, relative_path),
            }
        )
    return sorted(rows, key=lambda row: row["relative_path"])


def iter_manifest_file_paths(payload: Any, *, prefix: str = "") -> list[tuple[str, Path]]:
    """Recursively returns path-like values from a manifest payload."""

    paths: list[tuple[str, Path]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "artifact_index":
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(iter_manifest_file_paths(value, prefix=next_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            paths.extend(iter_manifest_file_paths(value, prefix=f"{prefix}[{index}]"))
    elif isinstance(payload, str) and (":" in payload or "\\" in payload or "/" in payload):
        path = Path(payload)
        if path.suffix or path.exists():
            paths.append((prefix, path))
    return paths


def artifact_category(
    relative_path: str,
    *,
    monitoring_bundle_directory_name: str,
) -> str:
    """Classifies a relative artifact path into a manifest category."""

    top_level = relative_path.split("/", 1)[0]
    return {
        "reports": "reports",
        "model": "model",
        "data": "data",
        "tables": "tables",
        "config": "configuration",
        "metadata": "metadata",
        "workbooks": "workbooks",
        "code": "rerun_code",
        "figures": "figures",
        monitoring_bundle_directory_name: "monitoring_bundle",
        "artifact_manifest.json": "manifest",
        "START_HERE.md": "orientation",
    }.get(top_level, "other")


def artifact_audience(category: str, relative_path: str) -> str:
    """Returns the likely reviewer audience for an artifact category."""

    if category == "reports":
        return "model builders, validators, business reviewers"
    if category == "model":
        return "model builders, scoring workflows"
    if category == "data":
        return "model builders, validators, downstream scoring users"
    if category == "tables":
        return "validators, auditors, technical reviewers"
    if category in {"configuration", "metadata", "manifest"}:
        return "auditors, technical reviewers"
    if category == "rerun_code":
        return "developers"
    if category == "monitoring_bundle":
        return "monitoring application"
    if relative_path == "START_HERE.md":
        return "all reviewers"
    return "technical reviewers"


def artifact_purpose(
    key_path: str,
    relative_path: str,
    *,
    monitoring_bundle_directory_name: str,
) -> str:
    """Returns a reviewer-friendly purpose statement for a manifest path."""

    file_name = Path(relative_path).name
    purpose_by_name = {
        "START_HERE.md": "Plain-English orientation guide for the run folder.",
        "artifact_manifest.json": "Machine-readable index of exported artifacts.",
        "interactive_report.html": "Standalone visual diagnostic report.",
        "decision_summary.md": "Decision-ready scorecard for model review.",
        "run_report.md": "Markdown summary of run metrics, warnings, and diagnostics.",
        "model_documentation_pack.md": "Development-facing model documentation summary.",
        "model_development_dossier.md": (
            "Audit-ready narrative dossier connecting purpose, data, model, "
            "validation evidence, lineage, and open review items."
        ),
        "validation_pack.md": "Validator-facing evidence index and review summary.",
        "quant_model.joblib": "Serialized fitted model object.",
        "feature_importance.csv": "Feature-level coefficients or importance values.",
        "feature_lineage_map.csv": (
            "Feature-level lineage, transformation, imputation, and documentation map."
        ),
        "model_summary.txt": "Text model summary from the fitted estimator.",
        "run_config.json": "Resolved configuration used for the run.",
        "configuration_template.xlsx": "Offline review workbook for configuration edits.",
        "metrics.json": "Structured metrics by split.",
        "statistical_tests.json": "Structured statistical-test payloads.",
        "step_manifest.json": "Ordered pipeline step stack.",
        "run_debug_trace.json": "Per-step debug trace and timing metadata.",
        "audit_events.jsonl": "Run-scoped audit event log for major GUI and workflow actions.",
        "reproducibility_manifest.json": "Hashes, package versions, and environment metadata.",
        "analysis_workbook.xlsx": (
            "Excel workbook containing metrics, predictions, and diagnostics."
        ),
        "generated_run.py": "Python entry point for rerunning the workflow without the GUI.",
        "HOW_TO_RERUN.md": "Plain-English rerun instructions.",
    }
    if file_name in purpose_by_name:
        return purpose_by_name[file_name]
    if "predictions" in file_name:
        return "Row-level model scoring output."
    if relative_path.startswith("tables/"):
        return "Diagnostic table exported from the model-development workflow."
    if relative_path.startswith("figures/"):
        return "Individual chart export."
    if relative_path.startswith(monitoring_bundle_directory_name):
        return "Copied artifact for the separate monitoring application."
    return f"Exported artifact recorded from `{key_path}`."


def path_string(path: Path | None) -> str | None:
    """Returns a stable string representation for optional artifact paths."""

    return str(path) if path is not None else None
