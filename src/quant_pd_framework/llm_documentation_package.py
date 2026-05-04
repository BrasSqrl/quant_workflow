"""LLM-ready model documentation package builder."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from quant_pd_framework.decision_summary import build_decision_summary

PACKAGE_ROOT = "llm_documentation_package"
PACKAGE_VERSION = "1.0"
DEFAULT_MAX_INCLUDED_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TABLE_PREVIEW_ROWS = 250
DEFAULT_MAX_TABLE_PREVIEW_COLUMNS = 80
DEFAULT_MAX_CHART_FILES = 60

ROW_LEVEL_ARTIFACT_KEYS = {
    "input_snapshot",
    "input_snapshot_csv",
    "input_snapshot_parquet",
    "predictions",
    "predictions_csv",
    "predictions_parquet",
    "full_data_predictions",
    "large_data_training_sample",
}
MODEL_BINARY_ARTIFACT_KEYS = {"model"}
HEAVY_DIRECTORY_ARTIFACT_KEYS = {
    "code_snapshot_dir",
    "monitoring_bundle_dir",
    "sample_development_dir",
    "full_data_scoring_dir",
    "large_data_metadata_dir",
}
HEAVY_REPORT_NAMES = {"interactive_report.html"}
LLM_TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".html"}
LLM_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}


def build_llm_documentation_package(
    snapshot: Mapping[str, Any],
    *,
    max_included_file_bytes: int = DEFAULT_MAX_INCLUDED_FILE_BYTES,
    max_chart_files: int = DEFAULT_MAX_CHART_FILES,
) -> bytes:
    """Builds a zipped evidence package for downstream LLM document drafting."""

    payload = build_llm_documentation_context_payload(snapshot)
    return build_llm_documentation_package_from_payload(
        payload,
        max_included_file_bytes=max_included_file_bytes,
        max_chart_files=max_chart_files,
    )


def build_llm_documentation_context_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Returns a JSON-safe context payload used by the package builder and UI cache."""

    diagnostics_tables = _mapping(snapshot.get("diagnostics_tables"))
    summary = build_decision_summary(snapshot)
    table_previews = {
        str(name): _frame_payload(table)
        for name, table in diagnostics_tables.items()
        if isinstance(table, pd.DataFrame) and not _looks_row_level_table(str(name), table)
    }
    return {
        "package_version": PACKAGE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run": {
            "run_id": snapshot.get("run_id", ""),
            "execution_mode": snapshot.get("execution_mode", ""),
            "model_type": snapshot.get("model_type", ""),
            "target_mode": snapshot.get("target_mode", ""),
            "target_column": snapshot.get("target_column", ""),
            "labels_available": bool(snapshot.get("labels_available", False)),
            "input_shape": dict(_mapping(snapshot.get("input_shape"))),
            "feature_summary": dict(_mapping(snapshot.get("feature_summary"))),
            "split_summary": _json_safe(snapshot.get("split_summary", {})),
            "run_timing": _json_safe(snapshot.get("run_timing", {})),
            "run_diagnostics": _json_safe(snapshot.get("run_diagnostics", {})),
        },
        "decision_summary": {
            "recommendation": summary["recommendation"],
            "level": summary["level"],
            "rationale": list(summary["rationale"]),
            "cards": _records_payload(pd.DataFrame(summary["cards"])),
            "metrics": _records_payload(summary["metric_frame"]),
            "issues": _records_payload(summary["issue_frame"]),
            "feature_drivers": _records_payload(summary["feature_frame"]),
            "validation_checklist": _records_payload(summary["validation_checklist_frame"]),
            "traceability_map": _records_payload(summary["traceability_frame"]),
        },
        "metrics": _json_safe(snapshot.get("metrics", {})),
        "statistical_tests": _json_safe(snapshot.get("statistical_tests", {})),
        "warnings": [str(warning) for warning in snapshot.get("warnings", [])],
        "feature_columns": [str(column) for column in snapshot.get("feature_columns", [])],
        "numeric_features": [str(column) for column in snapshot.get("numeric_features", [])],
        "categorical_features": [
            str(column) for column in snapshot.get("categorical_features", [])
        ],
        "feature_importance": _frame_payload(snapshot.get("feature_importance")),
        "diagnostic_table_previews": table_previews,
        "diagnostic_table_inventory": [
            {
                "table_name": str(name),
                "rows": int(table.shape[0]),
                "columns": int(table.shape[1]),
                "included_in_context": str(name) in table_previews,
            }
            for name, table in diagnostics_tables.items()
            if isinstance(table, pd.DataFrame)
        ],
        "artifacts": {
            str(key): str(value)
            for key, value in _mapping(snapshot.get("artifacts")).items()
            if value
        },
    }


def build_llm_documentation_package_from_payload(
    payload: Mapping[str, Any],
    *,
    max_included_file_bytes: int = DEFAULT_MAX_INCLUDED_FILE_BYTES,
    max_chart_files: int = DEFAULT_MAX_CHART_FILES,
) -> bytes:
    """Builds a zipped LLM documentation package from a JSON-safe payload."""

    output = BytesIO()
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    added_arcnames: set[str] = set()
    artifacts = _mapping(payload.get("artifacts"))
    run_root = _resolve_run_root(artifacts)

    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        _write_text(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/README_LLM_PACKAGE.md",
            _build_readme(),
            included,
            evidence_area="orientation",
            source="generated",
        )
        _write_json(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/model_document_context.json",
            payload,
            included,
            evidence_area="llm_context",
            source="generated",
        )
        _write_text(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/model_document_context.md",
            _build_context_markdown(payload),
            included,
            evidence_area="llm_context",
            source="generated",
        )
        _write_json(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/sr_11_7_evidence_mapping.json",
            _build_guidance_mapping(),
            included,
            evidence_area="regulatory_mapping",
            source="generated",
        )
        _write_text(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/prompt_generate_model_methodology.md",
            _build_prompt_template(),
            included,
            evidence_area="llm_prompt",
            source="generated",
        )
        _write_text(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/default_model_methodology_outline.md",
            _build_default_outline(),
            included,
            evidence_area="document_outline",
            source="generated",
        )
        _write_template_files(archive, added_arcnames, included)
        _write_diagnostic_table_previews(archive, added_arcnames, payload, included)
        _include_run_artifacts(
            archive=archive,
            added_arcnames=added_arcnames,
            artifacts=artifacts,
            run_root=run_root,
            included=included,
            skipped=skipped,
            max_included_file_bytes=max_included_file_bytes,
            max_chart_files=max_chart_files,
        )
        checklist_rows = _build_evidence_checklist(included, skipped)
        _write_csv(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/llm_evidence_checklist.csv",
            checklist_rows,
            included,
            evidence_area="evidence_checklist",
            source="generated",
        )
        _write_text(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/llm_evidence_checklist.md",
            _build_checklist_markdown(checklist_rows),
            included,
            evidence_area="evidence_checklist",
            source="generated",
        )
        _write_csv(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/source_citation_map.csv",
            included,
            included,
            evidence_area="citation_map",
            source="generated",
        )
        _write_json(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/llm_evidence_manifest.json",
            {
                "package_version": PACKAGE_VERSION,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "run_id": _mapping(payload.get("run")).get("run_id", ""),
                "usage": (
                    "Evidence package for LLM-assisted model methodology drafting. "
                    "LLM output must be reviewed by qualified model-development and "
                    "validation personnel."
                ),
                "privacy_policy": (
                    "Row-level input data, row-level predictions, serialized model binaries, "
                    "and full code snapshots are excluded by default."
                ),
                "included_files": included,
                "skipped_files": skipped,
            },
            included,
            evidence_area="manifest",
            source="generated",
        )

    return output.getvalue()


def _include_run_artifacts(
    *,
    archive: ZipFile,
    added_arcnames: set[str],
    artifacts: Mapping[str, Any],
    run_root: Path | None,
    included: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    max_included_file_bytes: int,
    max_chart_files: int,
) -> None:
    for key, value in artifacts.items():
        path = Path(str(value))
        if not path.exists():
            skipped.append(
                _skip_record(
                    key=key,
                    original_path=path,
                    reason="artifact path was not found on disk",
                )
            )
            continue
        if key in ROW_LEVEL_ARTIFACT_KEYS:
            skipped.append(
                _skip_record(
                    key=key,
                    original_path=path,
                    reason="row-level input or prediction data is excluded",
                )
            )
            continue
        if key in MODEL_BINARY_ARTIFACT_KEYS:
            skipped.append(
                _skip_record(
                    key=key,
                    original_path=path,
                    reason="serialized model binaries are excluded",
                )
            )
            continue
        if key in HEAVY_DIRECTORY_ARTIFACT_KEYS:
            skipped.append(
                _skip_record(
                    key=key,
                    original_path=path,
                    reason="large supporting directory is excluded",
                )
            )
            continue
        if path.is_dir():
            if key == "tables_dir":
                _include_table_directory(
                    archive=archive,
                    added_arcnames=added_arcnames,
                    path=path,
                    run_root=run_root,
                    included=included,
                    skipped=skipped,
                    max_included_file_bytes=max_included_file_bytes,
                )
            elif key == "figures_dir":
                _include_figure_directory(
                    archive=archive,
                    added_arcnames=added_arcnames,
                    path=path,
                    run_root=run_root,
                    included=included,
                    skipped=skipped,
                    max_included_file_bytes=max_included_file_bytes,
                    max_chart_files=max_chart_files,
                )
            else:
                skipped.append(
                    _skip_record(
                        key=key,
                        original_path=path,
                        reason="directory is not part of the LLM evidence allowlist",
                    )
                )
            continue
        _include_file(
            archive=archive,
            added_arcnames=added_arcnames,
            key=key,
            path=path,
            run_root=run_root,
            included=included,
            skipped=skipped,
            max_included_file_bytes=max_included_file_bytes,
        )


def _include_table_directory(
    *,
    archive: ZipFile,
    added_arcnames: set[str],
    path: Path,
    run_root: Path | None,
    included: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    max_included_file_bytes: int,
) -> None:
    for table_path in sorted(path.rglob("*")):
        if not table_path.is_file():
            continue
        if _looks_row_level_path(table_path):
            skipped.append(
                _skip_record(
                    key="tables_dir",
                    original_path=table_path,
                    reason="row-level table path is excluded",
                )
            )
            continue
        if table_path.suffix.lower() == ".parquet":
            _include_parquet_table_as_csv(
                archive=archive,
                added_arcnames=added_arcnames,
                path=table_path,
                run_root=run_root,
                included=included,
                skipped=skipped,
                max_included_file_bytes=max_included_file_bytes,
            )
            continue
        if table_path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            _include_file(
                archive=archive,
                added_arcnames=added_arcnames,
                key="tables_dir",
                path=table_path,
                run_root=run_root,
                included=included,
                skipped=skipped,
                max_included_file_bytes=max_included_file_bytes,
            )


def _include_figure_directory(
    *,
    archive: ZipFile,
    added_arcnames: set[str],
    path: Path,
    run_root: Path | None,
    included: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    max_included_file_bytes: int,
    max_chart_files: int,
) -> None:
    chart_count = 0
    for figure_path in sorted(path.rglob("*")):
        if not figure_path.is_file():
            continue
        suffix = figure_path.suffix.lower()
        if suffix not in LLM_IMAGE_SUFFIXES and suffix != ".html":
            continue
        if chart_count >= max_chart_files:
            skipped.append(
                _skip_record(
                    key="figures_dir",
                    original_path=figure_path,
                    reason=f"chart file cap of {max_chart_files} was reached",
                )
            )
            continue
        if _file_size(figure_path) > max_included_file_bytes:
            skipped.append(
                _skip_record(
                    key="figures_dir",
                    original_path=figure_path,
                    reason=(
                        "chart file exceeds package file-size cap "
                        f"({max_included_file_bytes} bytes)"
                    ),
                )
            )
            continue
        chart_count += 1
        _include_file(
            archive=archive,
            added_arcnames=added_arcnames,
            key="figures_dir",
            path=figure_path,
            run_root=run_root,
            included=included,
            skipped=skipped,
            max_included_file_bytes=max_included_file_bytes,
        )


def _include_file(
    *,
    archive: ZipFile,
    added_arcnames: set[str],
    key: str,
    path: Path,
    run_root: Path | None,
    included: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    max_included_file_bytes: int,
) -> None:
    file_name = path.name
    suffix = path.suffix.lower()
    if file_name in HEAVY_REPORT_NAMES:
        skipped.append(
            _skip_record(
                key=key,
                original_path=path,
                reason="large standalone interactive report is referenced but not duplicated",
            )
        )
        return
    if suffix not in LLM_TEXT_SUFFIXES and suffix not in LLM_IMAGE_SUFFIXES and suffix != ".py":
        skipped.append(
            _skip_record(
                key=key,
                original_path=path,
                reason=f"file suffix `{suffix or 'none'}` is not LLM package friendly",
            )
        )
        return
    size = _file_size(path)
    if size > max_included_file_bytes:
        skipped.append(
            _skip_record(
                key=key,
                original_path=path,
                reason=f"file exceeds package file-size cap ({max_included_file_bytes} bytes)",
                size_bytes=size,
            )
        )
        return
    arcname = _source_arcname(path, run_root)
    if arcname in added_arcnames:
        return
    archive.write(path, arcname)
    added_arcnames.add(arcname)
    included.append(
        {
            "package_path": arcname,
            "original_path": str(path),
            "artifact_key": key,
            "evidence_area": _evidence_area_for_path(path, key),
            "source": "run_artifact",
            "size_bytes": size,
            "llm_use": _llm_use_for_path(path, key),
        }
    )


def _include_parquet_table_as_csv(
    *,
    archive: ZipFile,
    added_arcnames: set[str],
    path: Path,
    run_root: Path | None,
    included: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    max_included_file_bytes: int,
) -> None:
    if _file_size(path) > max_included_file_bytes:
        skipped.append(
            _skip_record(
                key="tables_dir",
                original_path=path,
                reason="parquet table exceeds conversion file-size cap",
            )
        )
        return
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive against parquet engine issues.
        skipped.append(
            _skip_record(
                key="tables_dir",
                original_path=path,
                reason=f"parquet table could not be converted to CSV: {exc}",
            )
        )
        return
    if _looks_row_level_table(path.stem, frame):
        skipped.append(
            _skip_record(
                key="tables_dir",
                original_path=path,
                reason="row-level parquet table is excluded",
            )
        )
        return
    csv_text = frame.to_csv(index=False)
    relative = _relative_source_path(path, run_root).with_suffix(".csv")
    arcname = f"{PACKAGE_ROOT}/source_artifacts_converted/{_posix(relative)}"
    if arcname in added_arcnames:
        return
    archive.writestr(arcname, csv_text)
    added_arcnames.add(arcname)
    included.append(
        {
            "package_path": arcname,
            "original_path": str(path),
            "artifact_key": "tables_dir",
            "evidence_area": "diagnostic_tables",
            "source": "converted_parquet_table",
            "size_bytes": len(csv_text.encode("utf-8")),
            "llm_use": "Converted diagnostic table for LLM-readable evidence review.",
        }
    )


def _write_diagnostic_table_previews(
    archive: ZipFile,
    added_arcnames: set[str],
    payload: Mapping[str, Any],
    included: list[dict[str, Any]],
) -> None:
    table_previews = _mapping(payload.get("diagnostic_table_previews"))
    for table_name, table_payload in table_previews.items():
        rows = list(_mapping(table_payload).get("records", []))
        if not rows:
            continue
        safe_name = _safe_name(str(table_name))
        _write_csv(
            archive,
            added_arcnames,
            f"{PACKAGE_ROOT}/structured_table_previews/{safe_name}.csv",
            rows,
            included,
            evidence_area="diagnostic_table_preview",
            source="generated",
            llm_use=(
                "LLM-readable preview of a diagnostic table captured in the completed run."
            ),
        )


def _write_template_files(
    archive: ZipFile,
    added_arcnames: set[str],
    included: list[dict[str, Any]],
) -> None:
    template_files = {
        f"{PACKAGE_ROOT}/document_template/DROP_TABLE_OF_CONTENTS_HERE.md": (
            "# Drop Your Table Of Contents Here\n\n"
            "Replace this file with your institution's model technical document table "
            "of contents before sending the package to an LLM. The prompt instructs the "
            "LLM to fill that structure using only package evidence.\n"
        ),
        f"{PACKAGE_ROOT}/document_template/table_of_contents_instructions.md": (
            "# Table Of Contents Instructions\n\n"
            "Use this folder for a group-specific model document structure. If a custom "
            "table of contents is present, instruct the LLM to preserve the section order, "
            "headings, and terminology while filling each section from the evidence package. "
            "If no custom table of contents is supplied, use "
            "`default_model_methodology_outline.md`.\n"
        ),
        f"{PACKAGE_ROOT}/document_template/example_table_of_contents.md": (
            "# Example Model Technical Document Table Of Contents\n\n"
            "1. Executive Summary\n"
            "2. Model Purpose, Scope, And Intended Use\n"
            "3. Data, Population, And Target Definition\n"
            "4. Methodology And Model Design\n"
            "5. Variable Selection, Transformations, And Feature Governance\n"
            "6. Model Performance And Validation Evidence\n"
            "7. Sensitivity, Stability, And Limitations\n"
            "8. Implementation, Controls, And Ongoing Use Considerations\n"
            "9. Approvals, Open Items, And Appendices\n"
        ),
    }
    for arcname, text in template_files.items():
        _write_text(
            archive,
            added_arcnames,
            arcname,
            text,
            included,
            evidence_area="document_template",
            source="generated",
            llm_use="Template placeholder for institution-specific document structure.",
        )


def _build_context_markdown(payload: Mapping[str, Any]) -> str:
    run = _mapping(payload.get("run"))
    decision = _mapping(payload.get("decision_summary"))
    lines = [
        "# LLM Model Document Context",
        "",
        "This file is a compact, LLM-readable summary of the completed Quant Studio run.",
        "Use source files listed in `source_citation_map.csv` for citations.",
        "",
        "## Run Overview",
        "",
        f"- Run ID: `{run.get('run_id', 'n/a')}`",
        f"- Execution mode: `{run.get('execution_mode', 'n/a')}`",
        f"- Model type: `{run.get('model_type', 'n/a')}`",
        f"- Target mode: `{run.get('target_mode', 'n/a')}`",
        f"- Target column: `{run.get('target_column', 'n/a')}`",
        f"- Labels available: `{run.get('labels_available', False)}`",
        "",
        "## Decision Summary",
        "",
        f"- Recommendation: `{decision.get('recommendation', 'n/a')}`",
        f"- Decision level: `{str(decision.get('level', 'n/a')).replace('_', ' ').title()}`",
        "",
        "### Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in decision.get("rationale", []) or ["Not available."])
    lines.extend(["", "## Metrics", ""])
    lines.extend(_markdown_table(decision.get("metrics", [])))
    lines.extend(["", "## Decision Issues", ""])
    lines.extend(_markdown_table(decision.get("issues", [])))
    lines.extend(["", "## Top Feature Drivers", ""])
    lines.extend(_markdown_table(decision.get("feature_drivers", [])))
    lines.extend(["", "## Diagnostic Table Inventory", ""])
    lines.extend(_markdown_table(payload.get("diagnostic_table_inventory", [])))
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings or ["No run warnings were captured."])
    return "\n".join(lines).strip() + "\n"


def _build_readme() -> str:
    return """# LLM Documentation Package

This zip contains curated run evidence for drafting a model methodology or model
technical document with an LLM. It is designed to support model-risk documentation
aligned with guidance such as SR 11-7 and OCC model-risk expectations, but it does
not certify compliance.

Recommended use:

1. Review `llm_evidence_manifest.json` and `source_citation_map.csv`.
2. Add your institution-specific table of contents to `document_template/` if needed.
3. Use `prompt_generate_model_methodology.md` with the package contents.
4. Require the LLM to cite source files for factual claims.
5. Have model-development, validation, and governance reviewers verify the draft.

Privacy posture:

- Row-level input data is excluded by default.
- Row-level prediction output is excluded by default.
- Serialized model binaries are excluded by default.
- Full code snapshots and monitoring handoff folders are excluded by default.
"""


def _build_prompt_template() -> str:
    return """# Prompt: Generate Model Methodology Document

You are drafting a model methodology / technical model document for a regulated
financial-services model-risk review. Use only the evidence in this package.

Instructions:

1. If `document_template/DROP_TABLE_OF_CONTENTS_HERE.md` has been replaced with a
   custom table of contents, follow that structure exactly.
2. If no custom table of contents is supplied, use
   `default_model_methodology_outline.md`.
3. Cite the source package file for every factual claim. Prefer citations from
   `source_citation_map.csv`, `model_document_context.json`,
   `reports/model_development_dossier.md`, `config/run_config.json`,
   `metadata/reproducibility_manifest.json`, and governance tables.
4. Do not invent results, thresholds, approvals, owners, or limitations.
5. If evidence is missing, write `Evidence not found in package` and identify the
   missing source.
6. Treat the output as a draft requiring qualified human review. Do not state that
   the model is approved, valid, or compliant unless explicit approval evidence is
   present in the package.

Deliverable:

- A complete Markdown model methodology document.
- A short list of unresolved documentation gaps.
- A citation appendix mapping major sections to package source files.
"""


def _build_default_outline() -> str:
    return """# Default Model Methodology Outline

## 1. Executive Summary

## 2. Model Purpose, Scope, And Intended Use

## 3. Portfolio, Population, Target, And Performance Horizon

## 4. Data Sources, Preparation, And Split Design

## 5. Model Methodology And Candidate Model Rationale

## 6. Feature Selection, Transformations, Imputation, And Lineage

## 7. Model Performance, Calibration, Stability, And Statistical Testing

## 8. Sensitivity, Scenario, Explainability, And Limitations

## 9. Implementation, Reproducibility, And Controls

## 10. Validation Evidence, Open Items, And Approval Considerations

## 11. Appendices And Source Evidence Index
"""


def _build_guidance_mapping() -> list[dict[str, Any]]:
    return [
        {
            "guidance_area": "Model development, implementation, and use",
            "documentation_need": (
                "Purpose, intended use, model design, data, assumptions, and limitations."
            ),
            "primary_package_sources": [
                "model_document_context.md",
                "reports/model_development_dossier.md",
                "reports/model_documentation_pack.md",
                "config/run_config.json",
                "code/generated_run.py",
            ],
        },
        {
            "guidance_area": "Model validation",
            "documentation_need": (
                "Performance, conceptual soundness evidence, process verification, "
                "outcomes analysis, and limitations."
            ),
            "primary_package_sources": [
                "reports/validation_pack.md",
                "reports/decision_summary.md",
                "metadata/metrics.json",
                "metadata/statistical_tests.json",
                "tables/governance/validation_checklist.*",
                "tables/governance/evidence_traceability_map.*",
            ],
        },
        {
            "guidance_area": "Governance, policies, and controls",
            "documentation_need": (
                "Reproducibility, run controls, source evidence, lineage, "
                "configuration, and audit trail."
            ),
            "primary_package_sources": [
                "artifact_manifest.json",
                "metadata/reproducibility_manifest.json",
                "metadata/step_manifest.json",
                "metadata/run_debug_trace.json",
                "tables/governance/feature_lineage_map.*",
            ],
        },
        {
            "guidance_area": "CECL / CCAR supporting context when applicable",
            "documentation_need": (
                "Forecasting, segmentation, scenario, horizon, and loss-definition "
                "context where relevant to the configured model."
            ),
            "primary_package_sources": [
                "config/run_config.json",
                "model_document_context.json",
                "reports/model_development_dossier.md",
                "structured_table_previews/",
            ],
        },
    ]


def _build_evidence_checklist(
    included: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> list[dict[str, str]]:
    included_text = "\n".join(
        (
            f"{row.get('artifact_key', '')} {row.get('package_path', '')} "
            f"{row.get('original_path', '')}"
        )
        for row in included
    ).lower()
    skipped_text = "\n".join(
        f"{row.get('artifact_key', '')} {row.get('original_path', '')} {row.get('reason', '')}"
        for row in skipped
    ).lower()
    requirements = [
        ("Run configuration", "run_config.json", "Required to document selected settings."),
        ("Generated Python workflow", "generated_run.py", "Required for reproducibility."),
        ("Decision summary", "decision_summary.md", "Required for recommendation context."),
        (
            "Model development dossier",
            "model_development_dossier.md",
            "Required for methodology narrative.",
        ),
        ("Validation pack", "validation_pack.md", "Required for validation evidence."),
        ("Metrics", "metrics.json", "Required for quantitative performance evidence."),
        ("Statistical tests", "statistical_tests.json", "Required for test evidence."),
        (
            "Feature lineage",
            "feature_lineage_map",
            "Required for source-to-model feature traceability.",
        ),
        (
            "Reproducibility manifest",
            "reproducibility_manifest.json",
            "Required for environment and hash evidence.",
        ),
        ("Artifact manifest", "artifact_manifest.json", "Required for source file indexing."),
        (
            "Diagnostic table previews",
            "structured_table_previews",
            "Useful for LLM-readable diagnostics.",
        ),
        ("Chart exports", "figures/", "Useful when individual chart exports exist."),
        (
            "Table of contents placeholder",
            "drop_table_of_contents_here",
            "Supports custom document templates.",
        ),
    ]
    rows: list[dict[str, str]] = []
    for area, token, rationale in requirements:
        token_lower = token.lower()
        present = token_lower in included_text
        rows.append(
            {
                "evidence_area": area,
                "status": "present" if present else "not_found",
                "why_it_matters": rationale,
                "recommended_action": (
                    "Use the included evidence."
                    if present
                    else (
                        "Review skipped files and run artifacts; regenerate the model "
                        "run if needed."
                    )
                ),
            }
        )
    rows.append(
        {
            "evidence_area": "Privacy exclusions",
            "status": "applied" if "row-level" in skipped_text else "not_triggered",
            "why_it_matters": (
                "Raw data and row-level predictions should not be sent to an LLM by default."
            ),
            "recommended_action": (
                "Only add row-level data under an approved privacy and governance process."
            ),
        }
    )
    return rows


def _build_checklist_markdown(rows: list[dict[str, str]]) -> str:
    lines = ["# LLM Evidence Checklist", ""]
    lines.extend(_markdown_table(rows))
    return "\n".join(lines).strip() + "\n"


def _write_text(
    archive: ZipFile,
    added_arcnames: set[str],
    arcname: str,
    text: str,
    included: list[dict[str, Any]],
    *,
    evidence_area: str,
    source: str,
    llm_use: str = "Generated package guidance or model-document context.",
) -> None:
    if arcname in added_arcnames:
        return
    archive.writestr(arcname, text)
    added_arcnames.add(arcname)
    included.append(
        {
            "package_path": arcname,
            "original_path": "",
            "artifact_key": "",
            "evidence_area": evidence_area,
            "source": source,
            "size_bytes": len(text.encode("utf-8")),
            "llm_use": llm_use,
        }
    )


def _write_json(
    archive: ZipFile,
    added_arcnames: set[str],
    arcname: str,
    payload: Any,
    included: list[dict[str, Any]],
    *,
    evidence_area: str,
    source: str,
) -> None:
    _write_text(
        archive,
        added_arcnames,
        arcname,
        json.dumps(payload, indent=2, default=str),
        included,
        evidence_area=evidence_area,
        source=source,
        llm_use="Structured JSON context for LLM ingestion.",
    )


def _write_csv(
    archive: ZipFile,
    added_arcnames: set[str],
    arcname: str,
    rows: list[dict[str, Any]],
    included: list[dict[str, Any]],
    *,
    evidence_area: str,
    source: str,
    llm_use: str = "CSV evidence table for LLM ingestion and citation.",
) -> None:
    if arcname in added_arcnames:
        return
    text = _rows_to_csv(rows)
    archive.writestr(arcname, text)
    added_arcnames.add(arcname)
    included.append(
        {
            "package_path": arcname,
            "original_path": "",
            "artifact_key": "",
            "evidence_area": evidence_area,
            "source": source,
            "size_bytes": len(text.encode("utf-8")),
            "llm_use": llm_use,
        }
    )


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = sorted({str(key) for row in rows for key in row})
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
    return output.getvalue()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list | tuple | set | dict):
        return json.dumps(value, default=str)
    return value


def _markdown_table(rows: Any) -> list[str]:
    if not rows:
        return ["No records available."]
    records = list(rows)
    columns = sorted({str(key) for row in records if isinstance(row, Mapping) for key in row})
    if not columns:
        return ["No records available."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in records[:50]:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in columns)
            + " |"
        )
    if len(records) > 50:
        lines.append(f"\nShowing 50 of {len(records):,} records.")
    return lines


def _markdown_cell(value: Any) -> str:
    return str(_csv_value(value)).replace("\n", " ").replace("|", "\\|")


def _frame_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, pd.DataFrame):
        return {"rows": 0, "columns": [], "records": []}
    frame = value.iloc[
        :DEFAULT_MAX_TABLE_PREVIEW_ROWS,
        :DEFAULT_MAX_TABLE_PREVIEW_COLUMNS,
    ].copy()
    return {
        "rows": int(value.shape[0]),
        "columns": [str(column) for column in value.columns],
        "preview_rows": int(frame.shape[0]),
        "preview_columns": [str(column) for column in frame.columns],
        "records": _records_payload(frame),
    }


def _records_payload(frame: Any) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    safe_frame = frame.copy()
    safe_frame.columns = [str(column) for column in safe_frame.columns]
    return [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in safe_frame.to_dict(orient="records")
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _frame_payload(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_run_root(artifacts: Mapping[str, Any]) -> Path | None:
    output_root = artifacts.get("output_root")
    if output_root:
        path = Path(str(output_root))
        if path.exists():
            return path
    manifest = artifacts.get("manifest") or artifacts.get("artifact_manifest")
    if manifest:
        path = Path(str(manifest))
        if path.exists():
            return path.parent
    return None


def _source_arcname(path: Path, run_root: Path | None) -> str:
    relative = _relative_source_path(path, run_root)
    return f"{PACKAGE_ROOT}/source_artifacts/{_posix(relative)}"


def _relative_source_path(path: Path, run_root: Path | None) -> Path:
    if run_root is not None:
        try:
            return path.resolve().relative_to(run_root.resolve())
        except ValueError:
            pass
    return Path(_safe_name(path.name))


def _posix(path: Path) -> str:
    return path.as_posix().replace("..", "_")


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return safe.strip("._") or "artifact"


def _skip_record(
    *,
    key: str,
    original_path: Path,
    reason: str,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    return {
        "artifact_key": key,
        "original_path": str(original_path),
        "reason": reason,
        "size_bytes": _file_size(original_path) if size_bytes is None else size_bytes,
    }


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _looks_row_level_path(path: Path) -> bool:
    path_text = path.as_posix().lower()
    row_level_tokens = (
        "/data/",
        "input_snapshot",
        "predictions",
        "scored",
        "row_level",
        "full_data",
    )
    return any(token in path_text for token in row_level_tokens)


def _looks_row_level_table(name: str, table: pd.DataFrame) -> bool:
    name_lower = name.lower()
    if any(token in name_lower for token in ("prediction", "input_snapshot", "row_level")):
        return True
    columns = {str(column).lower() for column in table.columns}
    score_columns = {
        "predicted_probability",
        "predicted_class",
        "predicted_value",
        "prediction_score",
        "residual",
    }
    identifier_like = {"loan_id", "account_id", "customer_id", "entity_id", "split"}
    return bool(score_columns & columns) and bool(identifier_like & columns) and len(table) > 500


def _evidence_area_for_path(path: Path, key: str) -> str:
    text = path.as_posix().lower()
    if key in {"config", "runner_script", "rerun_readme"} or "/config/" in text or "/code/" in text:
        return "reproducibility"
    if "/metadata/" in text:
        return "metadata"
    if "/tables/" in text:
        return "diagnostic_tables"
    if "/figures/" in text:
        return "visual_evidence"
    if "/reports/" in text:
        return "narrative_reports"
    if "/model/" in text:
        return "model_evidence"
    if path.name == "START_HERE.md":
        return "orientation"
    return "supporting_evidence"


def _llm_use_for_path(path: Path, key: str) -> str:
    file_name = path.name
    use_by_name = {
        "run_config.json": "Resolved run configuration for methodology and control documentation.",
        "generated_run.py": (
            "Generated Python workflow for reproducibility and implementation description."
        ),
        "decision_summary.md": (
            "Decision recommendation, metric scorecard, issues, and evidence index."
        ),
        "model_development_dossier.md": "Audit-oriented narrative source for methodology drafting.",
        "validation_pack.md": "Validator-facing evidence source for model validation sections.",
        "model_documentation_pack.md": "Development-facing documentation source.",
        "metrics.json": "Structured model performance metrics.",
        "statistical_tests.json": "Structured statistical-test evidence.",
        "reproducibility_manifest.json": (
            "Environment, package, hash, and input-source audit evidence."
        ),
        "artifact_manifest.json": "Authoritative run artifact index.",
        "feature_lineage_map.csv": (
            "Feature source, transformation, imputation, and documentation lineage."
        ),
    }
    if file_name in use_by_name:
        return use_by_name[file_name]
    if key == "figures_dir":
        return "Visual evidence to interpret charts and model diagnostics."
    if key == "tables_dir":
        return "Diagnostic table evidence to support quantitative claims."
    return "Supporting run evidence for model-document drafting."
