"""Enterprise UX helpers for the Streamlit model-development workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Any

import pandas as pd
import streamlit as st

from quant_pd_framework.presentation import format_metric_value
from quant_pd_framework.streamlit_ui.artifact_summary import build_artifact_summary_frame
from quant_pd_framework.streamlit_ui.state import prepare_table_for_display
from quant_pd_framework.streamlit_ui.theme import render_metric_strip


class WorkflowStepId(StrEnum):
    DATA_SCHEMA = "data_schema"
    MODEL_CONFIGURATION = "model_configuration"
    READINESS_CHECK = "readiness_check"
    RESULTS_ARTIFACTS = "results_artifacts"
    DECISION_SUMMARY = "decision_summary"


class WorkflowStatus(StrEnum):
    NOT_STARTED = "not_started"
    NEEDS_ATTENTION = "needs_attention"
    READY = "ready"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class WorkflowStepState:
    step_id: WorkflowStepId
    label: str
    status: WorkflowStatus
    summary: str
    next_action: str


@dataclass(frozen=True, slots=True)
class WorkflowIssue:
    severity: str
    step_id: WorkflowStepId
    source: str
    message: str
    recommended_action: str
    field_path: str = ""


@dataclass(frozen=True, slots=True)
class ReviewerRecord:
    reviewer_name: str
    approval_status: str
    review_notes: str
    exception_notes: str


GUIDANCE_TOPICS: dict[str, str] = {
    "Execution modes": (
        "`fit_new_model` trains a new model. `score_existing_model` reuses an exported "
        "`model/quant_model.joblib` for new data. `search_feature_subsets` compares feature "
        "sets before committing to a development run."
    ),
    "Column roles": (
        "Exactly one enabled column should be marked `target_source`. Time-aware "
        "workflows also need a `date` role, and panel workflows need an `identifier` role."
    ),
    "Imputation": (
        "Column-level missing-value policies are fit on the training split and replayed "
        "on validation, test, and scored data. Missingness indicators are useful when "
        "the fact that a value is missing is itself predictive."
    ),
    "Transformations": (
        "Governed transformations are defined before the split-aware pipeline runs. "
        "They should be used when the transformed value has a business or statistical "
        "reason to exist, not only because it improves one metric."
    ),
    "Diagnostics": (
        "Diagnostics should be interpreted as evidence groups: data quality, model "
        "performance, calibration, stability, backtesting, explainability, and governance."
    ),
    "Cross-validation": (
        "Cross-validation is supplementary evidence. It should not replace the governed "
        "train/validation/test split used for final reporting and artifact export."
    ),
    "Export controls": (
        "The standard export package is designed to be portable. Disable individual "
        "figure files when speed matters and the full interactive report is sufficient."
    ),
}


def build_workflow_step_states(
    *,
    dataframe_loaded: bool,
    preview_config: Any | None,
    preview_error: str | None,
    preview_findings: list[Any],
    last_run_snapshot: dict[str, Any] | None,
    current_config: dict[str, Any] | None,
) -> list[WorkflowStepState]:
    """Builds the five-step status model shown above the workflow tabs."""

    has_blocking_findings = any(
        str(getattr(finding, "severity", "")).lower() == "error" for finding in preview_findings
    )
    has_warnings = any(
        str(getattr(finding, "severity", "")).lower() == "warning" for finding in preview_findings
    )

    if not dataframe_loaded:
        data_state = WorkflowStepState(
            WorkflowStepId.DATA_SCHEMA,
            "Dataset & Schema",
            WorkflowStatus.NOT_STARTED,
            "No dataset selected",
            "Choose a bundled, Data_Load, or uploaded dataset.",
        )
    elif preview_config is None and preview_error:
        data_state = WorkflowStepState(
            WorkflowStepId.DATA_SCHEMA,
            "Dataset & Schema",
            WorkflowStatus.NEEDS_ATTENTION,
            "Schema needs attention",
            "Review the column designer and target-source role.",
        )
    else:
        data_state = WorkflowStepState(
            WorkflowStepId.DATA_SCHEMA,
            "Dataset & Schema",
            WorkflowStatus.COMPLETE,
            "Dataset loaded",
            "Continue to model configuration.",
        )

    if not dataframe_loaded:
        model_state = WorkflowStepState(
            WorkflowStepId.MODEL_CONFIGURATION,
            "Model Configuration",
            WorkflowStatus.NOT_STARTED,
            "Waiting on dataset",
            "Complete Step 1 first.",
        )
    elif preview_config is None:
        model_state = WorkflowStepState(
            WorkflowStepId.MODEL_CONFIGURATION,
            "Model Configuration",
            WorkflowStatus.NEEDS_ATTENTION,
            "Configuration does not resolve",
            "Fix the readiness issue shown in Step 3.",
        )
    else:
        model_state = WorkflowStepState(
            WorkflowStepId.MODEL_CONFIGURATION,
            "Model Configuration",
            WorkflowStatus.COMPLETE,
            "Configuration resolved",
            "Review readiness and preflight summary.",
        )

    if preview_config is None:
        readiness_state = WorkflowStepState(
            WorkflowStepId.READINESS_CHECK,
            "Readiness Check & Run",
            WorkflowStatus.NOT_STARTED if not dataframe_loaded else WorkflowStatus.NEEDS_ATTENTION,
            "Readiness not clear",
            "Resolve configuration issues before running.",
        )
    elif preview_error or has_blocking_findings:
        readiness_state = WorkflowStepState(
            WorkflowStepId.READINESS_CHECK,
            "Readiness Check & Run",
            WorkflowStatus.NEEDS_ATTENTION,
            "Blocking issue found",
            "Open the issue center and fix blocking items.",
        )
    else:
        readiness_state = WorkflowStepState(
            WorkflowStepId.READINESS_CHECK,
            "Readiness Check & Run",
            WorkflowStatus.READY,
            "Ready with warnings" if has_warnings else "Ready to run",
            "Run the workflow when the preflight summary looks correct.",
        )

    if not last_run_snapshot:
        results_state = WorkflowStepState(
            WorkflowStepId.RESULTS_ARTIFACTS,
            "Results & Artifacts",
            WorkflowStatus.NOT_STARTED,
            "No completed run",
            "Run from Step 3 to generate outputs.",
        )
    elif current_config is not None and last_run_snapshot.get("config") != current_config:
        results_state = WorkflowStepState(
            WorkflowStepId.RESULTS_ARTIFACTS,
            "Results & Artifacts",
            WorkflowStatus.NEEDS_ATTENTION,
            "Results may be stale",
            "Rerun if the current configuration should replace the prior outputs.",
        )
    else:
        results_state = WorkflowStepState(
            WorkflowStepId.RESULTS_ARTIFACTS,
            "Results & Artifacts",
            WorkflowStatus.COMPLETE,
            "Latest run available",
            "Review diagnostics, model card, and artifacts.",
        )

    if not last_run_snapshot:
        decision_state = WorkflowStepState(
            WorkflowStepId.DECISION_SUMMARY,
            "Decision Summary",
            WorkflowStatus.NOT_STARTED,
            "No decision summary",
            "Complete a run before reviewing the decision scorecard.",
        )
    elif current_config is not None and last_run_snapshot.get("config") != current_config:
        decision_state = WorkflowStepState(
            WorkflowStepId.DECISION_SUMMARY,
            "Decision Summary",
            WorkflowStatus.NEEDS_ATTENTION,
            "Summary may be stale",
            "Rerun if the current configuration should replace the prior decision.",
        )
    else:
        decision_state = WorkflowStepState(
            WorkflowStepId.DECISION_SUMMARY,
            "Decision Summary",
            WorkflowStatus.COMPLETE,
            "Decision scorecard available",
            "Review the recommendation, issues, feature drivers, and evidence index.",
        )

    return [data_state, model_state, readiness_state, results_state, decision_state]


def collect_readiness_issues(
    *,
    preview_error: str | None,
    preview_findings: list[Any],
    profile_warnings: list[str] | None = None,
) -> list[WorkflowIssue]:
    """Converts build errors, guardrails, and profile warnings into one issue list."""

    issues: list[WorkflowIssue] = []
    if preview_error:
        issues.append(
            WorkflowIssue(
                severity="error",
                step_id=_step_for_message(preview_error),
                source="Configuration builder",
                message=preview_error,
                recommended_action=_recommended_action_for_message(preview_error),
            )
        )
    for finding in preview_findings:
        severity = str(getattr(finding, "severity", "warning")).lower()
        field_path = str(getattr(finding, "field_path", ""))
        issues.append(
            WorkflowIssue(
                severity=severity,
                step_id=_step_for_field_path(field_path),
                source=str(getattr(finding, "code", "workflow_guardrail")),
                message=str(getattr(finding, "message", finding)),
                field_path=field_path,
                recommended_action=_recommended_action_for_field_path(field_path),
            )
        )
    for warning in profile_warnings or []:
        issues.append(
            WorkflowIssue(
                severity="warning",
                step_id=WorkflowStepId.DATA_SCHEMA,
                source="Configuration profile",
                message=warning,
                recommended_action="Confirm the profile still matches the selected dataset.",
            )
        )
    return issues


def build_preflight_summary(
    *,
    dataframe: pd.DataFrame,
    data_source_label: str,
    preview_config: Any,
    edited_schema: pd.DataFrame,
    transformation_frame: pd.DataFrame,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    """Builds a one-page pre-run summary from the resolved configuration."""

    feature_rows = edited_schema.loc[
        edited_schema["role"].astype(str).str.lower().eq("feature")
        & edited_schema["enabled"].astype(bool)
    ]
    transformation_count = 0
    if not transformation_frame.empty and "enabled" in transformation_frame.columns:
        transformation_count = int(transformation_frame["enabled"].fillna(False).astype(bool).sum())
    cards = [
        {"label": "Dataset", "value": data_source_label or "Selected input"},
        {"label": "Rows", "value": f"{len(dataframe):,}"},
        {"label": "Columns", "value": f"{len(dataframe.columns):,}"},
        {
            "label": "Model",
            "value": preview_config.model.model_type.value.replace("_", " ").title(),
        },
        {"label": "Target Mode", "value": preview_config.target.mode.value.title()},
        {"label": "Features", "value": f"{len(feature_rows):,}"},
        {"label": "Transformations", "value": f"{transformation_count:,}"},
        {"label": "Diagnostics", "value": _enabled_count(preview_config.diagnostics)},
        {"label": "Export Profile", "value": preview_config.artifacts.export_profile.value.title()},
    ]
    rows = [
        ("Execution mode", preview_config.execution.mode.value),
        ("Target source", preview_config.target.source_column),
        ("Target output", preview_config.target.output_column),
        ("Split strategy", preview_config.split.data_structure.value),
        ("Split assignment", preview_config.split.split_strategy.value),
        ("Train/validation/test", _split_text(preview_config)),
        ("Output root", str(preview_config.artifacts.output_root)),
        (
            "Enhanced report visuals",
            preview_config.artifacts.include_enhanced_report_visuals,
        ),
        (
            "Advanced visual analytics",
            preview_config.artifacts.include_advanced_visual_analytics,
        ),
        ("Individual chart package", "On-demand in Step 5"),
        ("Keep all checkpoints", preview_config.artifacts.keep_all_checkpoints),
        ("Input snapshot", preview_config.artifacts.export_input_snapshot),
        ("Code snapshot", preview_config.artifacts.export_code_snapshot),
    ]
    return cards, pd.DataFrame(rows, columns=["area", "value"])


def build_model_suitability_explainer(
    *,
    dataframe: pd.DataFrame,
    preview_config: Any,
    edited_schema: pd.DataFrame,
    transformation_frame: pd.DataFrame,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    """Explains why the selected model setup is suitable or needs review."""

    feature_count = _enabled_feature_count(edited_schema)
    target_column = str(preview_config.target.source_column or "")
    target_mode = preview_config.target.mode.value
    model_type = preview_config.model.model_type.value
    event_text, events_per_feature = _event_density_text(
        dataframe=dataframe,
        target_column=target_column,
        target_mode=target_mode,
        feature_count=feature_count,
    )
    transform_count = _enabled_transformation_count(transformation_frame)
    rows = [
        {
            "area": "Target compatibility",
            "status": "pass",
            "explanation": (
                f"`{model_type}` is valid for `{target_mode}` target mode in the "
                "resolved configuration."
            ),
            "recommended_action": "No action needed.",
        },
        {
            "area": "Data structure",
            "status": _structure_status(preview_config),
            "explanation": _structure_explanation(preview_config),
            "recommended_action": _structure_action(preview_config),
        },
        {
            "area": "Sample and feature volume",
            "status": _sample_feature_status(len(dataframe), feature_count),
            "explanation": f"{len(dataframe):,} rows and {feature_count:,} enabled features.",
            "recommended_action": "Reduce features or use simpler models if warnings appear.",
        },
        {
            "area": "Event density",
            "status": (
                "warning"
                if events_per_feature is not None and events_per_feature < 10
                else "pass"
            ),
            "explanation": event_text,
            "recommended_action": "For binary models, consider fewer features or more events.",
        },
        {
            "area": "Transformation load",
            "status": "warning" if transform_count > 25 else "pass",
            "explanation": f"{transform_count:,} enabled governed transformations.",
            "recommended_action": (
                "Keep transformations with clear business or statistical purpose."
            ),
        },
    ]
    cards = [
        {"label": "Selected model", "value": model_type.replace("_", " ").title()},
        {"label": "Target mode", "value": target_mode.title()},
        {"label": "Data structure", "value": preview_config.split.data_structure.value.title()},
        {"label": "Enabled features", "value": f"{feature_count:,}"},
    ]
    return cards, pd.DataFrame(rows)


def build_configuration_risk_score(
    *,
    dataframe: pd.DataFrame,
    preview_config: Any,
    edited_schema: pd.DataFrame,
    transformation_frame: pd.DataFrame,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    """Scores configuration complexity risk before execution."""

    rows: list[dict[str, str]] = []
    score = 0
    feature_count = _enabled_feature_count(edited_schema)
    transform_count = _enabled_transformation_count(transformation_frame)
    categorical_risks = _high_cardinality_risk_count(dataframe)
    diagnostics_enabled = _enabled_bool_count(preview_config.diagnostics)
    if feature_count > 50:
        score += 15
        rows.append(_risk_row("Feature count", "medium", f"{feature_count:,} enabled features."))
    if len(dataframe) < max(feature_count * 20, 200):
        score += 20
        rows.append(
            _risk_row(
                "Rows vs features",
                "high",
                "Sample size is small relative to enabled feature count.",
            )
        )
    if transform_count > 25:
        score += 10
        rows.append(_risk_row("Transformations", "medium", f"{transform_count:,} enabled rows."))
    if categorical_risks:
        score += 15
        rows.append(
            _risk_row(
                "High-cardinality categoricals",
                "medium",
                f"{categorical_risks:,} text/category fields may expand memory.",
            )
        )
    if diagnostics_enabled > 12:
        score += 10
        rows.append(
            _risk_row("Diagnostics", "medium", f"{diagnostics_enabled:,} diagnostic toggles on.")
        )
    if preview_config.artifacts.include_advanced_visual_analytics:
        score += 10
        rows.append(_risk_row("Advanced visuals", "medium", "Advanced Visual Analytics is on."))
    if preview_config.performance.retain_full_working_data:
        score += 20
        rows.append(
            _risk_row(
                "Memory retention",
                "high",
                "Full diagnostic working dataframe retention is enabled.",
            )
        )
    if not rows:
        rows.append(
            {
                "area": "Overall",
                "severity": "low",
                "signal": "No major pre-run complexity risks detected.",
                "recommended_action": "Proceed after reviewing readiness warnings.",
            }
        )
    score = min(score, 100)
    band = "Low" if score < 25 else "Moderate" if score < 60 else "High"
    cards = [
        {"label": "Risk score", "value": f"{score}/100"},
        {"label": "Risk band", "value": band},
        {"label": "Risk items", "value": f"{len(rows):,}"},
        {"label": "Diagnostics on", "value": f"{diagnostics_enabled:,}"},
    ]
    return cards, pd.DataFrame(rows)


def build_runtime_artifact_estimate(
    *,
    dataframe: pd.DataFrame,
    preview_config: Any,
    transformation_frame: pd.DataFrame,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    """Builds rough runtime and artifact-size estimates for user planning."""

    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    memory_mb = _dataframe_memory_mb(dataframe)
    transform_count = _enabled_transformation_count(transformation_frame)
    diagnostics_enabled = _enabled_bool_count(preview_config.diagnostics)
    complexity_score = (
        row_count * max(column_count, 1) / 1_000_000
        + transform_count * 0.25
        + diagnostics_enabled * 0.5
        + _model_complexity_weight(preview_config.model.model_type.value)
    )
    runtime_band = (
        "Fast" if complexity_score < 8 else "Moderate" if complexity_score < 30 else "Long"
    )
    report_mb = _estimate_report_size_mb(preview_config, row_count, diagnostics_enabled)
    table_mb = max(memory_mb * 0.25, 1.0)
    cards = [
        {"label": "Runtime estimate", "value": runtime_band},
        {"label": "Input memory", "value": _format_mb(memory_mb)},
        {"label": "Report estimate", "value": _format_mb(report_mb)},
        {"label": "Table estimate", "value": _format_mb(table_mb)},
    ]
    rows = [
        ("Rows x columns", f"{row_count:,} x {column_count:,}"),
        ("Model complexity", preview_config.model.model_type.value),
        ("Enabled transformations", f"{transform_count:,}"),
        ("Enabled diagnostics", f"{diagnostics_enabled:,}"),
        ("Individual chart files", "On-demand from Step 5"),
        (
            "Enhanced report visuals",
            _yes_no(preview_config.artifacts.include_enhanced_report_visuals),
        ),
        (
            "Advanced visual analytics",
            _yes_no(preview_config.artifacts.include_advanced_visual_analytics),
        ),
        ("Checkpoint retention", _yes_no(preview_config.artifacts.keep_all_checkpoints)),
    ]
    return cards, pd.DataFrame(rows, columns=["driver", "estimate"])


def build_resource_readiness_check(
    *,
    dataframe: pd.DataFrame,
    preview_config: Any,
    large_data_mode: bool,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    """Builds memory, disk, and expensive-option planning checks before run."""

    memory_mb = _dataframe_memory_mb(dataframe)
    source_file_mb = _source_file_size_mb(dataframe)
    dataframe_multiplier = float(preview_config.performance.memory_estimate_dataframe_multiplier)
    file_multiplier = float(preview_config.performance.memory_estimate_file_multiplier)
    working_data_multiplier = dataframe_multiplier
    if not preview_config.performance.retain_full_working_data:
        working_data_multiplier = max(1.4, dataframe_multiplier * 0.65)
    estimated_peak_mb = max(memory_mb * working_data_multiplier, source_file_mb * file_multiplier)
    memory_limit_gb = preview_config.performance.memory_limit_gb
    memory_limit_mb = None if memory_limit_gb is None else float(memory_limit_gb) * 1024
    report_mb = _estimate_report_size_mb(
        preview_config,
        len(dataframe),
        _enabled_bool_count(preview_config.diagnostics),
    )
    table_mb = _estimate_table_output_mb(memory_mb, preview_config)
    disk_mb = report_mb + table_mb + max(memory_mb * 0.5, 5.0)
    high_cost_options = _high_cost_option_rows(preview_config)
    high_cost_count = sum(1 for row in high_cost_options if row["status"] == "warning")
    recommended_profile = _recommended_resource_profile(
        estimated_peak_mb=estimated_peak_mb,
        disk_mb=disk_mb,
        high_cost_count=high_cost_count,
        large_data_mode=large_data_mode,
    )
    rows = [
        _resource_row(
            "Memory estimate",
            (
                "pass"
                if memory_limit_mb is None or estimated_peak_mb <= memory_limit_mb
                else "warning"
            ),
            _format_mb(estimated_peak_mb),
            (
                "Lower diagnostic retention, use compact exports, or use a larger instance "
                "before starting the run."
            ),
        ),
        _resource_row(
            "Input dataframe memory",
            "pass",
            _format_mb(memory_mb),
            "Use Parquet and dtype optimization if this is materially larger than file size.",
        ),
        _resource_row(
            "Source file size",
            "pass",
            _format_mb(source_file_mb) if source_file_mb else "Not available",
            "Keep the original source metadata available for capacity planning.",
        ),
        _resource_row(
            "Large Data Mode",
            "pass" if large_data_mode or len(dataframe) < 500_000 else "warning",
            "On" if large_data_mode else "Off",
            "Use Large Data Mode for multi-GB file-backed runs.",
        ),
        _resource_row(
            "Checkpoint retention",
            "warning" if preview_config.artifacts.keep_all_checkpoints else "pass",
            (
                "Keep all checkpoints on"
                if preview_config.artifacts.keep_all_checkpoints
                else "Prune safe checkpoints"
            ),
            "Leave off unless debugging requires retained context files.",
        ),
        _resource_row(
            "Report visuals",
            "warning" if preview_config.artifacts.include_advanced_visual_analytics else "pass",
            _visual_setting_text(preview_config),
            "Disable advanced visuals for faster runs or smaller HTML reports.",
        ),
        _resource_row(
            "Interactive report estimate",
            "warning" if report_mb > 250 else "pass",
            _format_mb(report_mb),
            "Use standard visuals or Fast export profile if report portability matters.",
        ),
        _resource_row(
            "Tabular output estimate",
            "warning" if table_mb > 2_000 else "pass",
            _format_mb(table_mb),
            "Use sampled or metadata-only table exports when full tabular outputs are not needed.",
        ),
        _resource_row(
            "Disk estimate",
            "pass" if disk_mb < 5_000 else "warning",
            _format_mb(disk_mb),
            "Use sampled exports or disable extra figure files if disk space is constrained.",
        ),
        _resource_row(
            "Recommended run profile",
            "pass" if recommended_profile == "Standard" else "warning",
            recommended_profile,
            "Adjust export and diagnostic settings before running if the profile is not Standard.",
        ),
    ]
    rows.extend(high_cost_options)
    cards = [
        {"label": "Estimated peak memory", "value": _format_mb(estimated_peak_mb)},
        {
            "label": "Configured memory limit",
            "value": _format_mb(memory_limit_mb) if memory_limit_mb is not None else "Not set",
        },
        {"label": "Estimated disk output", "value": _format_mb(disk_mb)},
        {"label": "High-cost options", "value": f"{high_cost_count:,}"},
        {
            "label": "Recommended profile",
            "value": recommended_profile,
        },
    ]
    return cards, pd.DataFrame(rows)


def build_config_diff_frame(
    *,
    current_config: dict[str, Any] | None,
    baseline_config: dict[str, Any] | None,
    baseline_label: str,
) -> pd.DataFrame:
    """Returns grouped config differences between current and baseline configs."""

    if not current_config or not baseline_config:
        return pd.DataFrame(
            columns=["baseline", "section", "field", "baseline_value", "current_value"]
        )
    current_flat = _flatten_config(current_config)
    baseline_flat = _flatten_config(baseline_config)
    rows: list[dict[str, str]] = []
    for field in sorted(set(current_flat) | set(baseline_flat)):
        current_value = current_flat.get(field)
        baseline_value = baseline_flat.get(field)
        if current_value == baseline_value:
            continue
        section = field.split(".", 1)[0]
        rows.append(
            {
                "baseline": baseline_label,
                "section": section,
                "field": field,
                "baseline_value": _display_value(baseline_value),
                "current_value": _display_value(current_value),
            }
        )
    return pd.DataFrame(rows)


def build_artifact_explorer_frame(artifacts: dict[str, Any]) -> pd.DataFrame:
    """Builds a grouped artifact explorer table with user-facing purpose text."""

    frame = build_artifact_summary_frame(artifacts)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["purpose"] = frame["key"].map(_artifact_purpose).fillna("Supporting run output.")
    frame["download_hint"] = frame["status"].map(
        lambda status: "Available for download" if status == "Available" else "Not available"
    )
    return frame.loc[
        :,
        ["area", "artifact", "status", "purpose", "download_hint", "path", "key"],
    ]


def build_model_card_markdown(
    *,
    snapshot: dict[str, Any],
    reviewer_record: ReviewerRecord | None = None,
) -> str:
    """Creates a compact model card from a completed run snapshot."""

    metrics = snapshot.get("metrics", {})
    feature_columns = snapshot.get("feature_columns", [])
    warnings = snapshot.get("warnings", [])
    review_status = reviewer_record.approval_status if reviewer_record else "Not reviewed"
    review_notes = reviewer_record.review_notes if reviewer_record else ""
    exception_notes = reviewer_record.exception_notes if reviewer_record else ""
    reviewer_name = reviewer_record.reviewer_name if reviewer_record else ""
    lines = [
        "# Quant Studio Model Card",
        "",
        f"- Run ID: `{snapshot.get('run_id', '')}`",
        f"- Execution mode: `{snapshot.get('execution_mode', '')}`",
        f"- Model family: `{snapshot.get('model_type', '')}`",
        f"- Target mode: `{snapshot.get('target_mode', '')}`",
        f"- Target column: `{snapshot.get('target_column', '')}`",
        f"- Feature count: `{len(feature_columns)}`",
        f"- Labels available: `{'yes' if snapshot.get('labels_available') else 'no'}`",
        "",
        "## Primary Metrics",
    ]
    if metrics:
        for split_name, split_metrics in metrics.items():
            if isinstance(split_metrics, dict):
                metric_text = ", ".join(
                    f"{key}: {format_metric_value(value)}"
                    for key, value in split_metrics.items()
                    if isinstance(value, int | float | bool | str) or value is None
                )
                lines.append(f"- {split_name}: {metric_text}")
    else:
        lines.append("- No metric summary was captured in the Streamlit snapshot.")
    lines.extend(
        [
            "",
            "## Feature Set",
            ", ".join(str(feature) for feature in feature_columns[:50])
            or "No feature list captured.",
            "",
            "## Warnings And Limitations",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No run warnings were captured.")
    lines.extend(
        [
            "",
            "## Reviewer Decision",
            f"- Reviewer: {reviewer_name or 'Not recorded'}",
            f"- Status: {review_status}",
            f"- Notes: {review_notes or 'Not recorded'}",
            f"- Exceptions: {exception_notes or 'None recorded'}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_workflow_status_strip(states: list[WorkflowStepState]) -> None:
    """Renders the workflow status strip."""

    cards = [
        {
            "label": f"{index}. {state.label}",
            "value": _status_label(state.status),
        }
        for index, state in enumerate(states, start=1)
    ]
    render_metric_strip(cards, compact=True)
    with st.expander("Workflow status details", expanded=False):
        st.dataframe(
            prepare_table_for_display(
                pd.DataFrame(
                    [
                        {
                            "step": state.label,
                            "status": _status_label(state.status),
                            "summary": state.summary,
                            "next_action": state.next_action,
                        }
                        for state in states
                    ]
                )
            ),
            width="stretch",
            hide_index=True,
        )


def render_issue_center(issues: list[WorkflowIssue]) -> None:
    """Renders a centralized readiness issue list."""

    with st.expander("Readiness Issue Center", expanded=bool(issues)):
        if not issues:
            st.success("No readiness issues are currently registered.")
            return
        issue_frame = pd.DataFrame(
            [
                {
                    "severity": issue.severity,
                    "step": issue.step_id.value.replace("_", " ").title(),
                    "source": issue.source,
                    "field": issue.field_path,
                    "message": issue.message,
                    "recommended_action": issue.recommended_action,
                }
                for issue in issues
            ]
        )
        st.dataframe(
            prepare_table_for_display(issue_frame),
            width="stretch",
            hide_index=True,
        )


def render_preflight_summary(
    *,
    cards: list[dict[str, str]],
    details: pd.DataFrame,
) -> None:
    """Renders the pre-run summary inside Step 3."""

    with st.expander("Run Preflight Summary", expanded=True):
        render_metric_strip(cards, compact=True)
        st.dataframe(
            prepare_table_for_display(details),
            width="stretch",
            hide_index=True,
        )


def render_model_suitability_explainer(
    *,
    cards: list[dict[str, str]],
    details: pd.DataFrame,
) -> None:
    """Renders Step 2 model suitability guidance."""

    with st.expander("Model Suitability Explainer", expanded=True):
        render_metric_strip(cards, compact=True)
        st.dataframe(prepare_table_for_display(details), width="stretch", hide_index=True)


def render_configuration_risk_score(
    *,
    cards: list[dict[str, str]],
    details: pd.DataFrame,
) -> None:
    """Renders Step 2 configuration complexity risk guidance."""

    with st.expander("Configuration Risk Score", expanded=False):
        render_metric_strip(cards, compact=True)
        st.dataframe(prepare_table_for_display(details), width="stretch", hide_index=True)


def render_runtime_artifact_estimate(
    *,
    cards: list[dict[str, str]],
    details: pd.DataFrame,
) -> None:
    """Renders Step 2 rough runtime and output-size estimates."""

    with st.expander("Runtime / Artifact Size Estimate", expanded=False):
        render_metric_strip(cards, compact=True)
        st.caption(
            "Estimates are directional planning aids based on current selections, "
            "not a guaranteed runtime or storage forecast."
        )
        st.dataframe(prepare_table_for_display(details), width="stretch", hide_index=True)


def render_resource_readiness_check(
    *,
    cards: list[dict[str, str]],
    details: pd.DataFrame,
) -> None:
    """Renders Step 3 compute and storage readiness guidance."""

    with st.expander("Resource Planner / Run Cost Estimate", expanded=True):
        render_metric_strip(cards, compact=True)
        st.caption(
            "Use this before execution to spot memory pressure, storage growth, "
            "and high-cost diagnostic options."
        )
        st.dataframe(prepare_table_for_display(details), width="stretch", hide_index=True)


def render_guidance_center() -> None:
    """Renders compact in-app guidance without expanding the main workflow."""

    with st.expander("Guidance Library", expanded=False):
        selected_topic = st.selectbox(
            "Guidance topic",
            options=list(GUIDANCE_TOPICS),
            key="enterprise_guidance_topic",
        )
        st.markdown(GUIDANCE_TOPICS[selected_topic])


def _enabled_feature_count(edited_schema: pd.DataFrame) -> int:
    if edited_schema.empty or "role" not in edited_schema.columns:
        return 0
    enabled = (
        edited_schema["enabled"].map(_is_enabled_value)
        if "enabled" in edited_schema.columns
        else pd.Series(True, index=edited_schema.index)
    )
    return int((enabled & edited_schema["role"].astype(str).str.lower().eq("feature")).sum())


def _enabled_transformation_count(transformation_frame: pd.DataFrame) -> int:
    if transformation_frame.empty:
        return 0
    if "enabled" not in transformation_frame.columns:
        return int(len(transformation_frame))
    return int(transformation_frame["enabled"].map(_is_enabled_value).sum())


def _is_enabled_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _event_density_text(
    *,
    dataframe: pd.DataFrame,
    target_column: str,
    target_mode: str,
    feature_count: int,
) -> tuple[str, float | None]:
    if target_mode != "binary" or target_column not in dataframe.columns:
        return "Not applicable for this target mode or target source is unavailable.", None
    target = pd.to_numeric(dataframe[target_column], errors="coerce").dropna()
    if target.empty:
        return "Target has no numeric non-missing values in the current dataframe.", None
    event_count = float(target.sum())
    events_per_feature = event_count / max(feature_count, 1)
    return (
        f"{event_count:,.0f} positive events; {events_per_feature:.1f} events per feature.",
        events_per_feature,
    )


def _structure_status(preview_config: Any) -> str:
    data_structure = preview_config.split.data_structure.value
    if data_structure == "panel" and not preview_config.split.entity_column:
        return "warning"
    if data_structure in {"panel", "time_series"} and not preview_config.split.date_column:
        return "warning"
    return "pass"


def _structure_explanation(preview_config: Any) -> str:
    data_structure = preview_config.split.data_structure.value
    date_column = preview_config.split.date_column or "none"
    entity_column = preview_config.split.entity_column or "none"
    split_strategy = preview_config.split.split_strategy.value
    return (
        f"Data structure is `{data_structure}`; split strategy=`{split_strategy}`; "
        f"date=`{date_column}`, entity=`{entity_column}`."
    )


def _structure_action(preview_config: Any) -> str:
    if _structure_status(preview_config) == "pass":
        return "No action needed."
    return "Assign date and identifier roles in Column Designer where required."


def _sample_feature_status(row_count: int, feature_count: int) -> str:
    if row_count < max(feature_count * 20, 200):
        return "warning"
    return "pass"


def _high_cardinality_risk_count(dataframe: pd.DataFrame) -> int:
    if dataframe.empty:
        return 0
    threshold = max(50, int(len(dataframe) * 0.2))
    columns = dataframe.select_dtypes(include=["object", "string", "category"]).columns
    return int(sum(dataframe[column].nunique(dropna=True) > threshold for column in columns))


def _enabled_bool_count(config_object: Any) -> int:
    if not is_dataclass(config_object):
        return 0
    return sum(
        1
        for field in fields(config_object)
        if isinstance(getattr(config_object, field.name), bool)
        and getattr(config_object, field.name)
        and field.name != "enabled"
    )


def _risk_row(area: str, severity: str, signal: str) -> dict[str, str]:
    return {
        "area": area,
        "severity": severity,
        "signal": signal,
        "recommended_action": "Review this setting before running the workflow.",
    }


def _dataframe_memory_mb(dataframe: pd.DataFrame) -> float:
    if dataframe.empty:
        return 0.0
    return float(dataframe.memory_usage(deep=True).sum()) / (1024 * 1024)


def _model_complexity_weight(model_type: str) -> float:
    if model_type in {"xgboost", "random_forest", "extra_trees"}:
        return 12.0
    if model_type in {
        "scorecard_logistic_regression",
        "explainable_boosting_machine",
        "gam_spline_regression",
        "gam_spline_logistic",
    }:
        return 8.0
    if "forecast" in model_type or model_type in {"sarimax_forecast"}:
        return 10.0
    return 4.0


def _estimate_report_size_mb(
    preview_config: Any,
    row_count: int,
    diagnostics_enabled: int,
) -> float:
    size = 8.0 + diagnostics_enabled * 1.5 + min(row_count / 50_000, 20)
    if preview_config.artifacts.include_enhanced_report_visuals:
        size += 10.0
    if preview_config.artifacts.include_advanced_visual_analytics:
        size += 25.0
    return size


def _estimate_table_output_mb(memory_mb: float, preview_config: Any) -> float:
    policy = preview_config.artifacts.large_data_export_policy.value
    if policy == "metadata_only":
        return 5.0
    if policy == "sampled":
        return max(10.0, min(memory_mb * 0.2, 750.0))
    output_format = preview_config.artifacts.tabular_output_format.value
    format_multiplier = 0.35 if output_format == "parquet" else 0.8
    if output_format == "both":
        format_multiplier = 1.1
    return max(10.0, memory_mb * format_multiplier)


def _source_file_size_mb(dataframe: pd.DataFrame) -> float:
    metadata = dataframe.attrs.get("quant_studio_input_source", {})
    if not isinstance(metadata, dict):
        return 0.0
    try:
        return float(metadata.get("size_bytes") or 0) / (1024 * 1024)
    except (TypeError, ValueError):
        return 0.0


def _high_cost_option_rows(preview_config: Any) -> list[dict[str, str]]:
    candidates = [
        (
            "Advanced visual analytics",
            bool(preview_config.artifacts.include_advanced_visual_analytics),
            "Disable unless the review specifically needs advanced exploratory visuals.",
        ),
        (
            "Cross-validation",
            bool(getattr(preview_config.cross_validation, "enabled", False)),
            "Use after the baseline run is stable, especially on large datasets.",
        ),
        (
            "Robustness testing",
            bool(getattr(preview_config.robustness, "enabled", False)),
            "Run when validation requires perturbation evidence.",
        ),
        (
            "Scenario testing",
            bool(getattr(preview_config.scenario_testing, "enabled", False)),
            "Run only when scenario evidence is in scope.",
        ),
        (
            "Keep all checkpoints",
            bool(preview_config.artifacts.keep_all_checkpoints),
            "Leave off unless debugging checkpoint contents.",
        ),
    ]
    return [
        _resource_row(
            area,
            "warning" if enabled else "pass",
            "On" if enabled else "Off",
            action,
        )
        for area, enabled, action in candidates
    ]


def _recommended_resource_profile(
    *,
    estimated_peak_mb: float,
    disk_mb: float,
    high_cost_count: int,
    large_data_mode: bool,
) -> str:
    if estimated_peak_mb > 128 * 1024 or disk_mb > 20_000:
        return "Scale instance or reduce outputs"
    if estimated_peak_mb > 32 * 1024 or disk_mb > 5_000 or high_cost_count >= 3:
        return "Large-data cautious"
    if large_data_mode or high_cost_count:
        return "Review settings"
    return "Standard"


def _format_mb(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024:.1f} GB"
    return f"{value:.1f} MB"


def _yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _resource_row(
    area: str,
    status: str,
    signal: str,
    recommended_action: str,
) -> dict[str, str]:
    return {
        "area": area,
        "status": status,
        "signal": signal,
        "recommended_action": recommended_action if status != "pass" else "No action needed.",
    }


def _visual_setting_text(preview_config: Any) -> str:
    settings = []
    if preview_config.artifacts.include_enhanced_report_visuals:
        settings.append("enhanced")
    if preview_config.artifacts.include_advanced_visual_analytics:
        settings.append("advanced")
    return ", ".join(settings) if settings else "standard report visuals"


def _flatten_config(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        field_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, field_name))
        elif isinstance(value, list):
            flattened[field_name] = json.dumps(value, sort_keys=True, default=str)
        else:
            flattened[field_name] = value
    return flattened


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _step_for_message(message: str) -> WorkflowStepId:
    lowered = message.lower()
    if any(token in lowered for token in ["target", "schema", "column", "date", "identifier"]):
        return WorkflowStepId.DATA_SCHEMA
    return WorkflowStepId.MODEL_CONFIGURATION


def _step_for_field_path(field_path: str) -> WorkflowStepId:
    if field_path.startswith(("schema", "target", "split.date", "split.entity")):
        return WorkflowStepId.DATA_SCHEMA
    if field_path.startswith(("execution", "model", "comparison", "feature_policy")):
        return WorkflowStepId.MODEL_CONFIGURATION
    return WorkflowStepId.READINESS_CHECK


def _recommended_action_for_message(message: str) -> str:
    lowered = message.lower()
    if "target" in lowered:
        return "Open Column Designer and mark exactly one enabled target_source row."
    if "date" in lowered:
        return "Open Column Designer and assign a valid date role."
    if "identifier" in lowered:
        return "Open Column Designer and assign a valid identifier role."
    return "Review the affected configuration group and rerun readiness."


def _recommended_action_for_field_path(field_path: str) -> str:
    if field_path.startswith("target"):
        return "Review target mode, source, output, and positive target values."
    if field_path.startswith("split"):
        return "Review data structure and date or identifier roles."
    if field_path.startswith("model"):
        return "Review the selected model family and target-mode compatibility."
    if field_path.startswith("documentation"):
        return "Complete the required documentation fields or disable strict guardrails."
    return "Review the listed field before execution."


def _enabled_count(config_object: Any) -> str:
    if not is_dataclass(config_object):
        return "0 enabled"
    values = [
        getattr(config_object, field.name)
        for field in fields(config_object)
        if isinstance(getattr(config_object, field.name), bool) and field.name != "enabled"
    ]
    return f"{sum(values):,} enabled"


def _split_text(preview_config: Any) -> str:
    return (
        f"{preview_config.split.train_size:.0%} / "
        f"{preview_config.split.validation_size:.0%} / "
        f"{preview_config.split.test_size:.0%}"
    )


def _status_label(status: WorkflowStatus) -> str:
    return {
        WorkflowStatus.NOT_STARTED: "Not Started",
        WorkflowStatus.NEEDS_ATTENTION: "Needs Attention",
        WorkflowStatus.READY: "Ready",
        WorkflowStatus.COMPLETE: "Complete",
    }[status]


def _artifact_purpose(key: str) -> str:
    purpose_map = {
        "output_root": "Top-level run folder containing all exported evidence.",
        "decision_summary": "Decision-ready scorecard with recommendation and evidence links.",
        "interactive_report": "Standalone HTML report for sharing the validation dashboard.",
        "model": "Serialized fitted model object for reruns and scoring.",
        "config": "Resolved run configuration used by the orchestrator.",
        "reproducibility_manifest": "Hashes, package versions, and environment evidence.",
        "run_debug_trace": "Step-level runtime trace for debugging failed or slow runs.",
        "predictions": "Scored development-split predictions.",
        "statistical_tests": "Structured statistical-test output.",
        "documentation_pack": "Development documentation narrative generated from the run.",
        "validation_pack": "Validator-facing evidence pack.",
        "monitoring_bundle_dir": "Portable model bundle for the separate monitoring app.",
    }
    return purpose_map.get(key, "Supporting run output.")
