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
            "Readiness Check",
            WorkflowStatus.NOT_STARTED if not dataframe_loaded else WorkflowStatus.NEEDS_ATTENTION,
            "Readiness not clear",
            "Resolve configuration issues before running.",
        )
    elif preview_error or has_blocking_findings:
        readiness_state = WorkflowStepState(
            WorkflowStepId.READINESS_CHECK,
            "Readiness Check",
            WorkflowStatus.NEEDS_ATTENTION,
            "Blocking issue found",
            "Open the issue center and fix blocking items.",
        )
    else:
        readiness_state = WorkflowStepState(
            WorkflowStepId.READINESS_CHECK,
            "Readiness Check",
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
        ("Individual figure files", preview_config.artifacts.export_individual_figure_files),
        ("Keep all checkpoints", preview_config.artifacts.keep_all_checkpoints),
        ("Input snapshot", preview_config.artifacts.export_input_snapshot),
        ("Code snapshot", preview_config.artifacts.export_code_snapshot),
    ]
    return cards, pd.DataFrame(rows, columns=["area", "value"])


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


def render_guidance_center() -> None:
    """Renders compact in-app guidance without expanding the main workflow."""

    with st.expander("Guidance Library", expanded=False):
        selected_topic = st.selectbox(
            "Guidance topic",
            options=list(GUIDANCE_TOPICS),
            key="enterprise_guidance_topic",
        )
        st.markdown(GUIDANCE_TOPICS[selected_topic])


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
