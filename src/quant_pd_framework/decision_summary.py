"""Decision-summary helpers shared by the GUI and exported reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from quant_pd_framework.presentation import format_metric_value

PRIMARY_ARTIFACT_EVIDENCE: tuple[tuple[str, str, str], ...] = (
    ("interactive_report", "Interactive diagnostic report", "Detailed charts and tables"),
    ("decision_summary", "Decision summary", "Executive model decision scorecard"),
    ("documentation_pack", "Model documentation pack", "Development documentation"),
    ("validation_pack", "Validation pack", "Validator-facing evidence index"),
    ("model", "Model object", "Serialized fitted model for reuse"),
    ("config", "Run configuration", "Resolved configuration and rerun contract"),
    ("tests", "Statistical tests", "Machine-readable test outputs"),
    ("metrics", "Metrics", "Machine-readable performance metrics"),
    ("feature_importance", "Feature importance", "Feature-level influence evidence"),
    ("run_debug_trace", "Run debug trace", "Step timing and debugging evidence"),
)


def build_decision_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Builds a decision-ready model summary from a completed run snapshot."""

    metrics = _mapping(snapshot.get("metrics"))
    diagnostics_tables = _mapping(snapshot.get("diagnostics_tables"))
    warnings = [str(warning) for warning in snapshot.get("warnings", [])]
    labels_available = bool(snapshot.get("labels_available", False))
    target_mode = str(snapshot.get("target_mode", ""))
    execution_mode = str(snapshot.get("execution_mode", ""))
    feature_columns = list(snapshot.get("feature_columns", []))

    split_name, split_metrics = _preferred_metric_split(metrics)
    primary_metric_name, primary_metric_value = _primary_metric(target_mode, split_metrics)
    primary_metric_band = _metric_band(target_mode, primary_metric_name, primary_metric_value)
    issue_frame = _build_issue_frame(
        diagnostics_tables=diagnostics_tables,
        warnings=warnings,
        statistical_tests=_mapping(snapshot.get("statistical_tests")),
    )
    issue_counts = _issue_counts(issue_frame)
    recommendation, level, rationale = _recommendation(
        execution_mode=execution_mode,
        labels_available=labels_available,
        primary_metric_name=primary_metric_name,
        primary_metric_band=primary_metric_band,
        issue_counts=issue_counts,
        warning_count=len(warnings),
    )
    cards = [
        {"label": "Recommendation", "value": recommendation},
        {"label": "Decision Level", "value": level.replace("_", " ").title()},
        {"label": "Primary Split", "value": split_name.title() if split_name else "N/A"},
        {
            "label": _title_metric(primary_metric_name)
            if primary_metric_name
            else "Primary Metric",
            "value": format_metric_value(primary_metric_value),
        },
        {"label": "Metric Read", "value": primary_metric_band.title()},
        {"label": "Features", "value": f"{len(feature_columns):,}"},
        {"label": "Warnings", "value": f"{len(warnings):,}"},
        {"label": "Decision Issues", "value": f"{issue_counts['material']:,}"},
    ]
    return {
        "recommendation": recommendation,
        "level": level,
        "rationale": rationale,
        "cards": cards,
        "metric_frame": _build_metric_frame(metrics, target_mode),
        "issue_frame": issue_frame,
        "feature_frame": _build_feature_frame(snapshot, diagnostics_tables),
        "evidence_frame": _build_evidence_frame(_mapping(snapshot.get("artifacts"))),
    }


def build_decision_summary_markdown(snapshot: Mapping[str, Any]) -> str:
    """Renders the decision summary as a portable Markdown artifact."""

    summary = build_decision_summary(snapshot)
    lines = [
        "# Decision Summary",
        "",
        "## Decision Recommendation",
        "",
        f"- Recommendation: **{summary['recommendation']}**",
        f"- Decision level: `{summary['level'].replace('_', ' ').title()}`",
        f"- Run ID: `{snapshot.get('run_id', 'n/a')}`",
        f"- Execution mode: `{snapshot.get('execution_mode', 'n/a')}`",
        f"- Model family: `{snapshot.get('model_type', 'n/a')}`",
        f"- Target mode: `{snapshot.get('target_mode', 'n/a')}`",
        "",
        "### Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["rationale"])
    lines.extend(["", "## Decision Scorecard", ""])
    lines.extend(_frame_to_markdown(pd.DataFrame(summary["cards"])))
    lines.extend(["", "## Primary Metrics", ""])
    lines.extend(_frame_to_markdown(summary["metric_frame"]))
    lines.extend(["", "## Decision Issues", ""])
    lines.extend(_frame_to_markdown(summary["issue_frame"]))
    lines.extend(["", "## Top Feature Drivers", ""])
    lines.extend(_frame_to_markdown(summary["feature_frame"]))
    lines.extend(["", "## Evidence Index", ""])
    lines.extend(_frame_to_markdown(summary["evidence_frame"]))
    return "\n".join(lines).strip() + "\n"


def build_decision_summary_snapshot_from_context(
    context: Any,
    *,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the minimal decision-summary snapshot from a pipeline context."""

    feature_importance = getattr(context, "feature_importance", None)
    diagnostics_tables = getattr(context, "diagnostics_tables", {})
    return {
        "run_id": getattr(context, "run_id", ""),
        "execution_mode": context.config.execution.mode.value,
        "model_type": context.config.model.model_type.value,
        "target_mode": context.config.target.mode.value,
        "target_column": getattr(context, "target_column", ""),
        "feature_columns": list(getattr(context, "feature_columns", [])),
        "labels_available": bool(context.metadata.get("labels_available", False)),
        "metrics": getattr(context, "metrics", {}),
        "warnings": list(getattr(context, "warnings", [])),
        "diagnostics_tables": diagnostics_tables,
        "statistical_tests": getattr(context, "statistical_tests", {}),
        "feature_importance": (
            feature_importance.copy(deep=True)
            if isinstance(feature_importance, pd.DataFrame)
            else pd.DataFrame()
        ),
        "artifacts": dict(artifacts or getattr(context, "artifacts", {})),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _preferred_metric_split(
    metrics: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    for split_name in ("test", "validation", "holdout", "score", "train", "subset_search"):
        split_metrics = metrics.get(split_name)
        if isinstance(split_metrics, Mapping):
            return split_name, split_metrics
    for split_name, split_metrics in metrics.items():
        if isinstance(split_metrics, Mapping):
            return str(split_name), split_metrics
    return "", {}


def _primary_metric(
    target_mode: str,
    split_metrics: Mapping[str, Any],
) -> tuple[str, Any]:
    if target_mode == "binary":
        candidates = ["roc_auc", "ks_statistic", "average_precision", "brier_score"]
    else:
        candidates = ["rmse", "mae", "r2", "mean_absolute_error", "mean_squared_error"]
    for candidate in candidates:
        if candidate in split_metrics:
            return candidate, split_metrics[candidate]
    for metric_name, metric_value in split_metrics.items():
        if isinstance(metric_value, int | float) and not isinstance(metric_value, bool):
            return str(metric_name), metric_value
    return "", None


def _metric_band(target_mode: str, metric_name: str, value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None or not metric_name:
        return "unrated"
    metric_name = metric_name.lower()
    if metric_name in {"roc_auc", "auc"}:
        return _higher_is_better_band(numeric, great=0.80, good=0.70, watch=0.60)
    if metric_name in {"ks_statistic", "ks"}:
        return _higher_is_better_band(numeric, great=0.40, good=0.30, watch=0.20)
    if metric_name in {"brier_score"}:
        return _lower_is_better_band(numeric, great=0.10, good=0.18, watch=0.25)
    if metric_name in {"r2", "r_squared"}:
        return _higher_is_better_band(numeric, great=0.75, good=0.50, watch=0.25)
    if target_mode == "binary" and metric_name in {"accuracy", "f1_score", "precision", "recall"}:
        return _higher_is_better_band(numeric, great=0.80, good=0.70, watch=0.60)
    return "contextual"


def _higher_is_better_band(value: float, *, great: float, good: float, watch: float) -> str:
    if value >= great:
        return "great"
    if value >= good:
        return "good"
    if value >= watch:
        return "watch"
    return "bad"


def _lower_is_better_band(value: float, *, great: float, good: float, watch: float) -> str:
    if value <= great:
        return "great"
    if value <= good:
        return "good"
    if value <= watch:
        return "watch"
    return "bad"


def _build_metric_frame(metrics: Mapping[str, Any], target_mode: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split_name, split_metrics in metrics.items():
        if not isinstance(split_metrics, Mapping):
            continue
        for metric_name, metric_value in split_metrics.items():
            if not _is_scalar(metric_value):
                continue
            band = _metric_band(target_mode, str(metric_name), metric_value)
            rows.append(
                {
                    "split": str(split_name),
                    "metric": _title_metric(str(metric_name)),
                    "value": format_metric_value(metric_value),
                    "interpretation": _band_label(band),
                }
            )
    if not rows:
        rows.append(
            {
                "split": "n/a",
                "metric": "No metrics captured",
                "value": "N/A",
                "interpretation": "Review run logs and artifacts.",
            }
        )
    return pd.DataFrame(rows)


def _build_issue_frame(
    *,
    diagnostics_tables: Mapping[str, Any],
    warnings: list[str],
    statistical_tests: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    assumption_checks = diagnostics_tables.get("assumption_checks")
    if isinstance(assumption_checks, pd.DataFrame) and not assumption_checks.empty:
        status_column = "status" if "status" in assumption_checks.columns else ""
        if status_column:
            flagged = assumption_checks.loc[
                assumption_checks[status_column].astype(str).str.lower().isin(["fail", "warn"])
            ]
            for _, row in flagged.head(20).iterrows():
                rows.append(
                    {
                        "severity": str(row.get("status_label", row.get(status_column, ""))),
                        "source": "Suitability checks",
                        "subject": str(row.get("subject", row.get("check_name", ""))),
                        "message": str(
                            row.get(
                                "interpretation",
                                row.get("recommended_action", "Review suitability check."),
                            )
                        ),
                    }
                )
    workflow_guardrails = diagnostics_tables.get("workflow_guardrails")
    if isinstance(workflow_guardrails, pd.DataFrame) and not workflow_guardrails.empty:
        severity_column = "severity" if "severity" in workflow_guardrails.columns else ""
        if severity_column:
            flagged = workflow_guardrails.loc[
                workflow_guardrails[severity_column]
                .astype(str)
                .str.lower()
                .isin(["error", "warning"])
            ]
            for _, row in flagged.head(20).iterrows():
                rows.append(
                    {
                        "severity": str(row.get(severity_column, "")),
                        "source": "Workflow guardrails",
                        "subject": str(row.get("field_path", row.get("code", ""))),
                        "message": str(row.get("message", "Review workflow guardrail.")),
                    }
                )
    for test_name, payload in statistical_tests.items():
        if isinstance(payload, Mapping):
            status = str(payload.get("status", "")).lower()
            if status in {"fail", "failed", "warning", "warn"}:
                rows.append(
                    {
                        "severity": status,
                        "source": "Statistical tests",
                        "subject": str(test_name),
                        "message": str(payload.get("interpretation", payload.get("message", ""))),
                    }
                )
    for warning in warnings[:20]:
        rows.append(
            {
                "severity": "warning",
                "source": "Run warning",
                "subject": "Pipeline",
                "message": warning,
            }
        )
    if not rows:
        rows.append(
            {
                "severity": "none",
                "source": "Decision review",
                "subject": "Completed run",
                "message": "No blocking decision issues were detected in the captured summary.",
            }
        )
    return pd.DataFrame(rows)


def _issue_counts(issue_frame: pd.DataFrame) -> dict[str, int]:
    severities = issue_frame.get("severity", pd.Series(dtype=str)).astype(str).str.lower()
    blocking = int(severities.isin(["fail", "failed", "error", "bad"]).sum())
    warning = int(severities.isin(["warn", "warning", "watch"]).sum())
    material = blocking + warning
    if len(issue_frame) == 1 and severities.iloc[0] == "none":
        material = 0
    return {"blocking": blocking, "warning": warning, "material": material}


def _build_feature_frame(
    snapshot: Mapping[str, Any],
    diagnostics_tables: Mapping[str, Any],
) -> pd.DataFrame:
    table = snapshot.get("feature_importance")
    if not isinstance(table, pd.DataFrame) or table.empty:
        table = diagnostics_tables.get("feature_importance")
    if not isinstance(table, pd.DataFrame) or table.empty or "feature_name" not in table.columns:
        return pd.DataFrame(
            [
                {
                    "feature": "No feature-importance table captured",
                    "importance": "N/A",
                    "direction": "N/A",
                    "evidence": "Review model summary and diagnostics.",
                }
            ]
        )
    working = table.copy(deep=True)
    value_column = _first_existing(
        working,
        [
            "abs_coefficient",
            "abs_importance",
            "importance_value",
            "coefficient",
        ],
    )
    signed_column = _first_existing(working, ["coefficient", "importance_value"])
    if value_column:
        working["_rank_value"] = pd.to_numeric(working[value_column], errors="coerce").abs()
        working = working.sort_values("_rank_value", ascending=False)
    rows: list[dict[str, str]] = []
    for _, row in working.head(12).iterrows():
        signed_value = _to_float(row.get(signed_column)) if signed_column else None
        rows.append(
            {
                "feature": str(row.get("feature_name", "")),
                "importance": format_metric_value(row.get(value_column)) if value_column else "N/A",
                "direction": _direction_label(signed_value),
                "evidence": _feature_evidence(row),
            }
        )
    return pd.DataFrame(rows)


def _build_evidence_frame(artifacts: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for key, artifact, purpose in PRIMARY_ARTIFACT_EVIDENCE:
        value = artifacts.get(key)
        if value:
            rows.append(
                {
                    "artifact": artifact,
                    "purpose": purpose,
                    "location": str(value),
                }
            )
    if not rows:
        rows.append(
            {
                "artifact": "Artifact manifest",
                "purpose": "Use the exported artifact manifest to locate evidence files.",
                "location": "artifact_manifest.json",
            }
        )
    return pd.DataFrame(rows)


def _recommendation(
    *,
    execution_mode: str,
    labels_available: bool,
    primary_metric_name: str,
    primary_metric_band: str,
    issue_counts: Mapping[str, int],
    warning_count: int,
) -> tuple[str, str, list[str]]:
    rationale: list[str] = []
    if execution_mode == "search_feature_subsets":
        rationale.append(
            "Feature-subset search is comparison evidence only; use the selected "
            "subset as an input to a later full development run."
        )
        return "Use selected subset for development review", "review", rationale
    if execution_mode == "score_existing_model":
        rationale.append(
            "This run reused an existing model; the decision focus is data "
            "compatibility and scoring evidence, not model approval."
        )
        return "Review scoring compatibility", "review", rationale
    if not labels_available:
        rationale.append(
            "Labels were not available, so realized model performance cannot be "
            "confirmed in this run."
        )
        return "Proceed only with compatibility caveats", "caution", rationale
    if issue_counts["blocking"] > 0:
        rationale.append(f"{issue_counts['blocking']} blocking decision issue(s) were detected.")
        return "Revise before relying on model", "revise", rationale
    if primary_metric_band == "bad":
        rationale.append(f"The primary metric `{primary_metric_name}` is in the bad review band.")
        return "Revise model specification", "revise", rationale
    if primary_metric_band in {"watch", "contextual", "unrated"} or issue_counts["warning"]:
        rationale.append(
            "The run has caveats that should be addressed or explicitly accepted before approval."
        )
        if warning_count:
            rationale.append(f"{warning_count} run warning(s) were captured.")
        return "Proceed with documented caveats", "caution", rationale
    rationale.append("Primary metrics and captured decision checks do not show blocking issues.")
    return "Proceed to model documentation review", "proceed", rationale


def _title_metric(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def _band_label(band: str) -> str:
    return {
        "great": "Great",
        "good": "Good",
        "watch": "Watch",
        "bad": "Bad",
        "contextual": "Context-dependent",
        "unrated": "Not rated",
    }.get(band, band.title())


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _is_scalar(value: Any) -> bool:
    return isinstance(value, int | float | str | bool) or value is None


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return ""


def _direction_label(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value > 0:
        return "Positive"
    if value < 0:
        return "Negative"
    return "Neutral"


def _feature_evidence(row: pd.Series) -> str:
    p_value = _to_float(row.get("p_value"))
    if p_value is not None:
        if p_value < 0.01:
            return "Very strong statistical evidence"
        if p_value < 0.05:
            return "Statistically significant"
        if p_value < 0.10:
            return "Borderline statistical evidence"
        return "Weak statistical evidence"
    return str(row.get("importance_type", "Model-derived importance"))


def _frame_to_markdown(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows captured."]
    safe = frame.fillna("").astype(str)
    columns = [str(column) for column in safe.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in safe.iterrows():
        lines.append(
            "| " + " | ".join(_escape_markdown_cell(row[column]) for column in columns) + " |"
        )
    return lines


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
