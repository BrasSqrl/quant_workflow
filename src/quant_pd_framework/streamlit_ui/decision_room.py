"""Decision Room synthesis helpers for Step 5."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_decision_room_payload(
    snapshot: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Builds a meeting-ready payload from the existing decision summary."""

    issue_frame = _frame(summary.get("issue_frame"))
    feature_frame = _frame(summary.get("feature_frame"))
    validation_frame = _frame(summary.get("validation_checklist_frame"))
    evidence_frame = _frame(summary.get("evidence_frame"))
    warnings = list(snapshot.get("warnings", []))

    attention_items = _attention_items(issue_frame, validation_frame, warnings)
    top_features = _top_features(feature_frame)
    key_artifacts = _key_artifacts(evidence_frame)
    readiness = _readiness_label(summary, attention_items)

    return {
        "recommendation": str(summary.get("recommendation", "Review required")),
        "level": str(summary.get("level", "review")),
        "readiness": readiness,
        "headline_cards": [
            {"label": "Decision", "value": str(summary.get("recommendation", "N/A"))},
            {"label": "Readiness", "value": readiness},
            {"label": "Warnings", "value": f"{len(warnings):,}"},
            {"label": "Open Items", "value": f"{len(attention_items):,}"},
        ],
        "rationale": list(summary.get("rationale", [])),
        "attention_items": attention_items,
        "top_features": top_features,
        "key_artifacts": key_artifacts,
        "next_actions": _next_actions(attention_items, key_artifacts),
    }


def _readiness_label(summary: dict[str, Any], attention_items: list[dict[str, str]]) -> str:
    level = str(summary.get("level", "")).lower()
    if level == "proceed" and not attention_items:
        return "Committee Ready"
    if level in {"proceed", "caution"}:
        return "Validation Ready"
    if level == "revise":
        return "Needs Revision"
    return "Review Required"


def _attention_items(
    issue_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    warnings: list[Any],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if not issue_frame.empty:
        for _, row in issue_frame.head(5).iterrows():
            severity = str(row.get("severity", "")).strip()
            if severity.lower() == "none":
                continue
            items.append(
                {
                    "source": str(row.get("source", "Issue")),
                    "severity": severity or "warning",
                    "message": str(row.get("message", row.get("subject", ""))),
                }
            )
    if not validation_frame.empty and "status" in validation_frame.columns:
        flagged = validation_frame.loc[
            validation_frame["status"].astype(str).str.lower().str.contains(
                "attention|missing|fail",
                regex=True,
            )
        ]
        for _, row in flagged.head(5).iterrows():
            items.append(
                {
                    "source": str(row.get("review_area", "Validation checklist")),
                    "severity": str(row.get("status", "attention")),
                    "message": str(row.get("recommended_action", row.get("evidence", ""))),
                }
            )
    for warning in warnings[:3]:
        items.append(
            {
                "source": "Run warning",
                "severity": "warning",
                "message": str(warning),
            }
        )
    return items[:8]


def _top_features(feature_frame: pd.DataFrame) -> list[dict[str, str]]:
    if feature_frame.empty:
        return []
    rows: list[dict[str, str]] = []
    for _, row in feature_frame.head(5).iterrows():
        rows.append(
            {
                "feature": str(row.get("feature", row.get("feature_name", ""))),
                "value": str(row.get("value", row.get("importance", ""))),
                "interpretation": str(row.get("interpretation", row.get("direction", ""))),
            }
        )
    return rows


def _key_artifacts(evidence_frame: pd.DataFrame) -> list[dict[str, str]]:
    if evidence_frame.empty:
        return []
    preferred = (
        "interactive diagnostic report",
        "interactive report",
        "decision summary",
        "validation pack",
        "model object",
        "run configuration",
    )
    rows: list[dict[str, str]] = []
    for _, row in evidence_frame.iterrows():
        artifact_key = str(row.get("artifact_key", row.get("key", ""))).replace("_", " ")
        artifact_label = str(row.get("artifact", row.get("label", artifact_key)))
        artifact_search_text = f"{artifact_key} {artifact_label}".lower()
        if not any(candidate in artifact_search_text for candidate in preferred):
            continue
        rows.append(
            {
                "artifact": artifact_label,
                "status": str(row.get("status", "")),
                "path": str(row.get("path", row.get("artifact_path", row.get("location", "")))),
            }
        )
    return rows[:5]


def _next_actions(
    attention_items: list[dict[str, str]],
    key_artifacts: list[dict[str, str]],
) -> list[str]:
    actions = [
        "Review the primary metrics and decision rationale.",
        "Confirm top feature drivers are explainable and policy-compliant.",
        "Open the interactive report for detailed evidence.",
    ]
    if attention_items:
        actions.insert(0, "Resolve or document open attention items.")
    if key_artifacts:
        actions.append("Use the key artifacts list to assemble the reviewer packet.")
    return actions


def _frame(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()
