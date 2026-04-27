"""Tests for decision-summary scorecard helpers."""

from __future__ import annotations

import pandas as pd

from quant_pd_framework.decision_summary import (
    build_decision_summary,
    build_decision_summary_markdown,
)


def test_decision_summary_recommends_proceed_for_clean_binary_model() -> None:
    snapshot = {
        "run_id": "run_clean",
        "execution_mode": "fit_new_model",
        "model_type": "logistic_regression",
        "target_mode": "binary",
        "labels_available": True,
        "feature_columns": ["balance", "utilization"],
        "metrics": {"test": {"roc_auc": 0.82, "ks_statistic": 0.42}},
        "warnings": [],
        "diagnostics_tables": {
            "assumption_checks": pd.DataFrame(
                [{"status": "pass", "check_label": "Events per feature"}]
            )
        },
        "feature_importance": pd.DataFrame(
            {
                "feature_name": ["balance", "utilization"],
                "coefficient": [0.45, -0.22],
                "p_value": [0.01, 0.08],
            }
        ),
        "artifacts": {"interactive_report": "reports/interactive_report.html"},
    }

    summary = build_decision_summary(snapshot)

    assert summary["level"] == "proceed"
    assert summary["recommendation"] == "Proceed to model documentation review"
    assert summary["feature_frame"].iloc[0]["feature"] == "balance"


def test_decision_summary_flags_suitability_failures() -> None:
    snapshot = {
        "run_id": "run_issue",
        "execution_mode": "fit_new_model",
        "model_type": "logistic_regression",
        "target_mode": "binary",
        "labels_available": True,
        "feature_columns": ["balance"],
        "metrics": {"test": {"roc_auc": 0.84}},
        "warnings": [],
        "diagnostics_tables": {
            "assumption_checks": pd.DataFrame(
                [
                    {
                        "status": "fail",
                        "status_label": "Fail",
                        "check_name": "events_per_feature",
                        "subject": "train_target",
                        "interpretation": "Too few target events per feature.",
                    }
                ]
            )
        },
        "feature_importance": pd.DataFrame(),
        "artifacts": {},
    }

    summary = build_decision_summary(snapshot)
    markdown = build_decision_summary_markdown(snapshot)

    assert summary["level"] == "revise"
    assert "Revise before relying on model" in markdown
    assert "Too few target events" in markdown


def test_decision_summary_treats_subset_search_as_review_input() -> None:
    summary = build_decision_summary(
        {
            "run_id": "subset_run",
            "execution_mode": "search_feature_subsets",
            "model_type": "logistic_regression",
            "target_mode": "binary",
            "labels_available": True,
            "feature_columns": ["balance"],
            "metrics": {"subset_search": {"best_roc_auc": 0.81}},
            "warnings": [],
            "diagnostics_tables": {},
            "feature_importance": pd.DataFrame(),
            "artifacts": {},
        }
    )

    assert summary["level"] == "review"
    assert "selected subset" in summary["recommendation"].lower()
