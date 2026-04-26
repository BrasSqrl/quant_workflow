"""Tests for suitability-check audit explanations."""

from __future__ import annotations

from quant_pd_framework.steps.assumption_checks import AssumptionCheckStep


def test_assumption_check_rows_include_plain_language_audit_fields() -> None:
    row = AssumptionCheckStep()._row(
        check_name="events_per_feature",
        subject="train_target",
        observed_value=8.3,
        threshold=10.0,
        passed=False,
    )

    assert row["status_label"] == "Fail"
    assert row["check_label"] == "Events per feature"
    assert "fewer target events per feature" in row["interpretation"]
    assert "overfitting" in row["why_it_matters"]
    assert "Reduce selected features" in row["recommended_action"]
