"""Tests for in-app guidance registries and review helpers."""

from __future__ import annotations

import pandas as pd

from quant_pd_framework import ModelType
from quant_pd_framework.streamlit_ui.decision_room import build_decision_room_payload
from quant_pd_framework.streamlit_ui.glossary import (
    build_glossary_badges,
    get_glossary_entry,
    glossary_help_text,
)
from quant_pd_framework.streamlit_ui.model_story_cards import get_model_story_card
from quant_pd_framework.streamlit_ui.output_explainers import get_output_explainer
from quant_pd_framework.streamlit_ui.scorecard_workbench import (
    build_binning_quality_messages,
    build_manual_bin_override_string,
)


def test_every_model_type_has_a_story_card() -> None:
    for model_type in ModelType:
        card = get_model_story_card(model_type)

        assert card.model_type == model_type
        assert card.summary
        assert card.outputs_to_review
        assert card.validation_questions


def test_glossary_helpers_return_compact_definitions() -> None:
    entry = get_glossary_entry("WoE")
    help_text = glossary_help_text("AUC", "KS")
    badges = build_glossary_badges(["AUC", "auc", "KS"])

    assert entry is not None
    assert "Weight of Evidence" in entry.definition
    assert "AUC:" in help_text
    assert "KS:" in help_text
    assert badges.count("glossary-badge") == 2


def test_output_explainer_registry_covers_high_value_outputs() -> None:
    explainer = get_output_explainer("roc_curve")

    assert explainer is not None
    assert "ranks" in explainer.what_it_shows.lower()
    assert explainer.action


def test_binning_theater_builds_manual_override_edges() -> None:
    woe_table = pd.DataFrame(
        {
            "bucket_label": ["(-inf, 0.2]", "(0.2, 0.5]", "(0.5, inf]"],
            "total": [100, 200, 150],
        }
    )

    override = build_manual_bin_override_string(woe_table)

    assert "0.200" in override
    assert "0.500" in override


def test_binning_quality_messages_flag_weak_or_sparse_bins() -> None:
    messages = build_binning_quality_messages(
        summary_row=pd.Series(
            {
                "information_value": 0.01,
                "largest_bin_share": 0.65,
                "bad_rate_trend": "mixed",
            }
        ),
        woe_table=pd.DataFrame({"total": [100, 0]}),
    )

    message_text = " ".join(message["message"] for message in messages)

    assert "weak" in message_text.lower()
    assert "half" in message_text.lower()
    assert "no rows" in message_text.lower()


def test_decision_room_payload_summarizes_existing_decision_summary() -> None:
    payload = build_decision_room_payload(
        snapshot={"warnings": ["Review calibration drift."]},
        summary={
            "recommendation": "Proceed to model documentation review",
            "level": "caution",
            "rationale": ["Test AUC is in the good range."],
            "issue_frame": pd.DataFrame(
                [
                    {
                        "source": "Calibration",
                        "severity": "warning",
                        "message": "Calibration requires review.",
                    }
                ]
            ),
            "feature_frame": pd.DataFrame(
                [{"feature": "utilization", "importance": "0.42"}]
            ),
            "validation_checklist_frame": pd.DataFrame(
                [{"review_area": "Documentation", "status": "complete"}]
            ),
            "evidence_frame": pd.DataFrame(
                [
                    {
                        "artifact": "Interactive diagnostic report",
                        "purpose": "Detailed charts and tables",
                        "location": "reports/interactive_report.html",
                    }
                ]
            ),
        },
    )

    assert payload["readiness"] == "Validation Ready"
    assert payload["headline_cards"][0]["value"] == "Proceed to model documentation review"
    assert payload["attention_items"]
    assert payload["top_features"][0]["feature"] == "utilization"
    assert payload["key_artifacts"][0]["path"] == "reports/interactive_report.html"
