"""Reusable in-app glossary helpers for Streamlit guidance surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """Short definition suitable for inline UI help."""

    term: str
    definition: str


GLOSSARY_ENTRIES: dict[str, GlossaryEntry] = {
    "auc": GlossaryEntry(
        "AUC",
        "Ranking metric that measures how well scores separate event and non-event records.",
    ),
    "roc": GlossaryEntry(
        "ROC",
        "Curve comparing true-positive and false-positive rates across score thresholds.",
    ),
    "ks": GlossaryEntry(
        "KS",
        "Maximum separation between event and non-event cumulative score distributions.",
    ),
    "woe": GlossaryEntry(
        "WoE",
        "Weight of Evidence; a bucket value based on the log ratio of good-rate to bad-rate.",
    ),
    "iv": GlossaryEntry(
        "IV",
        "Information Value; a feature-level summary of bucket separation strength.",
    ),
    "psi": GlossaryEntry(
        "PSI",
        "Population Stability Index; a measure of distribution shift between populations.",
    ),
    "vif": GlossaryEntry(
        "VIF",
        "Variance Inflation Factor; a collinearity measure for numeric predictors.",
    ),
    "brier": GlossaryEntry(
        "Brier Score",
        "Average squared probability error for binary predictions; lower is better.",
    ),
    "calibration": GlossaryEntry(
        "Calibration",
        "Agreement between predicted probabilities and observed outcome rates.",
    ),
    "pdp": GlossaryEntry(
        "PDP",
        "Partial Dependence Plot; average model response as one feature changes.",
    ),
    "ice": GlossaryEntry(
        "ICE",
        "Individual Conditional Expectation; per-record feature-effect curves.",
    ),
    "ale": GlossaryEntry(
        "ALE",
        (
            "Accumulated Local Effects; local feature effects that can be "
            "more stable with correlated features."
        ),
    ),
    "tobit": GlossaryEntry(
        "Tobit",
        "Regression for continuous targets censored at a known lower or upper bound.",
    ),
    "probit": GlossaryEntry(
        "Probit",
        "Binary-response regression using a normal cumulative distribution link.",
    ),
    "quantile regression": GlossaryEntry(
        "Quantile Regression",
        "Regression that estimates a selected conditional percentile instead of the mean.",
    ),
    "scorecard": GlossaryEntry(
        "Scorecard",
        "Transparent points-based model usually built from binned variables and WoE.",
    ),
    "reason code": GlossaryEntry(
        "Reason Code",
        "Top model drivers explaining why an observation received its score.",
    ),
    "challenger": GlossaryEntry(
        "Challenger",
        "Alternative model compared with the selected or baseline model.",
    ),
    "lgd": GlossaryEntry(
        "LGD",
        "Loss given default; the severity of loss after a default event.",
    ),
    "pd": GlossaryEntry(
        "PD",
        "Probability of default.",
    ),
    "cecl": GlossaryEntry(
        "CECL",
        "Current Expected Credit Losses.",
    ),
    "ccar": GlossaryEntry(
        "CCAR",
        "Comprehensive Capital Analysis and Review.",
    ),
}


def get_glossary_entry(term: str) -> GlossaryEntry | None:
    """Returns a glossary entry for a term using forgiving key matching."""

    normalized = _normalize_term(term)
    return GLOSSARY_ENTRIES.get(normalized)


def glossary_help_text(*terms: str) -> str:
    """Builds a compact help string for Streamlit widget help text."""

    definitions = [
        f"{entry.term}: {entry.definition}"
        for term in terms
        if (entry := get_glossary_entry(term)) is not None
    ]
    return "\n\n".join(definitions)


def build_glossary_badges(terms: list[str]) -> str:
    """Builds hoverable glossary badges as HTML."""

    badges: list[str] = []
    seen: set[str] = set()
    for term in terms:
        entry = get_glossary_entry(term)
        if entry is None or entry.term in seen:
            continue
        seen.add(entry.term)
        badges.append(
            '<span class="glossary-badge" '
            f'title="{escape(entry.definition, quote=True)}">'
            f"{escape(entry.term)} <span>?</span></span>"
        )
    return "".join(badges)


def render_glossary_badges(terms: list[str], *, caption: str = "Key terms") -> None:
    """Renders compact hoverable glossary badges."""

    badges = build_glossary_badges(terms)
    if not badges:
        return
    st.markdown(
        f'<div class="glossary-strip"><strong>{escape(caption)}</strong>{badges}</div>',
        unsafe_allow_html=True,
    )


def _normalize_term(term: str) -> str:
    return term.strip().lower().replace("_", " ").replace("-", " ")
