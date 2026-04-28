"""High-value output explanation registry for Step 4 and Step 5."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True, slots=True)
class OutputExplainer:
    """Plain-English explanation for a chart or table."""

    key: str
    title: str
    what_it_shows: str
    how_to_read: str
    good_or_bad: str
    action: str


OUTPUT_EXPLAINERS: dict[str, OutputExplainer] = {
    "split_metrics": OutputExplainer(
        key="split_metrics",
        title="Split Metrics",
        what_it_shows="Primary model metrics across train, validation, and test splits.",
        how_to_read="Compare levels and gaps across splits rather than focusing on one number.",
        good_or_bad="Good results are strong and reasonably consistent across holdout splits.",
        action="Investigate large train-versus-test divergence before relying on the model.",
    ),
    "roc_curve": OutputExplainer(
        key="roc_curve",
        title="ROC / AUC",
        what_it_shows="How well the score ranks event records above non-event records.",
        how_to_read="A curve farther above the diagonal and higher AUC indicate stronger ranking.",
        good_or_bad="AUC near 0.5 is weak; higher values are better, subject to use-case context.",
        action="Use with KS, calibration, and threshold evidence before making a decision.",
    ),
    "precision_recall_curve": OutputExplainer(
        key="precision_recall_curve",
        title="Precision-Recall",
        what_it_shows="The precision and recall tradeoff across decision thresholds.",
        how_to_read=(
            "This is especially useful when events are rare or class imbalance is material."
        ),
        good_or_bad="Better models retain higher precision as recall increases.",
        action="Review alongside confusion-matrix and business cutoff requirements.",
    ),
    "threshold_analysis": OutputExplainer(
        key="threshold_analysis",
        title="Threshold Analysis",
        what_it_shows="How classification metrics change as the decision threshold moves.",
        how_to_read=(
            "Look for a threshold that balances risk appetite, capture, and false positives."
        ),
        good_or_bad="There is no universal best threshold; it must match the business decision.",
        action="Document the chosen threshold and rationale if it differs from the default.",
    ),
    "calibration": OutputExplainer(
        key="calibration",
        title="Calibration",
        what_it_shows="Whether predicted probabilities align with observed event rates.",
        how_to_read="Predicted and observed rates should move together across risk buckets.",
        good_or_bad="Large gaps suggest probability estimates may need recalibration or review.",
        action="Review Brier score, calibration curves, and calibration challenger methods.",
    ),
    "calibration_summary": OutputExplainer(
        key="calibration_summary",
        title="Calibration Summary",
        what_it_shows="Probability error and calibration-method comparison statistics.",
        how_to_read="Lower Brier/log-loss values generally indicate better probability accuracy.",
        good_or_bad="Poor calibration can exist even when AUC is strong.",
        action="Consider Platt or isotonic challenger calibration if enabled.",
    ),
    "lift_gain": OutputExplainer(
        key="lift_gain",
        title="Lift / Gain",
        what_it_shows="How events concentrate in the highest-risk score buckets.",
        how_to_read="Higher lift in top buckets means the model ranks risky records effectively.",
        good_or_bad="Flat lift suggests weak ranking power.",
        action="Use with ROC/AUC and KS for ranking-performance evidence.",
    ),
    "feature_importance": OutputExplainer(
        key="feature_importance",
        title="Feature Importance",
        what_it_shows="The features with the largest model influence or coefficients.",
        how_to_read="Review sign, magnitude, and whether the driver makes business sense.",
        good_or_bad="Strong drivers should be stable, explainable, and policy-compliant.",
        action="Investigate unexpected signs, leakage concerns, or unstable top drivers.",
    ),
    "permutation_importance": OutputExplainer(
        key="permutation_importance",
        title="Permutation Importance",
        what_it_shows="Performance loss after shuffling each feature on held-out data.",
        how_to_read="Larger performance drops imply the model relies more on that feature.",
        good_or_bad="Disagreement with model-specific importance can reveal instability.",
        action="Use as a challenger view of feature influence.",
    ),
    "partial_dependence": OutputExplainer(
        key="partial_dependence",
        title="Partial Dependence",
        what_it_shows="Average prediction movement as one feature changes.",
        how_to_read="Look for direction, non-linearity, and business-plausible shape.",
        good_or_bad="Erratic or counterintuitive curves need review before approval.",
        action="Compare with ICE/ALE and feature policy expectations.",
    ),
    "ice_curves": OutputExplainer(
        key="ice_curves",
        title="ICE Curves",
        what_it_shows="Feature-effect curves for individual records.",
        how_to_read="Spread across curves shows heterogeneity in feature impact.",
        good_or_bad="Very inconsistent curves may indicate interactions or unstable effects.",
        action="Use with PDP and interaction-strength diagnostics.",
    ),
    "accumulated_local_effects": OutputExplainer(
        key="accumulated_local_effects",
        title="Accumulated Local Effects",
        what_it_shows="Local feature effects accumulated across the feature range.",
        how_to_read="Useful when features are correlated and PDP may be misleading.",
        good_or_bad="Effects should be directionally sensible and not dominated by sparse regions.",
        action="Compare with PDP and bucket-level calibration.",
    ),
    "psi": OutputExplainer(
        key="psi",
        title="Population Stability Index",
        what_it_shows="Distribution shift between development and comparison samples.",
        how_to_read="Higher PSI means larger movement in a feature or score distribution.",
        good_or_bad="Moderate or high PSI requires investigation and documentation.",
        action="Review shifted variables, segments, and whether redevelopment is needed.",
    ),
    "vif": OutputExplainer(
        key="vif",
        title="Variance Inflation Factors",
        what_it_shows="Collinearity pressure among numeric predictors.",
        how_to_read="Higher VIF means a feature is strongly explained by other features.",
        good_or_bad="High VIF can destabilize coefficients and signs.",
        action="Remove, combine, or document highly collinear predictors.",
    ),
    "assumption_checks": OutputExplainer(
        key="assumption_checks",
        title="Suitability Checks",
        what_it_shows="Pre-fit and model-family checks for data and modeling suitability.",
        how_to_read="Failures should be reviewed before the model evidence is relied on.",
        good_or_bad="Pass is clean; Watch or Fail requires action or documented acceptance.",
        action="Resolve failed checks or document an exception.",
    ),
    "validation_checklist": OutputExplainer(
        key="validation_checklist",
        title="Validation Checklist",
        what_it_shows=(
            "Whether major evidence areas are complete, attention-needed, or not applicable."
        ),
        how_to_read="Start with attention-needed rows.",
        good_or_bad="Complete evidence is good; missing evidence weakens review readiness.",
        action="Use this as a routing checklist for validation follow-up.",
    ),
    "evidence_traceability_map": OutputExplainer(
        key="evidence_traceability_map",
        title="Traceability Map",
        what_it_shows="Which artifact or table answers each common review question.",
        how_to_read="Use the artifact path and table name to navigate the run folder.",
        good_or_bad="Missing links mean the evidence may be harder to audit.",
        action="Use it when distributing files to reviewers.",
    ),
    "report_payload_audit": OutputExplainer(
        key="report_payload_audit",
        title="Report Payload Audit",
        what_it_shows="Which HTML report charts were kept, downsampled, or skipped.",
        how_to_read="Skipped or downsampled rows explain why report visuals may be lighter.",
        good_or_bad="Downsampling is expected on large runs; skipped key charts need review.",
        action="Use exported diagnostic tables when HTML visuals are capped.",
    ),
    "scorecard_woe_table": OutputExplainer(
        key="scorecard_woe_table",
        title="Scorecard WoE Table",
        what_it_shows="Bucket-level good, bad, bad-rate, WoE, and IV component values.",
        how_to_read="Review bucket order, bad-rate movement, WoE movement, and sparse bins.",
        good_or_bad="Good bins are interpretable, large enough, and directionally stable.",
        action="Use manual bin overrides when bins are too sparse or non-monotonic.",
    ),
    "scorecard_points_table": OutputExplainer(
        key="scorecard_points_table",
        title="Scorecard Points Table",
        what_it_shows="Partial score contribution for each feature bucket.",
        how_to_read="Review whether higher/lower risk buckets receive sensible points.",
        good_or_bad="Unexpected points can indicate coefficient, WoE, or binning issues.",
        action="Compare with WoE table and feature-level scorecard summary.",
    ),
    "scorecard_feature_summary": OutputExplainer(
        key="scorecard_feature_summary",
        title="Scorecard Feature Summary",
        what_it_shows="Feature-level IV, bin count, point span, and bin quality indicators.",
        how_to_read="Use it to prioritize which binned features need review.",
        good_or_bad="High IV with stable bins is useful; extreme IV can indicate leakage.",
        action="Investigate high-risk or weak binning before finalizing a scorecard.",
    ),
}


def get_output_explainer(key: str) -> OutputExplainer | None:
    """Returns an explainer for a high-value output key."""

    return OUTPUT_EXPLAINERS.get(key)


def render_output_explainer(key: str) -> None:
    """Renders a collapsed explanation for a chart or table when available."""

    explainer = get_output_explainer(key)
    if explainer is None:
        return
    with st.expander("Explain this output", expanded=False):
        st.markdown(f"**What it shows:** {explainer.what_it_shows}")
        st.markdown(f"**How to read it:** {explainer.how_to_read}")
        st.markdown(f"**Good / bad signals:** {explainer.good_or_bad}")
        st.markdown(f"**Recommended action:** {explainer.action}")


def explainer_to_dict(explainer: OutputExplainer) -> dict[str, str]:
    """Converts an explainer to a testable dictionary."""

    return {
        "key": explainer.key,
        "title": explainer.title,
        "what_it_shows": explainer.what_it_shows,
        "how_to_read": explainer.how_to_read,
        "good_or_bad": explainer.good_or_bad,
        "action": explainer.action,
    }
