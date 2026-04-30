"""Model type story cards for Step 2 model-selection guidance."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import streamlit as st

from quant_pd_framework import ModelType
from quant_pd_framework.streamlit_ui.glossary import render_glossary_badges


@dataclass(frozen=True, slots=True)
class ModelStoryCard:
    """User-facing explanation for one selectable model type."""

    model_type: ModelType
    title: str
    target_mode: str
    summary: str
    best_for: tuple[str, ...]
    avoid_when: tuple[str, ...]
    key_settings: tuple[str, ...]
    outputs_to_review: tuple[str, ...]
    validation_questions: tuple[str, ...]
    glossary_terms: tuple[str, ...] = ()


MODEL_STORY_CARDS: dict[ModelType, ModelStoryCard] = {
    ModelType.LOGISTIC_REGRESSION: ModelStoryCard(
        model_type=ModelType.LOGISTIC_REGRESSION,
        title="Logistic Regression",
        target_mode="Binary",
        summary="Standard interpretable probability model for PD-style event outcomes.",
        best_for=(
            "Baseline PD development",
            "Coefficient and odds-direction review",
            "Calibration, ROC/AUC, KS, and threshold evidence",
        ),
        avoid_when=(
            "The target is continuous",
            "The relationship is strongly non-linear and no transformations are planned",
        ),
        key_settings=("Classification threshold", "Solver", "Class weight", "C"),
        outputs_to_review=("Feature importance", "ROC/AUC", "KS", "Calibration", "Lift/Gain"),
        validation_questions=(
            "Are coefficient signs directionally sensible?",
            "Is the selected threshold justified?",
            "Are calibration and discrimination both acceptable?",
        ),
        glossary_terms=("PD", "AUC", "KS", "Calibration"),
    ),
    ModelType.DISCRETE_TIME_HAZARD_MODEL: ModelStoryCard(
        model_type=ModelType.DISCRETE_TIME_HAZARD_MODEL,
        title="Discrete-Time Hazard Model",
        target_mode="Binary",
        summary="Lifetime PD model for event timing across time-indexed observations.",
        best_for=("Lifetime PD / CECL", "Panel or time-series default timing", "Vintage review"),
        avoid_when=("There is no reliable date field", "The data is purely cross-sectional"),
        key_settings=("Data structure", "Date column", "Split strategy", "Class weight"),
        outputs_to_review=("Lifetime PD curve", "Vintage curves", "Calibration", "Backtests"),
        validation_questions=(
            "Is the period-level target correctly defined?",
            "Does the split respect time ordering?",
            "Does lifetime risk accumulate plausibly?",
        ),
        glossary_terms=("PD", "CECL", "Calibration"),
    ),
    ModelType.ELASTIC_NET_LOGISTIC_REGRESSION: ModelStoryCard(
        model_type=ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
        title="Elastic-Net Logistic Regression",
        target_mode="Binary",
        summary="Regularized logistic model for larger or more correlated feature sets.",
        best_for=("Correlated predictors", "Feature shrinkage", "Sparse challenger models"),
        avoid_when=("Simple coefficient transparency is the primary goal", "Few features exist"),
        key_settings=("Elastic-net l1 ratio", "C", "Max iterations", "Class weight"),
        outputs_to_review=("Feature importance", "Coefficient stability", "ROC/AUC", "Calibration"),
        validation_questions=(
            "Did regularization improve stability?",
            "Are important features shrunk unexpectedly?",
            "Does performance improve enough to justify added tuning?",
        ),
        glossary_terms=("AUC", "Calibration", "Challenger"),
    ),
    ModelType.SCORECARD_LOGISTIC_REGRESSION: ModelStoryCard(
        model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION,
        title="Scorecard Logistic Regression",
        target_mode="Binary",
        summary="WoE-binned logistic scorecard with points, bins, and reason codes.",
        best_for=("Transparent scorecards", "WoE review", "Reason-code output"),
        avoid_when=("The target is continuous", "Maximum black-box performance is the goal"),
        key_settings=("Scorecard bins", "Monotonicity", "Min bin share", "PDO"),
        outputs_to_review=("WoE table", "Points table", "Binning workbench", "Reason codes"),
        validation_questions=(
            "Are bins monotonic and large enough?",
            "Do WoE values move in a business-sensible direction?",
            "Are score points and reason codes explainable?",
        ),
        glossary_terms=("WoE", "IV", "Scorecard", "Reason Code"),
    ),
    ModelType.PROBIT_REGRESSION: ModelStoryCard(
        model_type=ModelType.PROBIT_REGRESSION,
        title="Probit Regression",
        target_mode="Binary",
        summary="Interpretable binary challenger using a normal-link probability model.",
        best_for=("Parametric binary challenger", "Logistic-link sensitivity review"),
        avoid_when=("The target is continuous", "Tree-like non-linear effects dominate"),
        key_settings=("Max iterations", "Class weight", "Threshold"),
        outputs_to_review=("Feature importance", "Calibration", "ROC/AUC", "Model comparison"),
        validation_questions=(
            "Does probit materially change conclusions versus logistic?",
            "Are signs and rank ordering stable?",
        ),
        glossary_terms=("Probit", "AUC", "Calibration"),
    ),
    ModelType.GEE_LOGISTIC_REGRESSION: ModelStoryCard(
        model_type=ModelType.GEE_LOGISTIC_REGRESSION,
        title="GEE Logistic Regression",
        target_mode="Binary",
        summary=(
            "Cluster-aware logistic model for repeated observations using robust GEE estimates."
        ),
        best_for=(
            "Panel PD development",
            "Borrower-level repeated rows",
            "Cluster-robust inference",
        ),
        avoid_when=(
            "There is no natural grouping column",
            "You need subject-specific random effects rather than population-average effects",
        ),
        key_settings=("GEE group column", "Classification threshold", "Max iterations"),
        outputs_to_review=(
            "Model summary",
            "Robust coefficient significance",
            "Calibration",
            "AUC/KS",
        ),
        validation_questions=(
            "Does the group column represent the repeated-observation unit?",
            "Are signs and significance stable versus regular logistic regression?",
            "Does the population-average interpretation match the business use case?",
        ),
        glossary_terms=("GEE", "PD", "AUC", "KS"),
    ),
    ModelType.LINEAR_REGRESSION: ModelStoryCard(
        model_type=ModelType.LINEAR_REGRESSION,
        title="Linear Regression",
        target_mode="Continuous or Binary",
        summary="Simple linear baseline for continuous outcomes and forecast development.",
        best_for=("Continuous baselines", "Forecast benchmarks", "Transparent coefficients"),
        avoid_when=("The target is censored", "The target is bounded and highly skewed"),
        key_settings=("Split strategy", "Feature transformations", "Outlier diagnostics"),
        outputs_to_review=("RMSE", "MAE", "R-squared", "Residual diagnostics"),
        validation_questions=(
            "Are residuals biased by segment or time?",
            "Are outliers driving coefficients?",
            "Is a linear mean model aligned to the business question?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.RIDGE_REGRESSION: ModelStoryCard(
        model_type=ModelType.RIDGE_REGRESSION,
        title="Ridge Regression",
        target_mode="Continuous",
        summary=(
            "Regularized linear regression that shrinks correlated coefficients "
            "without dropping them."
        ),
        best_for=(
            "Continuous LGD or forecast baselines",
            "Correlated financial statement features",
        ),
        avoid_when=("Sparse feature selection is required", "The target is binary"),
        key_settings=("Regularization alpha", "Feature transformations", "Split strategy"),
        outputs_to_review=("RMSE", "MAE", "Coefficient magnitudes", "Residual diagnostics"),
        validation_questions=(
            "Did shrinkage reduce unstable coefficient swings?",
            "Are heavily correlated features still interpretable together?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.LASSO_REGRESSION: ModelStoryCard(
        model_type=ModelType.LASSO_REGRESSION,
        title="Lasso Regression",
        target_mode="Continuous",
        summary="Regularized linear regression that can shrink weak feature coefficients to zero.",
        best_for=("Continuous feature screening", "High-dimensional forecast baselines"),
        avoid_when=("Small coefficient changes are materially important", "The target is binary"),
        key_settings=("Regularization alpha", "Feature transformations", "Split strategy"),
        outputs_to_review=("Selected coefficients", "RMSE", "MAE", "Residual diagnostics"),
        validation_questions=(
            "Are zeroed features plausible exclusions?",
            "Does the simpler model retain acceptable holdout performance?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.ELASTIC_NET_REGRESSION: ModelStoryCard(
        model_type=ModelType.ELASTIC_NET_REGRESSION,
        title="Elastic-Net Regression",
        target_mode="Continuous",
        summary="Continuous regression combining ridge-style shrinkage and lasso-style selection.",
        best_for=("Correlated continuous predictors", "Feature screening with shrinkage"),
        avoid_when=(
            "A fully unregularized coefficient estimate is required",
            "The target is binary",
        ),
        key_settings=("Regularization alpha", "Elastic-net l1 ratio", "Feature transformations"),
        outputs_to_review=("Selected coefficients", "RMSE", "MAE", "Residual diagnostics"),
        validation_questions=(
            "Does the l1 ratio balance stability and sparsity?",
            "Are selected features stable across validation and test splits?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.BETA_REGRESSION: ModelStoryCard(
        model_type=ModelType.BETA_REGRESSION,
        title="Beta Regression",
        target_mode="Continuous",
        summary="Bounded-outcome model for rates or severities between zero and one.",
        best_for=("LGD severity rates", "Bounded proportions", "Recovery-adjusted severity"),
        avoid_when=("Values are outside 0 to 1", "There are many true zero outcomes"),
        key_settings=("Beta clip epsilon", "Feature transformations", "Residual diagnostics"),
        outputs_to_review=("RMSE", "MAE", "Model summary", "Residual diagnostics"),
        validation_questions=(
            "Are boundary values handled correctly?",
            "Is severity mostly positive and bounded?",
            "Would a two-stage LGD model be more appropriate?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.FRACTIONAL_LOGIT: ModelStoryCard(
        model_type=ModelType.FRACTIONAL_LOGIT,
        title="Fractional Logit",
        target_mode="Continuous",
        summary=(
            "Bounded-response GLM for rates, proportions, or LGD values constrained to 0 through 1."
        ),
        best_for=("LGD rates", "Utilization or recovery proportions", "Bounded forecasts"),
        avoid_when=(
            "The target is unbounded",
            "Boundary masses at zero or one need separate modeling",
        ),
        key_settings=("Beta clip epsilon", "Feature transformations", "Split strategy"),
        outputs_to_review=("Model summary", "Actual versus predicted", "Residual diagnostics"),
        validation_questions=(
            "Are predictions staying within the valid 0 to 1 range?",
            "Are boundary observations common enough to require a zero-one model?",
        ),
        glossary_terms=("LGD", "GLM"),
    ),
    ModelType.ZERO_ONE_INFLATED_BETA: ModelStoryCard(
        model_type=ModelType.ZERO_ONE_INFLATED_BETA,
        title="Zero-One Inflated Beta",
        target_mode="Continuous",
        summary=(
            "Three-part bounded LGD model for outcomes with true mass at zero, "
            "one, and the interior."
        ),
        best_for=("LGD with full recoveries and full losses", "Boundary-heavy severity modeling"),
        avoid_when=(
            "There are too few interior observations",
            "Zeros or ones are data-quality artifacts",
        ),
        key_settings=("Beta clip epsilon", "Max iterations", "Boundary diagnostics"),
        outputs_to_review=("Boundary models", "Interior beta model", "LGD segment diagnostics"),
        validation_questions=(
            "Are zero and one outcomes true economic states?",
            "Do boundary and interior drivers tell a coherent story?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.TWO_STAGE_LGD_MODEL: ModelStoryCard(
        model_type=ModelType.TWO_STAGE_LGD_MODEL,
        title="Two-Stage LGD Model",
        target_mode="Continuous",
        summary="LGD model that separates positive-loss probability from conditional severity.",
        best_for=("LGD with many zero losses", "Positive-loss and severity decomposition"),
        avoid_when=("There are too few positive-loss observations", "All outcomes are positive"),
        key_settings=("Class weight", "Beta clip epsilon", "LGD diagnostics"),
        outputs_to_review=("Stage-one coefficients", "Stage-two coefficients", "LGD segments"),
        validation_questions=(
            "Are there enough positive losses?",
            "Do stage-one and stage-two drivers make sense?",
            "Do segment errors reveal bias?",
        ),
        glossary_terms=("LGD",),
    ),
    ModelType.PANEL_REGRESSION: ModelStoryCard(
        model_type=ModelType.PANEL_REGRESSION,
        title="Panel Regression",
        target_mode="Continuous",
        summary="Documented regression workflow for entity-time panel forecasting.",
        best_for=("CCAR-style panels", "Repeated observations by entity", "Forecast baselines"),
        avoid_when=(
            "There is no entity/date structure",
            "A formal econometric panel estimator is required",
        ),
        key_settings=("Data structure", "Date column", "Entity column", "Split strategy"),
        outputs_to_review=("Residual diagnostics", "Time diagnostics", "Structural breaks"),
        validation_questions=(
            "Are entity and date roles assigned correctly?",
            "Does the split avoid future leakage?",
            "Are residuals stable through time?",
        ),
        glossary_terms=("CCAR",),
    ),
    ModelType.QUANTILE_REGRESSION: ModelStoryCard(
        model_type=ModelType.QUANTILE_REGRESSION,
        title="Quantile Regression",
        target_mode="Continuous",
        summary="Regression for a selected conditional percentile rather than the mean.",
        best_for=("Tail forecasts", "Downside or stressed outcomes", "Percentile-specific views"),
        avoid_when=(
            "The business question asks for the mean",
            "The selected alpha is not justified",
        ),
        key_settings=("Quantile alpha", "Split strategy", "Residual diagnostics"),
        outputs_to_review=("Continuous metrics", "Residual summary", "Coefficient outputs"),
        validation_questions=(
            "Does alpha match the business use case?",
            "Are results interpreted as a percentile, not an average?",
        ),
        glossary_terms=("Quantile Regression",),
    ),
    ModelType.TOBIT_REGRESSION: ModelStoryCard(
        model_type=ModelType.TOBIT_REGRESSION,
        title="Tobit Regression",
        target_mode="Continuous",
        summary="Regression for continuous outcomes censored at known bounds.",
        best_for=("Censored loss or balance outcomes", "Known lower or upper bounds"),
        avoid_when=("Zeros are true separate events", "The target is bounded but not censored"),
        key_settings=("Tobit left censor", "Tobit right censor", "Max iterations"),
        outputs_to_review=("Model summary", "Residual diagnostics", "Actual versus predicted"),
        validation_questions=(
            "Are censoring bounds tied to the data-generating process?",
            "Is censoring different from missingness or true zero?",
        ),
        glossary_terms=("Tobit",),
    ),
    ModelType.COX_PROPORTIONAL_HAZARDS: ModelStoryCard(
        model_type=ModelType.COX_PROPORTIONAL_HAZARDS,
        title="Cox Proportional Hazards",
        target_mode="Continuous",
        summary="Survival-style duration model that ranks time-to-event risk with hazard ratios.",
        best_for=("Duration-to-default or prepayment ranking", "Time-to-event challenger evidence"),
        avoid_when=(
            "You need censoring indicators not currently configured",
            "The target is not a positive duration",
        ),
        key_settings=("Date and duration definition", "Feature transformations", "Split strategy"),
        outputs_to_review=(
            "Hazard ratios",
            "Concordance-style ranking evidence",
            "Residual diagnostics",
        ),
        validation_questions=(
            "Is the target a positive time-to-event duration?",
            "Is the proportional hazards assumption reasonable enough for challenger use?",
        ),
        glossary_terms=("Survival",),
    ),
    ModelType.AFT_SURVIVAL_MODEL: ModelStoryCard(
        model_type=ModelType.AFT_SURVIVAL_MODEL,
        title="AFT Survival Model",
        target_mode="Continuous",
        summary=(
            "Log-duration regression that estimates how features accelerate "
            "or decelerate time to event."
        ),
        best_for=(
            "Duration baselines",
            "Forecasting expected time-to-event",
            "Transparent survival challenger",
        ),
        avoid_when=(
            "The target contains non-positive durations",
            "A censored survival likelihood is required",
        ),
        key_settings=("Duration target", "Feature transformations", "Split strategy"),
        outputs_to_review=("Duration predictions", "Residual diagnostics", "Feature importance"),
        validation_questions=(
            "Are durations strictly positive after cleaning?",
            "Do predicted durations align with observed event timing?",
        ),
        glossary_terms=("Survival",),
    ),
    ModelType.RANDOM_FOREST: ModelStoryCard(
        model_type=ModelType.RANDOM_FOREST,
        title="Random Forest",
        target_mode="Binary or Continuous",
        summary=(
            "Bagged tree ensemble for non-linear benchmark modeling with robust default settings."
        ),
        best_for=("Non-linear challenger models", "Interaction-heavy data", "Benchmarking"),
        avoid_when=(
            "Coefficient-level transparency is required",
            "The dataset is too small for tree ensembles",
        ),
        key_settings=("Tree / EBM estimators", "Tree / EBM max depth", "Class weight"),
        outputs_to_review=(
            "Feature importance",
            "Permutation importance",
            "PDP",
            "ALE",
            "Holdout metrics",
        ),
        validation_questions=(
            "Does the ensemble improve holdout performance versus interpretable baselines?",
            "Are top drivers stable and explainable?",
        ),
        glossary_terms=("Challenger", "PDP", "ALE"),
    ),
    ModelType.EXTRA_TREES: ModelStoryCard(
        model_type=ModelType.EXTRA_TREES,
        title="Extra Trees",
        target_mode="Binary or Continuous",
        summary=(
            "Highly randomized tree ensemble for fast non-linear challenger "
            "and sensitivity modeling."
        ),
        best_for=("Non-linear sensitivity checks", "Fast tree benchmarks", "Noisy feature spaces"),
        avoid_when=(
            "Coefficient-level transparency is required",
            "Randomized splits are hard to defend",
        ),
        key_settings=("Tree / EBM estimators", "Tree / EBM max depth", "Class weight"),
        outputs_to_review=(
            "Feature importance",
            "Permutation importance",
            "PDP",
            "ALE",
            "Holdout metrics",
        ),
        validation_questions=(
            "Does extra randomization produce stable results?",
            "Are performance gains consistent across validation and test splits?",
        ),
        glossary_terms=("Challenger", "PDP", "ALE"),
    ),
    ModelType.EXPLAINABLE_BOOSTING_MACHINE: ModelStoryCard(
        model_type=ModelType.EXPLAINABLE_BOOSTING_MACHINE,
        title="Explainable Boosting Machine",
        target_mode="Binary or Continuous",
        summary=(
            "No-new-dependency EBM-style shallow boosted-tree model used with PDP/ALE outputs "
            "for transparent non-linear review."
        ),
        best_for=("Interpretable non-linear challenger models", "Shape review through PDP/ALE"),
        avoid_when=(
            "You require the external interpret package's native EBM implementation",
            "The governance standard forbids boosted trees",
        ),
        key_settings=("Tree / EBM estimators", "Tree / EBM max depth", "Learning rate"),
        outputs_to_review=("Feature importance", "PDP", "ICE", "ALE", "Holdout metrics"),
        validation_questions=(
            "Do feature shapes make business sense?",
            "Does performance justify a non-linear model over scorecard/logistic baselines?",
        ),
        glossary_terms=("PDP", "ICE", "ALE", "Challenger"),
    ),
    ModelType.XGBOOST: ModelStoryCard(
        model_type=ModelType.XGBOOST,
        title="XGBoost",
        target_mode="Binary or Continuous",
        summary="Gradient-boosted tree model for non-linear challenger development.",
        best_for=("Non-linear challenger models", "Interactions", "Performance benchmarking"),
        avoid_when=("Maximum transparency is required", "The dataset is too small for tree tuning"),
        key_settings=("Estimators", "Learning rate", "Max depth", "Subsample", "Column sample"),
        outputs_to_review=("Feature importance", "Permutation importance", "PDP", "ICE", "ALE"),
        validation_questions=(
            "Does performance justify lower transparency?",
            "Are train/test results divergent?",
            "Do explainability outputs support the selected drivers?",
        ),
        glossary_terms=("Challenger", "PDP", "ICE", "ALE"),
    ),
}


def get_model_story_card(model_type: str | ModelType) -> ModelStoryCard:
    """Returns the story card for a model type."""

    resolved = model_type if isinstance(model_type, ModelType) else ModelType(str(model_type))
    return MODEL_STORY_CARDS[resolved]


def render_model_story_card(model_type: str | ModelType, *, advanced: bool) -> None:
    """Renders the selected model story card in Step 2."""

    card = get_model_story_card(model_type)
    st.markdown(
        f"""
        <div class="model-story-card">
          <span class="model-story-card__eyebrow">Model Type Story</span>
          <h3>{escape(card.title)}</h3>
          <p>{escape(card.summary)}</p>
          <div class="model-story-card__chips">
            <span>Target: {escape(card.target_mode)}</span>
            <span>Best use: {escape(card.best_for[0])}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_glossary_badges(list(card.glossary_terms), caption="Terms in this model")
    with st.expander("Model story details", expanded=advanced):
        _render_story_list("Best for", card.best_for)
        _render_story_list("Avoid when", card.avoid_when)
        _render_story_list("Key settings", card.key_settings)
        _render_story_list("Outputs to review", card.outputs_to_review)
        _render_story_list("Validation questions", card.validation_questions)


def story_card_to_dict(card: ModelStoryCard) -> dict[str, Any]:
    """Converts a story card to a testable dictionary."""

    return {
        "model_type": card.model_type.value,
        "title": card.title,
        "target_mode": card.target_mode,
        "summary": card.summary,
        "best_for": list(card.best_for),
        "avoid_when": list(card.avoid_when),
        "key_settings": list(card.key_settings),
        "outputs_to_review": list(card.outputs_to_review),
        "validation_questions": list(card.validation_questions),
    }


def _render_story_list(title: str, values: tuple[str, ...]) -> None:
    st.markdown(f"**{escape(title)}**")
    for value in values:
        st.markdown(f"- {value}")
