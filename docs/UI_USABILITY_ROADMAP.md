# UI Usability Roadmap

This is the implementation record for the current Quant Studio UI/UX
improvement cycle. Prior implementation roadmaps have been cleared because
their feature work has already been completed. This roadmap is intentionally
limited to five usability-focused features:

1. Model Type Story Cards
2. Binning Theater
3. Decision Room Mode
4. Explain This Output Buttons
5. Glossary Hover Layer

Status: implemented.

The goal is to improve user understanding without changing the modeling engine,
statistical calculations, exported model objects, or current workflow scope.

## Design Principles

- Keep the product focused on model development, validation support, and
  documentation.
- Add guidance where it reduces confusion at the moment of decision.
- Avoid turning the interface into a generic help site.
- Make explanations concise by default and expandable when deeper detail is
  needed.
- Keep all guidance consistent with existing documentation and actual code
  behavior.
- Preserve the five-step workflow structure.
- Avoid adding expensive runtime work unless a user explicitly opens a feature.

## Implementation Order

### 1. Glossary Hover Layer

Purpose:

Add lightweight contextual definitions for technical terms such as AUC, KS,
WoE, IV, PSI, Brier score, calibration, PDP, ICE, ALE, Tobit, probit,
quantile regression, scorecard, reason code, and challenger model.

Why first:

This creates shared explanation infrastructure that can be reused by story
cards, output explainers, and the Decision Room.

Implemented behavior:

- Added a centralized glossary registry in the Streamlit UI layer.
- Added reusable hover-badge and widget-help helpers for concise definitions.
- Applied the helper to high-value terms in Step 2, Step 4, and Step 5.

Likely files:

- `src/quant_pd_framework/streamlit_ui/glossary.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `src/quant_pd_framework/streamlit_ui/results.py`
- `docs/user_guides/GLOSSARY.md`

Acceptance criteria:

- Important modeling and metric terms have concise in-app definitions.
- Definitions do not clutter the default layout.
- The registry is reusable and testable.

### 2. Model Type Story Cards

Purpose:

When a user selects a model type, show a polished card explaining what the
model is for, when to avoid it, the required target mode, key settings, main
outputs, and common validation questions.

Why second:

It uses the glossary foundation and directly supports the model-selection
decision in Step 2.

Implemented behavior:

- Added a model story registry keyed by `ModelType`.
- Rendered the selected model's story inside Step 2 near the model selector.
- Included sections for `Best for`, `Avoid when`, `Target mode`, `Key
  settings`, `Outputs to review`, and `Validation questions`.
- Kept the card compact by default with expanded detail in Advanced mode.

Likely files:

- `src/quant_pd_framework/streamlit_ui/model_story_cards.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `docs/user_guides/MODEL_TYPE_REFERENCE_GUIDE.md`

Acceptance criteria:

- Every selectable model type has a story card.
- The selected model card updates immediately when the model type changes.
- The card reflects the actual constraints in `ModelConfig.validate(...)`.

### 3. Explain This Output Buttons

Purpose:

Add a consistent explanation affordance beside important Step 4 and Step 5
tables/charts. A user should be able to click or expand `Explain this output`
to understand what the output shows, how to read it, what good/bad looks like,
and what action to take.

Why third:

This uses the same explanation registry approach and improves interpretation
of existing outputs without changing calculations.

Implemented behavior:

- Added an output explainer registry keyed by high-value figure/table names.
- Rendered `Explain this output` expanders in Step 4 section views and selected
  Step 5 decision tabs.
- Kept explanations collapsed by default and limited to what the output shows,
  how to read it, good/bad signals, and recommended action.

Likely files:

- `src/quant_pd_framework/streamlit_ui/output_explainers.py`
- `src/quant_pd_framework/streamlit_ui/results.py`
- `src/quant_pd_framework/presentation.py`
- `docs/METRIC_CATALOG.md`
- `docs/STATISTICAL_TEST_CATALOG.md`

Acceptance criteria:

- High-value outputs have explanation buttons or expanders.
- The UI remains clean when explanations are collapsed.
- The explainer registry covers at least model performance, calibration,
  scorecard, feature effects, statistical tests, governance, and artifact
  outputs.

### 4. Binning Theater

Purpose:

Create a more interactive scorecard-binning review surface for
`scorecard_logistic_regression`. Users should be able to inspect bins, WoE,
bad rates, IV, monotonicity, points, and manual override recommendations in one
focused workspace.

Why fourth:

It is the most specialized and complex item. It benefits from the glossary and
output-explainer infrastructure already being in place.

Implemented behavior:

- Added a scorecard-specific review panel in the Scorecard / Binning Workbench.
- Showed selected feature bin quality, IV, bin count, largest-bin share, WoE
  span, selected WoE buckets, and selected points buckets.
- Provided a copyable manual-bin override string when internal edges can be
  inferred from the exported bucket labels.
- Added guidance text explaining weak IV, sparse buckets, large concentration,
  and monotonicity concerns.
- Kept the panel review-only; it does not refit the model live.

Likely files:

- `src/quant_pd_framework/streamlit_ui/results.py`
- `src/quant_pd_framework/streamlit_ui/scorecard_workbench.py`
- `src/quant_pd_framework/steps/diagnostics.py`
- `docs/user_guides/MODEL_TYPE_REFERENCE_GUIDE.md`

Acceptance criteria:

- Scorecard runs expose a focused binning review surface.
- Users can clearly see bucket size, bad rate, WoE, IV, and points by feature.
- The UI provides manual override guidance without silently changing the model.

### 5. Decision Room Mode

Purpose:

Add a meeting-ready Step 5 view for executive and committee review. This should
summarize the recommendation, key metrics, primary risks, validation readiness,
top feature drivers, key artifacts, and reviewer next actions without forcing
users through technical tables.

Why fifth:

It depends on the other explanation surfaces and should synthesize the mature
run outputs rather than inventing new analysis.

Implemented behavior:

- Added a `Decision Room` view as the default Step 5 landing tab.
- Rendered a decision header, KPI strip, attention items, top drivers, key
  artifacts, and next-action checklist.
- Preserved existing Step 5 tables and downloads.
- Reused the existing decision-summary generation for downloadable Markdown.

Likely files:

- `src/quant_pd_framework/streamlit_ui/results.py`
- `src/quant_pd_framework/streamlit_ui/decision_room.py`
- `src/quant_pd_framework/decision_summary.py`
- `docs/user_guides/VALIDATION_REVIEWER_GUIDE.md`

Acceptance criteria:

- Step 5 has a clear meeting-ready view.
- It does not replace detailed evidence tabs.
- It uses existing run metrics, warnings, validation checklist, artifacts, and
  feature importance.
- It is concise enough to be used live in a review meeting.

## Testing Plan

- Add unit tests for glossary, model story, and output explainer registries.
- Add Streamlit UI regression tests for Step 2 model story rendering.
- Add results UI tests for explanation rendering and Decision Room data
  synthesis.
- Add scorecard workbench tests using existing scorecard fixture outputs.
- Run targeted tests before full suite:
  - `tests/test_streamlit_ui_results.py`
  - `tests/test_presentation.py`
  - scorecard-related diagnostics tests
  - any new registry tests

## Documentation Plan

- Update `docs/user_guides/MODEL_TYPE_REFERENCE_GUIDE.md`.
- Update `docs/user_guides/VALIDATION_REVIEWER_GUIDE.md`.
- Update `docs/user_guides/GLOSSARY.md`.
- Update `docs/GUI_TO_CODE_TRACEABILITY_GUIDE.md`.
- Update `docs/UI_UX_STANDARD.md` if new reusable UI patterns are introduced.

## Deferred Ideas

The first implementation should not include live model refitting from the
Binning Theater, automatic AI-generated interpretations, or new model
approval logic. Those can be considered only after the static guidance layer is
stable.
