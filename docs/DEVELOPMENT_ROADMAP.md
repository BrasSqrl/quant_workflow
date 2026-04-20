# Development Roadmap

This roadmap is intentionally narrow. The platform is aimed at model development,
validation, challenger review, scenario testing, and documentation for PD, LGD,
CCAR, and CECL workflows. It is not intended to become a broad production
monitoring platform.

## Principles

- Keep the focus on development, validation, and documentation.
- Prefer interpretable and regulator-friendly capabilities over broad model sprawl.
- Add features only when they strengthen repeatability, rigor, or governance.
- Preserve parity between the Python API, the GUI, and the exported run bundle.

## Phase 1: Calibration Workflow

Status: implemented

Scope:

- calibration bin controls with quantile and uniform strategies
- method comparison between base probabilities, Platt scaling, and isotonic calibration
- calibration summary metrics including Brier score, log loss, ECE, MCE,
  calibration slope/intercept, and Hosmer-Lemeshow statistics
- validation-fit recalibration with held-out test evaluation
- recommended calibration method metadata in exported outputs
- GUI controls and saved-config support for calibration settings

## Phase 2: Scorecard Development Mode

Status: implemented

Scope:

- supervised and monotonic binning controls
- WoE transformation as an explicit development path
- score scaling and points mapping
- reason-code style feature contribution outputs
- scorecard-specific documentation views and exports

## Phase 3: Documentation Pack Generation

Status: implemented

Scope:

- model purpose and scope capture
- portfolio, segment, horizon, and target-definition metadata
- assumptions, exclusions, and limitations sections
- validation-ready document assembly using exported metrics and diagnostics
- standardized narrative sections for model build, challenger review, and scenario results

## Phase 4: Variable Selection Workflow

Status: implemented

Scope:

- univariate screening
- candidate-variable reduction workflows
- stability and correlation screens
- manual feature approval states
- exported rationale for included and rejected variables

## Phase 5: Lifetime PD / CECL Model Family

Status: implemented

Scope:

- discrete-time hazard or related lifetime-PD modeling support
- horizon-aware outputs for CECL-style documentation
- lifetime default views and related validation tables
- scenario-ready lifetime loss documentation hooks

## Phase 6: Reproducibility Manifest

Status: implemented

Scope:

- run fingerprint export with dataframe, config, and model hashes
- Python and platform version capture
- tracked package-version capture
- optional git-commit capture
- reproducibility manifest export in JSON and table form

## Phase 7: Feature Dictionary / Variable Catalog

Status: implemented

Scope:

- feature-level business definitions, source lineage, units, and ranges
- expected-sign and inclusion-rationale capture
- dictionary coverage export for modeled features
- GUI editing surface and workbook round-trip support

## Phase 8: Governed Transformation Layer

Status: implemented

Scope:

- train-fit, replayable feature transformations
- winsorization, log1p, ratio, interaction, and manual-bin transforms
- transformation audit table in diagnostics and exports
- config-driven transformation contract persisted in rerun bundles

## Phase 9: Manual Review Workflow

Status: implemented

Scope:

- manual approve/reject/force decisions after feature screening
- reviewer-name capture and exported review table
- optional review-completeness enforcement
- scorecard numeric bin overrides with rationale

## Phase 10: Model Suitability And Assumption Checks

Status: implemented

Scope:

- pre-fit class-balance and events-per-feature checks
- bounded-target and censoring diagnostics where applicable
- panel duplicate entity-date checks and time-gap summaries
- dominant-category concentration checks
- exported suitability table with optional fail-on-violation behavior

## Phase 11: Structured Validation-Pack Export

Status: implemented

Scope:

- validator-facing markdown pack separate from the development pack
- sections for data contract, assumptions, review decisions, challenger summary,
  scenario summary, and artifact index
- stronger handoff between model development and validation review

## Phase 12: Excel-Based Template Import / Export

Status: implemented

Scope:

- editable workbook export for schema, feature dictionary, transformations,
  manual review, and scorecard overrides
- GUI download/upload workflow for offline review and re-entry
- exported template workbook in each run folder

## Phase 13: Robustness And Stability Testing

Status: implemented

Scope:

- repeated train-resample fitting on the active model family
- held-out metric stability tables with mean, spread, and percentile summaries
- feature and coefficient stability summaries across repeated resamples
- stability charts grouped into the `Stability / Drift` reporting section
- GUI controls and saved-config support for robustness settings

## Phase 14: Interactive Scorecard / Binning Workbench

Status: implemented

Scope:

- dedicated scorecard workbench section in the GUI and exported HTML report
- feature-level scorecard summary table with IV, bin counts, points spread, and monotonic trend
- scorecard points distribution and reason-code frequency outputs
- bucket-level bad-rate, WoE, and partial-points views for the profiled scorecard features
- saved-config support for workbench output scope and feature count

## Notes

- The roadmap items above are now present in the codebase and exposed through
  both the Python API and the GUI.
- Future work should stay narrow and focus on better model development rigor,
  stronger credit-risk documentation, and additional regulator-friendly
  development utilities rather than production monitoring features.
