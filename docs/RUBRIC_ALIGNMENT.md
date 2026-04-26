# Rubric Alignment

This note records how the current codebase aligns with the engineering rubric in `ENGINEERING_RUBRIC.md`.

## Architecture

Aligned:

- the GUI remains a thin configuration layer over `QuantModelOrchestrator`
- pipeline logic is still expressed as explicit steps under `src/quant_pd_framework/steps/`
- fresh model development, existing-model scoring, and feature-subset search
  are represented as explicit execution modes instead of hidden special cases
- large-data behavior is isolated behind file-backed intake, governed sampling,
  and chunked scoring helpers rather than being embedded in Streamlit callbacks

## Configuration And Contracts

Aligned:

- configuration objects now expose explicit validation methods
- orchestrator initialization validates the full `FrameworkConfig`, not just the split
- saved configs loaded from disk are validated on load
- GUI-built configs are validated before the pipeline runs

## Reproducibility And Auditability

Aligned:

- run exports still include the model artifact, resolved config, step manifest, rerun bundle, and reports
- existing-model scoring runs preserve the scoring artifact and effective execution mode in the exported config and report
- artifact manifests, debug traces, diagnostic registries, model cards, reviewer
  records, and profile metadata make completed runs easier to inspect without
  reading raw JSON first

## Quant Workflow Fidelity

Aligned:

- the framework supports `fit_new_model`, `score_existing_model`, and
  `search_feature_subsets`
- labeled scoring runs produce full validation metrics and diagnostics
- unlabeled scoring runs produce score-only documentation outputs and skip invalid label-dependent diagnostics
- loaded model artifacts are checked for feature compatibility before scoring
- feature-subset-search runs produce comparison-only evidence so candidate
  selection does not get confused with final model-development evidence

## Testing

Aligned with one explicit change:

- the highest-value behavioral tests remain in place
- low-value generated-output clutter from prior test runs is being removed from the repository
- the suite covers training, rerun bundles, model variants, GUI config
  translation, existing-model scoring, cross-validation, large-data controls,
  configuration profiles, enterprise workflow surfaces, and Streamlit result
  rendering

## Documentation

Aligned:

- README content is being kept in sync with the execution modes and export behavior
- an executive summary file is included for non-technical stakeholders
- examples exist for both training and existing-model scoring workflows
- dedicated catalogs document statistical tests, model families, metrics,
  preprocessing/data treatment, GUI-to-code traceability, SageMaker setup, and
  the logistic-regression walkthrough

## Operational Hygiene

Aligned:

- generated artifacts, caches, and editable-install metadata are treated as cleanup targets rather than durable project content
- the repository keeps a bootstrap setup path for the GUI while avoiding reliance on generated run output inside source directories
- the root SageMaker text guide is a convenience mirror, while
  `docs/SAGEMAKER_SETUP.md` remains the detailed setup source
