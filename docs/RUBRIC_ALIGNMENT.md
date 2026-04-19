# Rubric Alignment

This note records how the current codebase aligns with the engineering rubric in `ENGINEERING_RUBRIC.md`.

## Architecture

Aligned:

- the GUI remains a thin configuration layer over `QuantModelOrchestrator`
- pipeline logic is still expressed as explicit steps under `src/quant_pd_framework/steps/`
- existing-model scoring is represented as an execution mode instead of a hidden special case

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

## Quant Workflow Fidelity

Aligned:

- the framework now supports both `fit_new_model` and `score_existing_model`
- labeled scoring runs produce full validation metrics and diagnostics
- unlabeled scoring runs produce score-only documentation outputs and skip invalid label-dependent diagnostics
- loaded model artifacts are checked for feature compatibility before scoring

## Testing

Aligned with one explicit change:

- the highest-value behavioral tests remain in place
- low-value generated-output clutter from prior test runs is being removed from the repository
- the suite covers training, rerun bundles, model variants, GUI config translation, and existing-model scoring

## Documentation

Aligned:

- README content is being kept in sync with the execution modes and export behavior
- an executive summary file is included for non-technical stakeholders
- examples exist for both training and existing-model scoring workflows

## Operational Hygiene

Aligned:

- generated artifacts, caches, and editable-install metadata are treated as cleanup targets rather than durable project content
- the repository keeps a bootstrap setup path for the GUI while avoiding reliance on generated run output inside source directories
