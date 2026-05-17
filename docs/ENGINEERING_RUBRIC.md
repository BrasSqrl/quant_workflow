# Engineering Rubric

This rubric defines the standard Quant Studio should meet. It is intentionally opinionated: the goal is not just working code, but maintainable, auditable, production-grade code for quantitative model workflows.

## 1. Architecture

The code should:

- keep domain logic in the framework, not in the GUI or launch scripts
- isolate each pipeline stage behind a single-responsibility step or helper
- prefer typed configuration and explicit contracts over ad hoc dictionaries
- support extension through composition and substitution rather than copy-paste branching

A design fails this standard if business logic is primarily embedded in UI callbacks, if step responsibilities are unclear, or if the same workflow logic is duplicated across modules.

## 2. Configuration And Contracts

The code should:

- validate configuration before execution begins
- fail early on invalid model-target combinations or impossible split settings
- treat model artifacts, saved configs, and rerun bundles as first-class contracts
- make execution mode explicit instead of inferring major behavior from side effects

A design fails this standard if invalid settings are only discovered deep inside runtime execution.

## 3. Reproducibility And Auditability

The code should:

- persist the effective run configuration
- persist model artifacts and scored outputs with stable filenames
- record the ordered step stack and major run metadata
- produce deterministic outputs where possible through explicit seeds and saved inputs

A design fails this standard if the same run cannot be reconstructed from exported artifacts.

## 4. Quant Workflow Fidelity

The code should:

- support both fresh model development and existing-model score-and-document workflows
- distinguish clearly between labeled validation and unlabeled score-only documentation
- prevent invalid diagnostics from running when labels are unavailable
- preserve compatibility between existing model artifacts and new scoring data

A design fails this standard if it produces misleading metrics or charts for unlabeled data.

## 5. Testing

The code should:

- keep a lean, high-value test suite focused on behavior, not file-count inflation
- isolate test outputs so test runs do not pollute the repository
- cover the critical workflows: training, export, rerun, GUI-config translation, and existing-model scoring
- prefer shared test helpers for repeated dataset and config construction

A design fails this standard if tests generate persistent clutter or mostly assert superficial text strings.

## 6. Documentation

The code should:

- document the supported workflows at both developer and end-user levels
- explain the difference between training a new model and scoring an existing one
- provide runnable examples for the main usage paths
- keep documentation aligned with the actual code and exported artifacts

A design fails this standard if the README describes behavior the code no longer implements.

## 7. Operational Hygiene

The repository should:

- avoid storing generated artifacts, caches, or editable-install metadata in source control
- keep top-level files purposeful and limited
- provide clear setup and launch paths for both developers and GUI users
- leave the working tree clean after normal testing and verification

A design fails this standard if routine development work leaves large volumes of generated clutter in the repo.

## Current Alignment Notes

The current Quant Studio codebase aligns to this rubric in the following ways:

- The GUI remains a configuration and review layer over the Python framework.
- The Streamlit app is organized by workflow step: Step 1 dataset/schema
  helpers live under `streamlit_ui/steps/step1_data_schema.py`, Step 2 model
  configuration helpers under `step2_model_config.py`, Step 3 readiness/run
  execution under `step3_readiness_run.py`, Step 4 results under
  `step4_results_artifacts.py`, and Step 5 decision summary under
  `step5_decision_summary.py`.
- Result rendering has dedicated panel modules under `streamlit_ui/result_panels/`,
  starting with the run registry and audit-trail panel, while `results.py`
  preserves the existing public import surface.
- Artifact layout and manifest/index logic lives under `quant_pd_framework.exporting`
  so `ArtifactExportStep` can focus on workflow orchestration.
- Large-data source handles and constants live under `quant_pd_framework.large_data_support`
  while `large_data.py` keeps compatibility re-exports for existing imports.
- Workflow execution is coordinated through explicit pipeline stages and
  checkpointed subprocess boundaries for memory isolation, restartability, and
  auditability.
- `fit_new_model`, `score_existing_model`, and `search_feature_subsets` are
  explicit execution modes rather than hidden branches.
- Large Data Mode uses file-backed intake, profile manifests, governed samples,
  chunked scoring, and certification metadata instead of keeping large files in
  Streamlit session state.
- Completed runs export the model artifact, resolved configuration, generated
  runner, artifact manifest, reproducibility metadata, diagnostic registry,
  decision summary, and traceability evidence.
- The user documentation is organized around setup, workflow use, validation
  review, artifacts, model types, statistical tests, large data, and
  GUI-to-code traceability.
