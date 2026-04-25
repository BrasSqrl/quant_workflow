# Quant Studio Performance And Maintainability Roadmap

This roadmap defines the current optimization phase. The objective is to keep
Quant Studio focused on model development and documentation while making large
runs faster, easier to debug, and easier to audit.

## 1. Run-Level Debug Trace

Status: implemented

Delivered:

- Add per-step timing, status, before/after shape snapshots, and failure details.
- Export `run_debug_trace.json` beside the existing run artifacts.
- Refresh the exported trace after artifact export completes so the export step
  itself appears in the trace.

Primary code:

- `src/quant_pd_framework/orchestrator.py`
- `src/quant_pd_framework/context.py`
- `src/quant_pd_framework/steps/export.py`

## 2. Export Profiles

Status: implemented

Delivered:

- Add `fast`, `standard`, and `audit` export profiles.
- Keep `standard` as the default behavior.
- Allow `fast` runs to skip expensive distribution assets such as workbooks,
  regulatory DOCX/PDF reports, input snapshots, and code snapshots while keeping
  the core report, metrics, tables, model, predictions, and manifests.

Primary code:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/config_io.py`
- `src/quant_pd_framework/steps/export.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `src/quant_pd_framework/streamlit_ui/config_builder.py`

## 3. Batched Explainability Scoring

Status: implemented

Delivered:

- Score modified PDP, ICE, ALE, two-way effect, marginal-effect, interaction,
  segmented-effect, and split-stability frames in batches.
- Reduce repeated preprocessing and model adapter calls for each grid point.
- Preserve the existing output tables and graph contracts.

Primary code:

- `src/quant_pd_framework/steps/diagnostics.py`
- `src/quant_pd_framework/diagnostics/scoring.py`

## 4. Lighter Report And Session Rendering

Status: implemented

Delivered:

- Avoid repeated Plotly theming work when a figure is already themed.
- Apply profile-aware HTML report limits for faster export paths.
- Store smaller Streamlit result snapshots for large runs while keeping full
  artifacts on disk.

Primary code:

- `src/quant_pd_framework/presentation.py`
- `src/quant_pd_framework/steps/export.py`
- `src/quant_pd_framework/streamlit_ui/state.py`

## 5. Focused Helper Modules

Status: implemented

Delivered:

- Move reusable export-profile decisions out of the artifact export step.
- Move batch scoring helpers out of the large diagnostics step.
- Keep public behavior stable while making debugging entry points clearer.

Primary code:

- `src/quant_pd_framework/export_profiles.py`
- `src/quant_pd_framework/diagnostics/scoring.py`

## 6. Performance Regression Tests

Status: implemented

Delivered:

- Verify debug trace artifacts and manifest links.
- Verify fast export profile behavior.
- Verify batched scoring helper behavior.
- Preserve existing artifact and Streamlit tests.

Primary code:

- `tests/test_performance_controls.py`

## 7. Documentation Alignment

Status: implemented

Delivered:

- Update README configuration references.
- Update GUI-to-code traceability documentation.
- Update executive summary with the new optimization and audit controls.

## 8. Parquet Intake And Data_Load Integration

Status: implemented

Delivered:

- Support `.parquet` and `.pq` files in upload and `Data_Load/` selection.
- Add Data_Load CSV-to-Parquet conversion for large local datasets.
- Keep input-source metadata in the reproducibility manifest.

Primary code:

- `src/quant_pd_framework/large_data.py`
- `src/quant_pd_framework/streamlit_ui/data.py`
- `src/quant_pd_framework/steps/ingestion.py`

## 9. Large Data Mode And Safe Defaults

Status: implemented

Delivered:

- Add a `Large data mode` GUI toggle.
- Default optional robustness and cross-validation refits off in large-data mode.
- Default per-figure HTML/PNG exports off in large-data mode.
- Reduce default diagnostic and export surfaces when large-data mode is enabled.

Primary code:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `src/quant_pd_framework/streamlit_ui/config_builder.py`

## 10. Memory Estimator And Guardrails

Status: implemented

Delivered:

- Add `large_data_memory_estimate` diagnostics.
- Add a configurable memory warning threshold.
- Preserve memory estimates in run metadata and exported diagnostic tables.

Primary code:

- `src/quant_pd_framework/large_data.py`
- `src/quant_pd_framework/steps/ingestion.py`

## 11. Dtype Optimization

Status: implemented

Delivered:

- Add optional numeric downcasting and low-cardinality string-to-category conversion.
- Export `dtype_optimization` audit rows with old/new dtypes and memory savings.

Primary code:

- `src/quant_pd_framework/large_data.py`
- `src/quant_pd_framework/steps/ingestion.py`

## 12. Diagnostic Sampling And Export Policies

Status: implemented

Delivered:

- Cap diagnostic sampling through `diagnostic_sample_rows` in large-data mode.
- Add tabular export policies for full, sampled, and metadata-only outputs.
- Export `tabular_export_policy` evidence for input and prediction artifacts.

Primary code:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/steps/diagnostics.py`
- `src/quant_pd_framework/steps/export.py`

## 13. Chunked CSV-To-Parquet Conversion

Status: implemented

Delivered:

- Convert CSV file-path inputs to Parquet in chunks using `pyarrow`.
- Record conversion metadata in run metadata and diagnostic tables.

Primary code:

- `src/quant_pd_framework/large_data.py`
- `src/quant_pd_framework/steps/ingestion.py`

## 14. Large-Output Parquet Exports

Status: implemented

Delivered:

- Add Parquet output for input snapshots, predictions, and diagnostic tables.
- Let users choose CSV, Parquet, or both.
- Keep full Parquet as the primary path when sampled CSV outputs are requested.

Primary code:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/steps/export.py`
- `src/quant_pd_framework/run.py`

## 15. SageMaker Sizing Documentation

Status: implemented

Delivered:

- Update SageMaker setup guidance for `Data_Load/`, Parquet, large-data mode,
  and practical RAM sizing for multi-GB inputs.

Primary docs:

- `docs/SAGEMAKER_SETUP.md`
- `SAGEMAKER_SETUP.txt`

## 16. File-Backed Large Data Engine

Status: implemented

Delivered:

- Add a file-backed `DatasetHandle` for Data_Load and CLI file-path runs.
- Read GUI previews without eager full-file pandas loads.
- Use DuckDB when available for sample selection, with PyArrow/Pandas fallback.
- Preserve the normal pandas workflow for standard runs.

Primary code:

- `src/quant_pd_framework/large_data.py`
- `src/quant_pd_framework/streamlit_ui/data.py`
- `src/quant_pd_framework/run.py`

## 17. Sample-Fit / Full-Data-Score Workflow

Status: implemented

Delivered:

- Load a governed training sample into pandas for model development.
- Replay deterministic preprocessing and learned imputation/transformation
  contracts on full-file chunks.
- Score the full file in chunks and write predictions directly to Parquet.
- Export `sample_development/`, `full_data_scoring/`, and
  `large_data_metadata/` folders for audit separation.

Primary code:

- `src/quant_pd_framework/steps/ingestion.py`
- `src/quant_pd_framework/steps/imputation.py`
- `src/quant_pd_framework/steps/transformations.py`
- `src/quant_pd_framework/steps/large_data_scoring.py`
- `src/quant_pd_framework/steps/export.py`

## 18. Maintainability Refactor For UI Execution

Status: implemented

Delivered:

- Move run execution input selection and spinner copy out of the main Streamlit
  controller.
- Move artifact-location summarization out of the result panels so paths are
  rendered as readable tables instead of raw JSON blocks.
- Keep the orchestration and model behavior unchanged while creating smaller
  debugging seams for future UI and artifact work.

Primary code:

- `src/quant_pd_framework/streamlit_ui/run_execution.py`
- `src/quant_pd_framework/streamlit_ui/artifact_summary.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `src/quant_pd_framework/streamlit_ui/results.py`

## 19. User-Facing Error Recovery

Status: implemented

Delivered:

- Classify common workflow failures into business-readable categories such as
  memory pressure, target setup, date/split setup, feature treatment, model
  convergence, large-data staging, and incomplete artifact export.
- Preserve the technical traceback in an expandable detail panel for debugging.
- Replace bare exception strings in the GUI with likely cause and recommended
  recovery actions.

Primary code:

- `src/quant_pd_framework/streamlit_ui/error_guidance.py`
- `src/quant_pd_framework/streamlit_ui/workflow_feedback.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`

## 20. Workflow UX Polish

Status: implemented

Delivered:

- Add a compact execution-plan panel before run execution that summarizes
  execution mode, model family, data source, fit strategy, and export profile.
- After successful runs, surface key output locations immediately in the GUI,
  including the run folder, interactive report, model object, config, debug
  trace, and predictions.
- Replace raw artifact JSON in the Governance panels with an organized artifact
  table that is easier to scan during validation or audit review.

Primary code:

- `src/quant_pd_framework/streamlit_ui/workflow_feedback.py`
- `src/quant_pd_framework/streamlit_ui/artifact_summary.py`
- `src/quant_pd_framework/streamlit_ui/results.py`

## 21. Diagnostics Module Decomposition

Status: implemented

Delivered:

- Move reusable diagnostic asset helpers out of the monolithic diagnostics step.
- Centralize diagnostic sampling, plotting row caps, asset-name sanitization,
  PSI calculation, and visual theming in `diagnostics/assets.py`.
- Keep model/statistical behavior stable while reducing the helper footprint in
  `steps/diagnostics.py`.

Primary code:

- `src/quant_pd_framework/diagnostics/assets.py`
- `src/quant_pd_framework/steps/diagnostics.py`

## 22. Export Layout Decomposition

Status: implemented

Delivered:

- Add a centralized export path layout object so artifact filenames and folders
  are built in one place.
- Route `ArtifactExportStep` through the path layout helper to reduce repeated
  path construction and make future export refactors safer.

Primary code:

- `src/quant_pd_framework/export_layout.py`
- `src/quant_pd_framework/steps/export.py`

## 23. Workflow Profiling And Benchmarks

Status: implemented

Delivered:

- Add a saved-run profiler that captures elapsed runtime, peak traced memory,
  slowest pipeline steps, artifact counts, diagnostic counts, and output paths.
- Add a synthetic Large Data Mode benchmark script that generates Parquet data,
  runs sample-fit/full-score execution, and writes benchmark JSON.

Primary code:

- `scripts/profile_workflow.py`
- `scripts/benchmark_large_data.py`

## 24. Diagnostic Registry

Status: implemented

Delivered:

- Add a runtime diagnostic registry that maps diagnostic names to config paths,
  families, expected tables, expected figures, target-mode restrictions,
  label requirements, and large-data behavior.
- Export a `diagnostic_registry` table with each run so users can see which
  diagnostics were emitted, disabled, or skipped.

Primary code:

- `src/quant_pd_framework/diagnostics/registry.py`
- `src/quant_pd_framework/steps/diagnostics.py`

## 25. Large-Data Scoring Resilience

Status: implemented

Delivered:

- Add chunk-progress metadata for full-data scoring.
- Wrap chunk failures with chunk number and progress-file guidance.
- Harden chunked Parquet writing so later chunks are aligned to the first
  schema, missing columns are filled, extra columns are ignored, and numeric
  coercion handles mixed chunk types.

Primary code:

- `src/quant_pd_framework/steps/large_data_scoring.py`
- `tests/test_large_data_controls.py`

## 26. Config Serialization Consolidation

Status: implemented

Delivered:

- Move JSON-friendly config serialization into a dedicated helper module.
- Publish the canonical `FrameworkConfig` section list so config round-trip
  tests can detect missing serialization coverage as the schema grows.

Primary code:

- `src/quant_pd_framework/config_serialization.py`
- `src/quant_pd_framework/config.py`
- `tests/test_large_data_controls.py`
