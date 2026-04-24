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
