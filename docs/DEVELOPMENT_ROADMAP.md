# Quant Studio Development Roadmap

This roadmap replaces the previous hardening roadmap for the current phase.
The focus of this phase is a third execution mode dedicated to feature-subset
comparison. The goal is to let a user evaluate candidate feature sets for a
single model family before committing to a normal model-development run.

## 1. Third Execution Mode: Feature Subset Search

Status: implemented

Delivered:

- a third execution mode named `search_feature_subsets`
- a typed config surface that defines the subset-search scope, size limits,
  ranking split, ranking metric, and significance-test behavior
- validation rules that keep this mode distinct from normal development and
  existing-model scoring

Primary code:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/config_io.py`
- `src/quant_pd_framework/gui_support.py`
- `app/streamlit_app.py`

## 2. Dedicated Subset-Search Pipeline

Status: implemented

Delivered:

- a dedicated pipeline step that enumerates eligible feature subsets for the
  currently selected model family
- ranking based on held-out comparison evidence rather than full development
  packaging
- guardrails that keep exhaustive search bounded to a practical candidate set

Primary code:

- `src/quant_pd_framework/steps/feature_subset_search.py`
- `src/quant_pd_framework/orchestrator.py`

## 3. Comparison-Only Metrics, Tests, And Visuals

Status: implemented

Delivered:

- candidate-level ranking outputs centered on:
  - ROC AUC
  - KS statistic
  - average precision
  - Brier score
  - log loss
  - feature-count parsimony
- ROC and KS comparison visuals for the leading candidates
- significance testing between the top-ranked candidates using paired-model
  comparison tests

Primary code:

- `src/quant_pd_framework/steps/feature_subset_search.py`
- `src/quant_pd_framework/diagnostic_frameworks.py`
- `src/quant_pd_framework/presentation.py`

## 4. Separate Export Surface For Subset Search

Status: implemented

Delivered:

- a dedicated export path for subset-search runs
- comparison-focused artifacts only, without normal development outputs such as
  fitted-model artifacts, validation packs, or full development documentation
- a standalone HTML report and markdown run summary tailored to feature-set
  comparison

Primary code:

- `src/quant_pd_framework/steps/export.py`
- `src/quant_pd_framework/presentation.py`

## 5. Dedicated GUI Experience For The Third Mode

Status: implemented

Delivered:

- subset-search controls in the GUI only when the third execution mode is
  selected
- a dedicated result view that emphasizes candidate ranking, frontier charts,
  ROC/KS comparison, and feature-inclusion frequency
- separation between subset-search evidence and normal model-development
  evidence

Primary code:

- `app/streamlit_app.py`
- `src/quant_pd_framework/gui_support.py`

## 6. Regression Coverage And Documentation Alignment

Status: implemented

Delivered:

- regression tests for config validation, orchestration, exports, and
  presentation behavior in the new mode
- documentation updates that describe when to use subset search versus a normal
  development run
- traceability updates so the new execution mode is visible in the repo's
  audit-facing documentation

Primary code:

- `tests/test_feature_subset_search_mode.py`
- `README.md`
- `docs/GUI_TO_CODE_TRACEABILITY_GUIDE.md`
- `EXECUTIVE_SUMMARY.txt`

## Notes

- The new mode is intended to answer a different question from a normal
  development run:
  `Which feature set should be taken forward for development?`
- The exported outputs intentionally stay separate from standard model
  development so the audit trail clearly distinguishes candidate-search
  evidence from final-model evidence.
