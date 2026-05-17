# Artifact Map

This guide explains where outputs are written, what the major files mean, and
which users usually need them.

## Run Folder Location

Artifacts are written under the configured artifact root, usually:

```text
artifacts/
```

Each run creates a timestamped folder such as:

```text
artifacts/run_2026-04-24_15-42-10_UTC/
```

The GUI also surfaces key output locations in Step 4, `Results & Artifacts`,
and summarizes the completed run in Step 5, `Decision Summary`.

## Most Important Files

| File | Meaning | Primary user |
| --- | --- | --- |
| `START_HERE.md` | First-read orientation for the run folder and audience-specific next steps. | All users |
| `reports/decision_summary.md` | Decision-ready scorecard with recommendation, key metrics, issues, feature drivers, and evidence links. | Model developer, validator, business reviewer |
| `reports/model_development_dossier.md` | Audit-ready narrative tying purpose, data, target, feature lineage, model methodology, validation evidence, limitations, and primary artifacts together. | Model developer, validator, auditor |
| `reports/interactive_report.html` | Standalone formal visual report with grouped diagnostics, companion charts, interpretation badges, and reviewer guidance. | Model developer, validator, business reviewer |
| `model/quant_model.joblib` | Saved fitted model object. | Developer, future scoring workflow, monitoring handoff |
| `config/run_config.json` | Effective configuration used for the run. | Developer, auditor, reproducibility review |
| `metadata/metrics.json` | Model metrics by split. | Developer, validator |
| `data/predictions/predictions.csv` or `data/predictions/predictions.parquet` | Row-level scores and predicted outputs. Compact by default so it does not duplicate every feature column. | Developer, reviewer, downstream user |
| `model/feature_importance.csv` | Coefficients, feature importance, or model-specific importance output. | Developer, validator |
| `model/feature_lineage_map.csv` | Direct CSV copy of the final feature-lineage map for quick review. | Developer, validator, auditor |
| `metadata/statistical_tests.json` | Machine-readable statistical test results. | Validator, auditor |
| `reports/model_documentation_pack.md` | Development-facing written summary. | Developer, documentation owner |
| `reports/validation_pack.md` | Validator-facing summary and evidence index. | Validation and risk teams |
| `artifact_manifest.json` | Index of exported files and directories. | Auditor, technical reviewer |
| `tables/governance/validation_checklist.*` | Checklist of completed, attention-needed, and not-applicable validation evidence areas. | Validator, model developer |
| `tables/governance/evidence_traceability_map.*` | Maps common review questions to the artifact or table that answers them. | Validator, auditor, technical reviewer |
| `tables/governance/feature_lineage_map.*` | Maps final model terms back to source features, transformations, imputation, variable-selection rationale, importance, and documentation fields. | Validator, auditor, technical reviewer |
| `tables/governance/report_payload_audit.*` | Records embedded report charts kept, downsampled, or skipped by report-size controls. | Developer, validator, support |
| `metadata/step_manifest.json` | Ordered pipeline step record. | Technical reviewer |
| `metadata/run_debug_trace.json` | Run start/completion time, total elapsed runtime, per-step timing, shape snapshots, memory estimates, and failure details. | Developer, support, performance reviewer |
| `metadata/audit_events.jsonl` | Run-scoped audit events copied from the global audit log, including profile actions, run start/completion/failure, package downloads, and review saves. | Validator, auditor, support |
| `checkpoints/checkpoint_manifest.json` | Stage manifest for checkpointed execution, including stage status, elapsed time, checkpoint-retention policy, pruned context evidence, and optional-stage failures. | Developer, support, auditor |
| `metadata/large_data/dataset_profile.json` | File-backed source profile with schema, preview metadata, row count where available, approximate null/cardinality/quantile evidence, and profile-cache hit/miss details. | Developer, support, auditor |
| `metadata/large_data/execution_plan.json` | Per-stage Large Data Mode plan showing whether each stage is file-backed, full-data, sample-based, in-memory, blocked, or override-driven. | Developer, support, auditor |
| `metadata/large_data/large_data_transformation_contract.json` | Transformation replay contract showing which governed transformations are compiled, sample-only, unsupported, or blocked in Large Data Mode. | Developer, support, auditor |
| `metadata/large_data/feature_screening_manifest.json` | Manifest for advisory large-data feature pre-screening and any auto-applied exclusions. | Developer, validator, auditor |
| `tables/feature_screening/large_data_feature_screening.*` | Advisory missingness, variance, cardinality, target-relationship, IV, and PSI evidence for large-data features. | Developer, validator, auditor |
| `metadata/large_data/partitioned_dataset_manifest.json` | Split-aware governed-sample Parquet partition manifest. | Developer, support, auditor |
| `metadata/large_data/prepared_dataset_manifest.json` | File-backed Large Data Mode bridge showing source, staged path, sample path, projected columns, target metadata, profile cache key, partition paths, artifact-size estimates, and row count where available. | Developer, support, auditor |
| `metadata/reproducibility_manifest.json` | Hashes, package versions, environment, and input metadata. | Auditor, reproducibility reviewer |
| `code/generated_run.py` | Python rerun script for running without the GUI. | Developer |
| `code/HOW_TO_RERUN.md` | Plain-English rerun instructions. | Developer, reviewer |

Step 5 also has an on-demand `Download LLM Package` button in the Export card.
That download creates a `.zip` for LLM-assisted model methodology drafting. It
includes LLM-readable context files, a model facts digest, section-level
evidence maps, evidence-backed approved claims, a documentation gap register, a
detailed regulatory crosswalk, model-type writing guidance, operator prompt
variants, citable-evidence and do-not-cite indexes, tone profiles, citation
rules, feature/metric/chart interpretation briefs, a target document schema,
evidence-strength policy, completion rules, controlled vocabulary,
draft-validation rules, document quality rubric, redaction policy, lightweight
draft validator script, deterministic DOCX helper scripts, a human review
checklist, a default model methodology outline, a table-of-contents drop zone
for institution-specific templates, diagnostic table previews, generated-run
code, run configuration, selected non-row-level evidence from the completed run,
DOCX build instructions, a model document style guide, a DOCX quality checklist,
figure and table placement manifests, section-specific evidence folders, a
package build profile, capped lightweight HTML chart assets, and a small
document-ready PNG subset. The downloaded folder is organized into numbered
folders from `00_START_HERE` through `09_generated_outputs` so users can
separate evidence, prompts, controls, visual assets, tools, and generated
outputs. If `Download
Individual Images` has already been prepared in the same UI session, the LLM
package reuses eligible chart files instead of rendering them again. It intentionally
excludes raw input snapshots, row-level predictions, serialized model binaries,
monitoring handoff bundles, and full code snapshots by default.

Step 5 also has an on-demand `Download Individual Images` button. That download
creates a separate chart zip after the run instead of adding PNG/HTML rendering
time to the model workflow. The zip contains `png/`, `html/`, one shared
`html/plotly.min.js`, and `figure_manifest.json`.

Step 5 also has an on-demand `Download OM Package` button for completed
`fit_new_model` runs. That download creates the `model_bundle_for_monitoring`
zip for the separate ongoing-monitoring application. The bundle is no longer
written automatically during every model fit, which avoids extra file copying
and Parquet-to-CSV conversion during the main workflow.

## Important Directories

| Directory | Meaning |
| --- | --- |
| `reports/` | Human-readable HTML, Markdown, DOCX, and PDF reports. |
| `model/` | Fitted model object, model summary, feature importance, and feature-lineage CSV. |
| `data/input/` | Input snapshot files when input export is enabled. |
| `data/predictions/` | Row-level and split-level prediction outputs. |
| `tables/` | Diagnostic tables grouped into topical subfolders. |
| `config/` | Resolved run configuration and offline configuration workbook. |
| `metadata/` | Metrics, statistical-test payloads, manifests, reproducibility data, and debug traces. |
| `checkpoints/` | Disk-backed stage manifest and, when retained, context checkpoints used by full and step-by-step workflow execution. |
| `workbooks/` | Optional Excel analysis workbook. |
| `code/code_snapshot/` | Copy of relevant source, examples, tests, and project metadata when code snapshot export is enabled. |
| `figures/` or equivalent figure folders | Separate chart HTML/PNG files for legacy code-driven runs. GUI users should use Step 5 `Download Individual Images` to generate these files on demand. |
| `data/sample_development/` | Large Data Mode sample-development evidence. |
| `data/sample_development/partitioned/` | Split-aware governed-sample Parquet partitions when partitioning is enabled. |
| `data/full_data_scoring/` | Large Data Mode full-file scoring outputs. |
| `metadata/large_data/` | Large Data Mode metadata, progress, and scoring summaries. |
| `artifacts/_background_jobs/` | Transient detached-worker manifests, stdout/stderr logs, and bounded Streamlit snapshots for active or recently completed Large Data Mode background runs. |
| `artifacts/_job_queue/` | Optional local worker-service queue used by `quant-pd-worker` when Large Data Mode worker service dispatch is enabled or detected. |
| `artifacts/_run_registry/` | File-backed run registry and global audit event log used by the Step 4 Run Registry panel. |
| `artifacts/certification/<timestamp>/` | CLI-generated Large Data Mode certification evidence with summary reports, scenario results, thresholds, environment profile, model capability matrix, run index, and linked benchmark folders. |

There is intentionally no separate `json/` folder. Configuration JSON lives in
`config/`, while metrics, manifests, statistical-test payloads, and debug JSON
live in `metadata/`.

## Run Registry And Audit Files

Step 4, `Results & Artifacts`, includes a `Run Registry` panel. It indexes runs
from the configured artifact root so users can find prior output folders,
metrics, reviewer status, and key artifacts without browsing the filesystem.

Global registry files live under:

```text
artifacts/_run_registry/
```

Key files:

| File | Meaning |
| --- | --- |
| `run_registry.json` | Searchable index of completed and known failed runs. |
| `audit_events.jsonl` | Append-only global audit log for high-value GUI and workflow actions. |

Each run folder also includes `metadata/audit_events.jsonl`, a run-scoped copy
of matching audit events. The audit log records major actions such as data
selection, profile load/save, schema/transformation table fingerprints, run
start/completion/failure, package downloads, and reviewer record saves. It does
not store raw input rows, row-level predictions, model binaries, passwords,
tokens, AWS credentials, or secrets.

## Large Data Certification Artifacts

The `quant-pd-certify-large-data` CLI and
`python scripts/certify_large_data.py` wrapper write acceptance evidence under:

```text
artifacts/certification/<timestamp>/
```

Key files:

| File | Meaning |
| --- | --- |
| `certification_summary.json` | Machine-readable suite result, status counts, thresholds, and overall pass/fail flag. |
| `certification_summary.md` | Human-readable certification summary. |
| `certification_report.html` | Standalone HTML report for reviewers. |
| `scenario_results.csv` | Scenario-level model, source, capability, timing, memory, artifact-size, threshold, and result-status evidence. |
| `model_capability_matrix.csv` | Static large-data model capability classification used to interpret benchmark evidence. |
| `effective_thresholds.json` | Thresholds after default suite and CLI overrides are applied. |
| `environment_profile.json` | Python, platform, CPU, memory, and working-directory evidence. |
| `run_index.csv` | Links to each scenario benchmark JSON, Markdown summary, and run artifact folder. |
| `benchmark_runs/` | Underlying scenario benchmark folders. |

`config/configuration_template.xlsx` is the offline review workbook. It includes
editable sheets for schema, feature dictionary, transformations, manual feature
review, and scorecard bin overrides. It also includes instruction,
allowed-value, transform-catalog, example, and required-column sheets plus
header comments and dropdown validation so offline edits are easier to upload
safely.

## Checkpointed Execution Artifacts

Quant Studio runs through checkpointed stages by default. The normal `Run full
workflow` button still performs the whole workflow, but each major stage runs
from a saved context so Python can release memory between stages. The
step-by-step mode uses the same files and runs one stage per click.

The `checkpoints/` folder contains:

- `00_initial_context.joblib`, the starting context when checkpoint retention is enabled
- `NN_stage_name.joblib`, the context after each completed stage when retained
- `checkpoint_manifest.json`, the stage status and latest checkpoint pointer

`Keep all checkpoints` is off by default to control storage. When it is off,
Quant Studio deletes stale `.joblib` context files after a newer safe checkpoint
exists and removes the final context checkpoint after successful completion. The
manifest remains available for proving which stages completed, which optional
stages failed, and which checkpoint contexts were pruned.

In Large Data Mode, checkpoint contexts spill large dataframe fields to
`checkpoints/tables/<checkpoint_name>/` as Parquet when the field exceeds the
configured `Large-data max in-memory rows` control. The `.joblib` context stores
lightweight table references and is rehydrated when the next stage loads the
checkpoint. This reduces duplicate checkpoint size without changing the stage
contract used by the modeling code.

Turn `Keep all checkpoints` on before the run when a developer or support user
needs to inspect context checkpoints after execution or rerun a later stage
without rebuilding earlier work. Checkpoint contexts are execution evidence, not
the primary business-facing report.

For the full list of checkpoint stages, their purpose, required versus optional
status, and the controls that influence each one, see
[Checkpoint Stage Guide](../CHECKPOINT_STAGE_GUIDE.md).

## Table Subfolders

Diagnostic tables are grouped by review topic:

- `tables/diagnostics/`
- `tables/model_performance/`
- `tables/calibration/`
- `tables/stability/`
- `tables/explainability/`
- `tables/statistical_tests/`
- `tables/backtesting/`
- `tables/governance/`
- `tables/feature_subset_search/`
- `tables/scorecard/`

## Export Profiles

| Profile | Use |
| --- | --- |
| `fast` | Faster iteration. Keeps core outputs and skips heavier distribution assets. |
| `standard` | Normal default. Balanced evidence package. |
| `audit` | Full review path when maximum supporting evidence is desired. |

Individual chart HTML/PNG export is on demand in the GUI. Use Step 5
`Download Individual Images` only when separate chart files are needed outside
the full report. The generated zip includes PNG files plus lightweight HTML
files that share one `plotly.min.js` file.

The full `reports/interactive_report.html` still includes grouped charts. The
separate chart zip is a duplicate distribution asset and is intentionally not
created during the model run. If `Include enhanced report visuals` or
`Advanced Visual Analytics` are enabled, the on-demand chart package mirrors
those same report-grade charts.

The Step 2 `Diagnostics & Exports` report-size controls protect the embedded
chart payload in `reports/interactive_report.html`:

- `Max points per report chart`
- `Max MB per report chart`
- `Max total report chart MB`

When a chart exceeds those limits, Quant Studio downsamples or skips that chart
inside the standalone HTML report and writes the decision to
`tables/governance/report_payload_audit.*`. Full diagnostic tables remain in
`tables/` even when the HTML preview is capped.

Prediction exports are compact by default. They preserve audit identifiers,
date/entity fields, target/split fields, low-cardinality segment fields, and
model score outputs. Turn off `Compact prediction exports` only when downstream
users explicitly need the full modeled feature matrix repeated in each scored
row.

When `Large tabular export policy = sampled`, CSV prediction files are sampled
for reviewer convenience while Parquet prediction files remain full.

## Interactive Report Layout

The standalone report is organized like a formal model-development evidence
package:

- The cover section summarizes the run, target mode, warning count, splits,
  and primary KPIs.
- A sticky report navigation bar lets reviewers jump by section instead of
  scrolling through the entire file. If JavaScript is unavailable, the sections
  remain visible as a readable long-form report.
- The `Executive Landing Page` is the default first view. It combines run
  context, primary metrics, review priorities, and an evidence map that links to
  each major section.
- Search filters report cards by chart, table, feature name, warning text, or
  section wording. Clearing the search restores the selected section.
- Reviewer mode can switch between validator, executive, and technical views.
  Executive mode hides supporting evidence so a meeting can focus on high-value
  outputs first.
- Compact view reduces descriptive text and locator rows for faster screen
  review. Show-all / print view opens every section sequentially for audit or
  PDF printing.
- Section badges show `Pass`, `Watch`, `Fail`, `Ready`, or `Not Run` status
  plus chart and table counts. These badges are navigation aids; the underlying
  diagnostic tables remain the source of record.
- Diagnostic sections use the same taxonomy as the live Results & Artifacts
  view: Model Performance, Calibration / Thresholds, Stability / Drift, Sample
  / Segmentation, Feature Effects / Explainability, Statistical Tests, Feature
  Subset Search, Scorecard / Binning Workbench, Credit-Risk Development, Data
  Quality, Backtesting / Time Diagnostics, and Governance / Export Bundle.
- Each chart and table card includes an artifact locator with the internal key
  and the likely exported file location. Table links use CSV for non-Parquet
  Step 1 inputs and Parquet for Parquet Step 1 inputs.
- Sections split cards into `Featured Evidence` and `Supporting Evidence` so a
  reviewer can start with the most decision-useful outputs without losing the
  audit trail.
- Companion charts are generated from existing run outputs where possible, such
  as annotated ROC, precision-recall, KS separation, calibration residual bars,
  PSI/VIF threshold bars, missingness heatmaps, feature-importance waterfalls,
  score-distribution violins, segment dumbbells, scenario tornados, and
  cross-validation violins.
- The optional `Advanced Visual Analytics` toggle adds a separate exploratory
  section with contribution beeswarms, interaction heatmaps, PDP/ICE matrices,
  segment calibration panels, score ridgelines, temporal score streams,
  correlation networks, lift/gain heatmaps, risk treemaps, model-comparison
  radar charts, scenario waterfalls, and feature-importance lollipop charts.
- Chart badges and guidance text provide practical interpretation context
  without changing the underlying statistical calculations.
- The `Include enhanced report visuals` toggle controls whether companion
  charts are added. Turn it off for faster development runs when the base
  diagnostics and tables are enough.
- `Advanced Visual Analytics` is off by default because it is exploratory and
  adds report-rendering work. Turn it on when richer model-insight visuals are
  worth the additional runtime and report length.
- Full tables remain available in the exported table files even when the HTML
  report previews only the first rows.
- Report-size controls may downsample or omit embedded charts from the HTML
  file to keep it shareable; review `report_payload_audit` for exact details.

## Feature-Subset Search Outputs

When `search_feature_subsets` is used, outputs are comparison-oriented rather
than final model-development artifacts. Important files include:

- `tables/feature_subset_search/subset_search_leaderboard.*`
- `tables/feature_subset_search/subset_search_top_candidate_comparison.*`
- `tables/feature_subset_search/subset_search_selection_rationale.*`
- `tables/feature_subset_search/subset_search_candidate_risk_flags.*`
- `tables/feature_subset_search/subset_search_candidates.*`
- `tables/feature_subset_search/subset_search_frontier.*`
- `tables/feature_subset_search/subset_search_feature_frequency.*`
- `tables/feature_subset_search/subset_search_significance_tests.*`
- `tables/feature_subset_search/subset_search_contribution_consistency.*`
- `tables/feature_subset_search/subset_search_redundancy_diagnostics.*`
- `tables/feature_subset_search/subset_search_excluded_feature_insights.*`
- `tables/feature_subset_search/subset_search_feature_family_view.*`
- `tables/feature_subset_search/subset_search_transformation_effectiveness.*`
- segment and time performance tables when eligible fields are available
- ROC/AUC, KS, leaderboard, calibration, risk-flag, transformation, and feature
  family charts in the report
- winning subset coefficient and feature summary where available

Use these outputs to choose candidate features, then run `fit_new_model` to
produce the final model package.

## Existing-Model Scoring Outputs

When `score_existing_model` is used:

- the model object is loaded from the provided path
- predictions are exported for the new data
- validation metrics are produced only when labels are available
- score-only documentation is produced when labels are not available

The most important files are `data/predictions/predictions.*`,
`config/run_config.json`, `reports/interactive_report.html`,
`reports/validation_pack.md`, and the manifests.

## What To Send To Which Audience

For a business reviewer:

- `reports/interactive_report.html`
- `reports/model_development_dossier.md`
- `reports/model_documentation_pack.md`
- `reports/validation_pack.md`

For a model validator:

- `reports/interactive_report.html`
- `metadata/metrics.json`
- `metadata/statistical_tests.json`
- `model/feature_importance.csv`
- `model/feature_lineage_map.csv`
- `config/run_config.json`
- `reports/validation_pack.md`
- `tables/governance/feature_lineage_map.*`
- `metadata/reproducibility_manifest.json`

For a technical reviewer:

- `config/run_config.json`
- `artifact_manifest.json`
- `metadata/step_manifest.json`
- `metadata/run_debug_trace.json`
- `code/generated_run.py`
- `code/code_snapshot/` if exported

For future scoring:

- `model/quant_model.joblib`
- `config/run_config.json`
- `code/generated_run.py`

For the separate monitoring application:

- Step 5 `Download OM Package`
