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
| `reports/interactive_report.html` | Standalone formal visual report with grouped diagnostics, companion charts, interpretation badges, and reviewer guidance. | Model developer, validator, business reviewer |
| `model/quant_model.joblib` | Saved fitted model object. | Developer, future scoring workflow, monitoring handoff |
| `config/run_config.json` | Effective configuration used for the run. | Developer, auditor, reproducibility review |
| `metadata/metrics.json` | Model metrics by split. | Developer, validator |
| `data/predictions/predictions.csv` or `data/predictions/predictions.parquet` | Row-level scores and predicted outputs. | Developer, reviewer, downstream user |
| `model/feature_importance.csv` | Coefficients, feature importance, or model-specific importance output. | Developer, validator |
| `metadata/statistical_tests.json` | Machine-readable statistical test results. | Validator, auditor |
| `reports/model_documentation_pack.md` | Development-facing written summary. | Developer, documentation owner |
| `reports/validation_pack.md` | Validator-facing summary and evidence index. | Validation and risk teams |
| `artifact_manifest.json` | Index of exported files and directories. | Auditor, technical reviewer |
| `metadata/step_manifest.json` | Ordered pipeline step record. | Technical reviewer |
| `metadata/run_debug_trace.json` | Per-step timing, shape snapshots, and failure details. | Developer, support, performance reviewer |
| `metadata/reproducibility_manifest.json` | Hashes, package versions, environment, and input metadata. | Auditor, reproducibility reviewer |
| `code/generated_run.py` | Python rerun script for running without the GUI. | Developer |
| `code/HOW_TO_RERUN.md` | Plain-English rerun instructions. | Developer, reviewer |

## Important Directories

| Directory | Meaning |
| --- | --- |
| `reports/` | Human-readable HTML, Markdown, DOCX, and PDF reports. |
| `model/` | Fitted model object, model summary, and feature importance. |
| `data/input/` | Input snapshot files when input export is enabled. |
| `data/predictions/` | Row-level and split-level prediction outputs. |
| `tables/` | Diagnostic tables grouped into topical subfolders. |
| `config/` | Resolved run configuration and offline configuration workbook. |
| `metadata/` | Metrics, statistical-test payloads, manifests, reproducibility data, and debug traces. |
| `workbooks/` | Optional Excel analysis workbook. |
| `code/code_snapshot/` | Copy of relevant source, examples, tests, and project metadata when code snapshot export is enabled. |
| `figures/` or equivalent figure folders | Separate chart HTML/PNG files when individual figure export is enabled. |
| `model_bundle_for_monitoring/` | Handoff bundle for the separate monitoring application, created for new fitted models. |
| `data/sample_development/` | Large Data Mode sample-development evidence. |
| `data/full_data_scoring/` | Large Data Mode full-file scoring outputs. |
| `metadata/large_data/` | Large Data Mode metadata, progress, and scoring summaries. |

There is intentionally no separate `json/` folder. Configuration JSON lives in
`config/`, while metrics, manifests, statistical-test payloads, and debug JSON
live in `metadata/`.

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

Individual figure HTML/PNG export is off by default. Turn it on only when
separate chart files are needed outside the full report.

The full `reports/interactive_report.html` still includes its grouped charts when
individual figure export is off. The separate figure files are only duplicate
distribution assets. If `Include enhanced report visuals` or `Advanced Visual
Analytics` are enabled, the optional separate figure files mirror those same
report-grade charts.

## Interactive Report Layout

The standalone report is organized like a formal model-development evidence
package:

- The cover section summarizes the run, target mode, warning count, splits,
  and primary KPIs.
- Diagnostic sections use the same taxonomy as the live Results & Artifacts
  view: Model Performance, Calibration / Thresholds, Stability / Drift, Sample
  / Segmentation, Feature Effects / Explainability, Statistical Tests, Feature
  Subset Search, Scorecard / Binning Workbench, Credit-Risk Development, Data
  Quality, Backtesting / Time Diagnostics, and Governance / Export Bundle.
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

## Feature-Subset Search Outputs

When `search_feature_subsets` is used, outputs are comparison-oriented rather
than final model-development artifacts. Important files include:

- `tables/feature_subset_search/subset_search_candidates.csv`
- `tables/feature_subset_search/subset_search_frontier.csv`
- `tables/feature_subset_search/subset_search_feature_frequency.csv`
- `tables/feature_subset_search/subset_search_significance_tests.csv`
- ROC/AUC and KS comparison charts in the report
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
- `reports/model_documentation_pack.md`
- `reports/validation_pack.md`

For a model validator:

- `reports/interactive_report.html`
- `metadata/metrics.json`
- `metadata/statistical_tests.json`
- `model/feature_importance.csv`
- `config/run_config.json`
- `reports/validation_pack.md`
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

- `model_bundle_for_monitoring/`
