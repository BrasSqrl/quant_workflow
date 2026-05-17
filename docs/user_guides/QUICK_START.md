# User Quick Start Guide

This guide gets a user from launch to a completed first model run. It uses the
bundled sample data so the workflow can be learned before moving to a real
dataset. The bundled sample is a 1,000-row synthetic commercial-loan panel:
100 loans observed across 10 quarter-end reporting periods with borrower
attributes, loan behavior fields, financial statement items, ratios, and a
binary `default_status` target.

For a visual, step-by-step PD logistic regression example, open
[PD Logistic Regression Walkthrough](./PD_LOGISTIC_REGRESSION_WALKTHROUGH.html).

## What You Will Produce

A successful first run writes a timestamped folder under `artifacts/` with:

- `START_HERE.md`
- `reports/interactive_report.html`
- `reports/decision_summary.md`
- `reports/model_development_dossier.md`
- `model/quant_model.joblib`
- `model/feature_lineage_map.csv`
- `config/run_config.json`
- `metadata/metrics.json`
- `data/predictions/predictions.csv` or `data/predictions/predictions.parquet`
- `metadata/run_debug_trace.json` with timing and memory estimates
- `checkpoints/checkpoint_manifest.json` with stage status and checkpoint-retention evidence
- `reports/model_documentation_pack.md`
- `reports/validation_pack.md`
- `artifact_manifest.json`

## Launch The App

On Windows, use:

```powershell
.\launch_gui.bat
```

If dependencies need to be prepared first, use:

```powershell
.\setup_gui.bat
```

On macOS, use:

```bash
bash scripts/bootstrap_macos.sh
bash scripts/run_macos_streamlit.sh
```

For macOS details, follow [macOS Setup](../MACOS_SETUP.md). In SageMaker or
Linux, follow [SageMaker Setup](../SAGEMAKER_SETUP.md).

## The Five-Step Workflow

Quant Studio opens to Step 1 by default.

| Step | Purpose |
| --- | --- |
| `1 Dataset & Schema` | Load data, preview rows, review the data contract scorecard, check leakage flags and schema fingerprint, define column roles, edit feature dictionary, preview transformations, and exchange the governance workbook. |
| `2 Model Configuration` | Choose execution mode, preset, model family, split strategy, diagnostics, export profile, governance options, explainability, scenarios, and documentation settings, then review model suitability, configuration risk, and rough runtime/output estimates. |
| `3 Readiness Check & Run` | Review issues, preflight summary, the resource planner/run cost estimate, guardrails, configuration diffs, and run the workflow. |
| `4 Results & Artifacts` | Review outputs, charts, tables, artifact locations, reviewer notes, and model-card downloads. |
| `5 Decision Summary` | Review the recommendation, primary metric scorecard, decision issues, top feature drivers, feature lineage, dossier, and supporting evidence index. |

## First Successful Run

1. Open Step 1, `Dataset & Schema`.
2. Leave bundled sample data selected.
3. Open `Data Review` and check the data contract, potential leakage flags, and schema fingerprint.
4. Open `Column Designer`.
5. Confirm `default_status` is the target source.
6. Confirm `as_of_date` is the date column.
7. Confirm `loan_id` is the identifier for the panel.
8. Confirm `legacy_text_field` is disabled or ignored.
9. Open `Transformation Studio` after `Feature Dictionary` if you want to add governed transforms. Start with `Recommendations` or `Recipe Library`, use `Custom Builder` for one-off transforms, then confirm status in `Pipeline Review`.
10. Open Step 2, `Model Configuration`.
11. Use `Execution mode = fit_new_model`.
12. Use `Model type = scorecard_logistic_regression`.
13. Use `Target mode = binary`.
14. Use `Data structure = panel`.
15. Review `Model Suitability Explainer`, `Configuration Risk Score`, and `Runtime / Artifact Size Estimate` after the configuration resolves cleanly.
16. Keep `Split strategy = Automatic` and the default train, validation, and
    test split for the first run. Use `Date cutoff`, `Explicit date windows`,
    or `Custom split column` when you need a documented out-of-time holdout.
17. Keep `Enable scorecard workbench` on so WoE bins, scorecard points, and
    reason-code outputs are produced.
18. Use Step 5 `Download Individual Images` after the run if separate chart
    PNG/HTML files are needed. They are generated on demand instead of during
    model execution.
19. Keep `Include enhanced report visuals` on for a polished first report, or turn it off for faster iteration runs.
20. Leave `Advanced Visual Analytics` off for the first run unless you want the
    extra exploratory chart section.
21. Leave the report-size controls at their defaults unless the HTML report must embed denser charts.
22. Leave `Compact prediction exports` on unless you need every modeled feature repeated in the scored output.
23. Leave `Retain full diagnostic working dataframe` off unless the machine has enough RAM and full-row diagnostic tables are required.
24. Leave `Keep all checkpoints` off unless support needs every saved context retained after the run.
25. Open Step 3, `Readiness Check & Run`.
26. Resolve blocking readiness issues if any appear.
27. Review `Resource Planner / Run Cost Estimate` for memory, Large Data Mode, checkpoint retention, high-cost options, report-visual, and disk-output warnings.
28. Leave `Workflow run style = Run full workflow` for the first run.
29. Click `Run Quant Model Workflow`.
30. Watch the `Run Status` panel for elapsed time, current stage, step
    progress, and the `Checkpoint Flow` chart. The flow chart highlights
    the active major stage and keeps completed, optional-failed, and failed
    stages visually distinct.
31. Use `Run checkpointed step-by-step` only when you want to run one saved
    stage per click, inspect failures between stages, or retry optional
    diagnostic groups without refitting the model.
32. Open Step 4, `Results & Artifacts`.
33. Review the overview, model performance, calibration, scorecard / binning,
    governance, and artifact explorer sections.
34. Open Step 5, `Decision Summary`.
35. Review the recommendation, decision issues, primary metrics, feature drivers, feature lineage, validation checklist, evidence index, traceability map, and dossier.
36. If separate chart files are needed, use Step 5 `Export` ->
    `Download Individual Images`.
37. If you need an LLM-assisted model methodology draft, use Step 5 `Export` ->
    `Download LLM Package`. The zip includes curated non-row-level evidence,
    section evidence maps, approved claims, documentation gaps, schema and
    evidence-strength controls, citable-evidence and do-not-cite indexes,
    completion rules, controlled vocabulary, draft-validation rules, operator
    prompt variants, citation rules, interpretation briefs, DOCX build
    instructions, a model document style guide, a DOCX quality checklist,
    figure and table placement manifests, deterministic DOCX helper scripts,
    section evidence folders, capped HTML and document-ready PNG chart evidence,
    a package build profile, a model facts digest, a quality rubric, a review
    checklist, and a table-of-contents drop zone for your group-specific
    document template. The package is organized into numbered folders so
    evidence, prompts, controls, visual assets, tools, and generated outputs are
    easy to separate.

## First Real-Data Run

For real data, keep the same workflow but change Step 1:

- upload CSV, Excel, or Parquet, or place the file in `Data_Load/` and choose `Select from Data_Load`
- rebuild the column roles around the real target, date, identifier, and feature fields
- update feature dictionary definitions and documentation fields before export
- use [Data Requirements Guide](./DATA_REQUIREMENTS.md) if the data does not load or roles are unclear

## Save A Profile

After a good setup in Step 2, save a configuration profile. Profiles preserve
the setup choices without storing raw data rows. They are useful when a user
wants the same modeling setup after closing and reopening the app.

## Offline Review Workbook

Step 1 includes a `Template Workbook` section. Download this workbook when
schema, feature dictionary, transformation, manual-review, or scorecard-bin
edits need to happen outside the GUI.

The workbook includes:

- editable sheets for `schema`, `feature_dictionary`, `transformations`,
  `feature_review`, and `scorecard_overrides`
- an `instructions` sheet that explains how each sheet should be used
- an `allowed_values` sheet for roles, dtypes, missing policies,
  transformation types, manual-review decisions, and boolean fields
- a `transform_catalog` sheet explaining each transformation, recipe group,
  when to use it, required parameters, large-data status, and expected output type
- an `examples` sheet with realistic row patterns
- a `required_columns` sheet that shows which headers must remain compatible
  for upload parsing

Do not rename the editable sheet headers. The workbook can be uploaded back
through the same Step 1 section after offline review is complete.

## Where To Go Next

- Use [Execution Mode Decision Guide](./EXECUTION_MODE_DECISION_GUIDE.md) if you are not sure whether to fit, score, or search feature subsets.
- Use [Artifact Map](./ARTIFACT_MAP.md) to understand output files.
- Use [Configuration Cookbook](./CONFIGURATION_COOKBOOK.md) for common PD, LGD, CECL, CCAR, existing-model, and large-data setups.
