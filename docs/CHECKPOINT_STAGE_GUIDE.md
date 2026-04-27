# Checkpoint Stage Guide

This guide explains the checkpoint stages shown in the Step 3 `Run Status`
panel and exported in `checkpoints/checkpoint_manifest.json`.

Checkpoint stages are execution boundaries. They are not the same as the five
GUI workflow tabs. The five GUI steps organize user setup and review. The
checkpoint stages organize what happens after the user clicks `Run Quant Model
Workflow`.

## What Required And Optional Mean

The `Checkpoint Flow` chart marks some stages as optional.

- Required stages are critical to the run. If a required stage fails, the run
  fails.
- Optional stages are non-blocking diagnostic or large-data add-on stages. If
  an optional stage fails, the manifest records the failure, but the app can
  continue to later stages when possible.

Optional does not always mean there is one single on/off toggle. Some optional
stages are controlled by a group of Step 2 settings, diagnostic-suite choices,
workspace mode, target mode, and whether the data supports the diagnostic.

## Normal Workflow Stages

These stages are used for `fit_new_model` and `score_existing_model`.

| Stage ID | UI label | Required? | What it does | Main controls |
| --- | --- | --- | --- | --- |
| `prepare_data` | Prepare data | Required | Loads data, applies schema, creates target, validates setup, cleans data, builds simple derived features, creates train/validation/test splits, runs assumption checks, fits imputation rules, applies governed transformations, and performs variable selection. | Step 1 `Dataset & Schema`; Step 2 `Core Setup`, `Split Strategy`, `Data Preparation`, `Selection & Documentation`; Column Designer; Transformations table. |
| `fit_model` | Fit or load model | Required | Fits the configured model for `fit_new_model`, or loads the existing model artifact for `score_existing_model`. | Step 2 `Core Setup`; Step 2 `Model Settings`; existing model path and existing config path when scoring an existing model. |
| `score_evaluate` | Score and evaluate | Required | Scores train/validation/test splits, calculates evaluation metrics, runs challenger comparison when enabled, and builds backtesting outputs. | Step 2 `Model Settings`; Step 2 `Challengers & Policies`; Step 2 calibration and backtest-related diagnostic settings. |
| `diagnostics_overview` | Diagnostics: overview | Required | Builds the core diagnostics overview used by results, reports, and exports. | Step 2 `Diagnostics & Exports`; target mode; label availability. |
| `diagnostics_governance` | Diagnostics: governance | Optional | Builds governance, policy, documentation, suitability, manual-review, regulatory-report, and reviewer-oriented outputs. | Step 2 `Governance & Review`; Step 2 `Selection & Documentation`; Step 2 `Output Options`. Some controls require `Advanced` workspace mode. |
| `diagnostics_data_quality` | Diagnostics: data quality | Optional | Builds data-quality, missingness, descriptive-statistic, correlation, VIF, WoE/IV, and related diagnostic outputs when enabled. | Step 2 `Diagnostics & Exports` -> `Diagnostic suites`; Column Designer; feature roles. |
| `diagnostics_performance` | Diagnostics: performance | Optional | Builds model-performance, calibration, threshold, lift/gain, residual, quantile, QQ, and segment-performance outputs when enabled and applicable. | Step 2 `Diagnostics & Exports` -> `Diagnostic suites`; calibration controls; threshold controls; target mode; label availability. |
| `diagnostics_stability_tests` | Diagnostics: stability and statistical tests | Optional | Builds stability, PSI, ADF, model-specification, forecasting statistical tests, structural-break, distribution, residual, outlier, dependency, and paired-comparison evidence when enabled. | Step 2 `Diagnostics & Exports` -> `Diagnostic suites`; advanced statistical-test defaults from presets; date and segment fields. |
| `diagnostics_comparison_explainability` | Diagnostics: comparison and explainability | Optional | Builds challenger-comparison views, feature importance, PDP, ICE, centered ICE, ALE, two-way effects, marginal effects, interaction strength, feature-bucket calibration, and scenario outputs when enabled. | Step 2 `Challengers & Policies`; Step 2 `Explainability & Scenarios`; `Advanced` workspace mode for most explainability controls. |
| `diagnostics_credit_risk` | Diagnostics: credit risk | Optional | Builds credit-risk development diagnostics such as vintage, migration, delinquency transition, cohort PD, LGD/recovery, and macro-sensitivity outputs when enabled and supported by the data. | Step 2 `Diagnostics & Exports` -> `Enable credit-risk development diagnostics`; migration state column; date field; target mode; segment fields. |
| `diagnostics_expanded_framework` | Diagnostics: expanded framework tests | Optional | Builds deeper framework diagnostics such as advanced imputation, imputation sensitivity, multiple-imputation pooling, feature-construction workbench, and preset recommendation surfaces. | Step 2 `Data Preparation`; Step 2 `Selection & Documentation`; Step 2 `Diagnostics & Exports`; `Advanced` workspace mode for deeper controls. |
| `cross_validation` | Cross-validation | Optional | Runs optional k-fold or time-aware fold diagnostics. This does not replace the final saved model. | Step 2 `Diagnostics & Exports` -> `Enable cross-validation diagnostics`. Disabled by default in Large Data Mode. |
| `large_data_full_scoring` | Large-data full scoring | Optional | Scores file-backed Large Data Mode inputs in chunks after the development model is fit on the governed sample. | Step 1 `Data Source` -> `Large Data Mode`; Step 2 `Diagnostics & Exports`; Step 2 `Output Options`. |
| `export_package` | Export package | Required | Writes the model object, reports, tables, manifests, rerun code, checkpoints, predictions, regulatory outputs, and monitoring handoff bundle when applicable. | Step 2 `Output Options`; export profile; tabular output format; individual figure export; enhanced report visuals; Advanced Visual Analytics. |

## Feature-Subset-Search Stages

When execution mode is `search_feature_subsets`, Quant Studio uses a shorter
comparison-only checkpoint sequence.

| Stage ID | UI label | Required? | What it does | Main controls |
| --- | --- | --- | --- | --- |
| `prepare_data` | Prepare data | Required | Loads data, applies schema, creates target, validates setup, cleans data, builds simple derived features, creates train/validation/test splits, runs assumption checks, fits imputation rules, and applies governed transformations. | Step 1 `Dataset & Schema`; Step 2 `Core Setup`; Step 2 `Split Strategy`; Step 2 `Data Preparation`; Column Designer; Transformations table. |
| `feature_subset_search` | Run feature subset search | Required | Enumerates candidate feature subsets for the selected supported model family, fits candidates, ranks them by the selected metric, and records comparison evidence. | Step 2 `Core Setup` -> `Execution mode = search_feature_subsets`; Step 2 `Feature Subset Search`; enabled feature roles. |
| `export_package` | Export package | Required | Writes subset-search ranking tables, comparison charts, winning-subset details, manifests, and report outputs. It does not create the final model-development package. | Step 2 `Output Options`; feature-subset-search settings. |

## Where Stage Evidence Is Stored

Each run writes checkpoint evidence under the run folder:

```text
artifacts/run_YYYY-MM-DD_HH-MM-SS_UTC/
  checkpoints/
    checkpoint_manifest.json
    00_initial_context.joblib
    NN_stage_name.joblib
  metadata/
    checkpoint_manifest.json
```

The checkpoint manifest records:

- stage order
- stage ID and label
- required versus optional status
- current stage status
- start and completion timestamps
- elapsed seconds
- latest context checkpoint path
- error message when a stage fails

The Step 3 `Checkpoint Flow` chart is rendered from the same stage-status
payload that is derived from this manifest, so the UI and exported run evidence
should agree.

## How To Use This During Review

Use the checkpoint stages to answer practical execution questions:

- Did the model fit before diagnostics failed?
- Did optional credit-risk or statistical-test diagnostics fail without
  blocking export?
- Was full-file Large Data Mode scoring attempted?
- Which stage produced the error shown in the UI?
- Can support inspect the last successful context checkpoint without rerunning
  earlier stages?

Use the five GUI workflow steps for setup and review navigation. Use checkpoint
stages for execution audit and debugging.
