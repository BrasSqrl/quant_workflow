# Execution Mode Decision Guide

Quant Studio has three execution modes. Choose the mode based on whether you
are building a final model, scoring with an existing model, or comparing
candidate feature subsets before final development.

## Quick Decision Table

| Goal | Use mode |
| --- | --- |
| Build and document a new fitted model | `fit_new_model` |
| Use a saved model on new data without refitting | `score_existing_model` |
| Try candidate feature combinations before choosing final features | `search_feature_subsets` |

## Mode 1: `fit_new_model`

Use this when:

- a new model must be trained
- final model-development evidence is needed
- the target and feature set are being finalized
- outputs should include a model object and development documentation

Required inputs:

- input dataframe or file
- target source column
- enabled features
- model family
- split strategy

Typical outputs:

- `model/quant_model.joblib`
- `config/run_config.json`
- `metadata/metrics.json`
- `data/predictions/predictions.csv` or `data/predictions/predictions.parquet`
- `reports/interactive_report.html`
- `reports/model_documentation_pack.md`
- `reports/validation_pack.md`
- `model_bundle_for_monitoring/`

Do not use this mode if the approved model already exists and must not be
rebuilt.

## Mode 2: `score_existing_model`

Use this when:

- a fitted `model/quant_model.joblib` already exists
- new data should be scored with the existing model
- diagnostics, documentation, and reports are needed without refitting
- model implementation must remain unchanged

Recommended inputs:

- existing `model/quant_model.joblib`
- matching prior `config/run_config.json`
- new scoring data with the raw features expected by the original model
- labels when validation metrics are desired

Behavior:

- the training step becomes a model-loading step
- saved schema and feature contracts can override the live editor when prior config is supplied
- labeled scoring runs produce validation diagnostics
- unlabeled scoring runs produce score-only documentation and skip invalid label-dependent diagnostics

Typical outputs:

- scored predictions
- score distributions
- validation diagnostics when labels are present
- score-only documentation when labels are absent
- refreshed reports and manifests

Do not use this mode for ongoing production monitoring. Quant Studio remains a
model-development and documentation tool.

## Mode 3: `search_feature_subsets`

Use this when:

- the user has a candidate feature pool
- the goal is to compare model performance across feature combinations
- the output should inform feature selection before final model development
- AUC/ROC, KS, frontier, frequency, and significance comparisons are needed

Required inputs:

- binary target
- selected model family that supports subset comparison
- candidate feature pool
- minimum and maximum subset sizes
- ranking split and ranking metric

Typical outputs:

- candidate ranking tables
- ROC/AUC and KS comparisons
- selected winning subset details
- feature-frequency views
- frontier charts
- paired significance tests

Important distinction:

`search_feature_subsets` is a comparison workflow. It does not replace a final
`fit_new_model` run. After choosing features, run `fit_new_model` with the
selected final feature set to produce the formal development package.

## Recommended Workflow

For most projects:

1. Use `search_feature_subsets` only if feature selection is uncertain.
2. Use `fit_new_model` to create the model-development package.
3. Use `score_existing_model` later when the same model must score new data.
