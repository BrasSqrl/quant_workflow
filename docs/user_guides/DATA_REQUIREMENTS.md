# Data Requirements Guide

This guide explains what input data Quant Studio expects and how to avoid the
most common setup problems.

## Supported Inputs

Quant Studio accepts:

- pandas dataframes through the Python API
- CSV files
- Excel files
- Parquet files
- files selected from the repo-level `Data_Load/` folder

For large files, prefer `Data_Load/` plus Parquet. Browser upload is available
but is less reliable for multi-GB data.

## Minimum Practical Columns

A normal model-development run needs:

| Column type | Required | Notes |
| --- | --- | --- |
| Target source | Yes for fitting and labeled scoring | The raw outcome column. For binary PD, it must map to default and non-default values. |
| Features | Yes | Numeric and categorical predictors used to fit or score the model. |
| Date | Required for time-series, panel, vintage, and backtesting views | Strongly recommended for development documentation even in cross-sectional data. |
| Identifier | Optional | Useful for traceability but should not enter the feature set. |
| Segment fields | Optional | Useful for segmentation, grouped imputation, stability, and scenario review. |

## Target Requirements

For binary PD-style modeling:

- identify the raw target source column
- define positive target values, usually values that mean default, loss, event, or bad outcome
- ensure both positive and negative classes exist for model fitting
- avoid using post-outcome leakage fields as features

For continuous LGD, regression, and forecasting:

- target values should be numeric
- censored or bounded targets should be matched to the right model family
- LGD-style bounded targets usually need values between `0` and `1`, depending on selected model

For existing-model scoring:

- labels are optional
- if labels are absent, the app should produce score-only documentation and skip label-dependent metrics
- if labels are present, the app can rerun validation diagnostics without rebuilding the model

## Column Role Rules

Use the Column Designer to set:

| Role | Use |
| --- | --- |
| `feature` | Predictive input used by the model. |
| `target_source` | Raw target field used to create the final modeled target. |
| `date` | Time field used for splits, backtests, vintage, drift, and time diagnostics. |
| `identifier` | Record, customer, loan, or account ID. Keep for traceability but not modeling. |
| `ignore` | Data that should not enter the model or exported modeling schema. |

Only one target source should be active for a standard run.

## Feature Requirements

Good feature columns should:

- be available at the time the model would be used
- be stable enough to explain and monitor during development review
- have business meaning that can be documented in the feature dictionary
- avoid directly encoding the target or post-event information
- have missingness and outliers reviewed before fitting

High-cardinality identifiers, free text placeholders, leakage fields, and
future-looking information should usually be disabled or ignored.

## Missing Values

Quant Studio supports several missing-value treatments:

- default scalar imputation
- median, mean, mode, zero, constant, forward-fill, and backward-fill rules
- grouped train-fit scalar imputation
- KNN imputation
- iterative model-based imputation
- missingness-indicator feature creation
- multiple-imputation review outputs
- Little's MCAR test and imputation sensitivity diagnostics

Use simple policies first unless there is a reason to use model-based
imputation. Always review exported imputation tables before treating advanced
imputation as final.

## Categorical Variables

Categorical fields can be included as features. Practical guidance:

- keep categories business-readable where possible
- watch for rare categories and dominant categories
- avoid unbounded IDs as categorical model features
- document category meaning in the feature dictionary when categories are not obvious

## Dates And Time Structure

Set a date column when:

- the model uses time-aware splits
- the dataset is panel or time-series
- vintage, drift, backtesting, or structural-break diagnostics matter
- CCAR or CECL forecasting workflow evidence is needed

For panel data, entity and date structure should be stable enough to identify
repeated observations over time.

## Large Files

For files above roughly 1 GB:

- place data in `Data_Load/` instead of browser upload
- prefer Parquet
- enable `Large Data Mode`
- consider converting CSV to Parquet before ingestion
- use sampled exports when full tabular outputs would be too large
- review [Large Data Playbook](./LARGE_DATA_PLAYBOOK.md)

## Common Data Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Readiness says no target source | No active column is marked `target_source` | Open Column Designer and mark the raw outcome column. |
| Target has one class | Positive values are wrong or data is filtered too narrowly | Check target mapping and source data. |
| Existing model scoring fails | New data does not have the saved model's expected raw features | Provide matching run config or add missing source features. |
| Model performance is suspiciously high | Leakage, identifiers, or post-event fields are enabled | Review features and disable invalid fields. |
| Memory pressure | File is too large for eager pandas loading | Use `Data_Load/`, Parquet, Large Data Mode, and sampled exports. |
