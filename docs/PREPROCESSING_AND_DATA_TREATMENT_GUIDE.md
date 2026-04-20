# Preprocessing and Data Treatment Guide

This guide documents how the framework transforms raw tabular input into a
model-ready dataset. The goal is transparency and reproducibility rather than
hidden automation.

Primary implementation files:

- `src/quant_pd_framework/orchestrator.py`
- `src/quant_pd_framework/steps/schema.py`
- `src/quant_pd_framework/steps/target.py`
- `src/quant_pd_framework/steps/validation.py`
- `src/quant_pd_framework/steps/cleaning.py`
- `src/quant_pd_framework/steps/feature_engineering.py`
- `src/quant_pd_framework/steps/splitting.py`
- `src/quant_pd_framework/steps/assumption_checks.py`
- `src/quant_pd_framework/steps/imputation.py`
- `src/quant_pd_framework/steps/transformations.py`
- `src/quant_pd_framework/steps/variable_selection.py`

## End-to-End Sequence

The default orchestrator executes the following preprocessing and preparation
steps before model fitting:

1. `IngestionStep`
2. `SchemaManagementStep`
3. `TargetConstructionStep`
4. `ValidationStep`
5. `CleaningStep`
6. `FeatureEngineeringStep`
7. `SplitStep`
8. `AssumptionCheckStep`
9. `ImputationStep`
10. `TransformationStep`
11. `VariableSelectionStep`

This order matters for auditability:

- schema comes before target construction because the framework must know what
  the columns are
- validation happens before expensive work
- splitting happens before imputation so train-fit imputation can be learned
  correctly
- assumption checks run after splitting so they inspect the development sample
- governed transformations run after train-fit imputation so transform fitting
  stays leakage-safe
- variable selection and manual review run after transformations so derived
  features can participate in screening

## 1. Ingestion

### What happens

The framework accepts:

- a pandas dataframe
- a CSV file
- an Excel file

The GUI upload path lives in `app/streamlit_app.py` and the pipeline ingestion
step lives in `src/quant_pd_framework/steps/ingestion.py`.

### Audit evidence

- `input_shape`
- optional exported `input_snapshot.csv`

## 2. Schema Management

### What happens

`SchemaManagementStep` standardizes the incoming dataframe according to
`SchemaConfig` and a list of `ColumnSpec` objects.

Each column can be:

- enabled or disabled
- renamed
- type-cast
- marked with a role
- created if missing
- preserved or replaced when renaming

### Column roles

Roles are defined by `ColumnRole`:

- `feature`
- `target_source`
- `date`
- `identifier`
- `ignore`

### Dtype handling

Supported dtype coercions:

- `string`
- `category`
- `float`
- `int`
- `bool`
- `datetime`

Boolean coercion accepts values such as:

- truthy: `1`, `true`, `t`, `yes`, `y`
- falsy: `0`, `false`, `f`, `no`, `n`

### Why it exists

This step makes later pipeline logic deterministic. The framework does not rely
on whatever raw column names and dtypes happened to arrive from Excel or CSV.

### Audit evidence

- `column_roles`
- `post_schema_columns`
- `dropped_columns`

## 3. Target Construction

### What happens

`TargetConstructionStep` creates the model target column defined by
`TargetConfig`.

Binary mode:

- maps user-declared positive values to `1`
- maps non-missing non-positive values to `0`
- preserves missing values as missing

Continuous mode:

- coerces the source column to numeric

### Special score-only behavior

When `ExecutionMode.SCORE_EXISTING_MODEL` is used and the target source column is
missing, the framework switches to score-only mode:

- `labels_available = False`
- label-based metrics are skipped downstream

### Audit evidence

- `target_source_column`
- `target_mode`
- `target_distribution`
- `labels_available`

## 4. Validation

### What happens

`ValidationStep` enforces structural rules before model fitting. Examples:

- the target must be present unless score-only mode allows omission
- time-aware workflows need a valid date column
- panel workflows need an identifier column
- discrete-time hazard requires time-series or panel structure

### Why it exists

This is where the framework fails fast on invalid workflow combinations instead
of producing ambiguous downstream errors.

## 5. Cleaning

### What happens

`CleaningStep` performs conservative hygiene rules:

- blank strings can become nulls
- string-like values can be trimmed
- duplicate rows can be dropped
- rows with missing targets can be dropped when labels are available
- feature columns that are entirely null can be dropped

### Protected columns

When dropping all-null columns, the step protects:

- the target column
- identifier columns

### Audit evidence

- `duplicate_rows_removed`
- `rows_removed_missing_target`
- `post_clean_shape`
- warnings for fully null dropped columns

## 6. Feature Engineering

### What happens

`FeatureEngineeringStep` performs simple, explicit feature creation.

Current supported behavior:

- derive date parts from configured date columns
- optionally drop raw date columns from model features
- create hazard-time features for discrete-time hazard models

Supported date parts:

- `year`
- `month`
- `quarter`
- `day`
- `dayofweek`

### Excluded from model features

The feature-engineering step excludes:

- the target column
- the target source column
- identifier columns
- ignored columns
- retained date columns

### Hazard-model additions

For `discrete_time_hazard_model`, the step adds:

- `hazard_period_index`
- `hazard_period_index_sq`

### Audit evidence

- `feature_columns`
- `numeric_features`
- `categorical_features`
- `feature_summary`
- `hazard_time_features` when applicable

## 7. Splitting

### What happens

`SplitStep` chooses between:

- randomized cross-sectional split
- chronological time-aware split

### Cross-sectional logic

For `DataStructure.CROSS_SECTIONAL`:

- uses `train_test_split`
- stratifies only when:
  - labels are available
  - target mode is binary
  - `SplitConfig.stratify = True`

### Time-aware logic

For `DataStructure.TIME_SERIES` and `DataStructure.PANEL`:

- parses the configured date column
- sorts by date
- sorts by date and entity for panels
- slices train, validation, and test in chronological order

### Why split comes before imputation

This is essential. Any learned fill value must be fit on the train split only so
validation and test results remain leakage-safe.

### Audit evidence

- `split_summary`
- warnings if a non-test split contains only one class in a time-ordered run

## 8. Assumption Checks

### What happens

`AssumptionCheckStep` records development-sample suitability diagnostics such as:

- non-null target count
- class balance and events per feature for binary workflows
- bounded-target and censoring checks for relevant continuous models
- duplicate entity-date checks for panel workflows
- dominant-category concentration checks

### Audit evidence

- table `assumption_checks`
- metadata `assumption_check_summary`

## 9. Imputation

### What happens

`ImputationStep` learns and applies missing-value rules after splitting.

Supported policies:

- `inherit_default`
- `none`
- `mean`
- `median`
- `mode`
- `constant`
- `forward_fill`
- `backward_fill`

### Default resolution

`inherit_default` resolves as:

- numeric feature -> `median`
- non-numeric feature -> `mode`

### Important governance rule

Scalar fill values are fit on the train split only and then reused across:

- validation
- test
- score-only datasets

### Directional fill rules

`forward_fill` and `backward_fill`:

- are only allowed for time-series or panel workflows
- apply within entity groups for panel data when an entity column exists

### Failure mode

If missing values remain in feature columns after the configured policy runs, the
pipeline fails rather than silently applying a fallback imputer.

### Audit evidence

- table `imputation_rules`
- metadata `imputation_summary`

## 10. Governed Transformations

### What happens

`TransformationStep` fits explicit, reproducible transforms on the train split
and replays them on validation, test, score-only datasets, and the working
diagnostic dataframe.

Supported transform families:

- `winsorize`
- `log1p`
- `ratio`
- `interaction`
- `manual_bins`

### Audit evidence

- table `governed_transformations`
- metadata `transformation_summary`

## 11. Variable Selection

### What happens

`VariableSelectionStep` is an optional train-split screening step, not an
undocumented automated search.

The step can apply:

- univariate screening
- correlation filtering
- feature-count cap
- locked include rules
- locked exclude rules
- manual approve / reject / force decisions after screening

### Univariate score

Binary targets:

- uses a univariate AUC-derived score:
  - `abs(auc - 0.5) * 2`

Continuous targets:

- uses absolute correlation with the target

### Correlation screen

The current correlation filter applies only to numeric features and removes a
feature when it is too correlated with a higher-ranked kept feature.

### Score-existing-model behavior

Variable selection is skipped in `score_existing_model` mode because the trained
feature contract must be preserved.

### Audit evidence

- table `variable_selection`
- table `manual_review_feature_decisions`
- metadata `variable_selection_summary`

## 12. How the GUI Controls Data Treatment

The GUI does not implement preprocessing itself. It collects settings and builds
the core config through:

- `GUIBuildInputs`
- `build_framework_config_from_editor(...)`

The column designer controls:

- roles
- dtypes
- missing-value policies
- synthetic columns
- renames

That configuration then drives the pipeline steps above.

The workspace also includes dedicated editors for:

- feature dictionary rows
- governed transformations
- manual review decisions
- scorecard bin overrides
- review-workbook import/export

## 13. Main Audit Trail Outputs

The preprocessing and data-treatment audit trail is spread across:

- `run_config.json`
- `step_manifest.json`
- `input_snapshot.csv`
- `imputation_rules`
- `assumption_checks`
- `governed_transformations`
- `variable_selection`
- `manual_review_feature_decisions`
- `feature_dictionary`
- `reproducibility_manifest.json`
- `data_quality_summary`
- `run_report.md`
- `model_documentation_pack.md`
- `validation_pack.md`
- `configuration_template.xlsx`

## 14. Review Questions for Validators

When reviewing a run, the highest-value questions are usually:

1. Did schema controls rename or disable any important fields?
2. Was the target constructed the way the business definition intended?
3. Was the split method appropriate for the data structure?
4. Were train-fit imputation rules applied consistently?
5. Did variable selection remove any required business variables?
6. Are date and identifier columns treated consistently with the declared
   workflow?

This guide is meant to make those questions answerable from exported artifacts.
