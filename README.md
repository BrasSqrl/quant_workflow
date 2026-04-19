# Quant PD Framework

Quant PD Framework is a Python project for quantitative model development,
validation, scenario analysis, and documentation on tabular data. Probability of
default modeling remains the default use case, but the framework now also
supports LGD and forecasting workflows that are common in CCAR and CECL
development. It is structured as a reusable, object-oriented framework rather
than a notebook-only prototype, so each primary modeling stage lives in its own
class and the entire run is coordinated by an orchestrator.

The project supports two working modes:

1. A Python API for developers who want to configure and run the pipeline in code.
2. A Streamlit GUI, branded in the app as `Quant Studio`, for users who want to load a dataframe, define schema rules, and run the workflow visually.

Inside those interfaces, the framework supports two execution modes:

1. `fit_new_model`
2. `score_existing_model`

## What The Framework Does

The framework assumes you begin with one of the following inputs:

- a pandas dataframe
- a CSV file
- an Excel file

The default workflow then moves through the main stages of a quantitative modeling process:

1. Ingestion
2. Schema management
3. Target construction
4. Validation
5. Cleaning
6. Feature engineering
7. Train/validation/test split
8. Model training
9. Evaluation
10. Backtesting
11. Diagnostics and visualization generation
12. Artifact export

When `score_existing_model` is used, the training stage becomes a
model-loading stage and the remaining steps run on newly scored data so the
existing model can still be documented, validated, stress tested, and exported
without refitting.

The framework now supports:

- logistic regression
- elastic-net logistic regression
- scorecard logistic regression
- probit regression
- linear regression
- beta regression
- two-stage LGD model
- panel regression
- quantile regression
- Tobit regression
- XGBoost

The framework can also reuse a previously exported `quant_model.joblib`
artifact so new data can be scored and validated without refitting the model.

For targets, the framework supports:

- binary mode for PD-style modeling
- continuous mode for LGD, regression, censored-regression, and forecasting workflows

The framework also includes development-focused workflow features:

- workflow presets for `PD Development`, `Lifetime PD / CECL`, `LGD Severity`,
  and `CCAR Forecasting`
- challenger-model comparison mode
- feature policy checks for required/excluded features, missingness, VIF, IV,
  expected signs, and monotonicity
- explainability outputs such as coefficient breakdowns, permutation
  importance, model-specific tables, and feature effect curves
- scenario testing for feature-level shocks on held-out data

## Design Goals

- Explicit pipeline structure: each modeling stage is visible and isolated.
- Extensibility: steps can be replaced or subclassed later.
- Reproducibility: each run exports metrics, artifacts, and the effective configuration.
- Separation of concerns: the GUI collects user input; the framework performs the modeling logic.
- Practical usability: schema control, dtype control, and target construction are first-class parts of the workflow.
- Fail-fast validation: configuration issues should be rejected before pipeline execution begins.
- Development focus: the platform is intentionally centered on model build,
  challenger review, scenario analysis, and documentation rather than ongoing
  production monitoring.

## Engineering Standards

The repository now includes an explicit engineering rubric and alignment note:

- [docs/ENGINEERING_RUBRIC.md](C:/Users/matth/Desktop/quant/docs/ENGINEERING_RUBRIC.md)
- [docs/RUBRIC_ALIGNMENT.md](C:/Users/matth/Desktop/quant/docs/RUBRIC_ALIGNMENT.md)
- [docs/UI_UX_STANDARD.md](C:/Users/matth/Desktop/quant/docs/UI_UX_STANDARD.md)

There is also an executive-level non-technical summary at:

- [EXECUTIVE_SUMMARY.txt](C:/Users/matth/Desktop/quant/EXECUTIVE_SUMMARY.txt)

## Repository Layout

```text
quant/
  .streamlit/
    config.toml
  app/
    streamlit_app.py
  docs/
    ENGINEERING_RUBRIC.md
    RUBRIC_ALIGNMENT.md
    UI_UX_STANDARD.md
  examples/
    run_development_workflow.py
    run_pipeline.py
    score_existing_model.py
  src/
    quant_pd_framework/
      __init__.py
      base.py
      config.py
      config_io.py
      context.py
      gui_launcher.py
      gui_support.py
      models.py
      orchestrator.py
      presets.py
      run.py
      sample_data.py
      steps/
        comparison.py
        ingestion.py
        schema.py
        target.py
        validation.py
        cleaning.py
        feature_engineering.py
        splitting.py
        training.py
        evaluation.py
        backtesting.py
        diagnostics.py
        export.py
  tests/
    test_existing_model_scoring.py
    test_development_features.py
    test_gui_launcher.py
    test_gui_support.py
    test_pipeline_smoke.py
    test_saved_run_bundle.py
    support.py
  EXECUTIVE_SUMMARY.txt
  launch_gui.bat
  setup_gui.bat
  pyproject.toml
  README.md
```

## Installation

### Python Version

The project requires Python `3.11+`.

### Base Install

Install the package and development dependencies:

```powershell
python -m pip install -e .[dev]
```

### Install With GUI Support

Install the package, tests, lint tooling, and Streamlit:

```powershell
python -m pip install -e .[dev,gui]
```

## Fast Start

### Easiest GUI Launch

The easiest Windows launch command is:

```powershell
.\launch_gui.bat
```

That root-level batch file now does two things:

- on the first run, it creates a local `.venv` in the project directory and installs `.[dev,gui]`
- on later runs, it reuses `.venv` and launches the GUI immediately
- if `.venv` exists but is incomplete or broken, it rebuilds the environment automatically
- if package installation into `.venv` fails but the current Python already has the GUI dependencies, it falls back to the local `src` tree and existing site-packages so the launcher still works

This makes the launcher double-click friendly while keeping the GUI isolated from your system Python environment.

If you want to rebuild the local environment manually, use:

```powershell
.\setup_gui.bat
```

`setup_gui.bat` also detects a broken `.venv` and recreates it before reinstalling dependencies.

You can also launch the installed console script:

```powershell
quant-pd-gui
```

For non-GUI execution from a saved config bundle, the installed CLI is:

```powershell
quant-pd-run --config path\to\run_config.json --input path\to\data.csv
```

If you prefer the raw Streamlit command, this still works:

```powershell
streamlit run app/streamlit_app.py
```

### Run The Example Pipeline

```powershell
python examples/run_pipeline.py
```

### Run The Development Workflow Example

```powershell
python examples/run_development_workflow.py
```

### Run From A Saved Config Bundle

If you already have an exported run folder, you can rerun it without the GUI:

```powershell
python -m quant_pd_framework.run --config artifacts\20260418T000000Z\run_config.json
```

If the run folder includes `input_snapshot.csv`, the runner picks it up automatically. Otherwise supply `--input`.

### Run Tests

```powershell
python -m pytest -q
```

### Run Lint Checks

```powershell
python -m ruff check .
```

## GUI Overview

The GUI is a thin front end over the Python framework. It does not implement modeling logic itself. Instead, it:

- loads CSV or Excel data
- previews the incoming dataframe
- constructs a schema editor from the incoming columns
  - lets the user assign column roles and dtypes
  - lets the user define per-column missing-value treatment
  - lets the user rename or disable columns
- lets the user add columns that should be created if missing
- collects target, split, cleaning, feature-engineering, diagnostics, and model settings
- builds a `FrameworkConfig`
- calls the `QuantModelOrchestrator`
- displays metrics, validation charts, feature drilldowns, diagnostic tables, predictions, and artifact paths

### GUI Design System

The current interface is intentionally styled as a premium light-mode fintech dashboard rather than a default Streamlit application.

The governing visual and interaction standard is documented in:

- [docs/UI_UX_STANDARD.md](C:/Users/matth/Desktop/quant/docs/UI_UX_STANDARD.md)

That standard drives both the live GUI and the exported standalone HTML report. The design system emphasizes:

- a light enterprise-fintech palette with stronger visual hierarchy
- grouped diagnostics instead of one long undifferentiated result page
- consistent Plotly theming across all charts
- metric cards, section shells, and filter controls that make scanning easier for model builders and validation teams
- the same section taxonomy in both Streamlit and exported reports so users do not have to relearn the layout

### Main GUI Controls

The GUI exposes the following decision areas:

- workflow preset
- execution mode
- file upload with a configured 50 GB per-file Streamlit limit
- existing model artifact path
- existing run config path
- model type
- target mode
- data structure
- target output name
- positive target values
- split percentages
- random state
- stratification toggle for cross-sectional data
- logistic, elastic-net, scorecard, quantile, XGBoost, and Tobit hyperparameters
- cleaning toggles
- date feature toggles
- per-column missing-value policy and constant-fill controls in the column designer
- challenger model selection and ranking metric
- feature policy inputs for required/excluded features, sign expectations, and monotonicity
- explainability controls for permutation importance and feature effect curves
- scenario table for held-out stress testing
- diagnostics and export toggles
- artifact output root
- schema editor for column-level configuration

### Workflow Presets

The preset selector is intended to narrow setup time for the main supported
development workflows:

- `PD Development`
- `Lifetime PD / CECL`
- `LGD Severity`
- `CCAR Forecasting`

Presets do not hide the underlying configuration. They simply seed the model,
target mode, data structure, comparison defaults, feature-policy defaults, and
explainability defaults so the user starts from a workflow-specific baseline.

### Diagnostic Studio Layout

After a run completes, the GUI presents results through a grouped validation workspace rather than a flat output dump.

The primary result sections are:

- Data Quality
- Sample / Segmentation
- Model Performance
- Calibration / Thresholds
- Stability / Drift
- Backtesting / Time Diagnostics
- Governance / Export Bundle

The live result surface also supports interactive filters for:

- split selection
- date range
- segment column and segment values
- feature lens
- decision threshold for binary models
- top-N display depth
- chart versus table visibility
- summary versus technical view depth

When `score_existing_model` is selected:

- provide an exported `quant_model.joblib` path
- optionally provide the matching prior `run_config.json`
- if the prior run config is supplied, its schema, target, feature, split, and model settings override the current GUI editor so the scoring run stays aligned with the original model contract

### GUI Relationship To The Python Code

The GUI is not the authoritative implementation of the workflow. It is a configuration shell over the Python package.

That means:

- GUI selections are converted into a `FrameworkConfig`
- the same `QuantModelOrchestrator` runs whether the pipeline starts from the GUI or from code
- exported run folders now include a Python rerun bundle so a user can leave the GUI entirely after setup

### Schema Editor Rules

The schema editor is one of the most important pieces of the GUI because it controls how the incoming dataframe is standardized before modeling.

Each schema row can define:

- whether the row is enabled
- the input column name
- the output column name
- the column role
- the target dtype
- the missing-value policy for that column
- the constant fill value when constant imputation is selected
- whether the column should be created if missing
- the default value for a created column
- whether the original source column should be kept after renaming

The GUI expects:

- exactly one enabled row marked as `target_source`
- one enabled row marked as `date` for time-series or panel runs
- one enabled row marked as `identifier` for panel runs

## Python API Usage

The Python API is the better choice when:

- you want to run the framework from scripts or notebooks
- you want version-controlled configuration
- you want to subclass or replace pipeline steps
- you want to integrate the framework into a larger application

### Minimal Example

```python
import pandas as pd

from quant_pd_framework import (
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    DiagnosticConfig,
    ExecutionConfig,
    ExecutionMode,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetMode,
    TargetConfig,
)

dataframe = pd.read_csv("loan_data.csv")

config = FrameworkConfig(
    schema=SchemaConfig(
        column_specs=[
            ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
            ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
            ColumnSpec(name="legacy_text_field", enabled=False),
            ColumnSpec(
                name="portfolio_segment",
                create_if_missing=True,
                default_value="retail",
                dtype="string",
            ),
        ]
    ),
    cleaning=CleaningConfig(),
    feature_engineering=FeatureEngineeringConfig(),
    diagnostics=DiagnosticConfig(),
    target=TargetConfig(
        source_column="default_status",
        mode=TargetMode.BINARY,
        output_column="default_flag",
        positive_values=["1", "default", "charged_off"],
    ),
    split=SplitConfig(
        data_structure=DataStructure.CROSS_SECTIONAL,
        date_column="as_of_date",
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
    ),
    execution=ExecutionConfig(mode=ExecutionMode.FIT_NEW_MODEL),
    model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
)

orchestrator = QuantModelOrchestrator(config=config)
context = orchestrator.run(dataframe)

print(context.metrics["test"])
print(context.artifacts["report"])
```

### Preset-Driven Development Example

The preset system is useful when you want a workflow-specific baseline without
hiding the underlying Python config.

```python
from quant_pd_framework import (
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    FrameworkConfig,
    PresetName,
    QuantModelOrchestrator,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
    get_preset_definition,
)
from quant_pd_framework.sample_data import build_sample_pd_dataframe

dataframe = build_sample_pd_dataframe()
preset = get_preset_definition(PresetName.PD_DEVELOPMENT)

config = FrameworkConfig(
    preset_name=preset.name,
    schema=SchemaConfig(
        column_specs=[
            ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
            ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
        ]
    ),
    cleaning=CleaningConfig(),
    feature_engineering=preset.feature_engineering,
    target=TargetConfig(
        source_column="default_status",
        mode=TargetMode.BINARY,
        output_column=preset.target_output_column,
        positive_values=[1],
    ),
    split=SplitConfig(
        data_structure=preset.data_structure,
        date_column="as_of_date",
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
    ),
    model=preset.model,
    comparison=preset.comparison,
    feature_policy=preset.feature_policy,
    explainability=preset.explainability,
    scenario_testing=ScenarioTestConfig(
        enabled=True,
        evaluation_split="test",
        scenarios=[
            ScenarioConfig(
                name="Higher Utilization",
                feature_shocks=[
                    ScenarioFeatureShock(
                        feature_name="utilization",
                        operation=ScenarioShockOperation.ADD,
                        value=0.10,
                    )
                ],
            )
        ],
    ),
    diagnostics=preset.diagnostics,
)

context = QuantModelOrchestrator(config=config).run(dataframe)
print(context.metadata.get("comparison_recommended_model"))
```

### Score New Data On An Existing Model

If you already have an exported model artifact from a prior run, you can reuse it on fresh data:

```python
from quant_pd_framework import ExecutionMode, QuantModelOrchestrator, load_framework_config

new_dataframe = ...

config = load_framework_config("artifacts/prior_run/run_config.json")
config.execution.mode = ExecutionMode.SCORE_EXISTING_MODEL
config.execution.existing_model_path = "artifacts/prior_run/quant_model.joblib"
config.execution.existing_config_path = "artifacts/prior_run/run_config.json"
config.artifacts.output_root = "artifacts/existing_model_scores"

context = QuantModelOrchestrator(config=config).run(new_dataframe)
print(context.metadata["labels_available"])
print(context.metrics["test"])
```

If the new dataframe still includes the realized target, the framework runs the
full labeled diagnostics package. If the target is missing, the run continues
in score-only mode and skips label-dependent tests while still exporting the
documentation bundle for the scored dataset.

### Run A Saved Bundle In Python

If you want to replay a previously exported run, you can load the saved config and execute it again:

```python
from quant_pd_framework import run_saved_config

context = run_saved_config(
    config_path="artifacts/20260418T000000Z/run_config.json",
    input_path="artifacts/20260418T000000Z/input_snapshot.csv",
    output_root="artifacts/reruns",
)

print(context.artifacts["output_root"])
```

## Configuration Reference

### `FrameworkConfig`

`FrameworkConfig` is the top-level configuration object. It aggregates:

- `schema`
- `cleaning`
- `feature_engineering`
- `target`
- `split`
- `preset_name`
- `execution`
- `model`
- `comparison`
- `feature_policy`
- `explainability`
- `scenario_testing`
- `diagnostics`
- `artifacts`

`FrameworkConfig.validate()` now performs fail-fast validation of the configuration contract before execution starts. The orchestrator, GUI config builder, and config loader all call that validation path.

### `SchemaConfig`

`SchemaConfig` controls how the incoming dataframe is standardized.

Important fields:

- `column_specs`
- `pass_through_unconfigured_columns`

### `ColumnSpec`

`ColumnSpec` is the main switchboard for column handling.

You can use it to:

- disable an existing column
- rename an incoming column
- assign a column role
- force a dtype
- set a per-column missing-value policy
- add a new column if it is missing
- decide whether the original source column should be retained

Examples:

- disable a column: `ColumnSpec(name="legacy_field", enabled=False)`
- rename a column: `ColumnSpec(name="borrower_age", source_name="age_years")`
- force float dtype: `ColumnSpec(name="balance", dtype="float")`
- set numeric imputation: `ColumnSpec(name="balance", missing_value_policy=MissingValuePolicy.MEDIAN)`
- set constant fill: `ColumnSpec(name="channel", missing_value_policy=MissingValuePolicy.CONSTANT, missing_value_fill_value="unknown")`
- mark a date column: `ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE)`
- mark an identifier: `ColumnSpec(name="loan_id", role=ColumnRole.IDENTIFIER)`
- create a new column: `ColumnSpec(name="segment", create_if_missing=True, default_value="retail")`

### Supported Missing-Value Policies

The column designer and Python config currently support:

- `inherit_default`
- `none`
- `mean`
- `median`
- `mode`
- `constant`
- `forward_fill`
- `backward_fill`

Behavior notes:

- `inherit_default` uses median for numeric features and mode for categorical features
- scalar imputation values are fit on the training split and then reused on validation, test, and score-only data
- `forward_fill` and `backward_fill` are only supported for time-series or panel workflows
- target values are still handled by `drop_rows_with_missing_target`; feature imputation is a separate step

### Supported Dtypes

The framework currently supports:

- `string`
- `category`
- `float`
- `int`
- `bool`
- `datetime`

### `CleaningConfig`

Cleaning rules currently include:

- trimming strings
- treating blank strings as null
- dropping duplicate rows
- dropping rows with missing target values
- dropping fully null feature columns

### `FeatureEngineeringConfig`

The default feature engineering logic is intentionally simple and transparent.

Current behavior:

- parse configured date columns
- optionally derive date parts such as year, month, quarter, day, and day-of-week
- optionally exclude raw date columns from model features

### `TargetConfig`

`TargetConfig` defines how the final target is built.

Important fields:

- `source_column`
- `mode`
- `output_column`
- `positive_values`
- `drop_source_column`

Use `mode="binary"` for PD-style modeling and `mode="continuous"` for linear or Tobit-style workflows.

If the target source is not already binary, provide `positive_values` so the framework knows which source values should map to default `1`.

### `SplitConfig`

`SplitConfig` controls how the dataset is partitioned into train, validation, and test sets.

Supported data structures:

- `cross_sectional`
- `time_series`
- `panel`

Behavior:

- cross-sectional data uses randomized splitting and can be stratified
- time-series data uses chronological splitting
- panel data uses chronological splitting and can also track an entity column

The split percentages must sum to `1.0`.

### `ExecutionConfig`

`ExecutionConfig` controls whether the run trains a new model or reuses an existing exported artifact.

Fields:

- `mode`
- `existing_model_path`
- `existing_config_path`

Supported modes:

- `fit_new_model`
- `score_existing_model`

Recommended scoring pattern:

- point `existing_model_path` at a prior exported `quant_model.joblib`
- point `existing_config_path` at the matching prior `run_config.json`
- let the framework reuse the prior schema, feature, target, split, and model settings so the new scoring run stays aligned with the original model contract

If `mode="score_existing_model"`, `existing_model_path` is required.

### `ModelConfig`

Supported model families:

- `logistic_regression`
- `elastic_net_logistic_regression`
- `scorecard_logistic_regression`
- `probit_regression`
- `linear_regression`
- `beta_regression`
- `two_stage_lgd_model`
- `panel_regression`
- `quantile_regression`
- `tobit_regression`
- `xgboost`

Exposed hyperparameters:

- `max_iter`
- `C`
- `solver`
- `l1_ratio`
- `class_weight`
- `threshold`
- `scorecard_bins`
- `beta_clip_epsilon`
- `lgd_positive_threshold`
- `quantile_alpha`
- `xgboost_n_estimators`
- `xgboost_learning_rate`
- `xgboost_max_depth`
- `xgboost_subsample`
- `xgboost_colsample_bytree`
- `tobit_left_censoring`
- `tobit_right_censoring`

Compatibility notes:

- binary targets support logistic, elastic-net logistic, scorecard logistic, probit, and XGBoost
- continuous targets support linear, beta, two-stage LGD, panel, quantile, Tobit, and XGBoost
- beta and two-stage LGD models require bounded continuous targets in `[0, 1]`
- panel regression requires `data_structure="panel"` and an entity column

### `ComparisonConfig`

`ComparisonConfig` controls optional challenger-model development runs.

Important fields:

- `enabled`
- `challenger_model_types`
- `ranking_metric`
- `ranking_split`

When enabled, challenger models are fit on the same prepared training split and
ranked on the selected validation or test metric. The resulting comparison table
and recommendation are exported with the rest of the documentation bundle.

### `FeaturePolicyConfig`

`FeaturePolicyConfig` lets the user encode lightweight model-governance rules
directly in the run configuration.

Important fields:

- `required_features`
- `excluded_features`
- `max_missing_pct`
- `max_vif`
- `minimum_information_value`
- `expected_signs`
- `monotonic_features`
- `error_on_violation`

These checks are intended for development governance and documentation, not
production monitoring.

### `ExplainabilityConfig`

`ExplainabilityConfig` controls optional interpretability outputs.

Important fields:

- `permutation_importance`
- `feature_effect_curves`
- `coefficient_breakdown`
- `top_n_features`
- `grid_points`
- `sample_size`

Depending on model family, this can produce coefficient tables, odds-ratio-style
breakdowns, permutation importance, feature-effect curves, WoE tables, and
two-stage LGD coefficient outputs.

### `ScenarioTestConfig`

`ScenarioTestConfig` controls optional what-if analysis on a held-out split.

Important fields:

- `evaluation_split`
- `scenarios`

Each scenario can contain one or more feature shocks using operations such as
`add`, `multiply`, and `set`. Scenario results are exported as summary tables,
segment impacts, and interactive visuals.

### `DiagnosticConfig`

`DiagnosticConfig` controls which analyses, charts, and export helpers are enabled.

Key toggles include:

- data quality summary
- descriptive statistics
- missingness analysis
- correlation analysis
- VIF analysis
- WoE/IV analysis
- PSI analysis
- ADF tests
- calibration analysis
- threshold analysis
- lift and gain analysis
- segment analysis
- residual analysis
- quantile analysis
- QQ analysis
- interactive HTML exports
- PNG chart exports
- Excel workbook export

### `ArtifactConfig`

Artifacts are written under the configured output root using a timestamped run directory.

Default artifact files:

- `quant_model.joblib`
- `metrics.json`
- `input_snapshot.csv`
- `predictions.csv`
- `feature_importance.csv`
- `backtest_summary.csv`
- `run_report.md`
- `interactive_report.html`
- `run_config.json`
- `statistical_tests.json`
- `analysis_workbook.xlsx`
- `artifact_manifest.json`
- `step_manifest.json`
- `generated_run.py`
- `HOW_TO_RERUN.md`
- `code_snapshot/`

## Pipeline Step Reference

Each major stage in the workflow is implemented as its own class.

### 1. Ingestion

Loads the starting dataframe from memory, CSV, or Excel and gives the rest of the framework a consistent input object.

### 2. Schema Management

Applies schema rules such as:

- renaming columns
- dropping columns
- creating columns
- enforcing dtypes

### 3. Target Construction

Builds the final modeling target from the configured source column. In binary
mode this creates a PD-style default flag. In continuous mode it standardizes
the source into the target used for LGD, regression, or forecasting workflows.

### 4. Validation

Checks preconditions before the remaining steps execute, such as:

- target existence
- target class diversity
- date-column availability for time-aware workflows
- entity-column availability for panel workflows

### 5. Cleaning

Applies general-purpose hygiene rules and removes clearly unusable data.

### 6. Feature Engineering

Builds lightweight derived features and finalizes which columns enter the model.

### 7. Splitting

Partitions the dataset into train, validation, and test sets according to the selected data structure.

### 8. Imputation

Fits the configured missing-value rules on the training split and applies them
consistently to every downstream split. The exported diagnostics bundle includes
an `imputation_rules` table so the treatment is fully documented.

### 9. Training

Fits the selected model family through a common adapter interface. Depending on configuration, this can mean:

- logistic regression
- elastic-net logistic regression
- scorecard logistic regression
- probit regression
- linear regression
- beta regression
- two-stage LGD model
- panel regression
- quantile regression
- Tobit regression
- XGBoost

When `execution.mode="score_existing_model"`, this stage instead loads a previously exported fitted model artifact and validates that the newly prepared dataframe still satisfies that model's raw feature contract.

### 10. Evaluation

Scores each split and computes metrics appropriate to the chosen target mode.

Binary-mode metrics include:

- ROC AUC
- average precision
- Brier score
- log loss
- KS statistic
- accuracy
- precision
- recall
- F1 score
- Matthews correlation
- confusion-matrix counts

Continuous-mode metrics include:

- RMSE
- MAE
- R-squared
- explained variance

When labels are unavailable in existing-model scoring mode, the framework still exports score-only summaries such as row counts, average score, predicted-positive rate, and score distributions while leaving label-dependent metrics blank.

### 11. Backtesting

Creates a simple risk-band summary on the test set by comparing predicted PD to observed default rate.

### 12. Diagnostics

Builds validation tables and interactive Plotly visuals such as:

- quantile plots
- calibration curves
- challenger model comparison tables and charts
- ROC and precision-recall curves
- threshold sweeps
- lift and gain charts
- missingness charts
- imputation rule tables
- correlation heatmaps
- feature importance charts
- permutation importance charts
- coefficient breakdowns
- feature effect curves
- residual plots
- QQ plots
- segment summaries
- ADF test tables
- PSI tables
- WoE/IV summaries
- feature policy checks
- scenario summaries and segment impacts

### 13. Artifact Export

Writes model artifacts, metrics, predictions, tables, figures, tests, workbook exports,
a markdown report, a standalone interactive HTML report, and a rerun-ready code bundle
to disk.

## Outputs

Each run writes a timestamped artifact directory under `artifacts/`.

Typical outputs include:

- trained model artifact
- exported input snapshot
- reused-model scoring outputs when an existing model is supplied
- per-split metrics
- per-row predictions
- model-specific feature importance or coefficient summary
- challenger comparison results and recommended model metadata
- backtest summary by risk band
- diagnostic tables as CSV
- feature policy check tables
- scenario summary tables and scenario definition tables
- statistical tests as JSON
- interactive HTML figures
- standalone grouped HTML validation report
- PNG figures when static export is enabled
- Excel workbook containing major tables
- human-readable run report
- resolved configuration used for the run
- ordered step manifest for the exact pipeline stack used
- generated Python launcher for non-GUI reruns
- local code snapshot containing the package, GUI, examples, tests, and project metadata

## Saved Run Bundles And Code Export

Each exported run directory is now intended to be portable. A completed run folder contains both the results and the Python material needed to inspect or replay the process outside Streamlit.

### What The Bundle Includes

- `run_config.json`
  The fully resolved configuration used for the run.
- `input_snapshot.csv`
  A CSV snapshot of the ingested dataframe, so the run can be reproduced without pointing back to the original upload.
- `step_manifest.json`
  The exact ordered list of pipeline steps used, including module and class names.
- `generated_run.py`
  A Python launcher that defaults to the exported config and input snapshot.
- `interactive_report.html`
  A grouped standalone validation dashboard that mirrors the visual taxonomy used in the GUI.
- `HOW_TO_RERUN.md`
  A short runbook describing the rerun path and the main editable files.
- `code_snapshot/`
  A copy of the framework source, GUI, tests, examples, README, and `pyproject.toml`.

### Why The Code Snapshot Exists

The GUI is useful for setup, but some users will want to inspect or modify step-level code after a run completes. The exported `code_snapshot/` addresses that concern directly.

You can edit files such as:

- `code_snapshot/src/quant_pd_framework/steps/diagnostics.py`
- `code_snapshot/src/quant_pd_framework/steps/export.py`
- `code_snapshot/src/quant_pd_framework/steps/training.py`
- `code_snapshot/src/quant_pd_framework/orchestrator.py`

The generated runner prepends `code_snapshot/src/` to `sys.path`, so those edits apply immediately when the user reruns the bundle.

### Easiest Non-GUI Rerun

From inside the exported run directory:

```powershell
python generated_run.py
```

By default this:

- reads `run_config.json`
- uses `input_snapshot.csv` if it exists
- writes the new results under `reruns/`
- imports the local code snapshot first

### Direct CLI Rerun

The packaged runner can also be invoked explicitly:

```powershell
python -m quant_pd_framework.run --config run_config.json --input input_snapshot.csv --output-root reruns
```

or:

```powershell
quant-pd-run --config run_config.json --input input_snapshot.csv --output-root reruns
```

If you want to score or retrain on a different dataset, keep the same config and change `--input`.

## Extension Points

The current project structure is intended to support future work without forcing a rewrite.

Common extension directions:

- add richer feature engineering
- add calibration steps
- add discrete-time hazard and survival-style credit models
- add more formal macroeconomic scenario libraries
- add model-governance templates for documentation sections and sign-off packs
- add alternate backtesting logic
- add experiment tracking or model registry integration

The most natural extension points are:

- `src/quant_pd_framework/config.py`
- `src/quant_pd_framework/orchestrator.py`
- `src/quant_pd_framework/steps/`

## Troubleshooting

### The GUI Does Not Launch

The default launcher now bootstraps a local `.venv` automatically. Start with:

```powershell
.\launch_gui.bat
```

If you want to rebuild the environment explicitly, run:

```powershell
.\setup_gui.bat
```

If `.venv` exists but is missing `pyvenv.cfg` or the venv Python executable, both scripts now treat it as invalid and rebuild it.

If you prefer to install manually without the batch scripts:

```powershell
python -m pip install -e .[dev,gui]
```

If that still fails, try:

```powershell
.venv\Scripts\python.exe -m quant_pd_framework.gui_launcher
```

or:

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

### I Need To Rerun A Completed GUI Run Without Streamlit

Open the exported run folder and use:

```powershell
python generated_run.py
```

If you prefer the packaged CLI instead:

```powershell
python -m quant_pd_framework.run --config run_config.json --input input_snapshot.csv --output-root reruns
```

If `input_snapshot.csv` is missing because input export was disabled, supply your own CSV or Excel path with `--input`.

### Excel Files Do Not Load

Excel support requires `openpyxl`, which is already included in the declared dependencies.

### Large File Uploads

The GUI is now configured for a `50 GB` Streamlit upload limit through
`.streamlit/config.toml`, the launcher command, and the uploader widget itself.

That limit is not the same as a guaranteed practical ingest size. Very large CSV
and Excel files can still fail if the machine does not have enough memory for
upload buffering and pandas parsing.

### Time-Series Or Panel Runs Fail Validation

Check that:

- one enabled schema row is marked as `date`
- the date column can actually be parsed as datetimes
- panel mode also has one enabled schema row marked as `identifier`

### Target Construction Fails

Usually this means one of the following:

- no target-source row was defined
- multiple target-source rows were defined
- the target source is not binary and `positive_values` were not supplied

## Workspace Cleanliness

Generated artifacts, pytest temp folders, `__pycache__`, editable-install metadata, and Ruff cache files are excluded through `.gitignore` so the working directory stays cleaner after future runs.

The automated tests were also refactored to use isolated temporary workspaces instead of writing durable output under the repository's `artifacts/` directory.

## Current Status

The current implementation includes:

- end-to-end PD workflow orchestration
- dataframe, CSV, and Excel inputs
- configurable schema and dtype handling
- cross-sectional, time-series, and panel split modes
- fresh-model and existing-model execution modes
- binary and continuous target modes
- workflow presets for PD, CECL, LGD, and CCAR development
- logistic, elastic-net logistic, scorecard logistic, probit, linear, beta, two-stage LGD, panel, quantile, Tobit, and XGBoost model options
- challenger comparison mode, feature policy checks, explainability outputs, and scenario testing
- expanded evaluation metrics and diagnostics
- labeled and score-only documentation paths when an existing model is reused
- interactive visualizations and richer exports
- workbook export
- rerun-ready code bundles with saved config, input snapshot, step manifest, generated launcher, and code snapshot
- explicit configuration validation and engineering-rubric documentation
- Streamlit GUI
- automated tests for pipeline flow, GUI config translation, GUI launching, model variants, existing-model scoring, and saved-bundle reruns
