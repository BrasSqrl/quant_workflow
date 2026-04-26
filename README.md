# Quant Studio

Quant Studio is a Python application and modeling framework for quantitative
model development, validation, scenario analysis, and documentation on tabular
data. Probability of default modeling remains the default use case, but the
platform also supports LGD and forecasting workflows that are common in CCAR
and CECL development. It is structured as a reusable, object-oriented framework
rather than a notebook-only prototype, so each primary modeling stage lives in
its own class and the entire run is coordinated by an orchestrator.

The underlying Python package and import path remain `quant_pd_framework`, and
the current distribution name remains `quant-pd-framework`, for compatibility
with the existing codebase and saved artifacts.

The project supports two working modes:

1. A Python API for developers who want to configure and run the pipeline in code.
2. A Streamlit GUI, branded in the app as `Quant Studio`, for users who want to load a dataframe, define schema rules, and run the workflow visually.

Inside those interfaces, the framework supports three execution modes:

1. `fit_new_model`
2. `score_existing_model`
3. `search_feature_subsets`

## What The Framework Does

The framework assumes you begin with one of the following inputs:

- a pandas dataframe
- a CSV file
- an Excel file
- a Parquet file

The default workflow then moves through the main stages of a quantitative modeling process:

1. Ingestion
2. Schema management
3. Target construction
4. Validation
5. Cleaning
6. Feature engineering
7. Train/validation/test split
8. Model-suitability and assumption checks
9. Imputation
10. Governed transformations
11. Variable selection and manual review
12. Model training
13. Evaluation
14. Backtesting
15. Diagnostics and visualization generation
16. Artifact export

When `score_existing_model` is used, the training stage becomes a
model-loading stage and the remaining steps run on newly scored data so the
existing model can still be documented, validated, stress tested, and exported
without refitting.

When `search_feature_subsets` is used, the normal training/evaluation/backtest
packaging path is replaced with a comparison-only workflow that:

- enumerates candidate feature subsets for the selected model family
- ranks them on held-out performance
- exports ROC/AUC, KS, feature-frequency, frontier, and significance-test views
- avoids writing full development artifacts such as a fitted model bundle or
  committee-ready report pack

The framework now supports:

- logistic regression
- discrete-time hazard model
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
- reproducibility manifest export with hashes, package versions, and optional
  git metadata
- feature dictionary / variable catalog support for business definitions,
  lineage, and inclusion rationale
- governed transformation layer for winsorization, log transforms, Yeo-Johnson,
  Box-Cox, natural splines, capped z-scores, piecewise-linear hinges, ratio
  features, lag and differencing features, EWMA and rolling features, rolling
  volatility features, percent-change features, interaction features, and
  manual bins
- variable-selection workflow with train-split screening and selection rationale
- manual review workflow for approve/reject decisions and scorecard bin
  overrides
- model-suitability and assumption checks before fitting
- scorecard development mode with monotonic binning, score scaling, and reason
  codes
- robustness and stability testing with repeated held-out resamples
- optional cross-validation diagnostics with stratified k-fold, regular k-fold,
  and time-aware expanding-window folds
- interactive scorecard workbench outputs for binning, WoE, points, and reason
  code review
- preset-aware workflow guardrails with pre-run readiness checks
- credit-risk-specific development diagnostics such as vintage curves,
  migration views, recovery segmentation, and macro sensitivity
- calibration workflow with method comparison, recalibration challengers, and
  recommended-method metadata
- advanced imputation and sensitivity testing with grouped train-fit fill rules,
  optional missingness-indicator features, KNN and iterative model-based
  imputation, multiple imputation with pooled surrogate summaries, Little's
  MCAR testing, and exported imputation-impact comparisons
- expanded statistical-test frameworks for distribution shifts, residual bias,
  outliers, dependency clustering, paired model-comparison significance,
  expanded specification testing, time-series econometrics, and structural
  breaks including CUSUM-style stability review
- feature-construction workbench outputs and preset recommendation tables for
  imputation, transformations, and tests
- documentation-pack generation for development-ready model summaries
- structured validation-pack generation for validator-facing review
- regulator-ready committee and validation report generation in DOCX and PDF
- normalized numerical warning capture with structured fit-health diagnostics
- polished regulator-ready reports with cover pages, section maps, section
  summaries, split appendices, and explicit numerical-stability sections
- Excel-based template import/export for offline governance edits
- reusable GUI configuration profiles for saving and reloading validated setup
  decisions across app launches
- feature policy checks for required/excluded features, missingness, VIF, IV,
  expected signs, and monotonicity
- explainability outputs such as coefficient breakdowns, permutation
  importance, PDP, ICE, centered ICE, ALE, two-way effects, marginal effects,
  interaction strength, and feature-bucket calibration
- scenario testing for feature-level shocks on held-out data
- large-data controls for Parquet intake, CSV-to-Parquet staging, DuckDB/PyArrow
  file-backed previews, dtype optimization, memory guardrails, sampled
  development, chunked full-data scoring, sampled exports, and Parquet output
  bundles

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

- [docs/ENGINEERING_RUBRIC.md](./docs/ENGINEERING_RUBRIC.md)
- [docs/RUBRIC_ALIGNMENT.md](./docs/RUBRIC_ALIGNMENT.md)
- [docs/UI_UX_STANDARD.md](./docs/UI_UX_STANDARD.md)
- [docs/UI_ENTERPRISE_REDESIGN.md](./docs/UI_ENTERPRISE_REDESIGN.md)
- [docs/DEVELOPMENT_ROADMAP.md](./docs/DEVELOPMENT_ROADMAP.md)

## Transparency and Auditability Guides

The repository now includes dedicated audit-oriented reference guides:

- [docs/STATISTICAL_TEST_CATALOG.md](./docs/STATISTICAL_TEST_CATALOG.md)
- [docs/MODEL_CATALOG.md](./docs/MODEL_CATALOG.md)
- [docs/METRIC_CATALOG.md](./docs/METRIC_CATALOG.md)
- [docs/PREPROCESSING_AND_DATA_TREATMENT_GUIDE.md](./docs/PREPROCESSING_AND_DATA_TREATMENT_GUIDE.md)
- [docs/GUI_TO_CODE_TRACEABILITY_GUIDE.md](./docs/GUI_TO_CODE_TRACEABILITY_GUIDE.md)
- [docs/LOGISTIC_REGRESSION_WALKTHROUGH.html](./docs/LOGISTIC_REGRESSION_WALKTHROUGH.html)
- [docs/SAGEMAKER_SETUP.md](./docs/SAGEMAKER_SETUP.md)

There is also an executive-level non-technical summary at:

- [EXECUTIVE_SUMMARY.txt](./EXECUTIVE_SUMMARY.txt)
- [SAGEMAKER_SETUP.txt](./SAGEMAKER_SETUP.txt)

## Reference Workflows

The repository now includes deterministic reference workflows for the core
intended development use cases:

- `PD Development`
- `LGD Severity`
- `Lifetime PD / CECL`
- `CCAR Forecasting`

These live under [examples/reference_workflows](./examples/reference_workflows) and
serve three purposes:

- copyable end-to-end examples for users
- locked regression targets for the test suite
- stable workflow contracts for CI hardening

Each reference workflow has:

- a deterministic synthetic input dataset defined in code
- a canonical config
- a locked expected-output contract under `examples/reference_workflows/expected`
- a golden-run regression test in `tests/test_reference_workflows.py`
- locked regulator-ready report artifacts and interactive-report sections
- required credit-risk development tables and figures where applicable
- a structured walkthrough pack under `examples/reference_workflows/packs`
- a `reference_example_pack.md` guide inside every exported reference run bundle

The repository also includes a GitHub Actions CI workflow at
[.github/workflows/ci.yml](./.github/workflows/ci.yml) that runs Ruff, the
golden reference checks, the Streamlit regression tests, and the full test
suite on push and pull request.

## Repository Layout

```text
quant/
  .github/
    workflows/
      ci.yml
  .streamlit/
    config.toml
  app/
    streamlit_app.py
  configs/
    saved_profiles/
      .gitkeep
  docs/
    DEVELOPMENT_ROADMAP.md
    ENGINEERING_RUBRIC.md
    GUI_TO_CODE_TRACEABILITY_GUIDE.md
    METRIC_CATALOG.md
    MODEL_CATALOG.md
    PREPROCESSING_AND_DATA_TREATMENT_GUIDE.md
    RUBRIC_ALIGNMENT.md
    STATISTICAL_TEST_CATALOG.md
    UI_ENTERPRISE_REDESIGN.md
    UI_UX_STANDARD.md
  examples/
    reference_workflows/
      expected/
        cecl_lifetime_pd.json
        ccar_forecasting.json
        lgd_severity.json
        pd_development.json
      packs/
        cecl_lifetime_pd.md
        ccar_forecasting.md
        lgd_severity.md
        pd_development.md
      cecl_lifetime_pd.py
      ccar_forecasting.py
      lgd_severity.py
      pd_development.py
      README.md
    run_development_workflow.py
    run_pipeline.py
    score_existing_model.py
  src/
    quant_pd_framework/
      __init__.py
      base.py
      config.py
      config_io.py
      config_serialization.py
      context.py
      export_layout.py
      gui_launcher.py
      gui_support.py
      models.py
      orchestrator.py
      presets.py
      reporting.py
      reference_workflows.py
      run.py
      sample_data.py
      diagnostics/
        assets.py
        registry.py
        scoring.py
      streamlit_ui/
        artifact_summary.py
        app_controller.py
        config_builder.py
        data.py
        error_guidance.py
        results.py
        run_execution.py
        state.py
        theme.py
        workflow_feedback.py
        workspace.py
      workflow_guardrails.py
      steps/
        assumption_checks.py
        comparison.py
        ingestion.py
        schema.py
        target.py
        validation.py
        cleaning.py
        feature_engineering.py
        feature_subset_search.py
        splitting.py
        transformations.py
        variable_selection.py
        training.py
        evaluation.py
        backtesting.py
        diagnostics.py
        export.py
  tests/
    test_artifact_contracts.py
    test_calibration_workflow.py
    test_existing_model_scoring.py
    test_feature_subset_search_mode.py
    test_development_features.py
    test_governance_extensions.py
    test_gui_launcher.py
    test_gui_support.py
    test_numerical_hardening_and_reporting.py
    test_performance_controls.py
    test_roadmap_features.py
    test_pipeline_smoke.py
    test_reference_workflows.py
    test_saved_run_bundle.py
    test_streamlit_app_e2e.py
    test_workflow_guardrails.py
    support.py
  scripts/
    bootstrap_sagemaker.sh
    benchmark_large_data.py
    profile_workflow.py
    run_sagemaker_streamlit.sh
    sagemaker_code_editor_lifecycle.sh
  requirements-sagemaker.txt
  EXECUTIVE_SUMMARY.txt
  SAGEMAKER_SETUP.txt
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

### Canonical Reference Examples

To run the canonical reference workflows directly:

```powershell
python examples\reference_workflows\pd_development.py
python examples\reference_workflows\lgd_severity.py
python examples\reference_workflows\cecl_lifetime_pd.py
python examples\reference_workflows\ccar_forecasting.py
```

Those examples write full artifact bundles under `artifacts/reference_workflows/`.

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

### Run In SageMaker Or Linux

For SageMaker, the recommended path is local VS Code Remote connected to
SageMaker compute. In the remote VS Code terminal:

```bash
bash scripts/bootstrap_sagemaker.sh
bash scripts/run_sagemaker_streamlit.sh
```

Then forward remote port `8501` from the local VS Code Ports panel and open the
forwarded local URL, usually `http://localhost:8501`.

The SageMaker bootstrap creates `.sagemaker_venv` in the repo and installs
Quant Studio there. This avoids changing SageMaker's own JupyterLab/Notebook
packages, which can otherwise produce resolver warnings from the preinstalled
IDE environment.

If you are not using local VS Code Remote and must use SageMaker JupyterLab's
browser proxy directly, use:

```bash
BASE_URL_PATH=/jupyterlab/default/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

For SageMaker Notebook Instances, the fallback proxy command is:

```bash
BASE_URL_PATH=/proxy/absolute/8501 bash scripts/run_sagemaker_streamlit.sh
```

The detailed guide is at [docs/SAGEMAKER_SETUP.md](./docs/SAGEMAKER_SETUP.md), and a
plain-text copy is kept at [SAGEMAKER_SETUP.txt](./SAGEMAKER_SETUP.txt).

### Run From A Saved Config Bundle

If you already have an exported run folder, you can rerun it without the GUI:

```powershell
python -m quant_pd_framework.run --config artifacts\20260418T000000Z\run_config.json
```

If the run folder includes `input_snapshot.csv` or `input_snapshot.parquet`,
the runner picks it up automatically. Otherwise supply `--input`.

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

- loads CSV, Excel, or Parquet data
- previews the incoming dataframe
- constructs a schema editor from the incoming columns
  - lets the user assign column roles and dtypes
  - lets the user define per-column missing-value treatment
  - lets the user rename or disable columns
- lets the user add columns that should be created if missing
- provides a feature-dictionary editor for business metadata and inclusion rationale
- provides a governed-transformation editor for reproducible feature engineering
- provides a workbook download/upload loop for offline governance review
- collects target, split, cleaning, feature-engineering, diagnostics, and model settings
- builds a `FrameworkConfig`
- calls the `QuantModelOrchestrator`
- displays metrics, validation charts, feature drilldowns, diagnostic tables, predictions, and artifact paths

The GUI is now organized as a thin entrypoint plus shared UI modules:

- `app/streamlit_app.py` is the Streamlit script entrypoint
- `src/quant_pd_framework/streamlit_ui/app_controller.py` coordinates the page
- `streamlit_ui/data.py` handles input loading and caching
- `streamlit_ui/state.py` owns typed session helpers and run snapshots
- `streamlit_ui/workspace.py` renders the dataset/schema workspace
- `streamlit_ui/results.py` renders readiness, results, and governance views
- `streamlit_ui/config_builder.py` assembles the preview configuration outside the raw UI flow
- `streamlit_ui/config_profiles.py` saves and reloads reusable GUI configuration profiles
- `streamlit_ui/theme.py` holds the shared visual system helpers
- `streamlit_ui/theme.py` also owns the command bar, four-step workflow tabs,
  main-canvas cards, and shared visual styling

### GUI Design System

The current interface is intentionally styled as a premium light-mode fintech dashboard rather than a default Streamlit application.

The governing visual and interaction standards are documented in:

- [docs/UI_UX_STANDARD.md](./docs/UI_UX_STANDARD.md)
- [docs/UI_ENTERPRISE_REDESIGN.md](./docs/UI_ENTERPRISE_REDESIGN.md)

That standard drives both the live GUI and the exported standalone HTML report. The design system emphasizes:

- a light enterprise-fintech palette with stronger visual hierarchy
- a command-bar header and four large clickable workflow steps
- a two-column Step 2 model-configuration workspace for faster scanning
- grouped diagnostics instead of one long undifferentiated result page
- main-canvas model configuration, readiness, and artifact workspaces instead
  of persistent side panes
- consistent Plotly theming across all charts
- metric cards, section shells, and filter controls that make scanning easier for model builders and validation teams
- the same section taxonomy in both Streamlit and exported reports so users do not have to relearn the layout

### Main GUI Controls

The GUI exposes the following decision areas:

- workspace mode (`guided` vs `advanced`)
- workflow preset
- reusable configuration profile save/load
- execution mode
- file upload with a configured 50 GB per-file Streamlit limit
- `Data_Load/` landing-zone file selection for CSV, Excel, and Parquet datasets
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
- scorecard monotonicity, minimum bin share, scaling, and reason-code settings
- cleaning toggles
- date feature toggles
- per-column missing-value policy and constant-fill controls in the column designer
- feature dictionary rows for business definitions, source systems, units,
  ranges, expected signs, and inclusion rationale
- governed transformations for winsorization, log1p, ratio, interaction, and
  manual-bin features
- challenger model selection and ranking metric
- feature policy inputs for required/excluded features, sign expectations, and monotonicity
- variable-selection controls for maximum features, univariate threshold,
  correlation threshold, and locked include/exclude fields
- suitability-check controls for class balance, events per feature, and category
  concentration
- manual feature-review decisions and scorecard bin overrides
- documentation-pack fields for model purpose, target definition, horizon,
  assumptions, exclusions, limitations, and reviewer notes
- explainability controls for permutation importance and feature effect curves
- scenario table for held-out stress testing
- calibration bin count, binning strategy, recalibration challengers, and
  calibration ranking metric
- reproducibility-manifest controls for tracked package versions and git capture
- diagnostics and export toggles
- Large Data Mode, memory guardrails, dtype optimization, CSV-to-Parquet
  staging, training sample size, full-data scoring chunk size, tabular output
  format, and sampled export policy
- artifact output root
- schema editor for column-level configuration

The workspace mode is intended to keep standard model-development runs compact:

- `guided` mode keeps advanced comparison, review, explainability, and
  documentation surfaces on the preset defaults
- `advanced` mode unlocks the full tuning and governance surface

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
They now also seed calibration, variable-selection, documentation, and
scorecard-development defaults where those are relevant to the workflow.

### Configuration Profiles

Step 2 includes a `Configuration Profiles` panel for saving and reloading GUI
setup decisions across app launches. A profile captures the resolved
`FrameworkConfig`, the column designer, feature dictionary, transformation
table, manual-review table, scorecard override table, and dataset fingerprint
metadata. It does not store raw source data rows.

Profiles can be:

- saved locally under `configs/saved_profiles/`
- downloaded as portable JSON
- loaded from the local saved-profile list
- imported from a downloaded JSON profile

Saved profile JSON files are git-ignored by default because they are
user-specific working artifacts. The folder is kept in the repository with a
`.gitkeep` file so the local save location is obvious.

When a profile is loaded against a different dataset, the GUI applies the saved
configuration but shows non-blocking warnings for missing columns, new columns,
or row-count changes. This supports reuse while keeping the dataset mismatch
visible before execution.

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

When `search_feature_subsets` is selected:

- choose one currently supported model family
- optionally narrow the candidate-feature pool
- set minimum and maximum subset size bounds
- choose the held-out ranking split and ranking metric
- review a comparison-only result surface centered on candidate ranking, ROC,
  KS, feature frequency, frontier charts, and paired significance tests

### GUI Relationship To The Python Code

The GUI is not the authoritative implementation of the workflow. It is a configuration shell over the Python package.

That means:

- GUI selections are converted into a `FrameworkConfig`
- the same `QuantModelOrchestrator` runs whether the pipeline starts from the GUI or from code
- exported run folders now include a Python rerun bundle so a user can leave the GUI entirely after setup
- the Streamlit script itself stays small while the UI behavior is split into
  reusable modules for state, rendering, caching, and config assembly

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
    input_path="artifacts/20260418T000000Z/input_snapshot.parquet",
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
- `calibration`
- `scorecard`
- `variable_selection`
- `documentation`
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
- optionally group scalar imputation by one or more segment columns
- optionally create a missingness-indicator feature
- add a new column if it is missing
- decide whether the original source column should be retained

Examples:

- disable a column: `ColumnSpec(name="legacy_field", enabled=False)`
- rename a column: `ColumnSpec(name="borrower_age", source_name="age_years")`
- force float dtype: `ColumnSpec(name="balance", dtype="float")`
- set numeric imputation: `ColumnSpec(name="balance", missing_value_policy=MissingValuePolicy.MEDIAN)`
- set constant fill: `ColumnSpec(name="channel", missing_value_policy=MissingValuePolicy.CONSTANT, missing_value_fill_value="unknown")`
- group imputation by segment: `ColumnSpec(name="balance", missing_value_policy=MissingValuePolicy.MEDIAN, missing_value_group_columns=["portfolio"])`
- create a missingness flag: `ColumnSpec(name="balance", create_missing_indicator=True)`
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
- `missing_value_group_columns` lets mean/median/mode-style imputation learn segment-specific fill values on the training split before falling back to the global learned fill value
- `create_missing_indicator=True` adds a numeric `<feature>__missing_indicator` feature before imputation is applied
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
- `search_feature_subsets`

Recommended scoring pattern:

- point `existing_model_path` at a prior exported `quant_model.joblib`
- point `existing_config_path` at the matching prior `run_config.json`
- let the framework reuse the prior schema, feature, target, split, and model settings so the new scoring run stays aligned with the original model contract

If `mode="score_existing_model"`, `existing_model_path` is required.

If `mode="search_feature_subsets"`, the run must use a binary target and a
supported binary model family. The subset-search workflow exports
comparison-only evidence rather than a fitted-model artifact bundle, with a
selected-candidate coefficient summary, selected-candidate ROC / KS visuals,
and ranked non-winning candidate tables for side-by-side comparison.

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

### `FeatureDictionaryConfig`

`FeatureDictionaryConfig` stores business metadata for modeled features so runs
can be reviewed as more than a list of column names.

Important fields:

- `entries`
- `require_documentation_for_selected_features`

Each `FeatureDictionaryEntry` can capture:

- `feature_name`
- `business_name`
- `definition`
- `source_system`
- `unit`
- `allowed_range`
- `missingness_meaning`
- `expected_sign`
- `inclusion_rationale`
- `notes`

When dictionary coverage is required, the pipeline fails if selected modeled
features do not have documented definitions.

### `TransformationConfig`

`TransformationConfig` controls governed feature transformations that are fit on
the training split and replayed consistently everywhere else.

Important fields:

- `enabled`
- `transformations`
- `error_on_failure`
- `auto_interactions_enabled`
- `include_numeric_numeric_interactions`
- `include_categorical_numeric_interactions`
- `max_auto_interactions`
- `max_categorical_levels`
- `min_interaction_score`

Supported transformation families:

- `winsorize`
- `log1p`
- `box_cox`
- `natural_spline`
- `yeo_johnson`
- `capped_zscore`
- `piecewise_linear`
- `ratio`
- `interaction`
- `lag`
- `difference`
- `ewma`
- `rolling_mean`
- `rolling_median`
- `rolling_min`
- `rolling_max`
- `rolling_std`
- `pct_change`
- `manual_bins`

Each `TransformationSpec` defines the source feature, optional secondary
feature, output feature, and transform-specific parameters such as quantiles,
categorical indicator values, lag windows, rolling windows, z-score caps, or
manual bin edges. Auto-generated screened interactions are persisted back into
the saved run config so existing-model scoring can replay the same features.

### `AdvancedImputationConfig` And Expanded Diagnostic Framework Configs

The framework now also exposes typed config objects for the newer roadmap
frameworks. The most important ones are:

- `AdvancedImputationConfig`
- `DistributionDiagnosticConfig`
- `ResidualDiagnosticConfig`
- `OutlierDiagnosticConfig`
- `DependencyDiagnosticConfig`
- `TimeSeriesDiagnosticConfig`
- `StructuralBreakConfig`
- `FeatureWorkbenchConfig`
- `PresetRecommendationConfig`

These configs control the deeper review surfaces added in the latest roadmap
phase, including model-based imputation, distribution and dependency testing,
residual and outlier analysis, structural-break review, engineered-feature
workbench outputs, and preset-aligned recommendation tables.

### `ExplainabilityConfig`

`ExplainabilityConfig` controls optional interpretability outputs.

Important fields:

- `permutation_importance`
- `feature_effect_curves`
- `partial_dependence`
- `ice_curves`
- `centered_ice_curves`
- `accumulated_local_effects`
- `two_way_effects`
- `effect_confidence_bands`
- `monotonicity_diagnostics`
- `segmented_effects`
- `effect_stability`
- `marginal_effects`
- `interaction_strength`
- `effect_calibration`
- `coefficient_breakdown`
- `top_n_features`
- `grid_points`
- `sample_size`
- `ice_sample_size`
- `effect_band_resamples`
- `two_way_grid_points`
- `max_effect_segments`

Depending on model family, this can produce coefficient tables, odds-ratio-style
breakdowns, permutation importance, PDP, ICE, centered ICE, ALE, two-way effect
heatmaps, confidence bands, monotonicity diagnostics, segmented and split-stable
effect curves, average marginal effects, interaction-strength tables,
effect-by-calibration views, WoE tables, and two-stage LGD coefficient outputs.

### `CalibrationConfig`

`CalibrationConfig` controls probability-alignment diagnostics and
recalibration challengers for binary workflows.

Important fields:

- `bin_count`
- `strategy`
- `platt_scaling`
- `isotonic_calibration`
- `ranking_metric`

When labels are available, the framework fits the selected calibration
challengers on the validation split, evaluates them on the held-out test split,
and exports:

- a bin-level calibration table
- a calibration summary table
- a calibration method-comparison chart
- recommended calibration method metadata
- calibrated score columns in the exported predictions output

### `ScorecardConfig`

`ScorecardConfig` controls scorecard-development behavior for
`scorecard_logistic_regression`.

Important fields:

- `monotonicity`
- `min_bin_share`
- `base_score`
- `points_to_double_odds`
- `odds_reference`
- `reason_code_count`

These settings drive monotonic binning, score scaling, points tables, and the
reason-code columns exported with scorecard predictions.

### `ScorecardWorkbenchConfig`

`ScorecardWorkbenchConfig` controls the dedicated scorecard binning workspace
for `scorecard_logistic_regression`.

Important fields:

- `enabled`
- `max_features`
- `include_score_distribution`
- `include_reason_code_analysis`

These settings control the scorecard-specific workbench outputs, including:

- feature-level IV and bin summary tables
- points-distribution visuals
- reason-code frequency views
- bucket-level bad-rate, WoE, and partial-points visuals for the profiled
  scorecard features

### `VariableSelectionConfig`

`VariableSelectionConfig` controls the optional train-split feature-screening
workflow.

Important fields:

- `enabled`
- `max_features`
- `min_univariate_score`
- `correlation_threshold`
- `locked_include_features`
- `locked_exclude_features`

The selection step exports a feature-by-feature rationale table so the chosen
development set is reviewable rather than hidden inside model training.

### `ManualReviewConfig`

`ManualReviewConfig` captures human review decisions that sit on top of
screening and scorecard-development outputs.

Important fields:

- `reviewer_name`
- `require_review_complete`
- `feature_decisions`
- `scorecard_bin_overrides`

Supported feature decisions:

- `approve`
- `reject`
- `force_include`
- `force_exclude`

Scorecard bin overrides allow explicit internal numeric cutoffs to replace the
default monotonic bin search for selected scorecard features.

### `SuitabilityCheckConfig`

`SuitabilityCheckConfig` controls pre-fit model-suitability and assumption
checks.

Important fields:

- `min_events_per_feature`
- `min_class_rate`
- `max_class_rate`
- `max_dominant_category_share`
- `min_non_null_target_rows`
- `error_on_failure`

These checks populate the exported `assumption_checks` table and can optionally
fail the run when the development sample is not suitable for the chosen model
family.

### `DocumentationConfig`

`DocumentationConfig` captures the narrative and governance fields used to build
the exported development documentation pack.

Important fields:

- `model_name`
- `model_owner`
- `business_purpose`
- `portfolio_name`
- `segment_name`
- `horizon_definition`
- `target_definition`
- `loss_definition`
- `assumptions`
- `exclusions`
- `limitations`
- `reviewer_notes`

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
- model specification tests
- forecasting statistical tests
- calibration analysis
- threshold analysis
- lift and gain analysis
- segment analysis
- residual analysis
- quantile analysis
- QQ analysis
- per-figure HTML exports
- per-figure PNG exports
- Excel workbook export

### `PerformanceConfig`

`PerformanceConfig` controls large-run safeguards and runtime tradeoffs.

Important fields include:

- `large_data_mode`
- `diagnostic_sample_rows`
- `optimize_dtypes`
- `downcast_numeric`
- `convert_low_cardinality_strings`
- `convert_csv_to_parquet`
- `csv_conversion_chunk_rows`
- `large_data_training_sample_rows`
- `large_data_score_chunk_rows`
- `large_data_project_columns`
- `large_data_auto_stage_parquet`
- `memory_limit_gb`
- `memory_estimate_file_multiplier`
- `memory_estimate_dataframe_multiplier`

When large-data mode is enabled in the GUI, Quant Studio defaults to safer
settings: file-backed Data_Load intake, sampled diagnostics, disabled
robustness and cross-validation refits, disabled per-figure file exports,
optional dtype optimization, reusable CSV-to-Parquet staging for file-path
inputs, configurable training sample size, chunked full-data scoring, and
sampled or Parquet-first tabular outputs.

For file-backed runs, the large-data workflow is intentionally split:

- `sample_development/` contains the governed sample loaded into pandas for
  model fitting, diagnostics, and documentation.
- `full_data_scoring/` contains chunked full-file scoring outputs, including
  full-data predictions written directly to Parquet.
- `large_data_metadata/` contains metadata describing the source path, staged
  Parquet path, projected columns, sample size, chunk size, row counts, and
  chunk-progress status.

### `RobustnessConfig`

`RobustnessConfig` controls repeated-resample robustness testing for new-model
development runs.

Important fields:

- `enabled`
- `resample_count`
- `sample_fraction`
- `sample_with_replacement`
- `evaluation_split`
- `metric_stability`
- `coefficient_stability`
- `random_state`

When enabled, the framework repeatedly refits the active model on train
resamples, scores a held-out split, and exports:

- resample-level metric distributions
- metric summary tables with mean, spread, and percentile values
- feature and coefficient stability tables
- stability charts under the `Stability / Drift` reporting section

### `CrossValidationConfig`

`CrossValidationConfig` controls optional fold-based validation diagnostics for
new-model development runs. It does not replace the final saved model. The final
model artifact is still fit once on the configured training split.

Important fields:

- `enabled`
- `fold_count`
- `strategy`
- `shuffle`
- `metric_stability`
- `coefficient_stability`
- `random_state`

When `strategy="auto"`, the framework uses:

- stratified k-fold validation for binary cross-sectional workflows
- regular k-fold validation for continuous cross-sectional workflows
- expanding-window time-series validation for time-series and panel workflows

When enabled, the framework exports:

- fold-level validation metrics
- metric distribution and summary tables
- feature and coefficient stability tables across folds
- metric and feature-stability charts

### `ReproducibilityConfig`

`ReproducibilityConfig` controls run-manifest metadata used for auditability and
reruns.

Important fields:

- `capture_git_metadata`
- `package_names`

When enabled, the export bundle includes hashes for the input dataframe, model
artifact, and resolved config, plus Python/platform metadata and tracked
package versions.

### `ArtifactConfig`

Artifacts are written under the configured output root using a timestamped run directory.
Run folders use a readable UTC date/time name such as
`run_2026-04-24_15-42-10_UTC`.

Important behavior:

- `interactive_report.html` is always exported for completed runs.
- `tabular_output_format` controls whether major tabular artifacts are written
  as CSV, Parquet, or both.
- `large_data_export_policy` can write full tabular outputs, sampled CSV
  outputs with full Parquet outputs, or metadata-only entries for very large
  tables.
- Individual figure `.html` and `.png` files are disabled by default and can be
  enabled through `ArtifactConfig.export_individual_figure_files` or the
  matching GUI toggle when separate chart files are needed.
- `export_profile` controls how much supporting evidence is packaged:
  `standard` preserves the normal governed bundle, `fast` skips heavier
  distribution assets such as Excel workbooks, regulatory DOCX/PDF reports,
  input snapshots, and code snapshots, and `audit` is reserved for full-review
  runs.
- `run_debug_trace.json` is exported for every run and records step timing,
  completion status, shape snapshots, and error details when a step fails.

Default artifact files:

- `quant_model.joblib`
- `metrics.json`
- `input_snapshot.csv`
- `input_snapshot.parquet` when Parquet or dual tabular output is selected
- `predictions.csv`
- `predictions.parquet` when Parquet or dual tabular output is selected
- `feature_importance.csv`
- `backtest_summary.csv`
- `run_report.md`
- `interactive_report.html`
- `run_config.json`
- `statistical_tests.json`
- `analysis_workbook.xlsx`
- `artifact_manifest.json`
  This now indexes the core run artifacts, export directories, figures, and
  rerun bundle in one file.
- `step_manifest.json`
- `run_debug_trace.json`
- `model_documentation_pack.md`
- `validation_pack.md`
- `reproducibility_manifest.json`
- `configuration_template.xlsx`
- `generated_run.py`
- `HOW_TO_RERUN.md`
- `code_snapshot/`
- `model_bundle_for_monitoring/`
  A versioned handoff bundle for the separate monitoring application. It
  includes the approved model artifact, resolved run config, generated runner,
  monitoring metadata, predictions, artifact manifest, and optional snapshot
  assets when those exports were enabled.

## Pipeline Step Reference

Each major stage in the workflow is implemented as its own class.

### 1. Ingestion

Loads the starting dataframe from memory, CSV, Excel, or Parquet and gives the
rest of the framework a consistent input object. File-path CSV inputs can be
converted to Parquet in chunks before ingestion when `convert_csv_to_parquet`
is enabled.

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

### 8. Assumption Checks

Runs pre-fit model-suitability diagnostics such as:

- non-null target count
- class balance and events per feature for binary models
- bounded-target and censoring checks for relevant continuous models
- duplicate entity-date checks for panel workflows
- dominant-category concentration checks

The exported diagnostics bundle includes an `assumption_checks` table that can
optionally fail the run when configured thresholds are breached.

### 9. Imputation

Fits the configured missing-value rules on the training split and applies them
consistently to every downstream split. The exported diagnostics bundle includes
an `imputation_rules` table so the treatment is fully documented.

The current advanced-imputation slice also supports:

- grouped train-fit scalar imputation using user-selected segment columns
- generated missingness-indicator features that can enter downstream selection
  and modeling
- KNN and iterative model-based numeric imputation with train-fit auxiliary
  feature selection and scalar fallback behavior
- multiple imputation with pooled surrogate coefficient and metric summaries
- approximate Little's MCAR testing for missing-completely-at-random review
- an `advanced_imputation_summary` table that exports the model-based
  imputation surface
- a `multiple_imputation_pooling_summary` table that exports pooled metric
  results
- a `multiple_imputation_pooled_coefficients` table that exports pooled
  coefficient estimates
- a `littles_mcar_test` table for missingness-randomness review
- an `imputation_group_rules` table that exports the learned group-specific
  fill values
- an `imputation_sensitivity_summary` table that ranks features where alternate
  fill choices materially change scores or held-out metrics
- an `imputation_sensitivity_detail` table with policy-by-policy score and
  metric deltas
- a `performance_hardening_actions` table when the framework samples or limits
  expensive large-run diagnostics

### 10. Governed Transformations

Fits and applies explicit, reproducible feature transformations such as:

- winsorization
- `log1p`
- `box_cox`
- `natural_spline`
- `yeo_johnson`
- `capped_zscore`
- `piecewise_linear`
- ratio features
- interaction features
- lag features
- difference features
- EWMA features
- rolling-mean features
- rolling-median features
- rolling-min features
- rolling-max features
- rolling-standard-deviation features
- percent-change features
- manual-bin categorical features

These transforms are fit on the training split, replayed on the remaining
splits, and exported through a `governed_transformations` audit table. When the
interaction engine is enabled, train-split screened candidates are exported
through `interaction_candidates` and the selected interactions are persisted
into the saved run config.

### 11. Variable Selection

Applies train-split screening rules before model fitting. This workflow can:

- rank variables by simple univariate power
- remove highly correlated numeric features
- enforce locked include and exclude lists
- cap the selected feature count
- apply manual approve/reject/force decisions after screening
- export both a selection rationale table and manual review decisions for documentation

### 12. Training

Fits the selected model family through a common adapter interface. Depending on configuration, this can mean:

- logistic regression
- discrete-time hazard model
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

When `execution.mode="search_feature_subsets"`, the normal training and
downstream final-model packaging path is replaced by the dedicated
`FeatureSubsetSearchStep`, which fits the selected model family across
candidate feature subsets and exports only comparison-ready evidence.

When scorecard development is active, this stage can also apply manual numeric
bin overrides supplied through the review workflow.

### 13. Evaluation

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

### 14. Backtesting

Creates a simple risk-band summary on the test set by comparing predicted PD to observed default rate.

### 15. Diagnostics

Builds validation tables and interactive Plotly visuals such as:

- suitability / assumption checks
- feature dictionary coverage tables
- governed transformation audit tables
- quantile plots
- lifetime PD curves for discrete-time hazard development
- calibration curves
- calibration summary tables with ECE, MCE, slope/intercept, and
  Hosmer-Lemeshow statistics
- base-versus-recalibrated method comparison charts
- challenger model comparison tables and charts
- ROC and precision-recall curves
- threshold sweeps
- lift and gain charts
- missingness charts
- imputation rule tables
- grouped imputation rule tables when segment-aware imputation is used
- generated missingness-indicator features when enabled in the column designer
- imputation sensitivity summary/detail tables
- interaction candidate tables when the interaction engine is enabled
- missingness stability, target-association, and indicator-correlation tables
- correlation heatmaps
- feature importance charts
- variable-selection tables
- manual review decision tables
- permutation importance charts
- coefficient breakdowns
- feature effect curves
- scorecard WoE, scaling, and points tables
- scorecard bin override tables
- residual plots
- QQ plots
- segment summaries
- ADF test tables
- model specification test tables and influence plots
- forecasting statistical test tables for time-aware workflows
- PSI tables
- WoE/IV summaries
- feature policy checks
- reproducibility manifest tables
- scenario summaries and segment impacts

### 16. Cross-Validation Diagnostics

Optionally fits temporary fold-level models on the training split to evaluate
metric and feature stability. This step does not replace `quant_model.joblib`;
the exported model remains the model fit by the normal training step.

Cross-sectional binary models use stratified k-fold validation by default.
Cross-sectional continuous models use regular k-fold validation. Time-series and
panel models use expanding-window validation to reduce look-ahead leakage risk.

### 17. Artifact Export

Writes model artifacts, metrics, predictions, tables, figures, tests, workbook exports,
a markdown report, a standalone interactive HTML report, a development
documentation pack, a structured validation pack, a reproducibility manifest, a
configuration workbook, and a rerun-ready code bundle to disk.

## Outputs

Each run writes a timestamped artifact directory under `artifacts/`.

Typical outputs include:

- trained model artifact
- exported input snapshot
- reused-model scoring outputs when an existing model is supplied
- per-split metrics
- per-row predictions
- scorecard points and reason codes when scorecard development mode is used
- feature dictionary coverage tables
- governed transformation audit tables
- assumption-check tables
- recommended calibration score columns and method metadata for binary runs
- variable-selection tables and selected-feature metadata
- manual feature-review decisions and scorecard override tables
- model-specific feature importance or coefficient summary
- cross-validation fold metrics and stability summaries when enabled
- challenger comparison results and recommended model metadata
- lifetime PD curve outputs for discrete-time hazard runs
- backtest summary by risk band
- diagnostic tables as CSV
- diagnostic tables as Parquet when Parquet or dual tabular output is selected
- feature policy check tables
- model numerical diagnostics and normalized numerical warning tables
- scenario summary tables and scenario definition tables
- statistical tests as JSON
- interactive HTML figures
- standalone grouped HTML validation report
- PNG figures when static export is enabled
- Excel workbook containing major tables
- human-readable run report
- documentation pack for development and review
- validator-facing validation pack
- reproducibility manifest with hashes and package versions
- input source metadata for uploaded, bundled, or `Data_Load/` datasets
- editable configuration workbook for offline governance
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
- `input_snapshot.parquet`
  A full Parquet snapshot when Parquet or dual tabular output is selected.
- `step_manifest.json`
  The exact ordered list of pipeline steps used, including module and class names.
- `generated_run.py`
  A Python launcher that defaults to the exported config and input snapshot.
- `interactive_report.html`
  A grouped standalone validation dashboard that mirrors the visual taxonomy used in the GUI.
- `model_documentation_pack.md`
  A development-ready narrative pack built from the run configuration, metrics,
  calibration review, and selected-feature summary.
- `validation_pack.md`
  A validator-facing markdown pack focused on assumptions, review decisions,
  challenger outcomes, and artifact index.
- `committee_report.docx/.pdf` and `validation_report.docx/.pdf`
  Polished regulator-ready documents with a cover page, report map, section
  summaries, split appendices, and estimation-health section.
- `reference_example_pack.md`
  A workflow-specific reading guide included in each reference example run.
- `reproducibility_manifest.json`
  Hashes, package versions, input source metadata, environment metadata, and
  optional git information.
- `configuration_template.xlsx`
  The editable workbook for schema, feature dictionary, transformations, and
  review tables.
- `artifact_manifest.json`
  A machine-readable index of the core artifacts, export directories, figures,
  regulator-ready reports, and rerun-bundle assets.
- `HOW_TO_RERUN.md`
  A short runbook describing the rerun path and the main editable files.
- `code_snapshot/`
  A copy of the framework source, GUI, tests, examples, README, and `pyproject.toml`.
- `model_bundle_for_monitoring/`
  A monitoring-ready handoff bundle containing `quant_model.joblib`,
  `run_config.json`, `generated_run.py`, `monitoring_metadata.json`,
  `artifact_manifest.json`, predictions in the selected primary tabular format,
  and optional input snapshot and `code_snapshot/` assets when those exports
  were enabled for the run.

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
- uses `input_snapshot.csv` or `input_snapshot.parquet` if either exists
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

If `input_snapshot.csv` and `input_snapshot.parquet` are missing because input
export was disabled, supply your own CSV, Excel, or Parquet path with `--input`.

### Excel Files Do Not Load

Excel support requires `openpyxl`, which is already included in the declared dependencies.

### Large File Uploads

The GUI is now configured for a `50 GB` Streamlit upload limit through
`.streamlit/config.toml`, the launcher command, and the uploader widget itself.

That limit is not the same as a guaranteed practical ingest size. Very large
CSV, Excel, and Parquet files can still fail if the machine does not have
enough memory for upload buffering and pandas parsing. Browser upload also
requires Streamlit to receive the file before pandas parses it, so `Data_Load/`
is preferred for multi-GB datasets.

To keep large runs usable, the framework now also applies lightweight
performance safeguards through `PerformanceConfig`, including:

- dataset-size warnings in the GUI
- file-backed Large Data Mode for `Data_Load/` and CLI file-path runs, avoiding
  eager full-file pandas loads before the user runs the workflow
- large-data mode defaults that turn off expensive optional refit diagnostics
  until the user opts back in
- memory-estimate tables and configurable pre-run warning thresholds
- dtype optimization audit tables for numeric downcasting and low-cardinality
  string-to-category conversion
- reusable chunked CSV-to-Parquet staging for file-path inputs
- governed sample-fit / full-data-score execution, where the development model
  is fit on a configurable sample and the full file is scored in chunks
- smaller dataset previews in the builder workspace
- capped multiple-imputation surrogate sampling for expensive diagnostics
- truncated HTML report table and figure previews so standalone reports remain usable
- lazy Streamlit result snapshots for large runs so the GUI does not keep every
  exported row in memory
- sampled CSV exports or full Parquet exports for predictions and input
  snapshots
- separate `sample_development/`, `full_data_scoring/`, and
  `large_data_metadata/` output folders for audit clarity
- export profiles that let users choose between faster development exports and
  fuller audit-oriented packaging

Explainability outputs such as PDP, ICE, ALE, two-way effects, marginal
effects, interaction strength, and macro sensitivity use batched scoring where
practical. This reduces repeated preprocessing through the model adapter while
preserving the exported tables and figures.

### Workflow Failure Guidance

The Streamlit app now classifies common run failures into user-facing recovery
messages instead of only showing the raw Python exception. The guidance covers
memory pressure, target setup, date and panel split setup, invalid feature
values, model convergence issues, large-data staging failures, and incomplete
artifact export. The original traceback remains available in an expandable
technical detail panel for debugging.

### Output Location Guidance

After a run completes, the GUI surfaces the most important output locations
immediately: the run folder, interactive report, model object, run
configuration, reproducibility manifest, debug trace, predictions, and any
Large Data Mode folders. The Governance result panel also renders artifact
locations as a readable table rather than a raw JSON block.

### Profiling And Benchmarking

Two scripts support performance investigation without changing the application
workflow:

```powershell
python scripts/profile_workflow.py --config artifacts\run_id\run_config.json --input Data_Load\sample.parquet --profile-output artifacts\profile.json
```

This profiles a saved configuration run, records elapsed time, peak traced
memory, slowest pipeline steps, diagnostic counts, and artifact locations.

```powershell
python scripts/benchmark_large_data.py --rows 100000 --features 12 --output-root artifacts\benchmarks
```

This generates synthetic Parquet data, runs Large Data Mode with sample-fit and
chunked full-data scoring, and writes a benchmark JSON file. It is intended for
before/after comparisons when optimizing diagnostics, exports, or large-data
scoring.

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
- dataframe, CSV, Excel, and Parquet inputs
- configurable schema and dtype handling
- cross-sectional, time-series, and panel split modes
- fresh-model, existing-model, and feature-subset-search execution modes
- binary and continuous target modes
- workflow presets for PD, CECL, LGD, and CCAR development
- logistic, discrete-time hazard, elastic-net logistic, scorecard logistic,
  probit, linear, beta, two-stage LGD, panel, quantile, Tobit, and XGBoost
  model options
- challenger comparison mode, feature policy checks, explainability outputs,
  calibration workflow, variable selection, scorecard development support,
  feature dictionary capture, governed transformations, manual review,
  suitability checks, documentation-pack generation, validation-pack export,
  reproducibility manifest export, and scenario testing
- expanded evaluation metrics and diagnostics
- labeled and score-only documentation paths when an existing model is reused
- interactive visualizations and richer exports
- workbook export
- feature subset search mode for comparison-only feature-set selection with selected-candidate and ranked non-winning comparison outputs
- rerun-ready code bundles with saved config, input snapshot, step manifest, generated launcher, and code snapshot
- explicit configuration validation and engineering-rubric documentation
- user-facing failure guidance with expandable technical tracebacks
- readable run-output location panels and artifact summary tables in the GUI
- diagnostic registry output that records enabled, emitted, disabled, and
  skipped diagnostic surfaces
- profiling and synthetic Large Data Mode benchmark scripts for performance
  regression review
- Streamlit GUI
- automated tests for pipeline flow, GUI config translation, GUI launching, model variants, existing-model scoring, and saved-bundle reruns
