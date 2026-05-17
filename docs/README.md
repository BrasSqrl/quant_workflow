# Quant Studio Documentation Home

This page is the starting point for Quant Studio documentation. It separates
quick-start material from setup guides, user workflow guides, technical
references, governance standards, release evidence, and root convenience files.

## Start Here

| Need | Read |
| --- | --- |
| Run the app for the first time | [User Quick Start Guide](./user_guides/QUICK_START.md) |
| Build a simple PD logistic regression model | [PD Logistic Regression Walkthrough](./user_guides/PD_LOGISTIC_REGRESSION_WALKTHROUGH.html) |
| Understand what data the app expects | [Data Requirements Guide](./user_guides/DATA_REQUIREMENTS.md) |
| Choose the right execution mode | [Execution Mode Decision Guide](./user_guides/EXECUTION_MODE_DECISION_GUIDE.md) |
| Find and understand exported files | [Artifact Map](./user_guides/ARTIFACT_MAP.md) |
| Pick a model family | [Model Selection Guide](./user_guides/MODEL_SELECTION_GUIDE.md) |
| Walk through each selectable model type | [Model Type Reference Guide](./user_guides/MODEL_TYPE_REFERENCE_GUIDE.md) |
| Configure a common workflow | [Configuration Cookbook](./user_guides/CONFIGURATION_COOKBOOK.md) |
| Review a completed model package and decision summary | [Validation Reviewer Guide](./user_guides/VALIDATION_REVIEWER_GUIDE.md) |
| Fix common app or run problems | [Troubleshooting Guide](./user_guides/TROUBLESHOOTING.md) |
| Work with multi-GB datasets | [Large Data Playbook](./user_guides/LARGE_DATA_PLAYBOOK.md) |
| Certify large-data behavior | [Large Data Certification Guide](./user_guides/LARGE_DATA_CERTIFICATION_GUIDE.md) |
| Decode terminology | [Glossary](./user_guides/GLOSSARY.md) |

## Recommended Reading Paths

For a new model builder:

1. [User Quick Start Guide](./user_guides/QUICK_START.md)
2. [PD Logistic Regression Walkthrough](./user_guides/PD_LOGISTIC_REGRESSION_WALKTHROUGH.html)
3. [Data Requirements Guide](./user_guides/DATA_REQUIREMENTS.md)
4. [Execution Mode Decision Guide](./user_guides/EXECUTION_MODE_DECISION_GUIDE.md)
5. [Model Selection Guide](./user_guides/MODEL_SELECTION_GUIDE.md)
6. [Configuration Cookbook](./user_guides/CONFIGURATION_COOKBOOK.md)
7. [Artifact Map](./user_guides/ARTIFACT_MAP.md)

For a validator or risk reviewer:

1. [Validation Reviewer Guide](./user_guides/VALIDATION_REVIEWER_GUIDE.md)
2. [Artifact Map](./user_guides/ARTIFACT_MAP.md)
3. [Metric Catalog](./METRIC_CATALOG.md)
4. [Statistical Test Catalog](./STATISTICAL_TEST_CATALOG.md)
5. [GUI-to-Code Traceability Guide](./GUI_TO_CODE_TRACEABILITY_GUIDE.md)

For a technical owner:

1. [Engineering Rubric](./ENGINEERING_RUBRIC.md)
2. [GUI-to-Code Traceability Guide](./GUI_TO_CODE_TRACEABILITY_GUIDE.md)
3. [Checkpoint Stage Guide](./CHECKPOINT_STAGE_GUIDE.md)
4. [Preprocessing and Data Treatment Guide](./PREPROCESSING_AND_DATA_TREATMENT_GUIDE.md)
5. [UI / UX Standard](./UI_UX_STANDARD.md)

For SageMaker use:

1. [SageMaker Setup](./SAGEMAKER_SETUP.md)
2. [Large Data Playbook](./user_guides/LARGE_DATA_PLAYBOOK.md)
3. [Large Data Certification Guide](./user_guides/LARGE_DATA_CERTIFICATION_GUIDE.md)
4. [Troubleshooting Guide](./user_guides/TROUBLESHOOTING.md)

For macOS use:

1. [macOS Setup](./MACOS_SETUP.md)
2. [User Quick Start Guide](./user_guides/QUICK_START.md)
3. [Troubleshooting Guide](./user_guides/TROUBLESHOOTING.md)

## Documentation Inventory

| Category | Document | Purpose |
| --- | --- | --- |
| Start-here | [README](../README.md) | Project overview, setup links, feature summary, and developer entry points. |
| Start-here | [Executive Summary](../EXECUTIVE_SUMMARY.txt) | Non-technical executive explanation of the application. |
| Start-here | [License](../LICENSE) | Noncommercial/commercial-permission license terms. |
| Start-here | [User Guide Index](./user_guides/README.md) | Task-oriented index for model builders, reviewers, and support users. |
| Start-here | [User Quick Start Guide](./user_guides/QUICK_START.md) | First successful Streamlit run with safe defaults. |
| Start-here | [PD Logistic Regression Walkthrough](./user_guides/PD_LOGISTIC_REGRESSION_WALKTHROUGH.html) | Complete offline visual walkthrough for a simple PD logistic regression run. |
| Setup | [macOS Setup](./MACOS_SETUP.md) | Mac installation and launch instructions. |
| Setup | [SageMaker Setup](./SAGEMAKER_SETUP.md) | SageMaker, VS Code Remote, port forwarding, and large-data setup instructions. |
| Setup | [Root macOS Setup Mirror](../MACOS_SETUP.txt) | Plain-text macOS convenience mirror. |
| Setup | [Root SageMaker Setup Mirror](../SAGEMAKER_SETUP.txt) | Plain-text SageMaker convenience mirror. |
| Setup | [SageMaker Requirements](../requirements-sagemaker.txt) | Isolated SageMaker runtime dependency list used by bootstrap scripts. |
| User workflow | [Data Requirements Guide](./user_guides/DATA_REQUIREMENTS.md) | Input formats, schema roles, target/date/entity expectations, and large-file intake. |
| User workflow | [Execution Mode Decision Guide](./user_guides/EXECUTION_MODE_DECISION_GUIDE.md) | When to fit a new model, score an existing model, or search feature subsets. |
| User workflow | [Artifact Map](./user_guides/ARTIFACT_MAP.md) | Where run outputs land and how to interpret the artifact folder. |
| User workflow | [Run Registry and Audit Guide](./user_guides/RUN_REGISTRY_AND_AUDIT_GUIDE.md) | How to browse prior runs and interpret the audit event log. |
| User workflow | [Configuration Cookbook](./user_guides/CONFIGURATION_COOKBOOK.md) | Practical setup recipes for common modeling workflows. |
| User workflow | [Model Selection Guide](./user_guides/MODEL_SELECTION_GUIDE.md) | How to choose model families by use case and target mode. |
| User workflow | [Model Type Reference Guide](./user_guides/MODEL_TYPE_REFERENCE_GUIDE.md) | Detailed model-type walkthroughs and usage notes. |
| User workflow | [Validation Reviewer Guide](./user_guides/VALIDATION_REVIEWER_GUIDE.md) | How validation/risk reviewers inspect completed runs. |
| User workflow | [Troubleshooting Guide](./user_guides/TROUBLESHOOTING.md) | Common launch, data, model, report, and SageMaker issues. |
| User workflow | [Glossary](./user_guides/GLOSSARY.md) | Common app, modeling, artifact, and validation terms. |
| Large data | [Large Data Playbook](./user_guides/LARGE_DATA_PLAYBOOK.md) | Large Data Mode, S3 intake, Parquet staging, worker mode, and guardrails. |
| Large data | [Large Data Certification Guide](./user_guides/LARGE_DATA_CERTIFICATION_GUIDE.md) | CLI certification harness and acceptance evidence for large datasets. |
| Technical reference | [GUI-to-Code Traceability Guide](./GUI_TO_CODE_TRACEABILITY_GUIDE.md) | Mapping from GUI controls to config fields, code paths, and audit surfaces. |
| Technical reference | [Checkpoint Stage Guide](./CHECKPOINT_STAGE_GUIDE.md) | Checkpoint flow stages, required/optional stage meaning, and controls. |
| Technical reference | [Preprocessing and Data Treatment Guide](./PREPROCESSING_AND_DATA_TREATMENT_GUIDE.md) | Schema, cleaning, imputation, transformations, selection, and treatment evidence. |
| Catalog | [Model Catalog](./MODEL_CATALOG.md) | Implemented model families and how they map to use cases. |
| Catalog | [Metric Catalog](./METRIC_CATALOG.md) | Metrics, diagnostics, and interpretation guidance. |
| Catalog | [Statistical Test Catalog](./STATISTICAL_TEST_CATALOG.md) | Statistical tests, when to use them, code references, and outcome interpretation. |
| Standard | [Engineering Rubric](./ENGINEERING_RUBRIC.md) | Code quality, architecture, developer module map, reproducibility, documentation, and operational standards. |
| Standard | [UI / UX Standard](./UI_UX_STANDARD.md) | Current GUI/report visual, layout, interaction, chart, table, and export standards. |
| Release evidence | [Release Notes v1.0.0](./RELEASE_NOTES_V1.0.0.md) | Release-ready baseline and scope. |
| Release evidence | [Release Validation Report](./RELEASE_VALIDATION_REPORT.md) | Validation evidence for the release baseline. |
| Examples | [Reference Workflow Index](../examples/reference_workflows/README.md) | Deterministic reference workflows and example packs. |
| Examples | [PD Development Pack](../examples/reference_workflows/packs/pd_development.md) | Reference pack for PD development. |
| Examples | [LGD Severity Pack](../examples/reference_workflows/packs/lgd_severity.md) | Reference pack for LGD severity. |
| Examples | [CECL Lifetime PD Pack](../examples/reference_workflows/packs/cecl_lifetime_pd.md) | Reference pack for CECL/lifetime PD. |
| Examples | [CCAR Forecasting Pack](../examples/reference_workflows/packs/ccar_forecasting.md) | Reference pack for CCAR forecasting. |

## Cleanup Policy

One-off feature plans should not remain as standalone documentation after the
related feature is implemented. Current standards, catalogs, release evidence,
setup guides, user guides, and reference workflow packs are retained because
they describe current application behavior.
