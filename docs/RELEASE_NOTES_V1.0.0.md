# Quant Studio v1.0.0 Release Notes

Release date: 2026-05-07

## Release Position

Quant Studio v1.0.0 is the first release-ready baseline for the current project
scope: model development, validation evidence generation, scenario analysis,
documentation, artifact export, existing-model scoring, feature-subset
comparison, and downstream LLM/ongoing-monitoring handoff packages.

This release should be treated as feature-frozen for the v1 line. Future work
should prioritize defect fixes, reproducibility hardening, performance
benchmarking on institution-specific datasets, and documentation clarity before
adding additional modeling families or workflow modes.

## Validated Workflows

The release-hardening pass validated these workflows:

| Workflow | Coverage |
| --- | --- |
| PD development | Canonical binary logistic-regression reference workflow completed and exported a full artifact bundle. |
| LGD severity | Canonical continuous two-stage LGD reference workflow completed and exported a full artifact bundle. |
| CECL lifetime PD | Canonical discrete-time hazard reference workflow completed and exported a full artifact bundle. |
| CCAR forecasting | Canonical panel-regression forecasting reference workflow completed and exported a full artifact bundle. |
| Existing-model scoring | Regression tests confirmed saved-model reuse and validation without refitting. |
| Feature subset search | Regression tests confirmed comparison-only candidate search exports and ranking outputs. |
| Large Data Mode controls | Regression tests confirmed file-backed controls, sampling safeguards, and large-data output behavior. |
| Streamlit workflow behavior | Regression tests covered Streamlit app workflow behavior and Step 5 artifact contracts. |

See [Release Validation Report](./RELEASE_VALIDATION_REPORT.md) for commands,
artifact audit results, and known validation boundaries.

## Main Capabilities In Scope

- Five-step Streamlit workflow for data/schema, model configuration, readiness
  and run, results/artifacts, and decision summary.
- Three execution modes: `fit_new_model`, `score_existing_model`, and
  `search_feature_subsets`.
- Model families covering PD, LGD, scorecard, regression, survival-style,
  forecasting, panel, tree-based, and gradient-boosting use cases.
- Governed preprocessing, imputation, transformations, feature policy,
  explainability, scenario testing, calibration, robustness, cross-validation,
  and statistical diagnostics.
- Checkpointed execution with optional step-by-step run control.
- Organized artifact folder with reports, model object, config, metadata,
  reproducibility evidence, tables, workbooks, generated rerun code, and code
  snapshot.
- On-demand LLM documentation package with numbered folders, evidence maps,
  approved claims, citation boundaries, figure/table placement manifests,
  DOCX helper scripts, prompts, validation controls, and generated-output
  workspace.
- On-demand ongoing-monitoring package for the separate monitoring application.
- Windows, macOS, and SageMaker setup documentation.

## Known Limitations

- Validation used synthetic bundled/reference workflows and automated regression
  suites. Institution-specific data, policies, performance thresholds, and
  approval standards still require local validation.
- Multi-GB production-scale testing was not performed as part of this v1.0.0
  hardening pass. Large Data Mode controls are covered by regression tests, but
  actual RAM/runtime limits depend on the deployed machine and dataset shape.
- Streamlit UI behavior was covered by automated tests; final user-acceptance
  testing should still be performed in the target desktop, SageMaker, or remote
  environment.
- LLM package outputs are drafting aids only. Any LLM-generated methodology
  document requires qualified human model-development, validation, and
  governance review.
- Commercial use is restricted by the project license and requires explicit
  written permission.

## Upgrade Notes

- The package version is now `1.0.0`.
- Existing artifacts using the `quant_pd_framework` import path remain
  compatible with the current package name and import surface.
- Existing run folders remain readable. New run folders include the current
  artifact layout and on-demand package behavior.
