# Reference Workflows

This directory contains canonical end-to-end examples for the main intended
development use cases of Quant Studio.

Each workflow is:

- deterministic
- based on a stable synthetic reference dataset
- wired to an expected-output contract under `expected/`
- used by the golden-run regression tests in `tests/test_reference_workflows.py`

The current workflows are:

- `pd_development.py`
- `lgd_severity.py`
- `cecl_lifetime_pd.py`
- `ccar_forecasting.py`

The directory also includes structured walkthrough packs under `packs/`. Each
pack explains:

- when to use the workflow
- how to run it
- which artifact files to review first
- which tables and figures matter most
- how to adapt the example to a real model-development use case

## How To Run

Run any workflow from the repository root:

```powershell
python examples\reference_workflows\pd_development.py
python examples\reference_workflows\lgd_severity.py
python examples\reference_workflows\cecl_lifetime_pd.py
python examples\reference_workflows\ccar_forecasting.py
```

Each script writes a full artifact bundle under `artifacts/reference_workflows/`.
Each run bundle now also includes a `reference_example_pack.md` file inside the
run directory so the output is self-explaining.

## Why These Exist

These workflows are not only examples. They are also the regression-hardening
surface for the project:

- they prove the framework still works on its intended use cases
- they freeze expected key metrics and artifact behavior
- they provide copyable starter patterns for real projects
