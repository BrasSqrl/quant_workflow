# Contributing

## Local Setup

```powershell
python -m pip install -e ".[dev,gui]"
pre-commit install
```

For normal GUI use, install with:

```powershell
python -m pip install -e ".[gui]"
```

## Common Commands

```powershell
python -m ruff check src tests scripts
python -m mypy
python -m pytest
python scripts/check_module_size.py
```

Run focused tests while changing a subsystem, then run the full suite before pushing
when runtime is reasonable.

## Security Expectations

`.joblib` files are trusted artifacts only when Quant Studio can verify the companion
`.sha256` sidecar. Do not bypass `quant_pd_framework.safe_serialization` when writing
or loading model, checkpoint, or background snapshot artifacts.

Do not commit secrets, private data, proprietary model output, or cloud credentials.
S3 access must use IAM roles, AWS profiles, or external environment variables.

## Secrets Baseline Review

The checked-in `.secrets.baseline` should contain reviewed false positives only. The
current baseline has no active findings. If a future change adds a baseline entry,
document why it is a false positive in the pull request or commit message. If the value
is a real secret, remove it from history where required and rotate it outside the repo.

## Module-Size Policy

New source modules under `src/quant_pd_framework` should stay at or below 1,500 lines.
Existing mega-modules are temporarily allowlisted by `scripts/check_module_size.py`;
they may not grow beyond their recorded baseline. Split code into focused modules when
new work would increase an allowlisted module.

## Broad-Exception Policy

New code should avoid `except Exception`. If a catch-all is required at an operational
boundary, it must either log useful context or include a `# noqa: BLE001` explanation.
Existing broad-exception debt is tracked by Ruff per-file ignores and should be reduced
as modules are refactored.

## Artifact Compatibility

Preserve existing artifact paths, filenames, manifest shapes, generated run scripts,
saved profiles, and workbook formats unless a migration is intentionally documented.
Quant Studio run folders are audit evidence, not temporary implementation detail.
