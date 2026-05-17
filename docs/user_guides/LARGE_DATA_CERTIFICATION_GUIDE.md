# Large Data Certification Guide

This guide explains how to run the CLI-first Large Data Mode certification
harness. The harness is separate from the Streamlit UI so large benchmark runs
do not make the app unresponsive.

## What The Harness Does

The certification harness:

- generates target-compatible synthetic datasets for Quant Studio model types
- runs Large Data Mode through the same workflow engine used by the application
- records elapsed time, memory evidence, artifact size, model capability status,
  source format, and failure point
- evaluates each scenario against configurable acceptance thresholds
- writes audit-ready JSON, Markdown, HTML, and CSV reports

The harness does not claim every model is full-data certified. It evaluates each
model family against its expected large-data behavior:

- `certified_pass` means a full-data-certified model completed within thresholds.
- `certified_fail` means a certified model failed or exceeded thresholds.
- `fallback_pass` means a non-certified model correctly used the fallback policy
  and completed within thresholds.
- `fallback_fail` means a fallback model failed or exceeded thresholds.
- `blocked_expected` means an unsafe configuration was blocked for the expected
  reason.
- `blocked_unexpected` means a blocked scenario failed for the wrong reason.
- `override_recorded` means a forced complex-model override was recorded as
  experimental evidence, not certified evidence.
- `benchmark_error` means the scenario could not produce usable benchmark
  evidence.

## Quick Smoke Run

Use the smoke run after installing the project or changing large-data code:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope small
```

Or, after editable install:

```bash
quant-pd-certify-large-data --preset smoke --model-scope small
```

The smoke run is intentionally small. It validates the harness, report writers,
and basic certified/fallback paths without generating large files.
Do not add the `1gb`, `5gb`, `10gb`, or materialized `50gb_projected` tiers to
normal CI; those tiers are intended for scheduled or manually triggered
capacity testing on appropriately sized machines.

## Full Model-Family Sweep

To test every selectable model family on a small synthetic dataset:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope all
```

To run only certified large-data candidates:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope certified
```

To run a specific list:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope model-list --models logistic_regression,xgboost,linear_regression
```

To add explicit certified-only block evidence for uncertified models:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope fallback --include-blocked-scenarios
```

To add forced override evidence for uncertified models:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope fallback --include-force-overrides
```

## Acceptance Tiers

The default suite is stored at:

```text
configs/large_data_certification/default_acceptance_suite.json
```

The default tiers are:

| Tier | Purpose |
| --- | --- |
| `smoke` | Small CI-safe or local validation run. |
| `1gb` | Workstation confidence tier. |
| `5gb` | High-memory development tier. |
| `10gb` | High-RAM acceptance tier. |
| `50gb_projected` | Planning tier that records projected scale without materializing 50 GB unless explicitly requested. |

Example larger runs:

```bash
python scripts/certify_large_data.py --preset 1gb --model-scope certified
python scripts/certify_large_data.py --preset 10gb --model-scope certified
python scripts/certify_large_data.py --preset 50gb_projected --model-scope certified
```

Only use `--materialize-50gb` when the machine is sized for a real 50 GB
synthetic file:

```bash
python scripts/certify_large_data.py --preset 50gb_projected --model-scope certified --materialize-50gb
```

## Sources And Formats

The harness supports generated local Parquet, generated local CSV, and
`Data_Load/` path simulation:

```bash
python scripts/certify_large_data.py --preset smoke --source-format parquet --source-kind local
python scripts/certify_large_data.py --preset smoke --source-format csv --source-kind local
python scripts/certify_large_data.py --preset smoke --source-format parquet --source-kind data_load
```

S3 certification is optional and uses environment credentials such as a
SageMaker IAM role, AWS CLI profile, or standard AWS environment variables.
Quant Studio does not store AWS secrets.

```bash
python scripts/certify_large_data.py --preset smoke --source-kind s3 --s3-uri s3://bucket/path/file.parquet
```

The S3 file must have the expected certification columns if it is used for a
full workflow run. For most acceptance testing, generated local Parquet is the
safest repeatable input.

## Threshold Overrides

Thresholds can be changed in the JSON suite file or overridden from the CLI:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope small --max-wall-time-seconds 900 --max-peak-memory-gb 8 --max-artifact-size-gb 2
```

Synthetic size controls can also be overridden:

```bash
python scripts/certify_large_data.py --preset smoke --rows 5000 --features 12 --sample-rows 1000 --chunk-rows 1000
```

## Output Location

Each run writes a timestamped folder:

```text
artifacts/certification/<timestamp>/
```

Important outputs:

| File | Purpose |
| --- | --- |
| `certification_summary.json` | Machine-readable certification summary and status counts. |
| `certification_summary.md` | Human-readable Markdown summary. |
| `certification_report.html` | Standalone HTML certification report. |
| `scenario_results.csv` | One row per certification scenario with thresholds, status, timings, memory, and artifact evidence. |
| `model_capability_matrix.csv` | Static model capability matrix compared with benchmark evidence. |
| `effective_thresholds.json` | Thresholds used after suite and CLI overrides. |
| `environment_profile.json` | Python, platform, CPU count, memory, and working-directory evidence. |
| `run_index.csv` | Links to each underlying benchmark JSON, Markdown, and run artifact folder. |
| `benchmark_runs/` | Underlying benchmark evidence for each scenario. |

## Interpreting Results

Start with `certification_summary.md` or `certification_report.html`. If a
scenario fails, open `scenario_results.csv`, filter to failed statuses, and then
open the linked benchmark JSON from `run_index.csv`.

For production claims, use the same machine class, storage path, source format,
model scope, and export settings that the intended production workflow will
use. A smoke pass proves the harness works; it does not prove 10-50 GB
production readiness.
