# Large Data Playbook

This guide explains how to run Quant Studio when files are too large for normal
browser upload or comfortable full in-memory processing.

## When To Use Large Data Mode

Enable `Large Data Mode` when:

- the file is above roughly 1 GB
- browser upload is slow or unreliable
- pandas memory usage is a concern
- full diagnostic exports would be too large
- the user wants sample-fit plus full-data scoring

For small files, normal mode is simpler.

For 50 GB-class files, do not use browser upload. Use one of these file-backed
sources instead:

- `Data_Load/` local landing-zone file
- direct local path from CLI or generated rerun code
- Step 1 `S3 path` using `s3://bucket/key.csv` or `s3://bucket/key.parquet`

S3 authentication comes from the runtime environment, such as a SageMaker IAM
role, an AWS CLI profile, or standard AWS environment variables. Quant Studio
does not collect or store AWS access keys.

Large Data Mode now uses a hybrid-certified approach. Certified model families
can use the optimized large-data path as it matures; other model families remain
available through governed sample development plus chunked full-file scoring.
If a complex model must be forced through a large-data path, use the explicit
override controls and document the reason before execution.

When a file-backed source is selected from `Data_Load/`, a local path, or S3,
Large Data Mode treats Streamlit as a control plane. The UI keeps only profile
metadata and capped previews in session state. The run itself uses file-backed
manifests, staged Parquet where needed, checkpoint subprocesses, and paged
result queries instead of repeatedly loading the full file into the browser
process.

## Recommended Large-Data Workflow

1. Place the source file in `Data_Load/`.
2. Or enter an S3 object path in Step 1 when the file lives in S3.
3. Prefer Parquet over CSV or Excel.
4. If starting from CSV, use CSV-to-Parquet staging.
5. Enable `Large Data Mode` before running.
6. Use a governed sample for development fitting unless the selected model path
   is certified for optimized full-data execution.
7. Score the full file in chunks.
8. Let tabular exports follow the Step 1 file type: Parquet inputs export
   Parquet artifacts; CSV, Excel, bundled-sample, and unknown inputs export CSV
   artifacts.
9. Use sampled export policy if full CSV artifacts would be too large.
10. Generate individual chart HTML/PNG files only from Step 5 if required.
11. Keep `Advanced Visual Analytics` off unless the added report visuals are
    specifically needed.
12. Keep report-size controls at conservative defaults unless a reviewer needs
    more points embedded directly in the standalone HTML report.
13. Review memory estimate, large-data metadata, model certification status,
    override audit if applicable, and `report_payload_audit`
    outputs.
14. Use the Step 4 `Full Row Browser` for paged prediction review instead of
    trying to render full prediction files in Streamlit.

## Large-Data Backend And Model Policy

Step 2 `Diagnostics & Exports` includes large-data execution controls:

| Setting | Default | Meaning |
| --- | --- | --- |
| `Large-data backend` | `auto` | Lets Quant Studio choose the safest path. `pandas_sample` forces governed sample fit plus full-file scoring. `disk_backed` is reserved for certified large-data paths and explicit overrides. |
| `Large-data model policy` | `allow_sample_fallback` | Keeps all model families selectable. Certified models use the optimized path when available; uncertified models use governed sample fit plus full-file scoring. |
| `certified_only` | off by default | Blocks uncertified model families before the run starts. |
| `force_full_data_override` | off by default | Allows a complex uncertified model only when the user confirms the risk and enters an override reason. |
| `Model CPU workers` | `0` | Lets XGBoost and supported tree ensembles use estimator-level CPU parallelism. Use a positive integer to cap worker count. |
| `DuckDB threads` | `0` | Lets DuckDB choose thread count for file-backed staging and paged result queries. Use a positive integer to cap CPU use. |
| `DuckDB memory limit (GB)` | `0` | Leaves DuckDB memory uncapped by Quant Studio. Enter a value to enforce a query/staging memory ceiling. |
| `Paged result rows` | `1,000` | Controls how many rows Step 4 queries from prediction Parquet/CSV tables at a time. |
| `Large-data max in-memory rows` | `250,000` | Caps dataframe materialization for UI snapshots and checkpoint spilling decisions. |
| `Use persistent large-data profile cache` | on | Stores schema, preview, row-count metadata, approximate null/cardinality/quantile evidence, and cache hit/miss evidence under `.quant_studio_cache/profiles/`. |
| `Large-data partition strategy` | `auto` | Writes split-aware governed-sample Parquet partitions by default. Use `none` when storage is more important than partition evidence. |
| `Run large-data feature pre-screening` | on | Records advisory missingness, low-variance, high-cardinality, target relationship, IV, PSI, and correlation evidence. |
| `Apply large-data pre-screening recommendations` | off | When enabled, automatically excludes features flagged by the pre-screen and records the exclusions in run metadata. |
| `Large-data worker mode` | `auto` | Uses a detected local worker service when available; otherwise starts the detached background process. |
| `Enable certified large-data fit planner` | on | Records the certified, sample-fallback, in-memory-only, or override fit basis for the selected model family. |
| `Checkpoint stage timeout (minutes)` | `60` | Stops any single checkpoint subprocess that runs longer than the configured limit. Increase this only when a large standard in-memory run is expected to spend more than one hour in scoring, diagnostics, or export. |

The current capability matrix records one of five statuses:

- `full_data_exact`: priority path for exact or aggregate-safe large-data
  fitting, including logistic, elastic-net logistic, scorecard logistic, linear,
  ridge, lasso, and elastic-net.
- `full_data_incremental`: CPU-parallel or external-memory-capable path, used
  first for XGBoost.
- `sample_fit_full_score`: governed sample fit followed by chunked full-file
  scoring.
- `in_memory_only`: estimator remains in-memory with the current dependency
  stack; random forest, extra trees, decision trees, and EBM are classified this
  way for large files.
- `blocked`: readiness prevents the run because policy or override evidence is
  incomplete.

Other model families remain selectable, but the run metadata records whether
they used governed sample fit, full-file scoring, or an experimental override.

Forced overrides are not treated as certified. They are written to the exported
metadata with the selected model, source file metadata, confirmation, reason,
and recommendation.

## Forcing Standard In-Memory Execution

Step 3 `Readiness Check & Run` includes `Force standard in-memory execution`
when Large Data Mode was auto-detected or enabled upstream. This is an advanced
escape hatch for users who intentionally want the standard pandas/scikit-learn
workflow to attempt the full local file in memory.

Use this only when the machine is sized for the full dataset and the model or
diagnostic path must run outside Large Data Mode. The run records
`large_data_auto_detected`, `large_data_user_override_disabled`,
`large_data_override_reason`, `large_data_standard_execution_override_reason`,
`large_data_effective_mode`, and `large_data_source_kind` in the exported run
configuration.

S3 inputs cannot use this override in the current implementation. S3 continues
to require Large Data Mode so the app does not try to buffer a full remote object
through Streamlit or pandas.

## S3 Intake

Use Step 1 `Specify S3 path` for S3-resident CSV, Excel, or Parquet files.
Examples:

```text
s3://my-credit-risk-bucket/modeling/loan_panel.csv
s3://my-credit-risk-bucket/modeling/loan_panel.xlsx
s3://my-credit-risk-bucket/modeling/loan_panel.parquet
```

Quant Studio reads only a capped preview for the UI, then stages the S3 object
locally when the run starts. CSV objects are streamed into the local Parquet
cache. Excel and Parquet objects are copied into the cache for repeatability
unless a future direct-scan backend is enabled.

The S3 local cache defaults to:

```text
.quant_studio_cache/s3
```

The cache key is based on the S3 URI plus available object metadata such as
size, modified time, and ETag. Re-running unchanged objects reuses the staged
Parquet file.

## What A Governed Sample Means

A governed sample is the documented development sample used to fit and review
the model when the full dataset is too large to use interactively.

It should be:

- created by a configured and repeatable sampling process
- large enough to represent the modeling population
- created before model fitting
- documented in exported large-data metadata
- separate from full-data scoring outputs

The model is developed on the governed sample, then the fitted preprocessing
and model contract are replayed on the full file in chunks for scoring.

## CSV Versus Parquet

CSV is row-oriented text. It is portable but expensive to parse and usually
larger on disk.

Parquet is columnar and typed. It is usually smaller, faster to scan, and
better for large tabular workflows. Quant Studio can convert file-path CSV
inputs to Parquet in chunks when configured.

## Memory Guidance

The 50 GB upload setting is a configured Streamlit ceiling, not a RAM promise.
In-memory pandas data can require several times the source file size.

Conservative starting points:

| Source file | Starting RAM guidance |
| --- | --- |
| Under 1 GB | 16 GB or more |
| 1-5 GB CSV | 32-64 GB and Large Data Mode |
| 5-10 GB CSV | 96-128 GB, Parquet staging, governed sample, sampled exports |
| 10 GB Parquet | Depends on compression and column types, but still plan for substantial RAM during sample development |

If available RAM is below the practical need, pandas may swap heavily, appear
frozen, or fail with memory errors. It is safer to reduce the sample, convert to
Parquet, and use larger compute.

For full-data model fitting, keep these defaults unless there is a specific
audit need to change them:

- `Optimize dtypes during ingestion`: on
- `Compact prediction exports`: on
- `Retain full diagnostic working dataframe`: off
- `Keep all checkpoints`: off
- individual chart HTML/PNG package: on-demand from Step 5 only
- Excel workbook export: off unless specifically requested
- report-size controls: leave defaults on unless the HTML report must embed
  denser charts
- `Tabular artifact format`: input-driven; Parquet only for original Parquet inputs
- `Workflow run style`: full workflow is acceptable because it uses the
  checkpointed stage engine; use step-by-step when debugging a specific stage

These settings do not sample the model training split. They reduce duplicated
copies, report/session memory, and artifact size around the full-data fit.

## Checkpointed Runtime Behavior

The default run button now uses checkpointed subprocess stages. This means the
app saves a stage context to disk, launches the next stage in a fresh Python
process, and avoids keeping every prior intermediate object alive in the same
process. It does not make pandas model fitting fully streaming, but it reduces
memory pressure from long end-to-end runs and makes failed diagnostics easier to
isolate.

For file-backed Large Data Mode full runs, Streamlit launches a detached
background worker and polls `job_manifest.json`. The browser stays responsive
while the worker writes the checkpoint manifest, stage progress, output paths,
failure details, and the final bounded Streamlit snapshot. The background job
can be cancelled from Step 3; cancellation takes effect at the next safe
checkpoint boundary.

If repeated high-volume runs need a more enterprise-like control plane, start
the optional local worker service in a separate terminal:

```bash
quant-pd-worker --queue-dir artifacts/_job_queue --workers 1
```

When `Large-data worker mode` is `auto`, Quant Studio uses this queue only when
it detects a fresh worker heartbeat. If no worker is detected, it falls back to
the detached process path. If `worker_service` is selected explicitly, Step 3
queues the run and waits for a worker to pick it up.

Large dataframe fields inside checkpoint contexts are spilled to Parquet table
references before joblib serialization when they exceed
`Large-data max in-memory rows`. Loading a checkpoint rehydrates those references
for the next stage. This keeps checkpoint files from duplicating large pandas
objects while preserving compatibility with the existing Python stage API.

To control disk usage, `Keep all checkpoints` is off by default. With the toggle
off, Quant Studio keeps the latest context checkpoint only while it is needed,
prunes older context files during the run, and removes the final context file
after successful completion. The checkpoint manifest stays in the run folder and
metadata folder for stage-status audit evidence.

During execution, the Run Status panel shows a `Checkpoint Flow` chart. Use it
to see the active stage, confirm which stages have already completed, and
identify whether a large run failed during fitting, diagnostics, scoring, or
export.

Step-by-step mode is useful when:

- the model fit succeeds but diagnostics fail
- optional visual analytics need to be retried separately
- support needs to inspect exactly which stage failed
- a large run should be advanced one auditable stage at a time

## Large-Data Outputs

Large Data Mode can export:

- `large_data_memory_estimate`
- `dtype_optimization`
- `csv_to_parquet_conversion`
- `large_data_source_profile`
- `large_data_model_certification`
- `large_data_execution_plan`
- `large_data_transformation_contract`
- `large_data_feature_screening`
- `large_data_fit_record`
- `large_data_override_audit` when a force override is used
- `metadata/large_data/dataset_profile.json`
- `metadata/large_data/execution_plan.json`
- `metadata/large_data/large_data_transformation_contract.json`
- `metadata/large_data/feature_screening_manifest.json`
- `metadata/large_data/partitioned_dataset_manifest.json`
- `metadata/large_data/prepared_dataset_manifest.json`
- `tables/feature_screening/large_data_feature_screening.*`
- `data/sample_development/`
- `data/sample_development/partitioned/`
- `data/full_data_scoring/`
- `metadata/large_data/`
- full or sampled predictions in the input-driven tabular format
- metadata-only entries for very large tables when configured

These outputs are intended to make sampling, conversion, memory decisions, and
chunked scoring auditable.

The execution plan records whether each major stage is file-backed,
sample-based, full-data, in-memory, blocked, or override-driven. The
transformation contract records whether each configured transformation can be
compiled to DuckDB SQL or aggregate replay, must remain sample-only, is
unsupported in disk-backed replay, or is blocked.

The prepared dataset manifest records the source identifier, staged path,
sample-development path, projected modeling columns, target column, row count
where available, transformation-contract keys, profile cache key, partition
paths, and artifact-size estimates. This is the audit bridge between Step 1
data intake and the file-backed execution artifacts.

Feature pre-screening is advisory by default. It does not remove user-selected
features unless `Apply large-data pre-screening recommendations` is enabled.
When auto-apply is enabled, exclusions are recorded in metadata and the
feature-screening table.

## Benchmarking Large Data Mode

Use the benchmark harness on SageMaker or a high-RAM workstation before making
capacity claims:

```bash
python scripts/benchmark_large_data.py --preset 1gb --output-root artifacts/benchmarks
python scripts/benchmark_large_data.py --preset 5gb --output-root artifacts/benchmarks
python scripts/benchmark_large_data.py --preset 10gb --output-root artifacts/benchmarks
python scripts/benchmark_large_data.py --preset 50gb_projected --output-root artifacts/benchmarks
```

Each run writes JSON and Markdown summaries under `artifacts/benchmarks/` with
row count, column count, file size, artifact size, model family, backend
policy, wall time, peak traced memory, phase timings, profile cache setting,
and failure point if one occurs. The `50gb_projected` preset records projected
scale without forcing a 50 GB local synthetic file unless the script is
customized for that environment.

## Certifying Large Data Mode

Use the certification harness when you need audit-ready acceptance evidence
across model families, source formats, and capability classes:

```bash
python scripts/certify_large_data.py --preset smoke --model-scope small
python scripts/certify_large_data.py --preset smoke --model-scope all
python scripts/certify_large_data.py --preset 10gb --model-scope certified
```

If the project is installed in editable mode, the console command is also
available:

```bash
quant-pd-certify-large-data --preset smoke --model-scope small
```

Certification outputs are written under
`artifacts/certification/<timestamp>/`. The folder includes
`certification_summary.json`, `certification_summary.md`,
`certification_report.html`, `scenario_results.csv`,
`model_capability_matrix.csv`, `effective_thresholds.json`,
`environment_profile.json`, `run_index.csv`, and links to each underlying
benchmark run folder.

The default suite lives at
`configs/large_data_certification/default_acceptance_suite.json`. It defines
the `smoke`, `1gb`, `5gb`, `10gb`, and `50gb_projected` tiers plus pass/fail
thresholds. The certification statuses are based on expected model behavior:
certified models must complete within thresholds, sample-fallback models must
use the fallback policy transparently, blocked configurations must block for
the expected reason, and forced overrides are recorded as experimental evidence
instead of certified evidence.

See [Large Data Certification Guide](./LARGE_DATA_CERTIFICATION_GUIDE.md) for
commands, threshold overrides, S3 notes, and interpretation guidance.

## Recommended Export Settings

For faster large-data runs:

- export profile: `fast` while iterating
- tabular output: input-driven from the Step 1 file type
- large tabular export policy: full, sampled, or metadata-only
- individual chart package: generate from Step 5 only if needed
- Advanced Visual Analytics: off
- lower `Max points per report chart` or `Max total report chart MB` if the
  standalone HTML report is too large
- optional robustness and cross-validation refits: off unless needed

For final review:

- export profile: `standard` or `audit`
- keep reproducibility manifest on
- keep run debug trace on
- preserve large-data metadata folders

## Practical Limits

Large Data Mode improves practicality, but it does not make every operation
streaming. Some model development and diagnostics still require a pandas sample.
For extremely large data, use governed sampling, Parquet, chunked scoring, and
right-sized SageMaker or workstation compute.
