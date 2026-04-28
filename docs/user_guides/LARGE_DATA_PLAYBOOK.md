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

Do not enable `Large Data Mode` when the requirement is to fit coefficients on
the full train split. Large Data Mode intentionally uses governed sample
development plus chunked full-file scoring. For full-data fitting, use the
memory-optimized normal workflow: Parquet input when available, dtype
optimization, compact prediction exports, capped diagnostic working snapshots,
and input-driven tabular outputs.

## Recommended Large-Data Workflow

1. Place the source file in `Data_Load/`.
2. Prefer Parquet over CSV or Excel.
3. If starting from CSV, use CSV-to-Parquet staging.
4. Enable `Large Data Mode` before running.
5. Use a governed sample for development fitting.
6. Score the full file in chunks.
7. Let tabular exports follow the Step 1 file type: Parquet inputs export
   Parquet artifacts; CSV, Excel, bundled-sample, and unknown inputs export CSV
   artifacts.
8. Use sampled export policy if full CSV artifacts would be too large.
9. Keep individual figure HTML/PNG export off unless required.
10. Keep `Advanced Visual Analytics` off unless the added report visuals are
    specifically needed.
11. Keep report-size controls at conservative defaults unless a reviewer needs
    more points embedded directly in the standalone HTML report.
12. Review memory estimate, large-data metadata, and `report_payload_audit`
    outputs.

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
- individual figure HTML/PNG export: off
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
- `data/sample_development/`
- `data/full_data_scoring/`
- `metadata/large_data/`
- full or sampled predictions in the input-driven tabular format
- metadata-only entries for very large tables when configured

These outputs are intended to make sampling, conversion, memory decisions, and
chunked scoring auditable.

## Recommended Export Settings

For faster large-data runs:

- export profile: `fast` while iterating
- tabular output: input-driven from the Step 1 file type
- large tabular export policy: full, sampled, or metadata-only
- individual figure files: off
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
