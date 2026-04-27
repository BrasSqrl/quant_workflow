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

## Recommended Large-Data Workflow

1. Place the source file in `Data_Load/`.
2. Prefer Parquet over CSV or Excel.
3. If starting from CSV, use CSV-to-Parquet staging.
4. Enable `Large Data Mode` before running.
5. Use a governed sample for development fitting.
6. Score the full file in chunks.
7. Export full large tables as Parquet.
8. Use sampled CSV exports if CSV review files are needed.
9. Keep individual figure HTML/PNG export off unless required.
10. Keep `Advanced Visual Analytics` off unless the added report visuals are
    specifically needed.
11. Review memory estimate and large-data metadata outputs.

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

## Large-Data Outputs

Large Data Mode can export:

- `large_data_memory_estimate`
- `dtype_optimization`
- `csv_to_parquet_conversion`
- `data/sample_development/`
- `data/full_data_scoring/`
- `metadata/large_data/`
- full Parquet predictions
- sampled CSV review files
- metadata-only entries for very large tables when configured

These outputs are intended to make sampling, conversion, memory decisions, and
chunked scoring auditable.

## Recommended Export Settings

For faster large-data runs:

- export profile: `fast` while iterating
- tabular output: Parquet or both only when CSV is needed
- large tabular export policy: sampled CSV plus full Parquet
- individual figure files: off
- Advanced Visual Analytics: off
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
