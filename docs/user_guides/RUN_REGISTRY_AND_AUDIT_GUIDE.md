# Run Registry And Audit Guide

Quant Studio keeps a lightweight local index of model runs and an append-only
audit log for high-value user and workflow actions. This helps users find prior
runs, compare where evidence lives, and show what happened during model
development without opening every artifact folder manually.

## Where To Find It In The GUI

Open Step 4, `Results & Artifacts`, and review the `Run Registry` panel.

The panel shows completed and known failed runs from the configured artifact
root. You can filter by status, model type, reviewer status, or text search.
Selecting a run shows:

- run ID, status, model type, target mode, and execution mode
- dataset/source label where available
- elapsed time and warning count
- reviewer status and reviewer name when a review record was saved
- primary metric summary
- key artifact locations such as report, config, model, metrics, and debug trace
- audit events tied to the selected run

Use `Refresh registry from artifacts` when older run folders were created before
the registry existed or when run folders were copied into the artifact root.

## Files Written

Global registry files are written under:

```text
artifacts/_run_registry/
```

Key files:

| File | Meaning |
| --- | --- |
| `run_registry.json` | Searchable index of completed and known failed runs. |
| `audit_events.jsonl` | Append-only global audit event log. |

Each run folder also gets a run-scoped audit file:

```text
artifacts/run_.../metadata/audit_events.jsonl
```

The registry is an index. The run folder remains the source of truth.

## What Is Audited

Quant Studio records high-value events that explain the user and workflow path:

- data source selected
- configuration profile saved or loaded
- schema, feature dictionary, transformation pipeline, feature review, and scorecard override changes by fingerprint
- workflow run started, completed, or failed
- background/checkpointed run started, completed, or failed
- review record saved
- decision summary downloaded
- individual image package prepared or downloaded
- LLM package prepared or downloaded
- OM package prepared or downloaded

Editable tables are debounced by fingerprint so normal Streamlit reruns do not
flood the audit log.

## What Is Not Audited

Audit events do not store raw input rows, row-level predictions, model binaries,
AWS credentials, secrets, passwords, or tokens. Sensitive-looking metadata keys
are redacted before writing JSONL.

## How Reviewers Should Use It

Use the registry to confirm which run is under review, where the run folder is,
and whether a review record has already been saved. Use the audit trail to
confirm major actions such as profile loading, run completion, package
downloads, and review sign-off.

The audit trail supports review and reproducibility, but it does not replace the
model evidence. Reviewers should still inspect the decision summary, interactive
report, model development dossier, validation checklist, feature lineage, and
reproducibility manifest.
