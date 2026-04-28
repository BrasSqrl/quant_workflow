# Enterprise Hardening Roadmap

This roadmap captures the current enterprise-readiness implementation focused
on report portability, validation transparency, evidence traceability, and
regression coverage.

## Scope

The work preserves Quant Studio's narrow purpose: model development,
validation, documentation, and export packaging. It does not add production
monitoring or model operations.

## Roadmap Items

| Item | Goal | Implementation |
| --- | --- | --- |
| 1. Interactive report size controls | Keep standalone HTML reports shareable for large runs without removing the report. | Added configurable chart point and payload caps, automatic chart downsampling/skipping, and a `report_payload_audit` table. |
| 2. Model validation checklist | Give validators a direct checklist of completed, attention-needed, and not-applicable evidence areas. | Added `validation_checklist` export and Step 5 checklist tab. |
| 3. Evidence traceability map | Make it clear which artifact answers each common review question. | Added `evidence_traceability_map` export and Step 5 traceability tab. |
| 8. More automated regression coverage | Protect report-size, checklist, traceability, and artifact-contract behavior from regressions. | Added unit tests for report payload limits and validation evidence, plus artifact-contract checks for exported governance tables. |

## Design Rules

- Report-size controls affect embedded HTML chart payloads; full diagnostic
  tables remain exported separately.
- Downsampled or skipped figures are documented in `report_payload_audit` so a
  reviewer can see exactly what happened.
- Step 5 remains a synthesis and review surface. It does not approve a model
  automatically.
- Traceability files are exported as normal diagnostic tables so they can be
  reviewed offline, sent separately, or referenced from validation materials.
- Regression tests cover the new artifact contracts instead of only checking
  visual appearance.

## User-Facing Controls

The report-size controls are in Step 2, `Diagnostics & Exports`:

- `Max points per report chart`
- `Max MB per report chart`
- `Max total report chart MB`

These controls are separate from `Export individual figure HTML and PNG files`.
Individual figure export controls whether duplicate per-chart files are written;
the report-size controls protect the embedded chart payload inside
`reports/interactive_report.html`.

## Exported Evidence

Each completed standard run now includes:

- `tables/governance/report_payload_audit.*`
- `tables/governance/validation_checklist.*`
- `tables/governance/evidence_traceability_map.*`
- updated `reports/decision_summary.md` sections for checklist and traceability

## Regression Tests

The implementation is covered by:

- `tests/test_report_payload.py`
- `tests/test_validation_evidence.py`
- `tests/test_artifact_contracts.py`
- existing decision-summary tests
