# Deferred Roadmap: Interactive Report Size Optimization

Status: partially implemented; retained for future optional distribution work

This roadmap captures planned work to keep `interactive_report.html` useful for
large Quant Studio runs without creating oversized standalone HTML files. The
core payload controls are now implemented through
`PerformanceConfig.html_max_points_per_figure`,
`PerformanceConfig.html_max_figure_payload_mb`,
`PerformanceConfig.html_max_total_figure_payload_mb`, and the exported
`report_payload_audit` table. This file remains as a parking lot for future
optional ideas such as externalized chart assets, downloadable report profiles,
or GUI report-size dashboards.

## Problem Statement

The interactive report is self-contained. It embeds Plotly JavaScript and the
data payload for each chart directly inside the HTML file. On large workflows,
especially workflows with enhanced visuals or advanced visual analytics enabled,
the serialized Plotly figure payloads can make the report hundreds of megabytes.

The main issue is chart payload size, not the Parquet source file itself. Table
previews are already capped, but figures can still embed large x/y arrays,
multiple traces, and repeated diagnostic or prediction-derived data.

## Goals

- Keep the interactive report readable and fast enough to open for large runs.
- Preserve audit value by explaining when charts are sampled, skipped, or
  externalized.
- Prevent one oversized figure from making the full report unusable.
- Give users clear controls for report richness versus runtime and file size.
- Keep default behavior safe for large datasets while allowing explicit opt-in
  for heavier visuals.

## Non-Goals

- Do not remove the interactive report.
- Do not remove enhanced visuals or advanced visual analytics as capabilities.
- Do not change model training, scoring, or statistical-test behavior.
- Do not make individual figure HTML/PNG export responsible for controlling the
  size of the main interactive report. That setting only controls separate
  duplicate chart files.

## Phase 1: Measure Report Payloads

Implemented. Quant Studio now measures report figure payloads and exports
`tables/governance/report_payload_audit.*`.

Implemented work:

- Measure serialized byte size for every Plotly figure before embedding.
- Export a `report_payload_audit` table with section, figure name, trace count,
  point count, serialized JSON size, and whether the figure was embedded,
  downsampled, externalized, or skipped.
- Add run-level metadata for total interactive report size, largest figures, and
  total embedded figure payload size.
- Surface the largest report contributors in the GUI after export.

Acceptance criteria:

- A user can identify exactly which charts made a report large.
- The audit table is exported with the rest of the run artifacts.
- Existing reports continue to render the same way unless later phases are
  enabled.

## Phase 2: Add Hard Figure Size Guardrails

Implemented. Quant Studio now applies global report point, per-chart payload,
and total chart payload limits before HTML embedding.

Implemented work:

- Add configuration fields such as:
  - `html_max_points_per_figure`
  - `html_max_serialized_figure_mb`
  - `html_max_total_figure_payload_mb`
  - `html_oversized_figure_policy`
- Downsample figure traces consistently before embedding when point limits are
  exceeded.
- Skip or summarize figures that still exceed the serialized-size cap after
  downsampling.
- Add a visible report note when a chart is sampled or skipped because of size.

Acceptance criteria:

- No single chart can silently inflate the report beyond configured limits.
- Skipped or sampled charts are explicitly documented in the report and audit
  table.
- Small and medium runs keep their current visual richness.

## Phase 3: Add Large-Run Report Slimming Defaults

Make report behavior automatically safer when the source data or generated
outputs are large.

Planned work:

- Detect large runs using source file size, row count, prediction-row count, and
  estimated report payload.
- Automatically apply a `large_report_mode` when thresholds are exceeded.
- In large report mode, reduce max figures per section, lower figure point caps,
  and prefer sampled diagnostic visuals.
- Keep `Include enhanced report visuals` and `Advanced Visual Analytics` as
  explicit user controls, but warn when enabling them may create a large report.

Acceptance criteria:

- Large reports are reduced by default without requiring users to know all
  report internals.
- Users can still opt into richer visuals when they accept the file-size tradeoff.
- The GUI clearly explains why large report mode was applied.

## Phase 4: Externalize Heavy Chart Payloads

Provide an alternative report format that keeps the HTML shell smaller.

Planned work:

- Write large chart payloads to adjacent JSON files under the run's report
  artifact folder.
- Keep the report shell as an HTML index that loads chart payloads on demand.
- Add a manifest mapping chart cards to external payload files.
- Preserve a self-contained HTML option for smaller runs and offline review
  situations where one file is preferred.

Acceptance criteria:

- Large reports can be opened as a folder-based report bundle instead of one
  oversized HTML file.
- The report remains portable when the full report folder is shared.
- The manifest makes external chart dependencies auditable.

## Phase 5: Improve User-Facing Report Controls

Add clear GUI controls for balancing report polish, file size, and runtime.

Planned work:

- Add a report richness selector such as `Compact`, `Standard`, and `Full`.
- Keep `Compact` focused on core model evidence, key diagnostics, and sampled
  charts.
- Keep `Standard` as the balanced default for normal-sized runs.
- Keep `Full` as an explicit opt-in for enhanced visuals and advanced analytics.
- Display an estimated report-size risk before run execution.

Acceptance criteria:

- Users can choose the level of report detail without understanding internal
  configuration names.
- Large-data runs recommend compact or standard report generation.
- Audit users still have a deliberate path to generate fuller reports.

## Phase 6: Add Regression Tests And Benchmarks

Protect the report-size behavior from future regressions.

Planned work:

- Add tests that verify oversized figures are sampled, skipped, or externalized
  according to configuration.
- Add synthetic large-report benchmark coverage.
- Add tests for `report_payload_audit` artifact content.
- Add tests that confirm small reports are not unnecessarily degraded.

Acceptance criteria:

- CI catches report bloat regressions.
- Report guardrails can be changed safely.
- The benchmark shows before/after size and runtime impact.

## Phase 7: Documentation Updates

Document the tradeoffs and operating guidance after implementation.

Planned work:

- Update the Large Data Playbook with report-size guidance.
- Update the Configuration Cookbook with report richness examples.
- Update the Artifact Map to explain report payload audit files and external
  chart bundles.
- Update troubleshooting guidance for oversized or slow-opening reports.

Acceptance criteria:

- Users understand why a report may be compacted, sampled, or split into a
  folder bundle.
- Audit reviewers know where to find skipped-chart rationale and payload
  evidence.
- Large-data setup guidance explains which controls reduce report size.

## Recommended Future Implementation Order

1. Implement report payload measurement and audit exports.
2. Add hard per-figure and total report-size guardrails.
3. Add large-run report slimming defaults.
4. Add user-facing compact/standard/full report controls.
5. Add folder-based external chart payload support.
6. Add regression tests and large-report benchmarks.
7. Update documentation after the behavior exists in code.

## Open Design Decisions

- Whether folder-based report bundles should replace the self-contained report
  for large runs or be an additional export.
- Whether large report mode should be automatic, user-confirmed, or controlled
  by preset.
- Whether report-size limits should be configured globally or per report
  section.
- Whether downsampling should use simple row sampling, quantile-preserving
  sampling, time-aware sampling, or chart-specific methods.
- Whether chart payload files should be JSON only or include pre-rendered static
  summaries for environments with limited browser performance.
