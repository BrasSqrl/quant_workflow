# Interactive Report Usability Roadmap

This roadmap tracks the usability upgrade for `reports/interactive_report.html`.
The goal is to keep the report audit-ready while making it easier to review
without a long uninterrupted scroll.

## Scope

1. Add sticky section navigation so reviewers can move between major evidence
   areas without scrolling through the full report.
2. Add an executive landing page with run context, KPIs, review priorities, and
   an evidence map.
3. Add section health badges with status and chart/table counts.
4. Split featured evidence from supporting evidence while preserving all
   exported content.
5. Add artifact locators beside each chart and table.
6. Add report search so users can filter by chart, table, feature, or section
   text.
7. Add a table-of-contents style evidence map.
8. Keep specific interpretation callouts and avoid generic filler text.
9. Add compact, reviewer, and print/show-all modes for different review needs.

## Implementation Standard

- The report must remain standalone HTML.
- Full tables remain exported under `tables/`; the HTML only previews them.
- Individual figure files remain optional; report cards should identify the
  optional figure path without requiring the files to exist.
- Print/show-all mode must expose every section sequentially for audit review.
- Search and tabs must be progressive enhancements; if JavaScript fails, the
  report should still show readable sections.

## Implemented Design

- Sticky top controls with search, reviewer mode, compact view, and show-all /
  print view.
- Tab-like section navigation using existing report sections.
- Overview section with review priorities and a section evidence map.
- Section cards with `Pass`, `Watch`, `Fail`, `Ready`, or `Not Run` status.
- Featured and supporting evidence classes used by reviewer and compact modes.
- Asset locator rows on every chart and table card.
- Print CSS that opens all sections and hides interactive controls.
