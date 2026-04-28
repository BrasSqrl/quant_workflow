# UI / UX Standard

This standard defines how the Quant Studio GUI and exported diagnostic reports should look and behave. The target experience is a premium light-mode fintech dashboard for model builders and validation teams.

The companion implementation standard for the latest Streamlit enterprise
redesign is [UI_ENTERPRISE_REDESIGN.md](./UI_ENTERPRISE_REDESIGN.md). That
document governs the command bar, five-step workflow navigation, main-canvas
configuration groups, readiness checkpoint, results/artifact workspace, and
decision-summary scorecard.

## 1. Visual Direction

The interface should feel:

- premium rather than playful
- analytical rather than ornamental
- modern rather than generic
- calm and precise rather than visually loud

The visual system should use:

- light backgrounds with layered surfaces
- strong navy text for primary hierarchy
- restrained gold accents for emphasis
- teal and blue chart colors for analytical content
- generous border radii and soft shadows

The interface should avoid:

- default Streamlit-looking layouts
- overly saturated colors
- flat white panels with no hierarchy
- charts that rely on random default palettes

## 2. Layout Standard

The GUI and exported report should use the same diagnostic taxonomy, with Step
4 `Results & Artifacts` acting as the live reviewer workspace and
Step 5 `Decision Summary` acting as the decision-ready synthesis layer.
`reports/interactive_report.html` is the distribution-ready companion report,
and `reports/decision_summary.md` is the portable scorecard version of Step 5.
The current reporting sections are:

1. Model Performance
2. Calibration / Thresholds
3. Stability / Drift
4. Sample / Segmentation
5. Feature Effects / Explainability
6. Statistical Tests
7. Feature Subset Search
8. Scorecard / Binning Workbench
9. Credit-Risk Development
10. Data Quality
11. Backtesting / Time Diagnostics
12. Governance / Export Bundle

Each section should include:

- a clear title
- a one-sentence explanation of why the section matters
- grouped charts first
- supporting tables second

The live GUI should also support two operator modes:

- `guided` for compact preset-led runs with advanced controls held to defaults
- `advanced` for specialist tuning, governance edits, and deeper challenger work

Summary views should prioritize featured charts and compact tables. Technical views should expose the full diagnostic inventory.

Guidance should be available at the point of use without taking over the
screen:

- Model Type Story Cards should sit near the Step 2 model selector and remain
  compact until expanded.
- Glossary badges should define technical terms through concise hover text.
- `Explain this output` panels should be collapsed by default and limited to
  high-value charts/tables.
- Scorecard Binning Theater should be review-only and should not silently
  change bins or refit a model.
- Step 5 should open on the Decision Room view before exposing detailed
  evidence tabs.

## 3. Chart Standard

All charts should:

- use the shared fintech color palette
- use consistent typography, spacing, legends, and margins
- have direct, business-readable titles
- use human-readable axis labels
- avoid unnecessary 3D, dual-axis clutter, and ornamental effects

Chart types should match the analytical question:

- bars for ranked comparisons and segment summaries
- waterfall bars for signed coefficient or importance review
- lines for trends, calibration, and backtests
- heatmaps for correlation-style matrix views
- violin plots for score and fold-metric distribution shape
- dumbbell plots for observed-versus-predicted segment gaps
- tornado charts for scenario sensitivity ranking
- small multiples for split-by-split feature-effect stability
- histograms for score distributions
- scatter plots for residual and actual-vs-predicted analysis

Report-only companion charts may be generated during presentation assembly when
the underlying diagnostics already exist. These companion views do not change
model fitting, metrics, or exported tables; they make the existing evidence
easier to review. The `Include enhanced report visuals` toggle controls this
presentation layer. Current companion views include annotated ROC, annotated
precision-recall, KS separation, calibration residual bars, PSI threshold bars,
VIF threshold bars, missingness-by-split heatmaps, feature-importance
waterfalls, score-distribution violins, segment-performance dumbbells,
scenario tornados, cross-validation metric violins, and feature-effect
stability small multiples.

The `Advanced Visual Analytics` toggle is off by default and should remain a
clearly optional exploratory layer. When enabled, it may add richer chart
families such as contribution beeswarms, interaction heatmaps, PDP/ICE matrices,
segment calibration small multiples, ridgelines, temporal streams, correlation
networks, risk treemaps, radar charts, waterfalls, and lollipop charts. These
views must be visually distinct from core validation evidence and must not imply
that extra model fitting or extra statistical testing occurred.

Charts that include practical thresholds should use interpretation badges:

- `Great` for strong diagnostic evidence
- `Good` for generally acceptable evidence
- `Watch` for borderline or context-dependent evidence
- `Bad` for likely remediation, justification, or rejection

When labels are unavailable, the interface must avoid implying realized performance and should instead emphasize score distributions, stability review, and documentation-safe summaries.

## 4. Table Standard

Tables should:

- be grouped under the same section as the related charts
- use readable names instead of internal keys
- be previewed in summary mode
- expose fuller detail in technical mode
- preserve the full CSV/XLSX exports in the artifact bundle

Table previews should show only the most useful subset by default and explicitly state when the full export contains more rows.

## 5. Interaction Standard

The live GUI should expose these controls where applicable:

- split selector
- view depth selector (`Summary` vs `Technical`)
- chart/table visibility controls
- date range filter
- segment selector
- feature drilldown selector
- threshold slider for binary models
- top-N controls for ranked content

Interactive controls should change what the user sees, not just what is visible. The dashboard should respond with filtered score distributions, segment views, threshold summaries, and time trends when the underlying data permits it.

## 6. Export Standard

Each run should export a polished standalone HTML diagnostic report in addition to the raw tables and chart files.

The exported report should:

- use the same grouping taxonomy as the GUI
- repeat the same visual language and chart styling
- use a formal regulatory-report cover treatment
- show chart badges and concise interpretation guidance only when the guidance is
  chart-specific and helps the reviewer interpret the result
- use collapsible diagnostic sections so dense evidence is easier to scan
- avoid static image fallback generation inside the report because dynamic
  Plotly rendering is faster and individual PNG exports are optional
- include KPI cards near the top
- present warnings and pipeline events clearly
- remain useful as a standalone validation artifact that can be shared directly

## 7. Copy Standard

Labels and section names should speak to model builders and validation teams directly. Prefer:

- `Calibration Curve` over raw implementation names
- `Population Stability Index` over internal abbreviations without context
- `Observed vs Predicted Over Time` over vague chart titles

Copy should be concise, specific, and aligned with actual model-governance use.

## 8. Quality Gate

A GUI or report change fails this standard if:

- outputs are presented as a flat list without analytical grouping
- charts use inconsistent palettes or unreadable defaults
- layout hierarchy is weak or visually confusing
- technical content is shown without context
- exported HTML feels disconnected from the live GUI
