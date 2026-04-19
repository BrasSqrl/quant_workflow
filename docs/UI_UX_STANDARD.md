# UI / UX Standard

This standard defines how the Quant PD Framework GUI and exported diagnostic reports should look and behave. The target experience is a premium light-mode fintech dashboard for model builders and validation teams.

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

The GUI and exported report should follow the same top-level structure:

1. Overview
2. Data Quality
3. Sample / Segmentation
4. Model Performance
5. Calibration / Thresholds
6. Stability / Drift
7. Backtesting / Time Diagnostics
8. Governance / Export Bundle

Each section should include:

- a clear title
- a one-sentence explanation of why the section matters
- grouped charts first
- supporting tables second

Summary views should prioritize featured charts and compact tables. Technical views should expose the full diagnostic inventory.

## 3. Chart Standard

All charts should:

- use the shared fintech color palette
- use consistent typography, spacing, legends, and margins
- have direct, business-readable titles
- use human-readable axis labels
- avoid unnecessary 3D, dual-axis clutter, and ornamental effects

Chart types should match the analytical question:

- bars for ranked comparisons and segment summaries
- lines for trends, calibration, and backtests
- heatmaps for correlation-style matrix views
- histograms for score distributions
- scatter plots for residual and actual-vs-predicted analysis

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
