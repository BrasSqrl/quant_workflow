# Quant Studio Enterprise UI Redesign Standard

This document governs the Streamlit UI cosmetic redesign. It is intentionally
limited to presentation, layout, and user-flow clarity. It must not change model
configuration behavior, pipeline execution, exported artifacts, or available
controls.

## Objective

Make the existing Streamlit application feel like a polished enterprise model
development product while replacing static side panes with a four-step workflow
that keeps the active work in the main canvas.

Primary goals:

- make the workflow easier to follow
- reduce visual clutter by removing persistent side panes
- improve the perceived speed of moving through the workflow
- surface next actions and output expectations without adding functionality
- expose workflow status, issue, preflight, diff, artifact, review, and model-card
  surfaces without changing the underlying modeling engine
- keep the current Python and Streamlit functionality intact

## Non-Goals

- no change to modeling functionality
- no removal of existing controls
- no change to config fields, defaults, or artifact contracts
- no JavaScript frontend work in this phase
- no backend or pipeline execution changes

## Layout Principles

The UI uses four clickable workflow stages:

1. **Dataset & Schema**
   Data source selection, dataset preview, column designer, feature dictionary,
   governed transformations, and the review workbook live here.

2. **Model Configuration**
   Former sidebar controls live in grouped main-canvas expanders: core setup,
   split strategy, model settings, feature subset search, preparation,
   diagnostics/export settings, governance, explainability, and documentation.
   The configuration-profile panel also lives here because it saves and reloads
   the Step 2 setup contract.
   On desktop, the groups render in two side-by-side columns so setup/model
   controls and governance/output controls can be scanned without a long single
   vertical stack.

3. **Readiness Check**
   Preview validation, centralized issue center, preflight summary,
   guardrail findings, execution-plan cards, and the primary run button live here.

4. **Results & Artifacts**
   Completed run diagnostics, tables, charts, downloads, artifact locations, and
   stale-result warnings live here. The artifact explorer, reviewer workspace,
   and model-card download also live here. Before a run exists, this stage shows
   an empty state.

## Visual Language

Use a premium fintech dashboard style:

- light mode
- very pale blue/gray application background
- white glass-like cards
- subtle shadows
- rounded corners between 16 and 24 pixels
- vivid professional blue as the primary accent
- green for valid/ready states
- red for blocking issues
- compact spacing and strong typographic hierarchy

Avoid:

- dark-mode bias
- generic purple gradients
- raw code or JSON presentation in primary surfaces
- long ungrouped vertical control stacks where a compact group works better

## Interaction Principles

- Existing controls remain available.
- Configuration expanders should look like structured setup groups, not
  unrelated drawers.
- The four top workflow stages should look like large segmented workflow pills.
- The workflow status strip should provide a compact management view of
  not-started, needs-attention, ready, and complete states.
- Readiness issues should appear in one consolidated table with recommended
  actions.
- Configuration profiles should behave like a small local library, not just a
  raw file picker.
- Artifact review should start from a grouped explorer before a user has to
  inspect file-system paths directly.
- Errors should appear as action-oriented cards, not flat bars when styling can
  improve readability.
- The run button remains prominent and visually tied to the Readiness Check step.

## Performance-Perception Principles

Cosmetic changes should reduce user effort even when actual execution speed is
unchanged:

- fewer visually competing elements above the fold
- stronger step labels
- clearer active states
- compact metric cards
- immediate guidance on the next action
- concise output-location summaries after execution
- local model-card generation and reviewer notes that reduce external
  documentation work

## Implementation Boundaries

Allowed implementation changes:

- CSS changes in `streamlit_ui/theme.py`
- markup wrappers for visual cards and workflow sections
- labels, captions, and non-functional guidance text
- workflow tabs that preserve existing widgets and widget keys
- runtime-only workflow status, issue center, preflight summary, diff viewer,
  artifact explorer, reviewer workspace, and model-card generation
- documentation updates

Disallowed implementation changes:

- changing config construction
- changing widget keys without need
- changing model defaults
- changing artifact names or paths
- changing pipeline step behavior

## Acceptance Criteria

The redesign is acceptable when:

- the app still launches through the existing Streamlit entrypoint
- current GUI tests pass
- all existing controls remain reachable
- the four top workflow stages are clickable and default to `Dataset & Schema`
- no persistent left or right control pane is required for normal operation
- model configuration controls render in the main canvas under Step 2
- readiness and the run button render in Step 3
- completed outputs and artifact locations render in Step 4
- no model workflow behavior changes
