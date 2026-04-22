# Quant Studio Development Roadmap

This roadmap replaces the prior phase roadmap for the current refactor cycle.
The focus of this phase is Streamlit UI architecture, maintainability, and
performance rather than new modeling breadth.

## 1. Split The App By Responsibility

Status: implemented

Delivered:

- a thin entrypoint at `app/streamlit_app.py`
- shared UI modules under `src/quant_pd_framework/streamlit_ui/`
- dedicated modules for theme, state/session helpers, data loading,
  workspace rendering, result rendering, and config assembly

Primary code:

- `app/streamlit_app.py`
- `src/quant_pd_framework/streamlit_ui/theme.py`
- `src/quant_pd_framework/streamlit_ui/state.py`
- `src/quant_pd_framework/streamlit_ui/data.py`
- `src/quant_pd_framework/streamlit_ui/workspace.py`
- `src/quant_pd_framework/streamlit_ui/results.py`
- `src/quant_pd_framework/streamlit_ui/config_builder.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`

## 2. Introduce A Typed Session-State Layer

Status: implemented

Delivered:

- typed workspace/session key objects
- shared helpers for run snapshot storage and retrieval
- centralized frame initialization and persistence instead of scattered raw
  `st.session_state[...]` writes

Primary code:

- `src/quant_pd_framework/streamlit_ui/state.py`

## 3. Replace Eager Tabs With Single-Section Rendering

Status: implemented

Delivered:

- the workspace no longer renders every tab body on each rerun
- run results and subset-search results now render one active section at a time
- this reduces unnecessary figure/table work on every interaction

Primary code:

- `src/quant_pd_framework/streamlit_ui/workspace.py`
- `src/quant_pd_framework/streamlit_ui/results.py`

## 4. Cache Expensive Data Loading And Artifact Reads

Status: implemented

Delivered:

- cached uploaded CSV/Excel parsing
- cached bundled sample loading
- cached text/binary artifact reads for run reports and downloadable assets

Primary code:

- `src/quant_pd_framework/streamlit_ui/data.py`
- `src/quant_pd_framework/streamlit_ui/state.py`

## 5. Reduce Sidebar-Induced Rerender Pressure

Status: partially implemented

Delivered:

- the heaviest view churn now comes from single-section rendering instead of
  full tab rerenders
- sidebar state is more centralized and less error-prone through typed key
  management

Remaining opportunity:

- further form-based staging of sidebar controls can still be added if the UI
  should trade some live control reactivity for fewer reruns

Primary code:

- `src/quant_pd_framework/streamlit_ui/app_controller.py`
- `src/quant_pd_framework/streamlit_ui/state.py`

## 6. Move Config Assembly Out Of The UI Layer

Status: implemented

Delivered:

- a dedicated config build path that assembles preview configuration and
  guardrail readiness outside the main controller flow
- the controller now delegates preview construction to a specialized module

Primary code:

- `src/quant_pd_framework/streamlit_ui/config_builder.py`
- `src/quant_pd_framework/streamlit_ui/app_controller.py`

## 7. Reduce Repeated Dataframe Copying And Display Conversion

Status: implemented

Delivered:

- sampling helpers now avoid unnecessary deep copies by default
- artifact report text is no longer eagerly loaded into the snapshot payload
- the render surface now leans on cached artifact access and narrower active
  sections

Primary code:

- `src/quant_pd_framework/streamlit_ui/data.py`
- `src/quant_pd_framework/streamlit_ui/state.py`
- `src/quant_pd_framework/streamlit_ui/results.py`

## 8. Make Artifact Downloads Lazy

Status: implemented

Delivered:

- governance views now expose a targeted artifact selector instead of eagerly
  materializing every download payload on each rerun
- binary artifact reads are cached behind the selected download action

Primary code:

- `src/quant_pd_framework/streamlit_ui/results.py`
- `src/quant_pd_framework/streamlit_ui/state.py`

## 9. Break Out Result Renderers Into Dedicated Modules

Status: implemented

Delivered:

- overview, governance, subset-search, section rendering, scorecard workbench,
  and feature drilldown logic now live outside the entrypoint
- result rendering can evolve independently of the controller and data-source
  logic

Primary code:

- `src/quant_pd_framework/streamlit_ui/results.py`

## 10. Remove Dead Or Drifted Entry-Point Helpers

Status: implemented

Delivered:

- the entrypoint no longer carries unused local helper logic
- stable helpers such as `build_editor_key` are re-exported intentionally for
  tests and tooling instead of remaining buried in a monolithic script

Primary code:

- `app/streamlit_app.py`
- `src/quant_pd_framework/streamlit_ui/data.py`
