"""Result and readiness renderers for the Streamlit UI."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from html import escape
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any
from zipfile import ZipFile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_pd_framework import ExecutionMode, TargetMode
from quant_pd_framework.decision_summary import (
    build_decision_summary,
    build_decision_summary_markdown,
)
from quant_pd_framework.figure_exports import (
    FigureExportAsset,
    build_figure_export_assets,
    build_individual_figure_zip,
)
from quant_pd_framework.large_data_runtime import (
    ResultTableRef,
    count_table_rows,
    distinct_column_values,
    query_table_page,
)
from quant_pd_framework.llm_documentation_package import (
    FIGURE_ASSETS_ROOT,
    build_llm_documentation_context_payload,
    build_llm_documentation_package_from_payload,
)
from quant_pd_framework.monitoring_package import (
    build_monitoring_package_from_payload,
    build_monitoring_package_payload,
)
from quant_pd_framework.presentation import (
    SECTION_SPECS,
    apply_advanced_visual_analytics,
    apply_fintech_figure_theme,
    build_asset_catalog,
    enhance_report_visualizations,
    format_metric_value,
    friendly_asset_title,
    prune_subset_search_highlight_assets,
    report_asset_badge,
    report_chart_guidance,
    summarize_run_kpis,
)
from quant_pd_framework.report_payload import optimize_report_visualizations
from quant_pd_framework.streamlit_ui.audit import record_gui_audit_event
from quant_pd_framework.streamlit_ui.data import sample_frame
from quant_pd_framework.streamlit_ui.decision_room import build_decision_room_payload
from quant_pd_framework.streamlit_ui.enterprise_workflow import (
    ReviewerRecord,
    build_artifact_explorer_frame,
    build_model_card_markdown,
)
from quant_pd_framework.streamlit_ui.glossary import render_glossary_badges
from quant_pd_framework.streamlit_ui.output_explainers import render_output_explainer
from quant_pd_framework.streamlit_ui.result_panels.registry_audit import (
    render_run_registry_panel as render_run_registry_panel,
)
from quant_pd_framework.streamlit_ui.run_execution import format_elapsed_seconds
from quant_pd_framework.streamlit_ui.scorecard_workbench import render_binning_theater
from quant_pd_framework.streamlit_ui.state import (
    build_plotly_key,
    prepare_table_for_display,
    read_binary_artifact,
    read_text_artifact,
    render_download_button,
    render_plotly_figure,
)
from quant_pd_framework.streamlit_ui.theme import render_metric_strip
from quant_pd_framework.workflow_guardrails import (
    build_guardrail_table,
    summarize_guardrail_counts,
)

SUITABILITY_DISPLAY_COLUMNS = [
    "status_label",
    "check_label",
    "subject",
    "observed_value",
    "threshold",
    "interpretation",
    "why_it_matters",
    "recommended_action",
    "details",
]

LLM_PACKAGE_MAX_CHARTS = 25
LLM_PACKAGE_INCLUDE_PNG = True
LLM_PACKAGE_PNG_LIMIT = 10
LLM_PACKAGE_CHART_ROOT = FIGURE_ASSETS_ROOT
LLM_PACKAGE_CHART_PRIORITY_PATTERNS = (
    ("roc", "auc"),
    ("ks",),
    ("calibration",),
    ("threshold",),
    ("lift", "gain"),
    ("feature_importance", "driver", "importance"),
    ("partial_dependence", "pdp"),
    ("scorecard", "woe", "bin"),
    ("actual_vs_predicted", "residual"),
    ("psi", "stability"),
    ("backtest",),
    ("quantile",),
)


@st.cache_data(show_spinner=False)
def _build_cached_individual_figure_package(
    run_id: str,
    visualization_signature: tuple[tuple[str, str], ...],
    _visualizations: dict[str, Any],
) -> bytes:
    """Builds individual chart files only when the user requests the package."""

    _ = (run_id, visualization_signature)
    return build_individual_figure_zip(_visualizations)


@st.cache_data(show_spinner=False)
def _build_cached_monitoring_package(
    run_id: str,
    artifact_signature: tuple[tuple[str, str, int], ...],
    payload: dict[str, Any],
) -> bytes:
    """Builds the ongoing-monitoring package only when the user requests it."""

    _ = (run_id, artifact_signature)
    return build_monitoring_package_from_payload(payload)


def format_memory_bytes(value: Any) -> str:
    """Formats byte counts for compact UI diagnostics."""

    if value is None:
        return "Not captured"
    try:
        bytes_value = float(value)
    except (TypeError, ValueError):
        return "Not captured"
    if bytes_value <= 0:
        return "0 B"

    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(bytes_value)} B"
    return f"{bytes_value:.1f} {units[unit_index]}"


def render_run_diagnostics_strip(snapshot: dict[str, Any]) -> None:
    """Renders run-level elapsed time and tracked dataframe memory diagnostics."""

    diagnostics = snapshot.get("run_diagnostics") or {}
    timing = snapshot.get("run_timing") or {}
    elapsed_seconds = diagnostics.get("elapsed_seconds", timing.get("elapsed_seconds"))
    peak_tracked_memory = diagnostics.get("peak_tracked_dataframe_memory_bytes")

    st.markdown("#### Run Diagnostics")
    render_metric_strip(
        [
            {
                "label": "Total run time",
                "value": format_elapsed_seconds(elapsed_seconds),
            },
            {
                "label": "Peak tracked dataframe memory",
                "value": format_memory_bytes(peak_tracked_memory),
            },
        ],
        compact=True,
    )
    if diagnostics.get("memory_profile_available"):
        st.caption(
            "Memory shown is tracked pandas dataframe memory from the debug trace, "
            "not total operating-system process RAM."
        )
    else:
        st.caption(
            "Peak tracked dataframe memory was not captured for this run. Enable "
            "`Capture memory profile in debug trace` to populate this value."
        )


def _snapshot_output_root(snapshot: dict[str, Any]) -> Path:
    artifacts = snapshot.get("artifacts", {})
    output_root = artifacts.get("output_root")
    if output_root:
        run_root = Path(str(output_root))
        return run_root.parent if run_root.name == str(snapshot.get("run_id", "")) else run_root
    config_root = snapshot.get("config", {}).get("artifacts", {}).get("output_root")
    return Path(str(config_root or "artifacts"))


def render_workflow_readiness(
    *,
    preview_config: Any,
    preview_findings: list[Any],
    preview_error: str | None,
) -> None:
    st.markdown(
        """
        <div class="workflow-stage">
          <div class="workflow-stage__index">3</div>
          <div class="workflow-stage__body">
            <span class="workflow-stage__kicker">Readiness Check & Run</span>
            <h2>Validate the configured workflow before execution</h2>
            <p>
              This summary uses the same typed configuration build that will be used at run time,
              including preset-specific guardrails and documentation requirements.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if preview_config is None:
        if preview_error:
            render_readiness_blocker(preview_error)
        st.info("Readiness details will appear after the current configuration resolves cleanly.")
        return
    preset_value = preview_config.preset_name.value if preview_config.preset_name else "custom"
    model_family = preview_config.model.model_type.value.replace("_", " ").title()
    data_structure = preview_config.split.data_structure.value.replace("_", " ").title()

    if not preview_config.workflow_guardrails.enabled:
        if preview_error:
            render_readiness_blocker(preview_error)
        render_metric_strip(
            [
                {"label": "Preset", "value": preset_value},
                {"label": "Guardrails", "value": "Disabled"},
                {"label": "Model Family", "value": model_family},
                {"label": "Data Structure", "value": data_structure},
            ],
            compact=True,
        )
        st.info("Workflow guardrails are currently disabled for this run.")
        return

    counts = summarize_guardrail_counts(preview_findings)
    render_metric_strip(
        [
            {"label": "Preset", "value": preset_value},
            {"label": "Errors", "value": f"{counts.get('error', 0):,}"},
            {"label": "Warnings", "value": f"{counts.get('warning', 0):,}"},
            {"label": "Model Family", "value": model_family},
            {"label": "Data Structure", "value": data_structure},
        ],
        compact=True,
    )
    if preview_error:
        render_readiness_blocker(preview_error)
    if not preview_findings:
        st.success("The current preset-specific readiness checks passed.")
        return

    readiness_table = build_guardrail_table(preview_findings)
    if counts.get("error", 0):
        st.error("Resolve the blocking guardrail findings before running the workflow.")
    elif counts.get("warning", 0):
        st.warning("The run is allowed, but review the preset warnings before execution.")
    st.dataframe(
        prepare_table_for_display(readiness_table),
        width="stretch",
        hide_index=True,
    )


def render_readiness_blocker(message: str) -> None:
    safe_message = escape(message)
    title = (
        "Target source required"
        if "target" in message.lower()
        else "Readiness issue requires attention"
    )
    st.markdown(
        f"""
        <div class="readiness-blocker-card">
          <span class="readiness-blocker-card__icon">!</span>
          <div>
            <strong>{escape(title)}</strong>
            <p>{safe_message}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def with_report_enhancement_visualizations(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Returns a view snapshot with report-grade companion charts added."""

    include_enhanced = snapshot.get("include_enhanced_report_visuals", True)
    include_advanced = snapshot.get("include_advanced_visual_analytics", False)
    if not include_enhanced and not include_advanced:
        return snapshot
    if snapshot.get("_report_enhancements_applied") and (
        not include_advanced or snapshot.get("_advanced_visual_analytics_applied")
    ):
        return snapshot
    visualizations = snapshot["visualizations"]
    if include_enhanced and not snapshot.get("_report_enhancements_applied"):
        visualizations = enhance_report_visualizations(
            metrics=snapshot["metrics"],
            diagnostics_tables=snapshot["diagnostics_tables"],
            visualizations=visualizations,
            target_mode=snapshot["target_mode"],
            labels_available=snapshot["labels_available"],
            predictions=snapshot.get("predictions"),
        )
        snapshot["_report_enhancements_applied"] = True
    if include_advanced and not snapshot.get("_advanced_visual_analytics_applied"):
        visualizations = apply_advanced_visual_analytics(
            metrics=snapshot["metrics"],
            diagnostics_tables=snapshot["diagnostics_tables"],
            visualizations=visualizations,
            target_mode=snapshot["target_mode"],
            labels_available=snapshot["labels_available"],
            predictions=snapshot.get("predictions"),
        )
        snapshot["_advanced_visual_analytics_applied"] = True
    snapshot["visualizations"] = visualizations
    return snapshot


def render_run_results(snapshot: dict[str, Any]) -> None:
    snapshot = with_report_enhancement_visualizations(snapshot)
    if snapshot["execution_mode"] == ExecutionMode.SEARCH_FEATURE_SUBSETS.value:
        render_subset_search_results(snapshot)
        return

    st.markdown(
        """
        <div class="workflow-stage">
          <div class="workflow-stage__index">4</div>
          <div class="workflow-stage__body">
            <span class="workflow-stage__kicker">Diagnostic Studio</span>
            <h2>Validation outputs organized by decision workflow</h2>
            <p>
              Review the run through grouped sections, interactive filters, and a
              polished export bundle that mirrors the live dashboard.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_run_diagnostics_strip(snapshot)

    asset_catalog = prune_subset_search_highlight_assets(
        build_asset_catalog(snapshot["diagnostics_tables"], snapshot["visualizations"])
    )
    prediction_frames = [
        frame for frame in snapshot["predictions"].values() if isinstance(frame, pd.DataFrame)
    ]
    all_predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    filter_state = render_result_filters(snapshot, all_predictions)
    filtered_predictions = apply_prediction_filters(snapshot, all_predictions, filter_state)
    render_large_data_paged_prediction_browser(snapshot)

    section_map: dict[str, str] = {"Overview": "overview", "Governance": "governance"}
    section_labels = ["Overview"]
    for section_id, payload in asset_catalog.items():
        if payload["figures"] or payload["tables"]:
            label = SECTION_SPECS[section_id]["title"]
            section_map[label] = section_id
            section_labels.append(label)
    section_labels.append("Governance")
    selected_section = st.radio(
        "Result section",
        options=section_labels,
        horizontal=True,
        key=f"{snapshot['run_id']}_result_section",
        label_visibility="collapsed",
    )

    selected_id = section_map[selected_section]
    if selected_id == "overview":
        render_overview_panel(snapshot, filtered_predictions, filter_state, asset_catalog)
    elif selected_id == "governance":
        render_governance_panel(snapshot, filtered_predictions)
    else:
        render_section_panel(
            snapshot=snapshot,
            section_id=selected_id,
            section_payload=asset_catalog[selected_id],
            filtered_predictions=filtered_predictions,
            filter_state=filter_state,
        )


def render_subset_search_results(snapshot: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="workflow-stage">
          <div class="workflow-stage__index">4</div>
          <div class="workflow-stage__body">
            <span class="workflow-stage__kicker">Subset Search Studio</span>
            <h2>Compare candidate feature sets before full model development</h2>
            <p>
              This execution mode is intentionally separate from normal development.
              The outputs below focus only on subset ranking, ROC and KS comparison,
              significance tests, and performance-versus-parsimony evidence.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_run_diagnostics_strip(snapshot)

    asset_catalog = build_asset_catalog(snapshot["diagnostics_tables"], snapshot["visualizations"])
    section_map: dict[str, str] = {"Overview": "overview", "Governance": "governance"}
    section_labels = ["Overview"]
    for section_id, payload in asset_catalog.items():
        if payload["figures"] or payload["tables"]:
            label = SECTION_SPECS[section_id]["title"]
            section_map[label] = section_id
            section_labels.append(label)
    section_labels.append("Governance")
    selected_section = st.radio(
        "Subset search section",
        options=section_labels,
        horizontal=True,
        key=f"{snapshot['run_id']}_subset_result_section",
        label_visibility="collapsed",
    )
    selected_id = section_map[selected_section]
    if selected_id == "overview":
        render_subset_search_overview(snapshot)
    elif selected_id == "governance":
        render_subset_search_governance(snapshot)
    else:
        render_subset_search_section(
            snapshot=snapshot,
            section_id=selected_id,
            section_payload=asset_catalog[selected_id],
        )


def render_decision_summary(snapshot: dict[str, Any]) -> None:
    """Renders the fifth workflow step: a decision-ready run scorecard."""

    snapshot = with_report_enhancement_visualizations(snapshot)
    summary = build_decision_summary(snapshot)
    st.markdown(
        """
        <div class="workflow-stage">
          <div class="workflow-stage__index">5</div>
          <div class="workflow-stage__body">
            <span class="workflow-stage__kicker">Decision Summary</span>
            <h2>Synthesize the completed run into a model decision scorecard</h2>
            <p>
              Use this page to review the recommendation, decision issues,
              primary metrics, top feature drivers, and the evidence files
              that support the model development decision.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_decision_level_message(
        recommendation=summary["recommendation"],
        level=summary["level"],
    )
    render_metric_strip(summary["cards"], compact=True)

    rationale_column, download_column = st.columns([0.7, 0.3], gap="large")
    with rationale_column:
        st.markdown("### Decision Rationale")
        for item in summary["rationale"]:
            st.markdown(f"- {item}")
    with download_column:
        st.markdown("### Export")
        _render_decision_summary_download(snapshot)
        _render_individual_images_download(snapshot)
        _render_llm_package_download(snapshot)
        _render_monitoring_package_download(snapshot)

    (
        decision_room_tab,
        metric_tab,
        issue_tab,
        feature_tab,
        lineage_tab,
        validation_tab,
        evidence_tab,
        traceability_tab,
        dossier_tab,
    ) = st.tabs(
        [
            "Decision Room",
            "Metrics",
            "Issues",
            "Feature Drivers",
            "Feature Lineage",
            "Validation Checklist",
            "Evidence Index",
            "Traceability Map",
            "Dossier",
        ]
    )
    with decision_room_tab:
        _render_decision_room(snapshot, summary)
    with metric_tab:
        render_output_explainer("split_metrics")
        st.dataframe(
            prepare_table_for_display(summary["metric_frame"]),
            width="stretch",
            hide_index=True,
        )
    with issue_tab:
        st.dataframe(
            prepare_table_for_display(summary["issue_frame"]),
            width="stretch",
            hide_index=True,
        )
    with feature_tab:
        render_output_explainer("feature_importance")
        st.dataframe(
            prepare_table_for_display(summary["feature_frame"]),
            width="stretch",
            hide_index=True,
        )
    with lineage_tab:
        lineage = snapshot["diagnostics_tables"].get("feature_lineage_map", pd.DataFrame())
        if lineage.empty:
            st.info("Feature lineage was not captured for this run.")
        else:
            st.caption(
                "This map connects final model terms back to source features, transformations, "
                "imputation policies, selection rationale, and feature documentation."
            )
            st.dataframe(
                prepare_table_for_display(lineage),
                width="stretch",
                hide_index=True,
            )
    with validation_tab:
        render_output_explainer("validation_checklist")
        st.dataframe(
            prepare_table_for_display(summary["validation_checklist_frame"]),
            width="stretch",
            hide_index=True,
        )
    with evidence_tab:
        st.dataframe(
            prepare_table_for_display(summary["evidence_frame"]),
            width="stretch",
            hide_index=True,
        )
    with traceability_tab:
        render_output_explainer("evidence_traceability_map")
        st.dataframe(
            prepare_table_for_display(summary["traceability_frame"]),
            width="stretch",
            hide_index=True,
        )
    with dossier_tab:
        dossier_path = snapshot.get("artifacts", {}).get("development_dossier", "")
        if dossier_path and Path(dossier_path).exists():
            dossier_text = read_text_artifact(dossier_path)
            st.markdown(dossier_text)
            render_download_button(
                "Download model development dossier",
                dossier_text,
                file_name=f"{snapshot.get('run_id', 'run')}_model_development_dossier.md",
                mime="text/markdown",
            )
        else:
            st.info("The model development dossier was not exported for this run.")


def _render_decision_summary_download(snapshot: dict[str, Any]) -> None:
    artifacts = snapshot.get("artifacts", {})
    st.download_button(
        "Download decision summary",
        data=build_decision_summary_markdown(snapshot),
        file_name=f"{snapshot.get('run_id', 'run')}_decision_summary.md",
        mime="text/markdown",
        width="stretch",
        key=f"{snapshot.get('run_id', 'run')}_decision_summary_download",
        help="Downloads the Markdown decision summary for the completed run.",
        on_click=record_gui_audit_event,
        kwargs={
            "output_root": _snapshot_output_root(snapshot),
            "event_type": "decision_summary_downloaded",
            "run_id": str(snapshot.get("run_id", "")),
            "artifact_root": artifacts.get("output_root"),
            "metadata": {"file_name": f"{snapshot.get('run_id', 'run')}_decision_summary.md"},
        },
    )
    st.caption("Downloads the Markdown decision summary immediately.")


def _render_individual_images_download(snapshot: dict[str, Any]) -> None:
    raw_visualizations = dict(snapshot.get("visualizations") or {})
    if not raw_visualizations:
        st.download_button(
            "Download Individual Images",
            data=b"",
            file_name=f"{snapshot.get('run_id', 'run')}_individual_images.zip",
            mime="application/zip",
            width="stretch",
            disabled=True,
            help="No chart visualizations are available for this completed run.",
        )
        return

    def build_package() -> bytes:
        visualizations = _download_visualizations(snapshot)
        return _build_cached_individual_figure_package(
            str(snapshot.get("run_id", "run")),
            _visualization_signature(visualizations),
            visualizations,
        )

    _render_lazy_zip_download(
        snapshot=snapshot,
        label="Download Individual Images",
        state_suffix="individual_images_package",
        file_name=f"{snapshot.get('run_id', 'run')}_individual_images.zip",
        help_text=(
            "Prepares a zip of standalone chart PNG and HTML files. Generation happens "
            "only when requested so model runs do not wait on individual image export."
        ),
        build_package=build_package,
        ready_label="Save Individual Images ZIP",
        caption=(
            "Includes PNG files plus lightweight HTML charts that share one Plotly "
            "JavaScript file."
        ),
        audit_event_prefix="individual_images_package",
    )


def _render_llm_package_download(snapshot: dict[str, Any]) -> None:
    artifacts = snapshot.get("artifacts", {})
    signature = _artifact_signature(artifacts)

    def build_package() -> bytes:
        return _build_llm_documentation_package_for_download(snapshot, signature)

    _render_lazy_zip_download(
        snapshot=snapshot,
        label="Download LLM Package",
        state_suffix="llm_documentation_package",
        file_name=f"{snapshot.get('run_id', 'run')}_llm_documentation_package.zip",
        help_text=(
            "Prepares an LLM-readable evidence package for drafting a model methodology "
            "document. Raw row-level data, row-level predictions, and model binaries are "
            "excluded by default."
        ),
        build_package=build_package,
        ready_label="Save LLM Package ZIP",
        caption=(
            "Includes prompt, outline, citation map, evidence checklist, TOC drop zone, "
            "build timing profile, and capped HTML plus document-ready PNG chart assets."
        ),
        show_spinner=False,
        audit_event_prefix="llm_package",
    )


def _build_llm_documentation_package_for_download(
    snapshot: dict[str, Any],
    artifact_signature: tuple[str, int],
) -> bytes:
    """Builds the LLM package with visible progress and chart export limits."""

    profile_rows: list[dict[str, Any]] = [
        {
            "stage": "package_options",
            "elapsed_seconds": 0.0,
            "max_charts": LLM_PACKAGE_MAX_CHARTS,
            "include_png": LLM_PACKAGE_INCLUDE_PNG,
            "png_figure_limit": LLM_PACKAGE_PNG_LIMIT,
            "artifact_signature": str(artifact_signature),
        }
    ]
    total_started = perf_counter()

    with st.status("Preparing LLM package...", expanded=True) as status:
        stage_started = perf_counter()
        status.write("Building curated model evidence context.")
        payload = build_llm_documentation_context_payload(snapshot)
        _record_llm_package_stage(
            profile_rows,
            "build_context_payload",
            stage_started,
            payload_keys=len(payload),
        )

        stage_started = perf_counter()
        status.write("Selecting optimized chart evidence.")
        visualizations = _prioritize_llm_visualizations(_download_visualizations(snapshot))
        _record_llm_package_stage(
            profile_rows,
            "optimize_visualizations",
            stage_started,
            available_charts=len(visualizations),
            max_charts=LLM_PACKAGE_MAX_CHARTS,
        )

        stage_started = perf_counter()
        reused_chart_assets = _prepared_llm_chart_assets_from_individual_package(snapshot)
        if reused_chart_assets:
            chart_assets = reused_chart_assets
            chart_source = "reused_prepared_individual_images"
            status.write("Reusing chart assets already prepared this session.")
        else:
            status.write("Rendering capped HTML and document-ready PNG chart assets.")
            chart_assets = build_figure_export_assets(
                visualizations,
                root_dir=LLM_PACKAGE_CHART_ROOT,
                include_html=True,
                include_png=LLM_PACKAGE_INCLUDE_PNG,
                max_figures=LLM_PACKAGE_MAX_CHARTS,
                png_figure_limit=LLM_PACKAGE_PNG_LIMIT,
            ).assets
            chart_source = "rendered_html_only"
        _record_llm_package_stage(
            profile_rows,
            "prepare_chart_assets",
            stage_started,
            chart_source=chart_source,
            asset_count=len(chart_assets),
            chart_count=sum(1 for asset in chart_assets if asset.file_format == "html"),
            png_count=sum(1 for asset in chart_assets if asset.file_format == "png"),
        )

        status.write("Writing controlled evidence zip.")
        package_bytes = build_llm_documentation_package_from_payload(
            payload,
            generated_chart_assets=chart_assets,
            build_profile=profile_rows,
        )
        total_elapsed = perf_counter() - total_started
        status.update(
            label=(
                "LLM package ready "
                f"({len(package_bytes) / (1024 * 1024):.1f} MB, "
                f"{format_elapsed_seconds(total_elapsed)})."
            ),
            state="complete",
            expanded=False,
        )
        return package_bytes


def _record_llm_package_stage(
    rows: list[dict[str, Any]],
    stage: str,
    started_at: float,
    **metadata: Any,
) -> None:
    row: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": round(perf_counter() - started_at, 3),
    }
    row.update(metadata)
    rows.append(row)


def _prioritize_llm_visualizations(visualizations: dict[str, Any]) -> dict[str, Any]:
    """Orders high-value documentation charts before lower-priority supporting visuals."""

    ranked_items = sorted(
        enumerate(visualizations.items()),
        key=lambda item: (_llm_chart_priority(str(item[1][0])), item[0]),
    )
    return {str(name): figure for _index, (name, figure) in ranked_items}


def _llm_chart_priority(name: str) -> int:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    for priority, patterns in enumerate(LLM_PACKAGE_CHART_PRIORITY_PATTERNS):
        if any(pattern in normalized for pattern in patterns):
            return priority
    return len(LLM_PACKAGE_CHART_PRIORITY_PATTERNS)


def _prepared_llm_chart_assets_from_individual_package(
    snapshot: dict[str, Any],
) -> tuple[FigureExportAsset, ...]:
    """Reuses prepared individual-image assets without re-rendering Plotly figures."""

    run_id = str(snapshot.get("run_id", "run"))
    state_key = f"{run_id}_individual_images_package_bytes"
    package_bytes = st.session_state.get(state_key)
    if not isinstance(package_bytes, bytes | bytearray):
        return ()

    try:
        with ZipFile(BytesIO(package_bytes)) as archive:
            names = archive.namelist()
            html_names = sorted(
                [
                    name
                    for name in names
                    if name.startswith("individual_images/html/")
                    and Path(name).suffix.lower() == ".html"
                    and Path(name).name != "plotly.min.js"
                ],
                key=lambda name: _llm_chart_priority(Path(name).stem),
            )
            selected_html_names = html_names[:LLM_PACKAGE_MAX_CHARTS]
            selected_stems = [Path(name).stem for name in selected_html_names]
            selected_png_stems = set(selected_stems[:LLM_PACKAGE_PNG_LIMIT])
            png_names = {
                Path(name).stem: name
                for name in names
                if name.startswith("individual_images/png/")
                and Path(name).suffix.lower() == ".png"
            }
            assets: list[FigureExportAsset] = []
            plotly_js_name = "individual_images/html/plotly.min.js"
            if plotly_js_name in names:
                data = archive.read(plotly_js_name)
                assets.append(
                    FigureExportAsset(
                        arcname=f"{LLM_PACKAGE_CHART_ROOT}/html/plotly.min.js",
                        data=data,
                        figure_name="plotly.min.js",
                        file_format="javascript",
                        size_bytes=len(data),
                    )
                )
            for archive_name in selected_html_names:
                data = archive.read(archive_name)
                path = Path(archive_name)
                assets.append(
                    FigureExportAsset(
                        arcname=f"{LLM_PACKAGE_CHART_ROOT}/html/{path.name}",
                        data=data,
                        figure_name=path.stem,
                        file_format="html",
                        size_bytes=len(data),
                    )
                )
            for stem in selected_stems:
                if stem not in selected_png_stems or stem not in png_names:
                    continue
                data = archive.read(png_names[stem])
                assets.append(
                    FigureExportAsset(
                        arcname=f"{LLM_PACKAGE_CHART_ROOT}/png/{stem}.png",
                        data=data,
                        figure_name=stem,
                        file_format="png",
                        size_bytes=len(data),
                    )
                )
    except Exception:  # pragma: no cover - corrupt session package should fall back.
        return ()

    if not any(asset.file_format == "html" for asset in assets):
        return ()
    skipped_figure_names = [Path(name).stem for name in html_names[len(selected_html_names) :]]
    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "source": "reused_individual_images_session_package",
        "available_figure_count": len(selected_stems) + len(skipped_figure_names),
        "figure_count": len(selected_stems),
        "skipped_figure_count": len(skipped_figure_names),
        "max_figures": LLM_PACKAGE_MAX_CHARTS,
        "html_enabled": True,
        "png_enabled": True,
        "png_figure_limit": LLM_PACKAGE_PNG_LIMIT,
        "selected_figures": selected_stems,
        "skipped_figures": skipped_figure_names,
        "figures": {
            figure_name: {
                "html": f"{LLM_PACKAGE_CHART_ROOT}/html/{figure_name}.html",
                **(
                    {"png": f"{LLM_PACKAGE_CHART_ROOT}/png/{figure_name}.png"}
                    if figure_name in selected_png_stems and figure_name in png_names
                    else {}
                ),
                "safe_name": figure_name,
            }
            for figure_name in selected_stems
        },
        "support_files": [
            {
                "path": f"{LLM_PACKAGE_CHART_ROOT}/html/plotly.min.js",
                "file_format": "javascript",
            }
        ],
        "warnings": [],
    }
    manifest_bytes = json.dumps(manifest, indent=2, default=str).encode("utf-8")
    assets.append(
        FigureExportAsset(
            arcname=f"{LLM_PACKAGE_CHART_ROOT}/figure_manifest.json",
            data=manifest_bytes,
            figure_name="figure_manifest",
            file_format="json",
            size_bytes=len(manifest_bytes),
        )
    )
    return tuple(assets)


def _render_lazy_zip_download(
    *,
    snapshot: dict[str, Any],
    label: str,
    state_suffix: str,
    file_name: str,
    help_text: str,
    build_package: Callable[[], bytes],
    ready_label: str,
    caption: str,
    show_spinner: bool = True,
    audit_event_prefix: str = "",
) -> None:
    run_id = str(snapshot.get("run_id", "run"))
    state_key = f"{run_id}_{state_suffix}_bytes"
    error_key = f"{run_id}_{state_suffix}_error"
    output_root = _snapshot_output_root(snapshot)
    artifact_root = snapshot.get("artifacts", {}).get("output_root")
    if st.button(label, key=f"{state_key}_prepare", width="stretch", help=help_text):
        st.session_state.pop(error_key, None)
        try:
            if show_spinner:
                with st.spinner(f"Preparing {label.lower()}..."):
                    st.session_state[state_key] = build_package()
            else:
                st.session_state[state_key] = build_package()
            if audit_event_prefix:
                package_bytes = st.session_state.get(state_key, b"")
                record_gui_audit_event(
                    output_root,
                    f"{audit_event_prefix}_prepared",
                    run_id=run_id,
                    artifact_root=artifact_root,
                    metadata={
                        "file_name": file_name,
                        "size_bytes": len(package_bytes)
                        if isinstance(package_bytes, bytes | bytearray)
                        else None,
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive UI fallback.
            st.session_state.pop(state_key, None)
            st.session_state[error_key] = str(exc)
            if audit_event_prefix:
                record_gui_audit_event(
                    output_root,
                    f"{audit_event_prefix}_prepare_failed",
                    run_id=run_id,
                    artifact_root=artifact_root,
                    metadata={"file_name": file_name, "error_message": str(exc)},
                )

    package_bytes = st.session_state.get(state_key)
    if package_bytes:
        st.download_button(
            ready_label,
            data=package_bytes,
            file_name=file_name,
            mime="application/zip",
            width="stretch",
            help=help_text,
            on_click=record_gui_audit_event if audit_event_prefix else None,
            kwargs={
                "output_root": output_root,
                "event_type": f"{audit_event_prefix}_downloaded",
                "run_id": run_id,
                "artifact_root": artifact_root,
                "metadata": {"file_name": file_name, "size_bytes": len(package_bytes)},
            }
            if audit_event_prefix
            else None,
        )
        st.caption(caption)
    elif st.session_state.get(error_key):
        st.warning(f"{label} could not be prepared: {st.session_state[error_key]}")
    else:
        st.caption("Click to prepare the zip package on demand.")


def _render_monitoring_package_download(snapshot: dict[str, Any]) -> None:
    if snapshot.get("execution_mode") != ExecutionMode.FIT_NEW_MODEL.value:
        st.download_button(
            "Download OM Package",
            data=b"",
            file_name=f"{snapshot.get('run_id', 'run')}_om_package.zip",
            mime="application/zip",
            width="stretch",
            disabled=True,
            help="Available only for completed new-model-fit runs.",
        )
        return

    artifacts = snapshot.get("artifacts", {})
    signature = _monitoring_artifact_signature(artifacts)

    def build_package() -> bytes:
        payload = build_monitoring_package_payload(snapshot)
        return _build_cached_monitoring_package(
            str(snapshot.get("run_id", "run")),
            signature,
            payload,
        )

    _render_lazy_zip_download(
        snapshot=snapshot,
        label="Download OM Package",
        state_suffix="om_package",
        file_name=f"{snapshot.get('run_id', 'run')}_om_package.zip",
        help_text=(
            "Downloads a model bundle for the separate ongoing-monitoring app. "
            "The bundle is created on demand so normal model runs do not spend "
            "time copying or converting monitoring files."
        ),
        build_package=build_package,
        ready_label="Save OM Package ZIP",
        audit_event_prefix="om_package",
        caption=(
            "Includes model, config, generated runner, manifest, metadata, and "
            "available CSV inputs/scores."
        ),
    )


def _artifact_signature(artifacts: dict[str, Any]) -> tuple[str, int]:
    signature_path = (
        artifacts.get("manifest")
        or artifacts.get("artifact_manifest")
        or artifacts.get("output_root")
        or ""
    )
    path = Path(str(signature_path)) if signature_path else Path()
    if signature_path and path.exists():
        try:
            return str(path), int(path.stat().st_mtime_ns)
        except OSError:
            return str(path), 0
    return str(signature_path), 0


def _visualization_signature(visualizations: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for name, figure in sorted(visualizations.items(), key=lambda item: str(item[0])):
        figure_type = type(figure).__name__
        trace_count = "0"
        try:
            trace_count = str(len(getattr(figure, "data", []) or []))
        except TypeError:
            trace_count = "0"
        rows.append((str(name), f"{figure_type}:{trace_count}"))
    return tuple(rows)


def _download_visualizations(snapshot: dict[str, Any]) -> dict[str, Any]:
    visualizations = dict(snapshot.get("visualizations") or {})
    if not visualizations:
        return {}
    limits = _download_visualization_limits(snapshot)
    optimized, _audit = optimize_report_visualizations(
        visualizations,
        max_points_per_figure=limits["max_points_per_figure"],
        max_figure_payload_mb=limits["max_figure_payload_mb"],
        max_total_figure_payload_mb=limits["max_total_figure_payload_mb"],
    )
    return optimized


def _download_visualization_limits(snapshot: dict[str, Any]) -> dict[str, float | int]:
    config_payload = snapshot.get("config")
    if not isinstance(config_payload, dict):
        config_payload = {}
    performance = dict(config_payload.get("performance", {}) or {})
    return {
        "max_points_per_figure": int(performance.get("html_max_points_per_figure", 7500)),
        "max_figure_payload_mb": float(performance.get("html_max_figure_payload_mb", 3.0)),
        "max_total_figure_payload_mb": float(
            performance.get("html_max_total_figure_payload_mb", 60.0)
        ),
    }


def _monitoring_artifact_signature(artifacts: dict[str, Any]) -> tuple[tuple[str, str, int], ...]:
    signature_keys = [
        "model",
        "config",
        "runner_script",
        "manifest",
        "artifact_manifest",
        "input_snapshot",
        "predictions",
        "code_snapshot_dir",
    ]
    rows: list[tuple[str, str, int]] = []
    for key in signature_keys:
        value = artifacts.get(key)
        path = Path(str(value)) if value else Path()
        timestamp = 0
        if value and path.exists():
            try:
                timestamp = int(path.stat().st_mtime_ns)
            except OSError:
                timestamp = 0
        rows.append((key, str(value or ""), timestamp))
    return tuple(rows)


def _render_decision_room(snapshot: dict[str, Any], summary: dict[str, Any]) -> None:
    payload = build_decision_room_payload(snapshot, summary)
    st.markdown(
        f"""
        <div class="decision-room-card">
          <span class="decision-room-card__eyebrow">Decision Room</span>
          <h3>{escape(payload["readiness"])}</h3>
          <p>
            Meeting-ready view of the model recommendation, open items, top
            drivers, key artifacts, and next actions. Detailed evidence remains
            available in the tabs beside this view.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_glossary_badges(
        ["AUC", "KS", "Calibration", "PSI", "WoE", "Challenger"],
        caption="Review terms",
    )
    render_metric_strip(payload["headline_cards"], compact=True)

    st.markdown("#### Decision Rationale")
    for item in payload["rationale"]:
        st.markdown(f"- {item}")

    columns = st.columns(3, gap="medium")
    with columns[0]:
        _render_decision_room_list(
            "Attention Items",
            [
                f"{item['source']}: {item['message']}"
                for item in payload["attention_items"]
            ]
            or ["No open attention items surfaced in the decision summary."],
        )
    with columns[1]:
        _render_decision_room_list(
            "Top Drivers",
            [
                f"{item['feature']} ({item['value']})"
                for item in payload["top_features"]
                if item["feature"]
            ]
            or ["No feature-driver table was available."],
        )
    with columns[2]:
        _render_decision_room_list(
            "Next Actions",
            payload["next_actions"],
        )

    if payload["key_artifacts"]:
        st.markdown("#### Key Artifacts")
        st.dataframe(
            prepare_table_for_display(pd.DataFrame(payload["key_artifacts"])),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No primary artifact links were available in the decision summary.")


def _render_decision_room_list(title: str, items: list[str]) -> None:
    list_items = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="decision-room-list-card">
          <h4>{escape(title)}</h4>
          <ul>{list_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_decision_level_message(*, recommendation: str, level: str) -> None:
    message = (
        f"Automated decision summary: **{recommendation}**. "
        "This is a synthesis aid for model builders and validation teams; it is not "
        "a substitute for formal approval."
    )
    if level == "proceed":
        st.success(message)
    elif level == "revise":
        st.warning(message)
    elif level == "caution":
        st.warning(message)
    else:
        st.info(message)


def render_subset_search_overview(snapshot: dict[str, Any]) -> None:
    kpis = summarize_run_kpis(
        metrics=snapshot["metrics"],
        input_rows=snapshot["input_shape"].get("rows"),
        feature_count=int(snapshot["feature_summary"].get("feature_count", 0)),
        labels_available=snapshot["labels_available"],
        execution_mode=snapshot["execution_mode"],
        model_type=snapshot["model_type"],
        target_mode=snapshot["target_mode"],
        warning_count=len(snapshot["warnings"]),
    )
    render_metric_strip(kpis)

    best_candidate = snapshot.get("subset_search_best_candidate", {})
    if best_candidate:
        st.markdown("### Best Candidate Snapshot")
        render_metric_strip(
            [
                {"label": "Candidate ID", "value": best_candidate.get("candidate_id", "N/A")},
                {
                    "label": "Feature Count",
                    "value": format_metric_value(best_candidate.get("feature_count")),
                },
                {
                    "label": "Validation ROC AUC",
                    "value": format_metric_value(best_candidate.get("ranking_roc_auc")),
                },
                {
                    "label": "Validation KS",
                    "value": format_metric_value(best_candidate.get("ranking_ks_statistic")),
                },
                {
                    "label": "Test ROC AUC",
                    "value": format_metric_value(best_candidate.get("test_roc_auc")),
                },
                {
                    "label": "Test KS",
                    "value": format_metric_value(best_candidate.get("test_ks_statistic")),
                },
            ],
            compact=True,
        )
        st.caption(f"Feature set: {best_candidate.get('feature_set', 'N/A')}")

    selected_candidate_table = snapshot["diagnostics_tables"].get(
        "subset_search_selected_candidate",
        pd.DataFrame(),
    )
    selected_coefficient_table = snapshot["diagnostics_tables"].get(
        "subset_search_selected_coefficients",
        pd.DataFrame(),
    )
    if not selected_candidate_table.empty or not selected_coefficient_table.empty:
        left_column, right_column = st.columns([0.95, 1.05], gap="large")
        with left_column:
            if not selected_candidate_table.empty:
                st.markdown("### Selected Candidate")
                st.dataframe(
                    prepare_table_for_display(selected_candidate_table),
                    width="stretch",
                    hide_index=True,
                )
        with right_column:
            if not selected_coefficient_table.empty:
                st.markdown("### Selected Candidate Coefficients")
                st.dataframe(
                    prepare_table_for_display(selected_coefficient_table),
                    width="stretch",
                    hide_index=True,
                )

    if snapshot["warnings"]:
        for warning in snapshot["warnings"]:
            st.warning(warning)

    overview_figure_keys = [
        "subset_search_selected_roc_curve",
        "subset_search_selected_ks_curve",
        "subset_search_metric_frontier",
        "subset_search_auc_frontier",
        "subset_search_ks_frontier",
        "subset_search_feature_frequency_chart",
    ]
    figures = [
        (friendly_asset_title(figure_key, kind="figure"), snapshot["visualizations"][figure_key])
        for figure_key in overview_figure_keys
        if figure_key in snapshot["visualizations"]
    ]
    if figures:
        columns = st.columns(2)
        for index, (title, figure) in enumerate(figures):
            with columns[index % 2]:
                st.markdown(f"#### {title}")
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(snapshot["run_id"], "subset_search_overview", title),
                )

    candidate_table = snapshot["diagnostics_tables"].get(
        "subset_search_nonwinning_candidates", pd.DataFrame()
    )
    if not candidate_table.empty:
        st.markdown("### Ranked Non-Winning Candidates")
        leading_candidates = candidate_table.head(25)
        st.dataframe(
            prepare_table_for_display(leading_candidates),
            width="stretch",
            hide_index=True,
        )


def render_subset_search_section(
    *,
    snapshot: dict[str, Any],
    section_id: str,
    section_payload: dict[str, Any],
) -> None:
    st.markdown(
        f"""
        <div class="section-subheader">
          <span class="section-kicker">{section_payload["title"]}</span>
          <p>{section_payload["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    figure_descriptors = choose_descriptors_for_view(section_payload["figures"], "Technical")
    table_descriptors = choose_descriptors_for_view(section_payload["tables"], "Technical")

    if figure_descriptors:
        columns = st.columns(2)
        for index, descriptor in enumerate(figure_descriptors):
            figure = snapshot["visualizations"].get(descriptor.key)
            if figure is None:
                continue
            with columns[index % 2]:
                st.markdown(f"#### {descriptor.title}")
                render_chart_review_context(descriptor)
                if descriptor.description:
                    st.caption(descriptor.description)
                render_output_explainer(descriptor.key)
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "subset_search_section",
                        section_id,
                        descriptor.key,
                    ),
                )

    if table_descriptors:
        st.markdown("### Reference Tables")
        for descriptor in table_descriptors:
            table = snapshot["diagnostics_tables"].get(descriptor.key)
            if table is None or table.empty:
                continue
            with st.expander(descriptor.title, expanded=descriptor.featured):
                if descriptor.description:
                    st.caption(descriptor.description)
                st.dataframe(
                    prepare_table_for_display(table),
                    width="stretch",
                    hide_index=True,
                )


def render_subset_search_governance(snapshot: dict[str, Any]) -> None:
    left_column, right_column = st.columns([1.1, 0.9], gap="large")

    with left_column:
        st.markdown("### Narrative Report")
        st.text_area(
            "Run report",
            value=_report_text(snapshot),
            height=320,
            label_visibility="collapsed",
        )

        candidate_table = snapshot["diagnostics_tables"].get(
            "subset_search_candidates", pd.DataFrame()
        )
        if not candidate_table.empty:
            st.markdown("### Candidate Ranking Preview")
            st.dataframe(
                prepare_table_for_display(candidate_table.head(100)),
                width="stretch",
                hide_index=True,
            )

    with right_column:
        if snapshot["warnings"]:
            st.markdown("### Warnings")
            for warning in snapshot["warnings"]:
                st.warning(warning)
        render_reviewer_workspace(snapshot)
        render_artifact_locations(snapshot)
        render_download_button(
            "Download Run Config",
            snapshot["config"],
            "run_config.json",
            "application/json",
        )
        render_download_button(
            "Download Markdown Report",
            _report_text(snapshot),
            "run_report.md",
            "text/markdown",
        )
        render_lazy_artifact_downloads(
            snapshot,
            artifact_names=["interactive_report", "tests", "input_snapshot"],
            key_prefix="subset_governance",
        )

        if snapshot["events"]:
            st.markdown("### Pipeline Events")
            for event in snapshot["events"]:
                st.caption(f"- {event}")


def render_result_filters(
    snapshot: dict[str, Any],
    all_predictions: pd.DataFrame,
) -> dict[str, Any]:
    split_options = ["all", *snapshot["predictions"].keys()]
    feature_options = snapshot["feature_columns"] if snapshot["feature_columns"] else ["(none)"]
    segment_default = snapshot.get("default_segment_column")
    segment_candidates = [
        column for column in snapshot["categorical_features"] if column in all_predictions.columns
    ]
    if (
        segment_default
        and segment_default in all_predictions.columns
        and segment_default not in segment_candidates
    ):
        segment_candidates.insert(0, segment_default)
    segment_options = ["(none)", *segment_candidates]
    date_candidates = (
        [
            column
            for column in all_predictions.columns
            if pd.api.types.is_datetime64_any_dtype(all_predictions[column])
        ]
        if not all_predictions.empty
        else []
    )

    st.markdown("### Interactive Filters")
    with st.expander("Adjust the live view", expanded=True):
        top_row = st.columns([1.1, 1.2, 1.2, 1.2])
        selected_split = top_row[0].selectbox("Split", options=split_options)
        view_mode = top_row[1].radio(
            "View depth",
            options=["Summary", "Technical"],
            horizontal=True,
        )
        display_surfaces = top_row[2].multiselect(
            "Display",
            options=["Charts", "Tables"],
            default=["Charts", "Tables"],
        )
        top_n = top_row[3].slider("Top-N features", min_value=5, max_value=25, value=10)

        second_row = st.columns([1.15, 1.15, 1.15, 1.15])
        selected_feature = second_row[0].selectbox("Feature lens", options=feature_options)
        segment_index = 0
        if segment_default and segment_default in segment_options:
            segment_index = segment_options.index(segment_default)
        selected_segment_column = second_row[1].selectbox(
            "Segment column",
            options=segment_options,
            index=segment_index,
        )
        selected_date_column = None
        date_range = None
        if date_candidates:
            default_date_column = (
                snapshot.get("date_column")
                if snapshot.get("date_column") in date_candidates
                else date_candidates[0]
            )
            selected_date_column = second_row[2].selectbox(
                "Date column",
                options=date_candidates,
                index=date_candidates.index(default_date_column),
            )
            date_series = all_predictions[selected_date_column].dropna()
            if not date_series.empty:
                date_range = second_row[3].date_input(
                    "Date range",
                    value=(date_series.min().date(), date_series.max().date()),
                )
        else:
            second_row[2].markdown(
                "<div class='filter-note'>No date field available</div>",
                unsafe_allow_html=True,
            )

        threshold = snapshot.get("threshold", 0.5)
        if snapshot["target_mode"] == TargetMode.BINARY.value:
            threshold = st.slider(
                "Decision threshold",
                min_value=0.05,
                max_value=0.95,
                value=float(snapshot.get("threshold", 0.5)),
                step=0.01,
            )

        selected_segments: list[str] = []
        if (
            selected_segment_column != "(none)"
            and selected_segment_column in all_predictions.columns
        ):
            segment_values = (
                all_predictions[selected_segment_column]
                .fillna("Missing")
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            selected_segments = st.multiselect(
                "Segment values",
                options=segment_values,
                default=segment_values[: min(6, len(segment_values))],
            )

    return {
        "selected_split": selected_split,
        "view_mode": view_mode,
        "display_surfaces": display_surfaces,
        "top_n": top_n,
        "selected_feature": selected_feature,
        "selected_segment_column": selected_segment_column,
        "selected_segments": selected_segments,
        "selected_date_column": selected_date_column,
        "date_range": date_range,
        "threshold": threshold,
    }


def apply_prediction_filters(
    snapshot: dict[str, Any],
    all_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> pd.DataFrame:
    selected_split = filter_state["selected_split"]
    if all_predictions.empty:
        return all_predictions
    filtered_predictions = (
        all_predictions
        if selected_split == "all"
        else snapshot["predictions"][selected_split].copy()
    )

    selected_segment_column = filter_state["selected_segment_column"]
    selected_segments = filter_state["selected_segments"]
    if (
        selected_segment_column != "(none)"
        and selected_segment_column in filtered_predictions.columns
        and selected_segments
    ):
        filtered_predictions = filtered_predictions.loc[
            filtered_predictions[selected_segment_column]
            .fillna("Missing")
            .astype(str)
            .isin(selected_segments)
        ]

    selected_date_column = filter_state["selected_date_column"]
    date_range = filter_state["date_range"]
    if selected_date_column and selected_date_column in filtered_predictions.columns:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_predictions = filtered_predictions.loc[
                filtered_predictions[selected_date_column].between(start_date, end_date)
            ]

    return filtered_predictions


def render_large_data_paged_prediction_browser(snapshot: dict[str, Any]) -> None:
    """Renders full row-level result access through file-backed paging."""

    refs = snapshot.get("prediction_table_refs") or {}
    streamlit_snapshot = snapshot.get("streamlit_snapshot") or {}
    if not streamlit_snapshot.get("large_data_mode") or not refs:
        return
    selected_ref_payload = refs.get("full_data_predictions") or refs.get("sample_predictions")
    if not selected_ref_payload:
        return
    table_ref = ResultTableRef.from_dict(selected_ref_payload)
    if not table_ref.path:
        return

    st.markdown("### Full Row Browser")
    st.caption(
        "Large-data rows are queried from the exported file-backed predictions table. "
        "The UI loads one page at a time instead of loading the full table into Streamlit."
    )
    with st.expander("Browse full prediction rows", expanded=False):
        page_size_default = int(streamlit_snapshot.get("result_page_rows") or 1000)
        controls = st.columns([1.0, 1.0, 1.1, 1.1])
        page_size = int(
            controls[0].number_input(
                "Rows per page",
                min_value=100,
                max_value=10000,
                value=max(100, min(page_size_default, 10000)),
                step=100,
                key=f"{snapshot['run_id']}_paged_result_rows",
            )
        )
        row_count = table_ref.row_count
        if row_count is None:
            try:
                row_count = count_table_rows(
                    table_ref,
                    duckdb_threads=int(streamlit_snapshot.get("duckdb_threads") or 0),
                    duckdb_memory_limit_gb=streamlit_snapshot.get("duckdb_memory_limit_gb"),
                )
            except Exception:
                row_count = None
        max_page = max(1, ((int(row_count) - 1) // page_size) + 1) if row_count else 1
        page_number = int(
            controls[1].number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=1,
                step=1,
                key=f"{snapshot['run_id']}_paged_result_page",
            )
        )
        sort_options = ["(none)", *table_ref.columns]
        sort_by = controls[2].selectbox(
            "Sort column",
            options=sort_options,
            key=f"{snapshot['run_id']}_paged_result_sort",
        )
        descending = controls[3].checkbox(
            "Descending",
            value=False,
            key=f"{snapshot['run_id']}_paged_result_desc",
        )

        filter_specs: list[dict[str, Any]] = []
        filter_columns = [
            column
            for column in [
                table_ref.split_column,
                table_ref.target_column,
                table_ref.score_column,
                *table_ref.segment_columns[:3],
            ]
            if column and column in table_ref.columns
        ]
        if filter_columns:
            filter_column = st.selectbox(
                "Optional equality filter",
                options=["(none)", *dict.fromkeys(filter_columns)],
                key=f"{snapshot['run_id']}_paged_result_filter_column",
            )
            if filter_column != "(none)":
                try:
                    values = distinct_column_values(
                        table_ref,
                        filter_column,
                        limit=250,
                        duckdb_threads=int(streamlit_snapshot.get("duckdb_threads") or 0),
                        duckdb_memory_limit_gb=streamlit_snapshot.get("duckdb_memory_limit_gb"),
                    )
                except Exception:
                    values = []
                if values:
                    selected_value = st.selectbox(
                        "Filter value",
                        options=values,
                        key=f"{snapshot['run_id']}_paged_result_filter_value",
                    )
                    filter_specs.append(
                        {"column": filter_column, "op": "eq", "value": selected_value}
                    )

        preferred_columns = [
            column
            for column in [
                table_ref.split_column,
                *table_ref.date_columns,
                *table_ref.segment_columns[:4],
                table_ref.target_column,
                table_ref.score_column,
                "predicted_probability",
                "predicted_value",
                "predicted_class",
                "scorecard_score",
                "scorecard_points",
            ]
            if column and column in table_ref.columns
        ]
        selected_columns = st.multiselect(
            "Displayed columns",
            options=table_ref.columns,
            default=list(dict.fromkeys(preferred_columns)) or table_ref.columns[:12],
            key=f"{snapshot['run_id']}_paged_result_columns",
        )
        try:
            page_frame = query_table_page(
                table_ref,
                columns=selected_columns or None,
                filters=filter_specs,
                sort_by=None if sort_by == "(none)" else sort_by,
                descending=descending,
                page=page_number,
                page_size=page_size,
                duckdb_threads=int(streamlit_snapshot.get("duckdb_threads") or 0),
                duckdb_memory_limit_gb=streamlit_snapshot.get("duckdb_memory_limit_gb"),
            )
            if filter_specs:
                filtered_count = count_table_rows(
                    table_ref,
                    filters=filter_specs,
                    duckdb_threads=int(streamlit_snapshot.get("duckdb_threads") or 0),
                    duckdb_memory_limit_gb=streamlit_snapshot.get("duckdb_memory_limit_gb"),
                )
                st.caption(f"Filtered rows: {filtered_count:,}")
            elif row_count is not None:
                st.caption(f"Rows available: {int(row_count):,}")
            st.dataframe(
                prepare_table_for_display(page_frame),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"Could not read the paged prediction table: {exc}")


def render_chart_review_context(descriptor: Any) -> None:
    """Adds compact interpretation context above live report charts."""

    guidance = report_chart_guidance(descriptor.key)
    if not guidance:
        return
    label, badge_class = report_asset_badge(
        descriptor.key,
        featured=bool(getattr(descriptor, "featured", False)),
    )
    st.markdown(
        f"""
        <div class="chart-review-context chart-review-context--{escape(badge_class)}">
          <span>{escape(label)}</span>
          <p>{escape(guidance)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_panel(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
    asset_catalog: dict[str, Any],
) -> None:
    kpis = summarize_run_kpis(
        metrics=snapshot["metrics"],
        input_rows=snapshot["input_shape"].get("rows"),
        feature_count=int(snapshot["feature_summary"].get("feature_count", 0)),
        labels_available=snapshot["labels_available"],
        execution_mode=snapshot["execution_mode"],
        model_type=snapshot["model_type"],
        target_mode=snapshot["target_mode"],
        warning_count=len(snapshot["warnings"]),
    )
    kpis.append({"label": "Filtered Rows", "value": f"{len(filtered_predictions):,}"})
    render_metric_strip(kpis)

    if snapshot["warnings"]:
        st.warning("\n".join(snapshot["warnings"]))
        if not snapshot["labels_available"]:
            st.info(
                "This view is operating in score-only documentation mode. "
                "Stability, segmentation, and score distribution outputs "
                "remain valid, while label-dependent diagnostics were skipped."
            )

    render_suitability_checks_panel(snapshot)

    if (
        snapshot["labels_available"]
        and snapshot["target_mode"] == TargetMode.BINARY.value
        and not filtered_predictions.empty
    ):
        render_dynamic_threshold_strip(snapshot, filtered_predictions, filter_state["threshold"])

    overview_figures = build_overview_figures(snapshot, filtered_predictions, filter_state)
    for figure_key in pick_overview_figure_keys(snapshot):
        if figure_key in snapshot["visualizations"]:
            overview_figures.append(
                (
                    friendly_asset_title(figure_key, kind="figure"),
                    snapshot["visualizations"][figure_key],
                )
            )

    unique_overview: list[tuple[str, go.Figure]] = []
    seen_titles: set[str] = set()
    for title, figure in overview_figures:
        if title in seen_titles:
            continue
        seen_titles.add(title)
        unique_overview.append((title, figure))

    if unique_overview:
        columns = st.columns(2)
        for index, (title, figure) in enumerate(unique_overview[:6]):
            with columns[index % 2]:
                st.markdown(f"#### {title}")
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(snapshot["run_id"], "overview", index, title),
                )

    st.markdown("### Feature Lens")
    selected_feature = filter_state["selected_feature"]
    if selected_feature == "(none)" or selected_feature not in filtered_predictions.columns:
        st.info("No feature is currently selected for drilldown.")
    else:
        render_feature_drilldown(snapshot, filtered_predictions, selected_feature)

    if "Tables" in filter_state["display_surfaces"]:
        featured_tables = choose_descriptors_for_view(
            asset_catalog["model_performance"]["tables"],
            filter_state["view_mode"],
        )[:2]
        if featured_tables:
            st.markdown("### Key Tables")
            for descriptor in featured_tables:
                table = filter_table_for_display(
                    snapshot["diagnostics_tables"][descriptor.key],
                    filter_state=filter_state,
                )
                if table.empty:
                    continue
                with st.expander(descriptor.title, expanded=False):
                    st.dataframe(
                        prepare_table_for_display(table.head(20)),
                        width="stretch",
                        hide_index=True,
                    )


def build_overview_figures(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> list[tuple[str, go.Figure]]:
    figures: list[tuple[str, go.Figure]] = []
    score_distribution = build_score_distribution_figure(
        snapshot,
        filtered_predictions,
        threshold=filter_state["threshold"],
    )
    if score_distribution is not None:
        figures.append(("Score Distribution", score_distribution))

    segment_chart = build_segment_snapshot_figure(snapshot, filtered_predictions, filter_state)
    if segment_chart is not None:
        figures.append(("Segment Snapshot", segment_chart))

    time_chart = build_time_trend_figure(snapshot, filtered_predictions, filter_state)
    if time_chart is not None:
        figures.append(("Trend Monitor", time_chart))
    return figures


def build_score_distribution_figure(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    *,
    threshold: float,
) -> go.Figure | None:
    score_column = resolve_score_column_for_display(snapshot, filtered_predictions)
    if score_column is None or filtered_predictions.empty:
        return None

    histogram_frame = sample_frame(filtered_predictions, 25000).copy()
    color_column = None
    if snapshot["labels_available"] and snapshot["target_column"] in histogram_frame.columns:
        color_column = snapshot["target_column"]
        histogram_frame[color_column] = histogram_frame[color_column].astype(str)
    elif "split" in histogram_frame.columns:
        color_column = "split"

    figure = px.histogram(
        histogram_frame,
        x=score_column,
        color=color_column,
        nbins=40,
        title="Score Distribution",
        labels={score_column: "Predicted Score"},
    )
    if snapshot["target_mode"] == TargetMode.BINARY.value:
        figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#C28A2C",
            annotation_text=f"Threshold {threshold:.2f}",
            annotation_position="top right",
        )
    return apply_fintech_figure_theme(figure, title="Score Distribution")


def build_segment_snapshot_figure(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> go.Figure | None:
    segment_column = filter_state["selected_segment_column"]
    score_column = resolve_score_column_for_display(snapshot, filtered_predictions)
    if (
        segment_column == "(none)"
        or segment_column not in filtered_predictions.columns
        or score_column is None
    ):
        return None

    aggregations: dict[str, tuple[str, str]] = {
        "observation_count": (score_column, "size"),
        "average_score": (score_column, "mean"),
    }
    if snapshot["labels_available"] and snapshot["target_column"] in filtered_predictions.columns:
        aggregations["average_actual"] = (snapshot["target_column"], "mean")

    segment_table = (
        filtered_predictions.assign(
            _segment=filtered_predictions[segment_column].fillna("Missing").astype(str)
        )
        .groupby("_segment", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("observation_count", ascending=False)
        .head(filter_state["top_n"])
    )
    if segment_table.empty:
        return None

    value_columns = ["average_score"]
    if "average_actual" in segment_table.columns:
        value_columns = ["average_actual", "average_score"]
    figure = px.bar(
        segment_table,
        x="_segment",
        y=value_columns,
        barmode="group",
        title=f"Observed vs Predicted by {segment_column}",
        labels={"_segment": "Segment", "value": "Rate"},
    )
    return apply_fintech_figure_theme(figure, title=f"Observed vs Predicted by {segment_column}")


def build_time_trend_figure(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> go.Figure | None:
    date_column = filter_state["selected_date_column"]
    score_column = resolve_score_column_for_display(snapshot, filtered_predictions)
    if (
        not date_column
        or date_column not in filtered_predictions.columns
        or score_column is None
        or filtered_predictions.empty
    ):
        return None

    aggregations: dict[str, tuple[str, str]] = {
        "average_score": (score_column, "mean"),
    }
    if snapshot["labels_available"] and snapshot["target_column"] in filtered_predictions.columns:
        aggregations["average_actual"] = (snapshot["target_column"], "mean")

    trend_table = (
        filtered_predictions.groupby(date_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(date_column)
    )
    if trend_table.empty:
        return None

    y_columns = ["average_score"]
    if "average_actual" in trend_table.columns:
        y_columns = ["average_actual", "average_score"]
    figure = px.line(
        trend_table,
        x=date_column,
        y=y_columns,
        markers=True,
        title="Observed vs Predicted Over Time",
    )
    return apply_fintech_figure_theme(figure, title="Observed vs Predicted Over Time")


def render_dynamic_threshold_strip(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    threshold: float,
) -> None:
    score_column = resolve_score_column_for_display(snapshot, filtered_predictions)
    if score_column is None:
        return
    threshold_metrics = compute_binary_threshold_metrics(
        filtered_predictions=filtered_predictions,
        score_column=score_column,
        target_column=snapshot["target_column"],
        threshold=threshold,
    )
    if threshold_metrics is None:
        return

    cards = [
        {"label": "Threshold", "value": f"{threshold:.2f}"},
        {"label": "Accuracy", "value": format_metric_value(threshold_metrics["accuracy"])},
        {"label": "Precision", "value": format_metric_value(threshold_metrics["precision"])},
        {"label": "Recall", "value": format_metric_value(threshold_metrics["recall"])},
        {"label": "F1 Score", "value": format_metric_value(threshold_metrics["f1_score"])},
        {
            "label": "Predicted Positive Rate",
            "value": format_metric_value(threshold_metrics["positive_rate"]),
        },
    ]
    st.markdown("### Decision Threshold Snapshot")
    render_metric_strip(cards, compact=True)


def build_suitability_display_table(table: pd.DataFrame) -> pd.DataFrame:
    """Returns a reviewer-friendly suitability table with failures first."""

    if table.empty:
        return table
    display_table = table.copy()
    if "check_label" not in display_table.columns and "check_name" in display_table.columns:
        display_table["check_label"] = (
            display_table["check_name"]
            .astype(str)
            .str.replace(
                "_",
                " ",
            )
            .str.title()
        )
    if "status_label" not in display_table.columns and "status" in display_table.columns:
        display_table["status_label"] = display_table["status"].map(
            {"fail": "Fail", "warn": "Watch", "pass": "Pass"}
        )
    status_rank = display_table.get("status", pd.Series(index=display_table.index, dtype=str)).map(
        {"fail": 0, "warn": 1, "pass": 2}
    )
    display_table = display_table.assign(_status_rank=status_rank.fillna(3))
    sort_columns = ["_status_rank"]
    for column_name in ["check_label", "check_name", "subject"]:
        if column_name in display_table.columns:
            sort_columns.append(column_name)
    display_table = display_table.sort_values(sort_columns, kind="stable").drop(
        columns=["_status_rank"]
    )
    columns = [
        column_name for column_name in SUITABILITY_DISPLAY_COLUMNS if column_name in display_table
    ]
    if not columns:
        return display_table
    return display_table[columns]


def render_suitability_checks_panel(snapshot: dict[str, Any]) -> None:
    """Highlights suitability failures in plain English in the Results overview."""

    table = snapshot["diagnostics_tables"].get("assumption_checks")
    if table is None or table.empty:
        return

    fail_count = int((table["status"] == "fail").sum()) if "status" in table.columns else 0
    warn_count = int((table["status"] == "warn").sum()) if "status" in table.columns else 0
    pass_count = int((table["status"] == "pass").sum()) if "status" in table.columns else 0
    st.markdown("### Suitability Checks")
    render_metric_strip(
        [
            {"label": "Failed Checks", "value": f"{fail_count:,}"},
            {"label": "Watch Checks", "value": f"{warn_count:,}"},
            {"label": "Passed Checks", "value": f"{pass_count:,}"},
        ],
        compact=True,
    )

    display_table = build_suitability_display_table(table)
    if fail_count:
        st.warning(
            "One or more model-suitability checks failed. Review the failed rows below "
            "before relying on the model evidence."
        )
        failed_table = display_table.loc[
            display_table.get("status_label", pd.Series(index=display_table.index)).eq("Fail")
        ]
        if failed_table.empty and "status" in table.columns:
            failed_table = build_suitability_display_table(table.loc[table["status"] == "fail"])
        st.dataframe(
            prepare_table_for_display(failed_table),
            width="stretch",
            hide_index=True,
        )
    elif warn_count:
        st.warning("Suitability checks passed without failures, but watch conditions need review.")
    else:
        st.success("Suitability checks passed against the configured thresholds.")

    with st.expander("All suitability check details", expanded=bool(fail_count)):
        st.dataframe(
            prepare_table_for_display(display_table),
            width="stretch",
            hide_index=True,
        )


def compute_binary_threshold_metrics(
    *,
    filtered_predictions: pd.DataFrame,
    score_column: str,
    target_column: str,
    threshold: float,
) -> dict[str, float] | None:
    required_columns = {score_column, target_column}
    if not required_columns.issubset(filtered_predictions.columns) or filtered_predictions.empty:
        return None

    scored = filtered_predictions[[score_column, target_column]].dropna()
    if scored.empty:
        return None

    predicted = (scored[score_column] >= threshold).astype(int)
    actual = scored[target_column].astype(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(scored) if len(scored) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "positive_rate": float(predicted.mean()) if len(predicted) else 0.0,
    }


def render_section_panel(
    *,
    snapshot: dict[str, Any],
    section_id: str,
    section_payload: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> None:
    st.markdown(
        f"""
        <div class="section-subheader">
          <span class="section-kicker">{section_payload["title"]}</span>
          <p>{section_payload["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_surfaces = set(filter_state["display_surfaces"])
    view_mode = filter_state["view_mode"]

    if section_id == "scorecard_workbench":
        render_scorecard_workbench_section(
            snapshot=snapshot,
            filtered_predictions=filtered_predictions,
            filter_state=filter_state,
        )
        return

    figure_descriptors = choose_descriptors_for_view(section_payload["figures"], view_mode)
    table_descriptors = choose_descriptors_for_view(section_payload["tables"], view_mode)

    if "Charts" in display_surfaces and figure_descriptors:
        columns = st.columns(2)
        for index, descriptor in enumerate(figure_descriptors):
            figure = snapshot["visualizations"].get(descriptor.key)
            if figure is None:
                continue
            with columns[index % 2]:
                st.markdown(f"#### {descriptor.title}")
                render_chart_review_context(descriptor)
                if descriptor.description:
                    st.caption(descriptor.description)
                render_output_explainer(descriptor.key)
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "section",
                        section_id,
                        descriptor.key,
                    ),
                )

    if "Tables" in display_surfaces and table_descriptors:
        st.markdown("### Reference Tables")
        for descriptor in table_descriptors:
            table = snapshot["diagnostics_tables"].get(descriptor.key)
            if table is None:
                continue
            table = filter_table_for_display(table, filter_state=filter_state)
            if table.empty:
                continue
            with st.expander(descriptor.title, expanded=view_mode == "Summary"):
                if descriptor.description:
                    st.caption(descriptor.description)
                render_output_explainer(descriptor.key)
                preview = table if view_mode == "Technical" else table.head(25)
                st.dataframe(
                    prepare_table_for_display(preview),
                    width="stretch",
                    hide_index=True,
                )
                if len(table) > len(preview):
                    st.caption(
                        f"Showing {len(preview):,} of {len(table):,} rows. "
                        "Use the export bundle for full detail."
                    )


def choose_descriptors_for_view(descriptors: list[Any], view_mode: str) -> list[Any]:
    if view_mode == "Technical":
        return descriptors
    featured = [descriptor for descriptor in descriptors if descriptor.featured]
    return featured or descriptors[:2]


def filter_table_for_display(
    table: pd.DataFrame,
    *,
    filter_state: dict[str, Any],
) -> pd.DataFrame:
    filtered = table.copy()
    if filter_state["selected_split"] != "all" and "split" in filtered.columns:
        filtered = filtered.loc[filtered["split"] == filter_state["selected_split"]]
    segment_column = filter_state["selected_segment_column"]
    selected_segments = filter_state["selected_segments"]
    if segment_column != "(none)" and segment_column in filtered.columns and selected_segments:
        filtered = filtered.loc[
            filtered[segment_column].fillna("Missing").astype(str).isin(selected_segments)
        ]
    return filtered


def render_scorecard_workbench_section(
    *,
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    filter_state: dict[str, Any],
) -> None:
    diagnostics_tables = snapshot["diagnostics_tables"]
    feature_summary = diagnostics_tables.get("scorecard_feature_summary", pd.DataFrame())
    woe_table = diagnostics_tables.get("scorecard_woe_table", pd.DataFrame())
    points_table = diagnostics_tables.get("scorecard_points_table", pd.DataFrame())
    scaling_summary = diagnostics_tables.get("scorecard_scaling_summary", pd.DataFrame())
    if feature_summary.empty or woe_table.empty or points_table.empty:
        st.info("Scorecard workbench assets are not available for this run.")
        return

    scaling_map = {
        str(row["metric"]): row["value"]
        for _, row in scaling_summary.iterrows()
        if "metric" in scaling_summary.columns and "value" in scaling_summary.columns
    }
    overview_cards = [
        {"label": "Profiled Features", "value": f"{len(feature_summary):,}"},
        {
            "label": "Average Bins",
            "value": format_metric_value(feature_summary["bin_count"].mean()),
        },
        {
            "label": "Manual Overrides",
            "value": f"{int(feature_summary['manual_override_applied'].sum()):,}",
        },
        {"label": "Base Score", "value": format_metric_value(scaling_map.get("base_score"))},
        {
            "label": "PDO",
            "value": format_metric_value(scaling_map.get("points_to_double_odds")),
        },
    ]
    render_metric_strip(overview_cards, compact=True)

    feature_options = feature_summary["feature_name"].astype(str).tolist()
    default_feature = (
        filter_state["selected_feature"]
        if filter_state["selected_feature"] in feature_options
        else feature_options[0]
    )
    selected_feature = st.selectbox(
        "Scorecard feature",
        options=feature_options,
        index=feature_options.index(default_feature),
        key=f"{snapshot['run_id']}_scorecard_workbench_feature",
    )

    selected_summary = (
        feature_summary.loc[feature_summary["feature_name"] == selected_feature]
        .head(1)
        .reset_index(drop=True)
    )
    if not selected_summary.empty:
        summary_row = selected_summary.iloc[0]
        feature_cards = [
            {
                "label": "Information Value",
                "value": format_metric_value(summary_row["information_value"]),
            },
            {"label": "Points Span", "value": format_metric_value(summary_row["points_span"])},
            {
                "label": "Largest Bin Share",
                "value": format_metric_value(summary_row["largest_bin_share"]),
            },
            {
                "label": "Bad-Rate Trend",
                "value": str(summary_row["bad_rate_trend"]).replace("_", " ").title(),
            },
        ]
        render_metric_strip(feature_cards, compact=True)
    render_glossary_badges(["WoE", "IV", "Scorecard", "Reason Code"], caption="Scorecard terms")
    render_output_explainer("scorecard_woe_table")
    render_binning_theater(
        selected_feature=selected_feature,
        feature_summary=feature_summary,
        woe_table=woe_table,
        points_table=points_table,
    )

    display_surfaces = set(filter_state["display_surfaces"])
    if "Charts" in display_surfaces:
        overview_figures: list[tuple[str, go.Figure]] = []
        if "scorecard_feature_iv" in snapshot["visualizations"]:
            overview_figures.append(
                ("Feature Information Value", snapshot["visualizations"]["scorecard_feature_iv"])
            )
        score_distribution = build_scorecard_distribution_figure(
            snapshot=snapshot,
            filtered_predictions=filtered_predictions,
        )
        if score_distribution is not None:
            overview_figures.append(("Points Distribution", score_distribution))
        reason_code_chart = build_scorecard_reason_code_chart(filtered_predictions)
        if reason_code_chart is not None:
            overview_figures.append(("Reason Code Frequency", reason_code_chart))

        if overview_figures:
            columns = st.columns(min(2, len(overview_figures)))
            for index, (title, figure) in enumerate(overview_figures):
                with columns[index % len(columns)]:
                    st.markdown(f"#### {title}")
                    render_plotly_figure(
                        figure,
                        key=build_plotly_key(
                            snapshot["run_id"],
                            "scorecard_workbench",
                            "overview",
                            title,
                        ),
                    )

        selected_woe = (
            woe_table.loc[woe_table["feature_name"] == selected_feature]
            .copy()
            .sort_values("bucket_rank")
        )
        selected_points = (
            points_table.loc[points_table["feature_name"] == selected_feature]
            .copy()
            .sort_values("bucket_rank")
        )
        feature_figures = build_scorecard_feature_figures(
            feature_name=selected_feature,
            woe_table=selected_woe,
            points_table=selected_points,
        )
        feature_columns = st.columns(3)
        for index, (title, figure) in enumerate(feature_figures):
            with feature_columns[index % len(feature_columns)]:
                st.markdown(f"#### {title}")
                render_plotly_figure(
                    figure,
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "scorecard_workbench",
                        selected_feature,
                        title,
                    ),
                )

    if "Tables" in display_surfaces:
        st.markdown("### Reference Tables")
        selected_woe = (
            woe_table.loc[woe_table["feature_name"] == selected_feature]
            .copy()
            .sort_values("bucket_rank")
        )
        selected_points = (
            points_table.loc[points_table["feature_name"] == selected_feature]
            .copy()
            .sort_values("bucket_rank")
        )
        reason_code_table = build_scorecard_reason_code_table(filtered_predictions)
        table_payloads = [
            ("Selected Feature Summary", selected_summary),
            ("WoE Detail", selected_woe),
            ("Points Detail", selected_points),
            ("Scaling Summary", scaling_summary),
        ]
        if not reason_code_table.empty:
            table_payloads.append(("Reason Code Frequency", reason_code_table.head(25)))
        explainer_keys = {
            "Selected Feature Summary": "scorecard_feature_summary",
            "WoE Detail": "scorecard_woe_table",
            "Points Detail": "scorecard_points_table",
        }

        for title, table in table_payloads:
            if table.empty:
                continue
            with st.expander(title, expanded=title == "Selected Feature Summary"):
                render_output_explainer(explainer_keys.get(title, ""))
                st.dataframe(
                    prepare_table_for_display(table),
                    width="stretch",
                    hide_index=True,
                )


def build_scorecard_distribution_figure(
    *,
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
) -> go.Figure | None:
    if "scorecard_points" not in filtered_predictions.columns or filtered_predictions.empty:
        return None
    histogram_frame = sample_frame(filtered_predictions, 25000).copy()
    color_column = None
    if snapshot["labels_available"] and snapshot["target_column"] in histogram_frame.columns:
        color_column = snapshot["target_column"]
        histogram_frame[color_column] = histogram_frame[color_column].astype(str)
    elif "split" in histogram_frame.columns:
        color_column = "split"
    figure = px.histogram(
        histogram_frame,
        x="scorecard_points",
        color=color_column,
        nbins=40,
        title="Scorecard Points Distribution",
        labels={"scorecard_points": "Scorecard Points"},
    )
    return apply_fintech_figure_theme(figure, title="Scorecard Points Distribution")


def build_scorecard_reason_code_table(filtered_predictions: pd.DataFrame) -> pd.DataFrame:
    reason_columns = sorted(
        column for column in filtered_predictions.columns if column.startswith("reason_code_")
    )
    if not reason_columns or filtered_predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    denominator = max(len(filtered_predictions), 1)
    for reason_column in reason_columns:
        rank_text = reason_column.rsplit("_", 1)[-1]
        rank_value = int(rank_text) if rank_text.isdigit() else reason_column
        counts = (
            filtered_predictions[reason_column]
            .replace("", pd.NA)
            .dropna()
            .astype(str)
            .value_counts()
        )
        for feature_name, count in counts.items():
            rows.append(
                {
                    "reason_code_rank": rank_value,
                    "feature_name": feature_name,
                    "count": int(count),
                    "share": float(count / denominator),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["count", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_scorecard_reason_code_chart(filtered_predictions: pd.DataFrame) -> go.Figure | None:
    reason_code_table = build_scorecard_reason_code_table(filtered_predictions)
    if reason_code_table.empty:
        return None
    chart_frame = reason_code_table.head(12)
    figure = px.bar(
        chart_frame,
        x="feature_name",
        y="count",
        color="reason_code_rank",
        barmode="group",
        title="Reason Code Frequency",
        labels={
            "feature_name": "Feature",
            "count": "Count",
            "reason_code_rank": "Reason Code Slot",
        },
    )
    return apply_fintech_figure_theme(figure, title="Reason Code Frequency")


def build_scorecard_feature_figures(
    *,
    feature_name: str,
    woe_table: pd.DataFrame,
    points_table: pd.DataFrame,
) -> list[tuple[str, go.Figure]]:
    if woe_table.empty or points_table.empty:
        return []
    return [
        (
            "Bad Rate by Bucket",
            apply_fintech_figure_theme(
                px.bar(
                    woe_table,
                    x="bucket_label",
                    y="bad_rate",
                    title=f"{feature_name}: bad rate by bucket",
                    labels={"bucket_label": "Bucket", "bad_rate": "Bad Rate"},
                ),
                title=f"{feature_name}: bad rate by bucket",
            ),
        ),
        (
            "WoE by Bucket",
            apply_fintech_figure_theme(
                px.line(
                    woe_table,
                    x="bucket_label",
                    y="woe",
                    markers=True,
                    title=f"{feature_name}: WoE by bucket",
                    labels={"bucket_label": "Bucket", "woe": "WoE"},
                ),
                title=f"{feature_name}: WoE by bucket",
            ),
        ),
        (
            "Points by Bucket",
            apply_fintech_figure_theme(
                px.bar(
                    points_table,
                    x="bucket_label",
                    y="partial_score_points",
                    title=f"{feature_name}: points by bucket",
                    labels={"bucket_label": "Bucket", "partial_score_points": "Partial Score"},
                ),
                title=f"{feature_name}: points by bucket",
            ),
        ),
    ]


def render_governance_panel(snapshot: dict[str, Any], filtered_predictions: pd.DataFrame) -> None:
    left_column, right_column = st.columns([1.1, 0.9], gap="large")
    with left_column:
        st.markdown("### Narrative Report")
        st.text_area(
            "Run report",
            value=_report_text(snapshot),
            height=320,
            label_visibility="collapsed",
        )
        st.markdown("### Predictions Preview")
        st.dataframe(
            prepare_table_for_display(filtered_predictions.head(250)),
            width="stretch",
            hide_index=True,
        )

    with right_column:
        if snapshot["warnings"]:
            st.markdown("### Warnings")
            for warning in snapshot["warnings"]:
                st.warning(warning)
        render_reviewer_workspace(snapshot)
        render_artifact_locations(snapshot)
        render_download_button(
            "Download Run Config",
            snapshot["config"],
            "run_config.json",
            "application/json",
        )
        render_download_button(
            "Download Markdown Report",
            _report_text(snapshot),
            "run_report.md",
            "text/markdown",
        )
        render_lazy_artifact_downloads(
            snapshot,
            artifact_names=[
                "interactive_report",
                "workbook",
                "predictions",
                "documentation_pack",
                "validation_pack",
                "committee_report_docx",
                "validation_report_docx",
                "committee_report_pdf",
                "validation_report_pdf",
                "reproducibility_manifest",
                "configuration_template",
            ],
            key_prefix="governance",
        )
        if snapshot["events"]:
            st.markdown("### Pipeline Events")
            for event in snapshot["events"]:
                st.caption(f"- {event}")


def render_feature_drilldown(
    snapshot: dict[str, Any],
    filtered_predictions: pd.DataFrame,
    selected_feature: str,
) -> None:
    score_column = resolve_score_column_for_display(snapshot, filtered_predictions)
    target_column = snapshot["target_column"]
    target_mode = snapshot["target_mode"]
    labels_available = (
        snapshot["labels_available"] and target_column in filtered_predictions.columns
    )
    if score_column is None:
        st.info("Predicted score outputs are not available for the current filtered selection.")
        return
    feature_series = filtered_predictions[selected_feature]

    if pd.api.types.is_numeric_dtype(feature_series):
        numeric_columns = [selected_feature, score_column]
        if labels_available:
            numeric_columns.append(target_column)
        sampled = sample_frame(filtered_predictions[numeric_columns], 20000)
        if target_mode == TargetMode.BINARY.value and labels_available:
            histogram = px.histogram(
                sampled,
                x=selected_feature,
                color=target_column,
                nbins=40,
                marginal="box",
                title=f"{selected_feature}: distribution by target",
            )
            render_plotly_figure(
                apply_fintech_figure_theme(
                    histogram,
                    title=f"{selected_feature}: distribution by target",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "distribution_by_target",
                ),
            )
            bucket_count = min(10, int(sampled[selected_feature].nunique()))
            if bucket_count >= 2:
                bucketed = sampled.copy()
                bucketed["feature_bucket"] = pd.qcut(
                    bucketed[selected_feature].rank(method="first"),
                    q=bucket_count,
                    duplicates="drop",
                )
                summary = (
                    bucketed.groupby("feature_bucket", dropna=False)
                    .agg(
                        observed_rate=(target_column, "mean"),
                        average_score=(score_column, "mean"),
                    )
                    .reset_index()
                )
                render_plotly_figure(
                    apply_fintech_figure_theme(
                        px.line(
                            summary,
                            x="feature_bucket",
                            y=["observed_rate", "average_score"],
                            title=f"{selected_feature}: observed vs predicted by quantile bucket",
                            markers=True,
                        ),
                        title=f"{selected_feature}: observed vs predicted by quantile bucket",
                    ),
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "feature_drilldown",
                        selected_feature,
                        "observed_vs_predicted_quantiles",
                    ),
                )
        elif labels_available:
            render_plotly_figure(
                apply_fintech_figure_theme(
                    px.scatter(
                        sampled,
                        x=selected_feature,
                        y=target_column,
                        title=f"{selected_feature}: actual relationship",
                        trendline="ols",
                        opacity=0.4,
                    ),
                    title=f"{selected_feature}: actual relationship",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "actual_relationship",
                ),
            )
            render_plotly_figure(
                apply_fintech_figure_theme(
                    px.scatter(
                        sampled,
                        x=selected_feature,
                        y=score_column,
                        title=f"{selected_feature}: predicted relationship",
                        trendline="ols",
                        opacity=0.4,
                    ),
                    title=f"{selected_feature}: predicted relationship",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "predicted_relationship",
                ),
            )
        else:
            render_plotly_figure(
                apply_fintech_figure_theme(
                    px.histogram(
                        sampled,
                        x=selected_feature,
                        nbins=40,
                        marginal="box",
                        title=f"{selected_feature}: distribution across scored observations",
                    ),
                    title=f"{selected_feature}: distribution across scored observations",
                ),
                key=build_plotly_key(
                    snapshot["run_id"],
                    "feature_drilldown",
                    selected_feature,
                    "distribution_scored_observations",
                ),
            )
            bucket_count = min(10, int(sampled[selected_feature].nunique()))
            if bucket_count >= 2:
                bucketed = sampled.copy()
                bucketed["feature_bucket"] = pd.qcut(
                    bucketed[selected_feature].rank(method="first"),
                    q=bucket_count,
                    duplicates="drop",
                )
                summary = (
                    bucketed.groupby("feature_bucket", dropna=False)
                    .agg(
                        average_score=(score_column, "mean"),
                        observation_count=(score_column, "size"),
                    )
                    .reset_index()
                )
                render_plotly_figure(
                    apply_fintech_figure_theme(
                        px.line(
                            summary,
                            x="feature_bucket",
                            y="average_score",
                            title=f"{selected_feature}: average score by quantile bucket",
                            markers=True,
                        ),
                        title=f"{selected_feature}: average score by quantile bucket",
                    ),
                    key=build_plotly_key(
                        snapshot["run_id"],
                        "feature_drilldown",
                        selected_feature,
                        "average_score_quantiles",
                    ),
                )
    else:
        aggregations: dict[str, tuple[str, str]] = {
            "observation_count": (score_column, "size"),
            "average_score": (score_column, "mean"),
        }
        if labels_available:
            aggregations["average_actual"] = (target_column, "mean")
        categorical_summary = (
            filtered_predictions.assign(_segment=feature_series.fillna("Missing").astype(str))
            .groupby("_segment", dropna=False)
            .agg(**aggregations)
            .reset_index()
            .sort_values("observation_count", ascending=False)
            .head(10)
        )
        y_columns = ["average_score"]
        if labels_available and "average_actual" in categorical_summary.columns:
            y_columns = ["average_actual", "average_score"]
        render_plotly_figure(
            apply_fintech_figure_theme(
                px.bar(
                    categorical_summary,
                    x="_segment",
                    y=y_columns,
                    barmode="group",
                    title=(
                        f"{selected_feature}: observed vs predicted by category"
                        if labels_available
                        else f"{selected_feature}: average score by category"
                    ),
                ),
                title=(
                    f"{selected_feature}: observed vs predicted by category"
                    if labels_available
                    else f"{selected_feature}: average score by category"
                ),
            ),
            key=build_plotly_key(
                snapshot["run_id"],
                "feature_drilldown",
                selected_feature,
                "category_summary",
            ),
        )
        st.dataframe(
            prepare_table_for_display(categorical_summary),
            width="stretch",
            hide_index=True,
        )


def resolve_score_column_for_display(
    snapshot: dict[str, Any],
    prediction_frame: pd.DataFrame,
) -> str | None:
    if prediction_frame.empty and len(prediction_frame.columns) == 0:
        return None

    configured_score_column = str(snapshot.get("score_column") or "")
    target_mode = str(snapshot.get("target_mode") or "")
    candidates: list[str] = []
    if configured_score_column:
        candidates.append(configured_score_column)

    if target_mode == TargetMode.BINARY.value:
        candidates.extend(
            [
                "predicted_probability_recommended",
                "predicted_probability",
                "prediction_score",
                "score",
            ]
        )
    else:
        candidates.extend(["predicted_value", "prediction_score", "score"])

    for candidate in dict.fromkeys(candidates):
        if _is_numeric_display_column(prediction_frame, candidate):
            return candidate

    for column_name in prediction_frame.columns:
        if not _is_numeric_display_column(prediction_frame, column_name):
            continue
        normalized = str(column_name).lower()
        if any(
            token in normalized
            for token in (
                "predicted_probability",
                "probability_recommended",
                "predicted_value",
                "score",
            )
        ):
            return str(column_name)

    return None


def _is_numeric_display_column(prediction_frame: pd.DataFrame, column_name: str) -> bool:
    if column_name not in prediction_frame.columns:
        return False
    return pd.api.types.is_numeric_dtype(prediction_frame[column_name])


def pick_overview_figure_keys(snapshot: dict[str, Any]) -> list[str]:
    if snapshot["target_mode"] == TargetMode.BINARY.value:
        preferred = [
            "feature_importance_overview",
            "split_metric_overview",
            "roc_curve",
            "calibration_curve",
            "quantile_backtest",
            "psi_profile",
        ]
    else:
        preferred = [
            "feature_importance_overview",
            "split_metric_overview",
            "actual_vs_predicted",
            "residuals_vs_predicted",
            "quantile_backtest",
            "psi_profile",
        ]
    return [key for key in preferred if key in snapshot["visualizations"]]


def render_lazy_artifact_downloads(
    snapshot: dict[str, Any],
    *,
    artifact_names: list[str],
    key_prefix: str,
) -> None:
    available = [
        artifact_name
        for artifact_name in artifact_names
        if snapshot["artifacts"].get(artifact_name)
        and Path(snapshot["artifacts"][artifact_name]).exists()
    ]
    if not available:
        return
    selected_artifact = st.selectbox(
        "Download artifact",
        options=available,
        format_func=lambda value: value.replace("_", " ").title(),
        key=f"{snapshot['run_id']}_{key_prefix}_artifact",
    )
    selected_path = snapshot["artifacts"][selected_artifact]
    st.download_button(
        f"Download {selected_artifact.replace('_', ' ').title()}",
        data=read_binary_artifact(selected_path),
        file_name=Path(selected_path).name,
        mime="application/octet-stream",
    )


def render_artifact_locations(snapshot: dict[str, Any]) -> None:
    st.markdown("### Artifact Explorer")
    artifact_table = build_artifact_explorer_frame(snapshot["artifacts"])
    available_count = int(artifact_table["status"].eq("Available").sum())
    st.caption(
        f"{available_count:,} artifact locations are available for this run. "
        "Use this table to find the run folder, model object, reports, manifests, "
        "and any Large Data Mode output folders."
    )
    area_options = ["All", *sorted(artifact_table["area"].dropna().unique().tolist())]
    selected_area = st.selectbox(
        "Artifact group",
        options=area_options,
        key=f"{snapshot['run_id']}_artifact_explorer_group",
    )
    display_table = artifact_table
    if selected_area != "All":
        display_table = artifact_table.loc[artifact_table["area"] == selected_area]
    st.dataframe(
        prepare_table_for_display(display_table),
        width="stretch",
        hide_index=True,
    )
    available_downloads = display_table.loc[
        display_table["status"].eq("Available") & display_table["path"].astype(bool)
    ]
    if available_downloads.empty:
        return
    selected_key = st.selectbox(
        "Download available artifact",
        options=available_downloads["key"].tolist(),
        format_func=lambda value: str(value).replace("_", " ").title(),
        key=f"{snapshot['run_id']}_artifact_explorer_download",
    )
    selected_path = str(
        available_downloads.loc[available_downloads["key"] == selected_key, "path"].iloc[0]
    )
    selected_path_obj = Path(selected_path)
    if selected_path_obj.is_file():
        st.download_button(
            f"Download {selected_key.replace('_', ' ').title()}",
            data=read_binary_artifact(selected_path),
            file_name=selected_path_obj.name,
            mime="application/octet-stream",
            key=f"{snapshot['run_id']}_{selected_key}_explorer_download_button",
        )
    else:
        st.caption("Directory artifacts are shown for navigation and are not downloaded directly.")


def render_reviewer_workspace(snapshot: dict[str, Any]) -> None:
    with st.expander("Reviewer / Approval Workspace", expanded=False):
        reviewer_name = st.text_input(
            "Reviewer name",
            value="",
            key=f"{snapshot['run_id']}_reviewer_name",
        )
        approval_status = st.selectbox(
            "Approval status",
            options=[
                "Not reviewed",
                "Approved",
                "Approved with exceptions",
                "Rejected",
                "Needs remediation",
            ],
            key=f"{snapshot['run_id']}_approval_status",
        )
        review_notes = st.text_area(
            "Review notes",
            value="",
            height=90,
            key=f"{snapshot['run_id']}_review_notes",
        )
        exception_notes = st.text_area(
            "Exception notes",
            value="",
            height=90,
            key=f"{snapshot['run_id']}_exception_notes",
        )
        reviewer_record = ReviewerRecord(
            reviewer_name=reviewer_name.strip(),
            approval_status=approval_status,
            review_notes=review_notes.strip(),
            exception_notes=exception_notes.strip(),
        )
        model_card = build_model_card_markdown(
            snapshot=snapshot,
            reviewer_record=reviewer_record,
        )
        st.download_button(
            "Download model card",
            data=model_card,
            file_name=f"{snapshot['run_id']}_model_card.md",
            mime="text/markdown",
            key=f"{snapshot['run_id']}_download_model_card",
        )
        if st.button(
            "Save review record to run folder",
            key=f"{snapshot['run_id']}_save_review_record",
        ):
            output_root = snapshot.get("artifacts", {}).get("output_root")
            if not output_root:
                st.error("No run folder is recorded for this snapshot.")
                return
            review_path = Path(output_root) / "review_workspace.json"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(json.dumps(asdict(reviewer_record), indent=2), encoding="utf-8")
            record_gui_audit_event(
                _snapshot_output_root(snapshot),
                "review_record_saved",
                run_id=str(snapshot.get("run_id", "")),
                artifact_root=output_root,
                metadata={
                    "reviewer_name": reviewer_record.reviewer_name,
                    "approval_status": reviewer_record.approval_status,
                    "review_notes_chars": len(reviewer_record.review_notes),
                    "exception_notes_chars": len(reviewer_record.exception_notes),
                    "review_path": str(review_path),
                },
            )
            st.success(f"Saved review record to {review_path}.")


def _report_text(snapshot: dict[str, Any]) -> str:
    report_path = snapshot.get("report_path", "")
    if report_path and Path(report_path).exists():
        return read_text_artifact(report_path)
    return ""
