"""Export profile decisions shared by artifact writers and report builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ExportProfile, PerformanceConfig

if TYPE_CHECKING:
    from .context import PipelineContext


def excel_workbook_enabled(context: PipelineContext) -> bool:
    """Returns whether the Excel analysis workbook should be generated."""

    return bool(
        context.config.diagnostics.export_excel_workbook
        and context.config.artifacts.export_profile != ExportProfile.FAST
    )


def regulatory_reports_enabled(context: PipelineContext) -> bool:
    """Returns whether DOCX/PDF regulatory reports should be generated."""

    return bool(
        context.config.regulatory_reporting.enabled
        and context.config.artifacts.export_profile != ExportProfile.FAST
    )


def code_snapshot_enabled(context: PipelineContext) -> bool:
    """Returns whether the rerun code snapshot should be generated."""

    return bool(
        context.config.artifacts.export_code_snapshot
        and context.config.artifacts.export_profile != ExportProfile.FAST
    )


def input_snapshot_enabled(context: PipelineContext) -> bool:
    """Returns whether the raw input snapshot should be exported."""

    return bool(
        context.config.artifacts.export_input_snapshot
        and context.config.artifacts.export_profile != ExportProfile.FAST
    )


def resolve_html_report_limits(context: PipelineContext) -> dict[str, Any]:
    """Resolves profile-aware HTML report preview and asset caps."""

    performance = getattr(context.config, "performance", PerformanceConfig())
    limits = {
        "table_preview_rows": performance.html_table_preview_rows,
        "max_figures_per_section": performance.html_max_figures_per_section,
        "max_tables_per_section": performance.html_max_tables_per_section,
        "max_points_per_figure": performance.html_max_points_per_figure,
        "max_figure_payload_mb": performance.html_max_figure_payload_mb,
        "max_total_figure_payload_mb": performance.html_max_total_figure_payload_mb,
    }
    if context.config.artifacts.export_profile == ExportProfile.FAST:
        limits["table_preview_rows"] = min(limits["table_preview_rows"], 8)
        limits["max_figures_per_section"] = min(limits["max_figures_per_section"], 2)
        limits["max_tables_per_section"] = min(limits["max_tables_per_section"], 4)
        limits["max_points_per_figure"] = min(limits["max_points_per_figure"], 2500)
        limits["max_figure_payload_mb"] = min(limits["max_figure_payload_mb"], 1.5)
        limits["max_total_figure_payload_mb"] = min(
            limits["max_total_figure_payload_mb"],
            20.0,
        )
    return limits
