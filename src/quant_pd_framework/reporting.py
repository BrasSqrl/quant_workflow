"""Helpers for generating regulator-ready DOCX and PDF report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from textwrap import wrap
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .context import PipelineContext


@dataclass(slots=True)
class ReportSection:
    """One structured report section used across markdown, DOCX, and PDF exports."""

    title: str
    summary: str
    lines: list[str]
    appendix: bool = False


@dataclass(slots=True)
class ReportArtifactPayload:
    """Normalized content for one regulator-ready report artifact set."""

    title: str
    subtitle: str
    markdown: str
    docx_bytes: bytes | None
    pdf_bytes: bytes | None


def build_regulatory_report_bundle(
    context: PipelineContext,
) -> dict[str, ReportArtifactPayload]:
    """Builds the committee-ready and validation-ready report payloads."""

    config = context.config.regulatory_reporting
    committee_title = f"Committee Report: {context.config.documentation.model_name}"
    validation_title = f"Validation Report: {context.config.documentation.model_name}"
    committee_subtitle = (
        f"Quant Studio | Template {config.committee_template_name} | Run {context.run_id}"
    )
    validation_subtitle = (
        f"Quant Studio | Template {config.validation_template_name} | Run {context.run_id}"
    )
    committee_sections = _build_committee_sections(context)
    validation_sections = _build_validation_sections(context)
    committee_cover_lines = _build_cover_lines(
        context,
        audience_label="Committee",
        template_name=config.committee_template_name,
        sections=committee_sections,
    )
    validation_cover_lines = _build_cover_lines(
        context,
        audience_label="Validation",
        template_name=config.validation_template_name,
        sections=validation_sections,
    )
    committee_markdown = _build_report_markdown(
        title=committee_title,
        subtitle=committee_subtitle,
        cover_lines=committee_cover_lines,
        report_map=_build_report_map(committee_sections),
        sections=committee_sections,
    )
    validation_markdown = _build_report_markdown(
        title=validation_title,
        subtitle=validation_subtitle,
        cover_lines=validation_cover_lines,
        report_map=_build_report_map(validation_sections),
        sections=validation_sections,
    )
    return {
        "committee": ReportArtifactPayload(
            title=committee_title,
            subtitle=committee_subtitle,
            markdown=committee_markdown,
            docx_bytes=(
                _build_docx_bytes(
                    title=committee_title,
                    subtitle=committee_subtitle,
                    cover_lines=committee_cover_lines,
                    report_map=_build_report_map(committee_sections),
                    sections=committee_sections,
                )
                if config.export_docx
                else None
            ),
            pdf_bytes=(
                _build_pdf_bytes(
                    title=committee_title,
                    subtitle=committee_subtitle,
                    cover_lines=committee_cover_lines,
                    report_map=_build_report_map(committee_sections),
                    sections=committee_sections,
                )
                if config.export_pdf
                else None
            ),
        ),
        "validation": ReportArtifactPayload(
            title=validation_title,
            subtitle=validation_subtitle,
            markdown=validation_markdown,
            docx_bytes=(
                _build_docx_bytes(
                    title=validation_title,
                    subtitle=validation_subtitle,
                    cover_lines=validation_cover_lines,
                    report_map=_build_report_map(validation_sections),
                    sections=validation_sections,
                )
                if config.export_docx
                else None
            ),
            pdf_bytes=(
                _build_pdf_bytes(
                    title=validation_title,
                    subtitle=validation_subtitle,
                    cover_lines=validation_cover_lines,
                    report_map=_build_report_map(validation_sections),
                    sections=validation_sections,
                )
                if config.export_pdf
                else None
            ),
        ),
    }


def _build_cover_lines(
    context: PipelineContext,
    *,
    audience_label: str,
    template_name: str,
    sections: list[ReportSection],
) -> list[str]:
    guardrail_summary = context.metadata.get("workflow_guardrail_summary", {})
    numerical_warning_count = len(
        context.diagnostics_tables.get("numerical_warning_summary", pd.DataFrame())
    )
    blocking_count = int(guardrail_summary.get("blocking_count", 0))
    warning_count = int(guardrail_summary.get("warning_count", 0))
    return [
        f"Audience: {audience_label}",
        f"Template: {template_name}",
        f"Run ID: {context.run_id}",
        f"Model family: {context.config.model.model_type.value}",
        f"Target mode: {context.config.target.mode.value}",
        f"Execution mode: {context.config.execution.mode.value}",
        f"Selected features: {len(context.feature_columns)}",
        f"Guardrail blockers: {blocking_count}",
        f"Guardrail warnings: {warning_count}",
        f"Normalized numerical warnings: {numerical_warning_count}",
        f"Report sections: {len(sections)}",
    ]


def _build_report_map(sections: list[ReportSection]) -> list[str]:
    return [display_title for display_title, _ in _iter_display_sections(sections)]


def _iter_display_sections(
    sections: list[ReportSection],
) -> list[tuple[str, ReportSection]]:
    display_sections: list[tuple[str, ReportSection]] = []
    numbered_index = 1
    for section in sections:
        if section.appendix:
            display_title = section.title
        else:
            display_title = f"{numbered_index}. {section.title}"
            numbered_index += 1
        display_sections.append((display_title, section))
    return display_sections


def _build_committee_sections(context: PipelineContext) -> list[ReportSection]:
    documentation = context.config.documentation
    reporting = context.config.regulatory_reporting
    split_metrics = context.metrics.get("test") or next(iter(context.metrics.values()), {})
    sections = [
        ReportSection(
            "Executive Summary",
            "Confirm the model purpose, operating context, and core run identifiers first.",
            [
                f"Run ID: {context.run_id}",
                f"Model family: {context.config.model.model_type.value}",
                f"Execution mode: {context.config.execution.mode.value}",
                f"Primary purpose: {documentation.business_purpose or 'Not provided.'}",
                f"Portfolio: {documentation.portfolio_name or 'Not provided.'}",
                f"Target mode: {context.config.target.mode.value}",
            ],
        ),
        ReportSection(
            "Performance Highlights",
            "Use these held-out metrics to decide whether the run merits deeper review.",
            _metric_lines(split_metrics),
        ),
        ReportSection(
            "Development Scope",
            "Confirm the sample, target horizon, and feature footprint before comparing outputs.",
            [
                f"Input rows: {context.metadata.get('input_shape', {}).get('rows', 'n/a')}",
                f"Input columns: {context.metadata.get('input_shape', {}).get('columns', 'n/a')}",
                f"Selected features: {len(context.feature_columns)}",
                f"Data structure: {context.config.split.data_structure.value}",
                f"Target definition: {documentation.target_definition or 'Not provided.'}",
                f"Horizon definition: {documentation.horizon_definition or 'Not provided.'}",
            ],
        ),
    ]
    numerical_lines = _build_numerical_stability_lines(context)
    if numerical_lines:
        sections.append(
            ReportSection(
                "Numerical Stability",
                "Review normalized warnings and fit-health checks before relying on the outputs.",
                numerical_lines,
            )
        )

    if reporting.include_challenger_section and context.comparison_results is not None:
        comparison_lines = [
            f"Recommended model: {context.metadata.get('comparison_recommended_model', 'n/a')}",
            f"Ranking split: {context.config.comparison.ranking_split}",
        ]
        comparison_lines.extend(
            _table_snapshot_lines(
                context.comparison_results,
                columns=["model_type", "split", "ranking_metric", "ranking_value"],
                max_rows=6,
            )
        )
        sections.append(
            ReportSection(
                "Challenger Review",
                "Compare the selected model against configured challengers on the ranking split.",
                comparison_lines,
            )
        )

    if reporting.include_scenario_section and context.scenario_results:
        scenario_lines = []
        for scenario_name, table in context.scenario_results.items():
            scenario_lines.append(f"{scenario_name}: {len(table)} rows exported.")
            scenario_columns = [
                column
                for column in ("split", "scenario_name", "mean_delta")
                if column in table.columns
            ]
            scenario_lines.extend(
                _table_snapshot_lines(
                    table,
                    columns=scenario_columns,
                    max_rows=3,
                )
            )
        sections.append(
            ReportSection(
                "Scenario Analysis",
                (
                    "Use held-out shocks to assess directional sensitivity before "
                    "finalizing the model."
                ),
                scenario_lines,
            )
        )

    if reporting.include_assumptions_section:
        sections.append(
            ReportSection(
                "Assumptions And Limitations",
                (
                    "Capture the governing assumptions and explicit constraints "
                    "that frame interpretation."
                ),
                _list_or_default(documentation.assumptions, "No assumptions were recorded.")
                + _section_break("Limitations")
                + _list_or_default(documentation.limitations, "No limitations were recorded."),
            )
        )

    if reporting.include_appendix_section:
        sections.extend(_build_appendix_sections(context))
    return sections


def _build_validation_sections(context: PipelineContext) -> list[ReportSection]:
    documentation = context.config.documentation
    reporting = context.config.regulatory_reporting
    assumption_checks = context.diagnostics_tables.get("assumption_checks", pd.DataFrame())
    guardrails = context.diagnostics_tables.get("workflow_guardrails", pd.DataFrame())
    feature_policy = context.diagnostics_tables.get("feature_policy_checks", pd.DataFrame())
    calibration_summary = context.diagnostics_tables.get("calibration_summary", pd.DataFrame())
    numerical_warnings = context.diagnostics_tables.get("numerical_warning_summary", pd.DataFrame())
    numerical_diagnostics = context.diagnostics_tables.get(
        "model_numerical_diagnostics", pd.DataFrame()
    )
    backtest_summary = (
        context.backtest_summary if context.backtest_summary is not None else pd.DataFrame()
    )
    backtest_columns = [
        column
        for column in ("split", "bucket", "observed_rate", "predicted_rate")
        if column in backtest_summary.columns
    ]

    sections = [
        ReportSection(
            "Validation Overview",
            "Confirm ownership, model family, and execution context before the detailed review.",
            [
                f"Run ID: {context.run_id}",
                f"Model owner: {documentation.model_owner or 'Not provided.'}",
                f"Model family: {context.config.model.model_type.value}",
                f"Target mode: {context.config.target.mode.value}",
                f"Execution mode: {context.config.execution.mode.value}",
            ],
        ),
        ReportSection(
            "Use Case And Data Contract",
            "Use this section to confirm scope, target construction, and dataset framing.",
            [
                f"Business purpose: {documentation.business_purpose or 'Not provided.'}",
                f"Target definition: {documentation.target_definition or 'Not provided.'}",
                f"Horizon definition: {documentation.horizon_definition or 'Not provided.'}",
                f"Loss definition: {documentation.loss_definition or 'Not provided.'}",
                f"Data structure: {context.config.split.data_structure.value}",
                f"Feature count: {len(context.feature_columns)}",
            ],
        ),
        ReportSection(
            "Guardrails And Suitability",
            (
                "Blocking findings here should be resolved before treating the run "
                "as development-ready."
            ),
            _table_snapshot_lines(
                guardrails,
                columns=["severity", "code", "message"],
                max_rows=8,
            )
            + _table_snapshot_lines(
                assumption_checks,
                columns=["check_name", "status", "message"],
                max_rows=8,
            )
            + _table_snapshot_lines(
                feature_policy,
                columns=["feature_name", "status", "check_type", "message"],
                max_rows=8,
            ),
        ),
        ReportSection(
            "Performance Diagnostics",
            "Use the validation or held-out metrics here as the primary performance checkpoint.",
            _metric_lines(context.metrics.get("validation") or context.metrics.get("test") or {}),
        ),
        ReportSection(
            "Backtesting And Calibration",
            (
                "Review calibration and observed-versus-predicted alignment before "
                "threshold decisions."
            ),
            _table_snapshot_lines(
                calibration_summary,
                columns=["method", "split", "brier_score", "log_loss"],
                max_rows=6,
            )
            + _table_snapshot_lines(
                backtest_summary,
                columns=backtest_columns,
                max_rows=6,
            ),
        ),
        ReportSection(
            "Numerical Stability And Estimation Health",
            (
                "Focus on this section when coefficients, standard errors, or "
                "convergence are in question."
            ),
            _table_snapshot_lines(
                numerical_warnings,
                columns=["source", "warning_code", "message"],
                max_rows=6,
            )
            + _table_snapshot_lines(
                numerical_diagnostics,
                columns=["source", "diagnostic_name", "value", "status"],
                max_rows=10,
            )
            or ["No numerical warnings or fit-health concerns were recorded."],
        ),
    ]

    if reporting.include_challenger_section and context.comparison_results is not None:
        sections.append(
            ReportSection(
                "Challenger Review",
                (
                    "Compare challenger ranking rows here before concluding the "
                    "incumbent is preferred."
                ),
                _table_snapshot_lines(
                    context.comparison_results,
                    columns=["model_type", "split", "ranking_metric", "ranking_value"],
                    max_rows=8,
                ),
            )
        )

    if reporting.include_scenario_section and context.scenario_results:
        scenario_lines = []
        for scenario_name, table in context.scenario_results.items():
            scenario_lines.append(f"{scenario_name}: {len(table)} rows exported.")
            scenario_columns = [
                column
                for column in ("split", "scenario_name", "mean_delta")
                if column in table.columns
            ]
            scenario_lines.extend(
                _table_snapshot_lines(
                    table,
                    columns=scenario_columns,
                    max_rows=4,
                )
            )
        sections.append(
            ReportSection(
                "Scenario Testing",
                "Use these outputs to understand how configured shocks change the held-out view.",
                scenario_lines,
            )
        )

    if reporting.include_assumptions_section:
        sections.append(
            ReportSection(
                "Assumptions, Exclusions, And Limitations",
                "This section records the stated assumptions, exclusions, and known constraints.",
                _list_or_default(documentation.assumptions, "No assumptions were recorded.")
                + _section_break("Exclusions")
                + _list_or_default(documentation.exclusions, "No exclusions were recorded.")
                + _section_break("Limitations")
                + _list_or_default(documentation.limitations, "No limitations were recorded."),
            )
        )

    if reporting.include_appendix_section:
        sections.extend(_build_appendix_sections(context))
    return sections


def _build_appendix_sections(context: PipelineContext) -> list[ReportSection]:
    feature_importance = (
        context.feature_importance if context.feature_importance is not None else pd.DataFrame()
    )
    variable_selection = context.diagnostics_tables.get("variable_selection", pd.DataFrame())
    reproducibility = context.diagnostics_tables.get("reproducibility_manifest", pd.DataFrame())
    numerical_warnings = context.diagnostics_tables.get("numerical_warning_summary", pd.DataFrame())
    numerical_diagnostics = context.diagnostics_tables.get(
        "model_numerical_diagnostics", pd.DataFrame()
    )
    feature_importance_columns = [
        column
        for column in ("feature_name", "importance_value", "coefficient")
        if column in feature_importance.columns
    ]
    variable_selection_columns = [
        column
        for column in ("feature_name", "selected", "selection_reason")
        if column in variable_selection.columns
    ]
    appendices = [
        ReportSection(
            "Appendix A. Feature And Selection Snapshot",
            "Use this appendix to review the modeled feature footprint and selection evidence.",
            [
                f"Numeric features: {len(context.numeric_features)}",
                f"Categorical features: {len(context.categorical_features)}",
                f"Warnings captured: {len(context.warnings)}",
            ]
            + _section_break("Feature Importance Snapshot")
            + _table_snapshot_lines(
                feature_importance,
                columns=feature_importance_columns,
                max_rows=8,
            )
            + _section_break("Variable Selection Snapshot")
            + _table_snapshot_lines(
                variable_selection,
                columns=variable_selection_columns,
                max_rows=8,
            ),
            appendix=True,
        ),
        ReportSection(
            "Appendix B. Reproducibility And Artifact Map",
            "Use this appendix to trace the run fingerprint and the main exported deliverables.",
            _section_break("Reproducibility Manifest Snapshot")
            + _table_snapshot_lines(
                reproducibility,
                columns=["field", "value"],
                max_rows=8,
            )
            + _section_break("Artifact Deliverables")
            + _build_artifact_index_lines(context),
            appendix=True,
        ),
        ReportSection(
            "Appendix C. Numerical Diagnostics Snapshot",
            "Use this appendix when you need the detailed numerical warning and fit-health trail.",
            _table_snapshot_lines(
                numerical_diagnostics,
                columns=["source", "diagnostic_name", "value", "status"],
                max_rows=8,
            )
            + _table_snapshot_lines(
                numerical_warnings,
                columns=["source", "warning_code", "message"],
                max_rows=6,
            ),
            appendix=True,
        ),
    ]
    return appendices


def _build_artifact_index_lines(context: PipelineContext) -> list[str]:
    artifacts = context.config.artifacts
    return [
        f"Run report: {artifacts.report_file_name}",
        f"Interactive HTML report: {artifacts.interactive_report_file_name}",
        f"Decision summary: {artifacts.decision_summary_file_name}",
        f"Development documentation pack: {artifacts.documentation_pack_file_name}",
        f"Validation pack: {artifacts.validation_pack_file_name}",
        (
            "Committee report: "
            f"{artifacts.committee_report_docx_file_name} / "
            f"{artifacts.committee_report_pdf_file_name}"
        ),
        (
            "Validation report: "
            f"{artifacts.validation_report_docx_file_name} / "
            f"{artifacts.validation_report_pdf_file_name}"
        ),
        f"Reproducibility manifest: {artifacts.reproducibility_manifest_file_name}",
        f"Configuration workbook: {artifacts.template_workbook_file_name}",
    ]


def _build_numerical_stability_lines(context: PipelineContext) -> list[str]:
    warning_table = context.diagnostics_tables.get("numerical_warning_summary", pd.DataFrame())
    diagnostics_table = context.diagnostics_tables.get(
        "model_numerical_diagnostics", pd.DataFrame()
    )
    lines = [
        f"Normalized numerical warnings: {len(warning_table)}",
        f"Recorded fit-health diagnostics: {len(diagnostics_table)}",
    ]
    lines.extend(
        _table_snapshot_lines(
            warning_table,
            columns=["source", "warning_code", "message"],
            max_rows=4,
        )
    )
    lines.extend(
        _table_snapshot_lines(
            diagnostics_table,
            columns=["source", "diagnostic_name", "value", "status"],
            max_rows=6,
        )
    )
    return lines


def _metric_lines(metrics: dict[str, float | int | None]) -> list[str]:
    if not metrics:
        return ["No metric snapshot was available."]
    return [f"{metric_name}: {metric_value}" for metric_name, metric_value in metrics.items()]


def _list_or_default(values: list[str], default_line: str) -> list[str]:
    if not values:
        return [default_line]
    return values


def _section_break(label: str) -> list[str]:
    return [f"{label}:"]


def _table_snapshot_lines(
    table: pd.DataFrame,
    *,
    columns: list[str],
    max_rows: int,
) -> list[str]:
    if table.empty or not columns:
        return []
    selected_columns = [column for column in columns if column in table.columns]
    if not selected_columns:
        return []
    snapshot = table.loc[:, selected_columns].head(max_rows)
    lines: list[str] = []
    for _, row in snapshot.iterrows():
        parts = [f"{column}={row[column]}" for column in selected_columns]
        lines.append("; ".join(parts))
    return lines


def _build_report_markdown(
    *,
    title: str,
    subtitle: str,
    cover_lines: list[str],
    report_map: list[str],
    sections: list[ReportSection],
) -> str:
    lines = [f"# {title}", "", subtitle, "", "## Report At A Glance", ""]
    lines.extend(f"- {line}" for line in cover_lines)
    lines.extend(["", "## Report Map", ""])
    lines.extend(f"- {line}" for line in report_map)
    lines.append("")
    for display_title, section in _iter_display_sections(sections):
        lines.extend([f"## {display_title}", ""])
        lines.append(f"> Section Summary: {section.summary}")
        lines.append("")
        if section.lines:
            lines.extend(f"- {line}" for line in section.lines)
        else:
            lines.append("- No content was available for this section.")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_docx_bytes(
    *,
    title: str,
    subtitle: str,
    cover_lines: list[str],
    report_map: list[str],
    sections: list[ReportSection],
) -> bytes:
    body_parts = [
        _build_docx_cover_title_xml(title),
        _build_docx_subtitle_xml(subtitle),
        _build_docx_spacer_xml(),
        _build_docx_banner_xml("Report At A Glance"),
    ]
    body_parts.extend(_build_docx_paragraph_xml(line) for line in cover_lines)
    body_parts.extend(
        [
            _build_docx_spacer_xml(),
            _build_docx_banner_xml("Report Map"),
        ]
    )
    body_parts.extend(_build_docx_paragraph_xml(line) for line in report_map)
    body_parts.append(_build_docx_page_break_xml())
    for display_title, section in _iter_display_sections(sections):
        body_parts.append(_build_docx_banner_xml(display_title))
        body_parts.append(_build_docx_summary_xml(section.summary))
        if section.lines:
            body_parts.extend(_build_docx_paragraph_xml(line) for line in section.lines)
        else:
            body_parts.append(_build_docx_paragraph_xml("No content was available."))
        body_parts.append(_build_docx_spacer_xml())
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {"".join(body_parts)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1080" w:bottom="1440" w:left="1080"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override
    PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship
    Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>
"""
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def _build_docx_cover_title_xml(text: str) -> str:
    return f"""
    <w:p>
      <w:pPr>
        <w:spacing w:before="400" w:after="160"/>
      </w:pPr>
      <w:r>
        <w:rPr><w:b/><w:sz w:val="42"/><w:color w:val="16324F"/></w:rPr>
        <w:t xml:space="preserve">{xml_escape(text)}</w:t>
      </w:r>
    </w:p>
    """


def _build_docx_subtitle_xml(text: str) -> str:
    return f"""
    <w:p>
      <w:pPr>
        <w:spacing w:after="220"/>
      </w:pPr>
      <w:r>
        <w:rPr><w:sz w:val="22"/><w:color w:val="5F6B7A"/></w:rPr>
        <w:t xml:space="preserve">{xml_escape(text)}</w:t>
      </w:r>
    </w:p>
    """


def _build_docx_banner_xml(text: str) -> str:
    return f"""
    <w:p>
      <w:pPr>
        <w:spacing w:before="200" w:after="120"/>
        <w:shd w:fill="16324F"/>
      </w:pPr>
      <w:r>
        <w:rPr><w:b/><w:color w:val="FFFFFF"/><w:sz w:val="24"/></w:rPr>
        <w:t xml:space="preserve">{xml_escape(text)}</w:t>
      </w:r>
    </w:p>
    """


def _build_docx_summary_xml(text: str) -> str:
    return f"""
    <w:p>
      <w:pPr>
        <w:spacing w:after="120"/>
      </w:pPr>
      <w:r>
        <w:rPr><w:i/><w:color w:val="46627A"/><w:sz w:val="20"/></w:rPr>
        <w:t xml:space="preserve">Section Summary: {xml_escape(text)}</w:t>
      </w:r>
    </w:p>
    """


def _build_docx_paragraph_xml(text: str) -> str:
    return f"""
    <w:p>
      <w:pPr>
        <w:spacing w:after="80"/>
      </w:pPr>
      <w:r>
        <w:t xml:space="preserve">{xml_escape(text)}</w:t>
      </w:r>
    </w:p>
    """


def _build_docx_spacer_xml() -> str:
    return """
    <w:p>
      <w:pPr><w:spacing w:after="120"/></w:pPr>
      <w:r><w:t xml:space="preserve"></w:t></w:r>
    </w:p>
    """


def _build_docx_page_break_xml() -> str:
    return """
    <w:p>
      <w:r><w:br w:type="page"/></w:r>
    </w:p>
    """


def _build_pdf_bytes(
    *,
    title: str,
    subtitle: str,
    cover_lines: list[str],
    report_map: list[str],
    sections: list[ReportSection],
) -> bytes:
    line_specs = _build_pdf_line_specs(
        title=title,
        subtitle=subtitle,
        cover_lines=cover_lines,
        report_map=report_map,
        sections=sections,
    )
    pages = _paginate_pdf_line_specs(line_specs)

    objects: list[str] = []

    def add_object(payload: str) -> int:
        objects.append(payload)
        return len(objects)

    font_object = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_object = add_object("")
    page_numbers: list[int] = []

    page_count = len(pages)
    for page_index, page_specs in enumerate(pages, start=1):
        content_stream = _build_pdf_content_stream(
            page_specs,
            title=title,
            page_number=page_index,
            page_count=page_count,
        )
        content_length = len(content_stream.encode("latin-1"))
        content_object = add_object(
            f"<< /Length {content_length} >>\nstream\n{content_stream}\nendstream"
        )
        page_object = add_object(
            f"<< /Type /Page /Parent {pages_object} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
            f"/Contents {content_object} 0 R >>"
        )
        page_numbers.append(page_object)

    objects[pages_object - 1] = (
        f"<< /Type /Pages /Count {len(page_numbers)} /Kids "
        f"[{' '.join(f'{page} 0 R' for page in page_numbers)}] >>"
    )
    catalog_object = add_object(f"<< /Type /Catalog /Pages {pages_object} 0 R >>")

    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n{payload}\nendobj\n".encode("latin-1"))
    xref_offset = buffer.tell()
    buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    buffer.write(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_object} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("latin-1")
    )
    return buffer.getvalue()


def _build_pdf_line_specs(
    *,
    title: str,
    subtitle: str,
    cover_lines: list[str],
    report_map: list[str],
    sections: list[ReportSection],
) -> list[tuple[str, str]]:
    line_specs: list[tuple[str, str]] = [
        ("cover_title", title),
        ("cover_subtitle", subtitle),
        ("spacer", ""),
        ("section", "Report At A Glance"),
    ]
    line_specs.extend(("body", line) for line in cover_lines)
    line_specs.extend([("spacer", ""), ("section", "Report Map")])
    line_specs.extend(("body", line) for line in report_map)
    line_specs.extend([("page_break", "")])
    for display_title, section in _iter_display_sections(sections):
        line_specs.append(("section", display_title))
        line_specs.append(("summary", f"Section Summary: {section.summary}"))
        section_lines = section.lines or ["No content was available."]
        line_specs.extend(("body", line) for line in section_lines)
        line_specs.append(("spacer", ""))
    return line_specs


def _paginate_pdf_line_specs(
    line_specs: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    style_height = {
        "cover_title": 30,
        "cover_subtitle": 18,
        "section": 18,
        "summary": 16,
        "body": 14,
        "spacer": 10,
    }
    style_wrap_width = {
        "cover_title": 55,
        "cover_subtitle": 90,
        "section": 90,
        "summary": 92,
        "body": 95,
    }
    available_height = 700
    pages: list[list[tuple[str, str]]] = []
    current_page: list[tuple[str, str]] = []
    current_height = 0

    for style, text in line_specs:
        if style == "page_break":
            if current_page:
                pages.append(current_page)
            current_page = []
            current_height = 0
            continue
        wrapped_lines = wrap(text, width=style_wrap_width.get(style, 95)) or [""]
        required_height = style_height.get(style, 14) * len(wrapped_lines)
        if current_page and current_height + required_height > available_height:
            pages.append(current_page)
            current_page = []
            current_height = 0
        for wrapped_line in wrapped_lines:
            current_page.append((style, wrapped_line))
            current_height += style_height.get(style, 14)
    if current_page:
        pages.append(current_page)
    return pages or [[("body", "")]]


def _build_pdf_content_stream(
    line_specs: list[tuple[str, str]],
    *,
    title: str,
    page_number: int,
    page_count: int,
) -> str:
    def escape_pdf_text(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_lines = [
        "BT",
        "54 750 Td",
    ]
    style_tokens = {
        "cover_title": {"font_size": 20, "leading": 24},
        "cover_subtitle": {"font_size": 11, "leading": 16},
        "section": {"font_size": 13, "leading": 18},
        "summary": {"font_size": 9, "leading": 13},
        "body": {"font_size": 10, "leading": 14},
        "spacer": {"font_size": 10, "leading": 10},
    }
    first_line = True
    for style, line in line_specs:
        token = style_tokens.get(style, style_tokens["body"])
        if first_line:
            content_lines.append(f"/F1 {token['font_size']} Tf")
            content_lines.append(f"{token['leading']} TL")
            content_lines.append(f"({escape_pdf_text(line)}) Tj")
            first_line = False
            continue
        content_lines.append("T*")
        content_lines.append(f"/F1 {token['font_size']} Tf")
        content_lines.append(f"{token['leading']} TL")
        content_lines.append(f"({escape_pdf_text(line)}) Tj")
    content_lines.extend(
        [
            "ET",
            "BT",
            "/F1 9 Tf",
            "54 24 Td",
            f"(Generated by Quant Studio: {escape_pdf_text(title)}) Tj",
            "ET",
            "BT",
            "/F1 9 Tf",
            "472 24 Td",
            f"(Page {page_number} of {page_count}) Tj",
            "ET",
        ]
    )
    return "\n".join(content_lines)
