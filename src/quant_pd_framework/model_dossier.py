"""Model-development dossier generation for audit-ready run packages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .decision_summary import build_decision_summary, build_decision_summary_snapshot_from_context
from .feature_lineage import summarize_feature_lineage

if TYPE_CHECKING:
    from .context import PipelineContext


def build_model_development_dossier(
    context: PipelineContext,
    *,
    feature_lineage: pd.DataFrame,
) -> str:
    """Creates a concise but audit-oriented model-development dossier."""

    documentation = context.config.documentation
    preset_name = context.config.preset_name.value if context.config.preset_name else "custom"
    decision_summary = build_decision_summary(build_decision_summary_snapshot_from_context(context))
    lineage_summary = summarize_feature_lineage(feature_lineage)
    split_summary = context.metadata.get("split_summary", {})
    assumption_summary = context.metadata.get("assumption_check_summary", {})
    guardrail_summary = context.metadata.get("workflow_guardrail_summary", {})
    transformation_summary = context.metadata.get("transformation_summary", {})
    imputation_summary = context.metadata.get("imputation_summary", {})
    resource_estimate = context.metadata.get("large_data_memory_estimate", {})

    lines = [
        f"# Model Development Dossier: {documentation.model_name}",
        "",
        "## 1. Executive Overview",
        "",
        f"- Run ID: `{context.run_id}`",
        f"- Execution mode: `{context.config.execution.mode.value}`",
        f"- Preset: `{preset_name}`",
        f"- Model type: `{context.config.model.model_type.value}`",
        f"- Target mode: `{context.config.target.mode.value}`",
        f"- Target column: `{context.target_column or 'n/a'}`",
        f"- Recommendation: `{decision_summary['recommendation']}`",
        f"- Decision level: `{decision_summary['level'].replace('_', ' ').title()}`",
        "",
        "## 2. Business Purpose And Scope",
        "",
        documentation.business_purpose or "Not provided.",
        "",
        f"- Model owner: `{documentation.model_owner or 'n/a'}`",
        f"- Portfolio: `{documentation.portfolio_name or 'n/a'}`",
        f"- Segment: `{documentation.segment_name or 'n/a'}`",
        "",
        "## 3. Target, Horizon, And Population",
        "",
        f"- Target definition: {documentation.target_definition or 'Not provided.'}",
        f"- Horizon definition: {documentation.horizon_definition or 'Not provided.'}",
        f"- Loss definition: {documentation.loss_definition or 'Not provided.'}",
        f"- Input rows: `{context.metadata.get('input_shape', {}).get('rows', 'n/a')}`",
        f"- Input columns: `{context.metadata.get('input_shape', {}).get('columns', 'n/a')}`",
        "",
    ]

    if split_summary:
        lines.extend(["### Development Splits", ""])
        for split_name, payload in split_summary.items():
            if isinstance(payload, dict):
                lines.append(
                    f"- {split_name.title()}: `{payload.get('rows', 'n/a')}` rows"
                    f" ({payload.get('target_rate', payload.get('target_mean', 'n/a'))})"
                )
        lines.append("")

    lines.extend(
        [
            "## 4. Feature Governance And Lineage",
            "",
            f"- Final model terms: `{lineage_summary['model_feature_count']}`",
            f"- Source features represented: `{lineage_summary['source_feature_count']}`",
            f"- Transformed terms: `{lineage_summary['transformed_feature_count']}`",
            (
                "- Terms with imputation policy evidence: "
                f"`{lineage_summary['imputed_feature_count']}`"
            ),
            f"- Terms with business definitions: `{lineage_summary['documented_feature_count']}`",
            (
                "- Terms missing business definitions: "
                f"`{lineage_summary['undocumented_feature_count']}`"
            ),
            f"- Governed transformations applied: `{transformation_summary.get('count', 0)}`",
            f"- Features with imputation rules: `{imputation_summary.get('feature_count', 0)}`",
            "",
        ]
    )
    top_lineage = _lineage_preview(feature_lineage)
    if top_lineage:
        lines.extend(["### Top Feature Lineage Preview", "", *top_lineage, ""])
    lines.append(
        "Full lineage is exported as `tables/governance/feature_lineage_map.*` and "
        "`model/feature_lineage_map.csv`."
    )
    lines.append("")

    lines.extend(["## 5. Modeling Methodology", ""])
    lines.append(
        f"The selected estimator is `{context.config.model.model_type.value}` for a "
        f"`{context.config.target.mode.value}` target. The resolved configuration is exported "
        "in `config/run_config.json`, and the rerunnable Python entry point is exported in "
        "`code/generated_run.py`."
    )
    lines.append("")
    if context.comparison_results is not None:
        lines.append(
            f"- Challenger comparison was enabled; recommended challenger result: "
            f"`{context.metadata.get('comparison_recommended_model', 'n/a')}`."
        )
    if context.config.cross_validation.enabled:
        lines.append("- Cross-validation was enabled and exported with the diagnostic tables.")
    if context.config.scenario_testing.enabled:
        lines.append("- Scenario testing was enabled and exported with the diagnostic tables.")
    lines.append("")

    lines.extend(["## 6. Performance And Validation Evidence", ""])
    lines.extend(_metrics_lines(context.metrics))
    lines.append("")
    lines.extend(
        [
            f"- Assumption-check failures: `{assumption_summary.get('fail_count', 0)}`",
            f"- Assumption-check warnings: `{assumption_summary.get('warn_count', 0)}`",
            f"- Workflow guardrail errors: `{guardrail_summary.get('error_count', 0)}`",
            f"- Workflow guardrail warnings: `{guardrail_summary.get('warning_count', 0)}`",
            "",
        ]
    )

    if resource_estimate:
        lines.extend(
            [
                "## 7. Resource And Reproducibility Controls",
                "",
                (
                    "- Estimated peak memory: "
                    f"`{resource_estimate.get('estimated_peak_memory_gb', 'n/a')}` GB"
                ),
                (
                    "- Configured memory limit: "
                    f"`{resource_estimate.get('configured_memory_limit_gb', 'n/a')}` GB"
                ),
                f"- Memory estimate status: `{resource_estimate.get('status', 'not_evaluated')}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 7. Resource And Reproducibility Controls",
                "",
                "- Resource estimates were not available in run metadata.",
                "",
            ]
        )
    lines.extend(
        [
            (
                "- `metadata/reproducibility_manifest.json` records hashes, package "
                "versions, and environment metadata."
            ),
            (
                "- `metadata/run_debug_trace.json` records step timing and tracked "
                "dataframe memory when enabled."
            ),
            "- `metadata/step_manifest.json` records the exact ordered workflow stage stack.",
            "",
            "## 8. Assumptions, Exclusions, Limitations",
            "",
        ]
    )
    lines.extend(_list_section("Assumptions", documentation.assumptions))
    lines.extend(_list_section("Exclusions", documentation.exclusions))
    lines.extend(_list_section("Limitations", documentation.limitations))

    lines.extend(["## 9. Open Review Items", ""])
    review_items = _review_items(context, lineage_summary)
    lines.extend(f"- {item}" for item in review_items)
    lines.append("")

    lines.extend(
        [
            "## 10. Primary Artifacts",
            "",
            "- `reports/interactive_report.html`: chart-first diagnostic report.",
            "- `reports/decision_summary.md`: model decision scorecard.",
            "- `reports/model_documentation_pack.md`: development documentation summary.",
            "- `reports/validation_pack.md`: validation evidence summary.",
            "- `reports/model_development_dossier.md`: this dossier.",
            "- `tables/governance/feature_lineage_map.*`: feature lineage and documentation table.",
            "- `model/quant_model.joblib`: serialized model object.",
            "- `config/run_config.json`: resolved workflow configuration.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _metrics_lines(metrics: dict[str, dict[str, Any]]) -> list[str]:
    if not metrics:
        return ["- No metrics were captured."]
    rows: list[str] = []
    for split_name, split_metrics in metrics.items():
        metric_text = ", ".join(
            f"{metric_name}: `{metric_value}`"
            for metric_name, metric_value in split_metrics.items()
        )
        rows.append(f"- {split_name.title()}: {metric_text or 'No metrics captured.'}")
    return rows


def _lineage_preview(feature_lineage: pd.DataFrame) -> list[str]:
    if feature_lineage.empty:
        return []
    preview_columns = [
        "model_feature_name",
        "source_feature",
        "lineage_type",
        "importance_value",
        "review_guidance",
    ]
    available_columns = [column for column in preview_columns if column in feature_lineage.columns]
    rows: list[str] = []
    for _, row in feature_lineage.loc[:, available_columns].head(10).iterrows():
        rows.append(
            "- "
            + " | ".join(
                f"{column}: `{_render_value(row.get(column))}`" for column in available_columns
            )
        )
    return rows


def _list_section(title: str, values: list[str]) -> list[str]:
    rows = [f"### {title}", ""]
    if values:
        rows.extend(f"- {value}" for value in values)
    else:
        rows.append("- None recorded.")
    rows.append("")
    return rows


def _review_items(context: PipelineContext, lineage_summary: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if lineage_summary.get("undocumented_feature_count", 0):
        items.append("Complete missing feature definitions before external validation review.")
    if context.warnings:
        items.append("Resolve or explicitly accept run warnings listed in the run report.")
    if context.metadata.get("assumption_check_summary", {}).get("fail_count", 0):
        items.append("Document the disposition of failed assumption checks.")
    if context.metadata.get("workflow_guardrail_summary", {}).get("warning_count", 0):
        items.append("Review workflow guardrail warnings and document acceptance rationale.")
    if not items:
        items.append(
            "No automated open review items were identified; perform standard model review."
        )
    return items


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)
