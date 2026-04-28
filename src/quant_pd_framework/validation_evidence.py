"""Validation checklist and evidence traceability outputs for completed runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .config import ExecutionMode, TargetMode
from .context import PipelineContext
from .tabular_policy import resolve_tabular_output_format

VALIDATION_CHECKLIST_COLUMNS = [
    "review_area",
    "status",
    "interpretation",
    "evidence",
    "recommended_action",
]

EVIDENCE_TRACEABILITY_COLUMNS = [
    "evidence_area",
    "question_answered",
    "artifact_key",
    "artifact_location",
    "table_name",
    "reviewer_use",
]


def publish_validation_evidence_tables(context: PipelineContext) -> None:
    """Adds audit-ready validation checklist and evidence map tables to the run."""

    context.diagnostics_tables["validation_checklist"] = build_validation_checklist(context)
    context.diagnostics_tables["evidence_traceability_map"] = build_evidence_traceability_map(
        context
    )


def build_validation_checklist(context: PipelineContext) -> pd.DataFrame:
    """Builds a reviewer-oriented checklist from the artifacts available in context."""

    tables = context.diagnostics_tables
    execution_mode = context.config.execution.mode
    target_mode = context.config.target.mode
    labels_available = bool(context.metadata.get("labels_available", False))
    has_model = context.model is not None
    has_subset_candidates = _has_table(tables, "subset_search_candidates")
    is_subset_search = execution_mode == ExecutionMode.SEARCH_FEATURE_SUBSETS
    rows = [
        _check_row(
            "Input data and schema",
            complete=context.raw_data is not None or context.large_data_handle is not None,
            evidence=_join_evidence(
                "input_shape metadata",
                "feature_dictionary" if _has_table(tables, "feature_dictionary") else "",
                "schema editor selections",
            ),
            action="Confirm ID, date, target, feature, and dropped-column roles before approval.",
        ),
        _check_row(
            "Target and label availability",
            complete=bool(context.target_column)
            and (labels_available or target_mode != TargetMode.BINARY),
            attention=target_mode == TargetMode.BINARY and not labels_available,
            not_applicable=target_mode != TargetMode.BINARY and not labels_available,
            evidence=_join_evidence(
                f"target_column={context.target_column or 'not set'}",
                f"labels_available={labels_available}",
            ),
            action=(
                "Use labeled validation or test data before making performance claims."
                if target_mode == TargetMode.BINARY
                else "Review continuous target construction and split metrics."
            ),
        ),
        _check_row(
            "Split strategy",
            complete=bool(context.split_frames) or bool(context.metadata.get("split_summary")),
            evidence="split_summary metadata and per-split metrics",
            action="Confirm train, validation, and test definitions match the modeling purpose.",
        ),
        _check_row(
            "Model fit or model load",
            complete=has_model or has_subset_candidates,
            evidence=(
                "model/quant_model.joblib"
                if has_model
                else "subset_search_candidates comparison table"
            ),
            action=(
                "Subset search is comparison evidence only; rerun fit_new_model for final approval."
                if is_subset_search
                else "Confirm the model object and run configuration are retained."
            ),
        ),
        _check_row(
            "Performance metrics",
            complete=bool(context.metrics),
            evidence="metadata/metrics.json and Step 4 metric panels",
            action=(
                "Review the primary holdout split and compare train versus "
                "validation/test results."
            ),
        ),
        _check_row(
            "Calibration and threshold evidence",
            complete=_has_any_table(
                tables,
                ["calibration", "calibration_summary", "threshold_analysis", "lift_gain"],
            ),
            not_applicable=target_mode != TargetMode.BINARY or not labels_available,
            evidence=_available_tables(
                tables,
                ["calibration", "calibration_summary", "threshold_analysis", "lift_gain"],
            ),
            action="For PD models, review calibration, lift/gain, and threshold tradeoffs.",
        ),
        _check_row(
            "Stability and backtesting evidence",
            complete=context.backtest_summary is not None
            or _has_any_table(tables, ["psi", "adf_tests", "structural_break_tests"]),
            evidence=_join_evidence(
                "backtest_summary.csv" if context.backtest_summary is not None else "",
                _available_tables(tables, ["psi", "adf_tests", "structural_break_tests"]),
            ),
            action="Review time, population, and segment stability before relying on the model.",
        ),
        _check_row(
            "Feature governance",
            complete=_has_any_table(
                tables,
                [
                    "feature_dictionary",
                    "feature_policy_checks",
                    "variable_selection",
                    "manual_review_feature_decisions",
                ],
            ),
            evidence=_available_tables(
                tables,
                [
                    "feature_dictionary",
                    "feature_policy_checks",
                    "variable_selection",
                    "manual_review_feature_decisions",
                ],
            ),
            action=(
                "Confirm selected features have business rationale and policy "
                "exceptions are resolved."
            ),
        ),
        _check_row(
            "Explainability evidence",
            complete=context.feature_importance is not None
            or _has_any_table(
                tables,
                ["permutation_importance", "partial_dependence", "feature_effect_curves"],
            ),
            evidence=_join_evidence(
                "feature_importance.csv" if context.feature_importance is not None else "",
                _available_tables(
                    tables,
                    ["permutation_importance", "partial_dependence", "feature_effect_curves"],
                ),
            ),
            action="Review top drivers, effect direction, and nonlinear or interaction evidence.",
        ),
        _check_row(
            "Scenario and challenger review",
            complete=_has_any_table(tables, ["scenario_summary", "model_comparison"]),
            not_applicable=not (
                context.config.scenario_testing.enabled or context.config.comparison.enabled
            ),
            evidence=_available_tables(tables, ["scenario_summary", "model_comparison"]),
            action="Use scenario and challenger evidence when policy requires sensitivity review.",
        ),
        _check_row(
            "Reproducibility package",
            complete=True,
            evidence="run_config.json, generated_run.py, code_snapshot, reproducibility_manifest",
            action="Confirm the exported config and generated runner reproduce the intended run.",
        ),
        _check_row(
            "Report size and payload controls",
            complete=_has_table(tables, "report_payload_audit"),
            evidence="report_payload_audit table and interactive_report.html",
            action=(
                "Review skipped or downsampled report figures when the interactive report "
                "must be shared as a standalone file."
            ),
        ),
    ]
    return pd.DataFrame(rows, columns=VALIDATION_CHECKLIST_COLUMNS)


def build_evidence_traceability_map(context: PipelineContext) -> pd.DataFrame:
    """Maps common review questions to exported artifacts and diagnostic tables."""

    prediction_format = resolve_tabular_output_format(context.metadata).value
    prediction_name = "predictions.parquet" if prediction_format == "parquet" else "predictions.csv"
    table_extension = "parquet" if prediction_format == "parquet" else "csv"
    rows = [
        _evidence_row(
            "Run setup",
            "Which configuration produced the run?",
            "config",
            "config/run_config.json",
            "",
            "Use to reproduce setup choices or compare against a saved profile.",
        ),
        _evidence_row(
            "Model object",
            "What fitted model should be reused for existing-model scoring?",
            "model",
            "model/quant_model.joblib",
            "",
            "Use only for fit_new_model runs that produced a final model.",
        ),
        _evidence_row(
            "Predictions",
            "Which rows were scored and what score was assigned?",
            "predictions",
            f"data/predictions/{prediction_name}",
            "",
            "Use for row-level audit, cutoff review, and downstream validation.",
        ),
        _evidence_row(
            "Performance",
            "How did the model perform by split?",
            "metrics",
            "metadata/metrics.json",
            "",
            "Use as the machine-readable source for KPI cards and decision summary.",
        ),
        _evidence_row(
            "Diagnostic report",
            "Where are the grouped charts and table previews?",
            "interactive_report",
            "reports/interactive_report.html",
            "",
            "Use for reviewer navigation and model-development evidence packaging.",
        ),
        _evidence_row(
            "Decision synthesis",
            "What does the app recommend reviewing before approval?",
            "decision_summary",
            "reports/decision_summary.md",
            "validation_checklist",
            "Use as the concise decision scorecard and review checklist.",
        ),
        _evidence_row(
            "Validation evidence",
            "Which evidence areas are complete, attention-needed, or not applicable?",
            "validation_checklist",
            f"tables/governance/validation_checklist.{table_extension}",
            "validation_checklist",
            "Use as the reviewer checklist before model sign-off.",
        ),
        _evidence_row(
            "Traceability",
            "Which artifact answers each common review question?",
            "evidence_traceability_map",
            f"tables/governance/evidence_traceability_map.{table_extension}",
            "evidence_traceability_map",
            "Use to route files to model development, validation, audit, or IT.",
        ),
        _evidence_row(
            "Statistical tests",
            "What formal tests were run and what were their raw outputs?",
            "tests",
            "metadata/statistical_tests.json",
            "",
            "Use with tables/statistical_tests for detailed test interpretation.",
        ),
        _evidence_row(
            "Feature influence",
            "Which features drove model outputs?",
            "feature_importance",
            "model/feature_importance.csv",
            "feature_importance",
            "Use for model rationale, challenger comparison, and policy review.",
        ),
        _evidence_row(
            "Checkpoint execution",
            "Which runtime stages completed, failed, or were skipped?",
            "step_manifest",
            "metadata/step_manifest.json",
            "",
            "Use to debug staged execution and optional diagnostic coverage.",
        ),
        _evidence_row(
            "Runtime debug trace",
            "Which stage consumed time or memory?",
            "run_debug_trace",
            "metadata/run_debug_trace.json",
            "",
            "Use for performance tuning and support investigations.",
        ),
        _evidence_row(
            "Report payload audit",
            "Were report figures capped, downsampled, or skipped for file size?",
            "report_payload_audit",
            f"tables/governance/report_payload_audit.{table_extension}",
            "report_payload_audit",
            "Use when validating why a shared HTML report is smaller than full diagnostics.",
        ),
        _evidence_row(
            "Rerun code",
            "How can the workflow be rerun without the GUI?",
            "runner_script",
            "code/generated_run.py",
            "",
            "Use for controlled reruns, code review, or custom Python modifications.",
        ),
    ]
    return pd.DataFrame(rows, columns=EVIDENCE_TRACEABILITY_COLUMNS)


def _check_row(
    review_area: str,
    *,
    complete: bool = False,
    attention: bool = False,
    not_applicable: bool = False,
    evidence: str,
    action: str,
) -> dict[str, str]:
    if not_applicable:
        status = "Not applicable"
        interpretation = "Info"
    elif attention or not complete:
        status = "Attention needed"
        interpretation = "Watch"
    else:
        status = "Complete"
        interpretation = "Good"
    return {
        "review_area": review_area,
        "status": status,
        "interpretation": interpretation,
        "evidence": evidence or "No specific evidence table was produced.",
        "recommended_action": action,
    }


def _evidence_row(
    evidence_area: str,
    question_answered: str,
    artifact_key: str,
    artifact_location: str,
    table_name: str,
    reviewer_use: str,
) -> dict[str, str]:
    return {
        "evidence_area": evidence_area,
        "question_answered": question_answered,
        "artifact_key": artifact_key,
        "artifact_location": artifact_location,
        "table_name": table_name,
        "reviewer_use": reviewer_use,
    }


def _has_table(tables: Mapping[str, pd.DataFrame], table_name: str) -> bool:
    table = tables.get(table_name)
    return isinstance(table, pd.DataFrame) and not table.empty


def _has_any_table(tables: Mapping[str, pd.DataFrame], table_names: list[str]) -> bool:
    return any(_has_table(tables, table_name) for table_name in table_names)


def _available_tables(tables: Mapping[str, pd.DataFrame], table_names: list[str]) -> str:
    available = [table_name for table_name in table_names if _has_table(tables, table_name)]
    return ", ".join(available)


def _join_evidence(*items: Any) -> str:
    return ", ".join(str(item) for item in items if str(item or "").strip())
