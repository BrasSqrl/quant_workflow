"""Centralized artifact path layout for export steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_pd_framework.config import ArtifactConfig


@dataclass(frozen=True)
class ExportPathLayout:
    output_root: Path
    tables_dir: Path
    figures_dir: Path
    html_dir: Path
    png_dir: Path
    json_dir: Path
    model_path: Path
    metrics_path: Path
    input_snapshot_path: Path
    input_snapshot_parquet_path: Path
    predictions_path: Path
    predictions_parquet_path: Path
    feature_importance_path: Path
    backtest_path: Path
    report_path: Path
    interactive_report_path: Path
    config_path: Path
    tests_path: Path
    workbook_path: Path
    model_summary_path: Path
    manifest_path: Path
    step_manifest_path: Path
    documentation_pack_path: Path
    validation_pack_path: Path
    committee_report_docx_path: Path
    validation_report_docx_path: Path
    committee_report_pdf_path: Path
    validation_report_pdf_path: Path
    reproducibility_manifest_path: Path
    run_debug_trace_path: Path
    template_workbook_path: Path
    runner_script_path: Path
    rerun_readme_path: Path
    code_snapshot_dir: Path


def build_export_path_layout(
    artifacts: ArtifactConfig,
    output_root: Path,
) -> ExportPathLayout:
    """Builds all export paths from config in one auditable place."""

    figures_dir = output_root / artifacts.figures_directory_name
    return ExportPathLayout(
        output_root=output_root,
        tables_dir=output_root / artifacts.tables_directory_name,
        figures_dir=figures_dir,
        html_dir=figures_dir / artifacts.html_directory_name,
        png_dir=figures_dir / artifacts.png_directory_name,
        json_dir=output_root / artifacts.json_directory_name,
        model_path=output_root / artifacts.model_file_name,
        metrics_path=output_root / artifacts.metrics_file_name,
        input_snapshot_path=output_root / artifacts.input_snapshot_file_name,
        input_snapshot_parquet_path=output_root / artifacts.input_snapshot_parquet_file_name,
        predictions_path=output_root / artifacts.predictions_file_name,
        predictions_parquet_path=output_root / artifacts.predictions_parquet_file_name,
        feature_importance_path=output_root / artifacts.feature_importance_file_name,
        backtest_path=output_root / artifacts.backtest_file_name,
        report_path=output_root / artifacts.report_file_name,
        interactive_report_path=output_root / artifacts.interactive_report_file_name,
        config_path=output_root / artifacts.config_file_name,
        tests_path=output_root / artifacts.statistical_tests_file_name,
        workbook_path=output_root / artifacts.workbook_file_name,
        model_summary_path=output_root / artifacts.model_summary_file_name,
        manifest_path=output_root / artifacts.manifest_file_name,
        step_manifest_path=output_root / artifacts.step_manifest_file_name,
        documentation_pack_path=output_root / artifacts.documentation_pack_file_name,
        validation_pack_path=output_root / artifacts.validation_pack_file_name,
        committee_report_docx_path=output_root / artifacts.committee_report_docx_file_name,
        validation_report_docx_path=output_root / artifacts.validation_report_docx_file_name,
        committee_report_pdf_path=output_root / artifacts.committee_report_pdf_file_name,
        validation_report_pdf_path=output_root / artifacts.validation_report_pdf_file_name,
        reproducibility_manifest_path=output_root / artifacts.reproducibility_manifest_file_name,
        run_debug_trace_path=output_root / artifacts.run_debug_trace_file_name,
        template_workbook_path=output_root / artifacts.template_workbook_file_name,
        runner_script_path=output_root / artifacts.runner_script_file_name,
        rerun_readme_path=output_root / artifacts.rerun_readme_file_name,
        code_snapshot_dir=output_root / artifacts.code_snapshot_directory_name,
    )
