"""Centralized artifact path layout for export steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_pd_framework.config import ArtifactConfig


@dataclass(frozen=True)
class ExportPathLayout:
    output_root: Path
    reports_dir: Path
    model_dir: Path
    data_dir: Path
    data_input_dir: Path
    data_predictions_dir: Path
    tables_dir: Path
    figures_dir: Path
    html_dir: Path
    png_dir: Path
    config_dir: Path
    metadata_dir: Path
    workbooks_dir: Path
    code_dir: Path
    start_here_path: Path
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
    decision_summary_path: Path
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

    reports_dir = output_root / "reports"
    model_dir = output_root / "model"
    data_dir = output_root / "data"
    data_input_dir = data_dir / "input"
    data_predictions_dir = data_dir / "predictions"
    tables_dir = output_root / artifacts.tables_directory_name
    figures_dir = output_root / artifacts.figures_directory_name
    config_dir = output_root / "config"
    metadata_dir = output_root / "metadata"
    workbooks_dir = output_root / "workbooks"
    code_dir = output_root / "code"
    return ExportPathLayout(
        output_root=output_root,
        reports_dir=reports_dir,
        model_dir=model_dir,
        data_dir=data_dir,
        data_input_dir=data_input_dir,
        data_predictions_dir=data_predictions_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        html_dir=figures_dir / artifacts.html_directory_name,
        png_dir=figures_dir / artifacts.png_directory_name,
        config_dir=config_dir,
        metadata_dir=metadata_dir,
        workbooks_dir=workbooks_dir,
        code_dir=code_dir,
        start_here_path=output_root / "START_HERE.md",
        model_path=model_dir / artifacts.model_file_name,
        metrics_path=metadata_dir / artifacts.metrics_file_name,
        input_snapshot_path=data_input_dir / artifacts.input_snapshot_file_name,
        input_snapshot_parquet_path=data_input_dir / artifacts.input_snapshot_parquet_file_name,
        predictions_path=data_predictions_dir / artifacts.predictions_file_name,
        predictions_parquet_path=data_predictions_dir / artifacts.predictions_parquet_file_name,
        feature_importance_path=model_dir / artifacts.feature_importance_file_name,
        backtest_path=tables_dir / "backtesting" / artifacts.backtest_file_name,
        report_path=reports_dir / artifacts.report_file_name,
        interactive_report_path=reports_dir / artifacts.interactive_report_file_name,
        config_path=config_dir / artifacts.config_file_name,
        tests_path=metadata_dir / artifacts.statistical_tests_file_name,
        workbook_path=workbooks_dir / artifacts.workbook_file_name,
        model_summary_path=model_dir / artifacts.model_summary_file_name,
        manifest_path=output_root / artifacts.manifest_file_name,
        step_manifest_path=metadata_dir / artifacts.step_manifest_file_name,
        decision_summary_path=reports_dir / artifacts.decision_summary_file_name,
        documentation_pack_path=reports_dir / artifacts.documentation_pack_file_name,
        validation_pack_path=reports_dir / artifacts.validation_pack_file_name,
        committee_report_docx_path=reports_dir / artifacts.committee_report_docx_file_name,
        validation_report_docx_path=reports_dir / artifacts.validation_report_docx_file_name,
        committee_report_pdf_path=reports_dir / artifacts.committee_report_pdf_file_name,
        validation_report_pdf_path=reports_dir / artifacts.validation_report_pdf_file_name,
        reproducibility_manifest_path=metadata_dir / artifacts.reproducibility_manifest_file_name,
        run_debug_trace_path=metadata_dir / artifacts.run_debug_trace_file_name,
        template_workbook_path=config_dir / artifacts.template_workbook_file_name,
        runner_script_path=code_dir / artifacts.runner_script_file_name,
        rerun_readme_path=code_dir / artifacts.rerun_readme_file_name,
        code_snapshot_dir=code_dir / artifacts.code_snapshot_directory_name,
    )
