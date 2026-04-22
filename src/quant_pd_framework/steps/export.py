"""Writes model artifacts, scored output, and a human-readable run report."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import textwrap
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.io as pio

from ..base import BasePipelineStep
from ..config import ExecutionMode
from ..context import PipelineContext
from ..gui_support import (
    build_column_editor_frame_from_schema,
    build_feature_dictionary_frame_from_config,
    build_feature_review_frame_from_config,
    build_scorecard_override_frame_from_config,
    build_template_workbook_bytes,
    build_transformation_frame_from_config,
)
from ..presentation import build_interactive_report_html
from ..reporting import build_regulatory_report_bundle


class ArtifactExportStep(BasePipelineStep):
    """
    Persists the outputs a quant practitioner usually needs after a model run.

    The exported bundle includes the trained model, scored observations, metrics,
    feature importance, backtesting output, diagnostics, and a rerun-ready code bundle.
    """

    name = "artifact_export"
    MONITORING_BUNDLE_DIRECTORY_NAME = "model_bundle_for_monitoring"
    MONITORING_METADATA_FILE_NAME = "monitoring_metadata.json"
    MONITORING_BUNDLE_VERSION = "1.0"

    def run(self, context: PipelineContext) -> PipelineContext:
        output_root = context.config.artifacts.output_root / context.run_id
        output_root.mkdir(parents=True, exist_ok=True)

        tables_dir = output_root / context.config.artifacts.tables_directory_name
        figures_dir = output_root / context.config.artifacts.figures_directory_name
        html_dir = figures_dir / context.config.artifacts.html_directory_name
        png_dir = figures_dir / context.config.artifacts.png_directory_name
        json_dir = output_root / context.config.artifacts.json_directory_name
        tables_dir.mkdir(parents=True, exist_ok=True)
        html_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        if context.config.diagnostics.static_image_exports:
            png_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_root / context.config.artifacts.model_file_name
        metrics_path = output_root / context.config.artifacts.metrics_file_name
        input_snapshot_path = output_root / context.config.artifacts.input_snapshot_file_name
        predictions_path = output_root / context.config.artifacts.predictions_file_name
        feature_importance_path = (
            output_root / context.config.artifacts.feature_importance_file_name
        )
        backtest_path = output_root / context.config.artifacts.backtest_file_name
        report_path = output_root / context.config.artifacts.report_file_name
        interactive_report_path = (
            output_root / context.config.artifacts.interactive_report_file_name
        )
        config_path = output_root / context.config.artifacts.config_file_name
        tests_path = output_root / context.config.artifacts.statistical_tests_file_name
        workbook_path = output_root / context.config.artifacts.workbook_file_name
        model_summary_path = output_root / context.config.artifacts.model_summary_file_name
        manifest_path = output_root / context.config.artifacts.manifest_file_name
        step_manifest_path = output_root / context.config.artifacts.step_manifest_file_name
        documentation_pack_path = (
            output_root / context.config.artifacts.documentation_pack_file_name
        )
        validation_pack_path = output_root / context.config.artifacts.validation_pack_file_name
        committee_report_docx_path = (
            output_root / context.config.artifacts.committee_report_docx_file_name
        )
        validation_report_docx_path = (
            output_root / context.config.artifacts.validation_report_docx_file_name
        )
        committee_report_pdf_path = (
            output_root / context.config.artifacts.committee_report_pdf_file_name
        )
        validation_report_pdf_path = (
            output_root / context.config.artifacts.validation_report_pdf_file_name
        )
        reproducibility_manifest_path = (
            output_root / context.config.artifacts.reproducibility_manifest_file_name
        )
        template_workbook_path = (
            output_root / context.config.artifacts.template_workbook_file_name
        )
        runner_script_path = output_root / context.config.artifacts.runner_script_file_name
        rerun_readme_path = output_root / context.config.artifacts.rerun_readme_file_name
        code_snapshot_dir = output_root / context.config.artifacts.code_snapshot_directory_name

        if context.config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            return self._run_subset_search_export(
                context=context,
                output_root=output_root,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                html_dir=html_dir,
                png_dir=png_dir,
                json_dir=json_dir,
                metrics_path=metrics_path,
                input_snapshot_path=input_snapshot_path,
                report_path=report_path,
                interactive_report_path=interactive_report_path,
                config_path=config_path,
                tests_path=tests_path,
                manifest_path=manifest_path,
                step_manifest_path=step_manifest_path,
            )

        if context.model is None:
            raise ValueError("Cannot export artifacts without a fitted model.")
        if (
            context.feature_importance is None
            or context.backtest_summary is None
            or not context.diagnostics_tables
        ):
            raise ValueError("Cannot export artifacts before diagnostics finish.")

        joblib.dump(context.model, model_path)
        self._write_json(metrics_path, context.metrics)
        self._write_json(config_path, self._build_export_config_payload(context, model_path))
        self._write_json(tests_path, context.statistical_tests)

        if context.config.artifacts.export_input_snapshot and context.raw_data is not None:
            context.raw_data.to_csv(input_snapshot_path, index=False)

        predictions = pd.concat(context.predictions.values(), ignore_index=True)
        predictions.to_csv(predictions_path, index=False)
        context.feature_importance.to_csv(feature_importance_path, index=False)
        context.backtest_summary.to_csv(backtest_path, index=False)
        for split_name, split_frame in context.predictions.items():
            split_frame.to_csv(tables_dir / f"predictions_{split_name}.csv", index=False)
        for table_name, table in context.diagnostics_tables.items():
            table.to_csv(tables_dir / f"{self._sanitize_name(table_name)}.csv", index=False)

        visualization_manifest = self._export_visualizations(context, html_dir, png_dir)
        if context.config.diagnostics.export_excel_workbook:
            self._export_excel_workbook(context, workbook_path, predictions)
        if context.model_summary is not None:
            if isinstance(context.model_summary, pd.DataFrame):
                context.model_summary.to_csv(model_summary_path.with_suffix(".csv"), index=False)
            else:
                model_summary_path.write_text(str(context.model_summary), encoding="utf-8")

        report_path.write_text(self._build_report(context), encoding="utf-8")
        documentation_pack_path.write_text(
            self._build_documentation_pack(context),
            encoding="utf-8",
        )
        validation_pack_path.write_text(
            self._build_validation_pack(context),
            encoding="utf-8",
        )
        regulatory_report_manifest = self._export_regulatory_reports(
            context=context,
            committee_report_docx_path=committee_report_docx_path,
            validation_report_docx_path=validation_report_docx_path,
            committee_report_pdf_path=committee_report_pdf_path,
            validation_report_pdf_path=validation_report_pdf_path,
        )
        reproducibility_manifest = self._build_reproducibility_manifest(
            context=context,
            model_path=model_path,
            config_path=config_path,
            input_snapshot_path=input_snapshot_path if input_snapshot_path.exists() else None,
        )
        self._write_json(reproducibility_manifest_path, reproducibility_manifest)
        context.diagnostics_tables["reproducibility_manifest"] = pd.DataFrame(
            reproducibility_manifest["rows"]
        )
        interactive_report_path.write_text(
            self._build_interactive_report(context),
            encoding="utf-8",
        )
        template_workbook_path.write_bytes(self._build_template_workbook(context))
        step_manifest = self._build_step_manifest(context)
        self._write_json(step_manifest_path, step_manifest)
        manifest = {
            "core_artifacts": {
                "output_root": str(output_root),
                "model": str(model_path),
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
                "feature_importance": str(feature_importance_path),
                "backtest": str(backtest_path),
                "report": str(report_path),
                "config": str(config_path),
                "tests": str(tests_path),
                "step_manifest": str(step_manifest_path),
                "artifact_manifest": str(manifest_path),
            },
            "directories": {
                "tables": str(tables_dir),
                "figures": str(figures_dir),
                "figures_html": str(html_dir),
                "json": str(json_dir),
            },
            **visualization_manifest,
            "interactive_report": str(interactive_report_path),
            "documentation_pack": str(documentation_pack_path),
            "validation_pack": str(validation_pack_path),
            "regulatory_reports": regulatory_report_manifest,
            "reproducibility_manifest": str(reproducibility_manifest_path),
            "configuration_template": str(template_workbook_path),
            "rerun_bundle": {
                "step_manifest": str(step_manifest_path),
                "runner_script": str(runner_script_path),
                "rerun_readme": str(rerun_readme_path),
            },
        }
        if context.config.diagnostics.export_excel_workbook:
            manifest["core_artifacts"]["analysis_workbook"] = str(workbook_path)
        if context.config.artifacts.export_input_snapshot and input_snapshot_path.exists():
            manifest["rerun_bundle"]["input_snapshot"] = str(input_snapshot_path)
        self._write_json(manifest_path, manifest)

        runner_script_path.write_text(
            self._build_generated_runner_script(context), encoding="utf-8"
        )
        rerun_readme_path.write_text(self._build_rerun_readme(context), encoding="utf-8")

        if context.config.artifacts.export_code_snapshot:
            self._export_code_snapshot(code_snapshot_dir)
            manifest["rerun_bundle"]["code_snapshot"] = str(code_snapshot_dir)
            self._write_json(manifest_path, manifest)

        monitoring_bundle: dict[str, Any] | None = None
        if context.config.execution.mode == ExecutionMode.FIT_NEW_MODEL:
            monitoring_bundle = self._export_monitoring_bundle(
                context=context,
                output_root=output_root,
                model_path=model_path,
                config_path=config_path,
                runner_script_path=runner_script_path,
                manifest_path=manifest_path,
                input_snapshot_path=input_snapshot_path if input_snapshot_path.exists() else None,
                predictions_path=predictions_path,
                code_snapshot_dir=code_snapshot_dir
                if context.config.artifacts.export_code_snapshot
                else None,
            )
            manifest["monitoring_bundle"] = {
                "directory": str(monitoring_bundle["bundle_dir"]),
                "metadata": str(monitoring_bundle["metadata_path"]),
                "bundle_version": monitoring_bundle["metadata"]["bundle_version"],
            }
            self._write_json(manifest_path, manifest)
            self._refresh_monitoring_bundle_manifest_copy(
                bundle_dir=monitoring_bundle["bundle_dir"],
                manifest_path=manifest_path,
            )

        context.artifacts = {
            "output_root": output_root,
            "model": model_path,
            "metrics": metrics_path,
            "input_snapshot": input_snapshot_path if input_snapshot_path.exists() else None,
            "predictions": predictions_path,
            "feature_importance": feature_importance_path,
            "backtest": backtest_path,
            "report": report_path,
            "documentation_pack": documentation_pack_path,
            "validation_pack": validation_pack_path,
            "committee_report_docx": (
                committee_report_docx_path if committee_report_docx_path.exists() else None
            ),
            "validation_report_docx": (
                validation_report_docx_path if validation_report_docx_path.exists() else None
            ),
            "committee_report_pdf": (
                committee_report_pdf_path if committee_report_pdf_path.exists() else None
            ),
            "validation_report_pdf": (
                validation_report_pdf_path if validation_report_pdf_path.exists() else None
            ),
            "interactive_report": interactive_report_path,
            "config": config_path,
            "tests": tests_path,
            "reproducibility_manifest": reproducibility_manifest_path,
            "configuration_template": template_workbook_path,
            "tables_dir": tables_dir,
            "figures_dir": figures_dir,
            "workbook": workbook_path if context.config.diagnostics.export_excel_workbook else None,
            "manifest": manifest_path,
            "step_manifest": step_manifest_path,
            "runner_script": runner_script_path,
            "rerun_readme": rerun_readme_path,
            "code_snapshot_dir": code_snapshot_dir
            if context.config.artifacts.export_code_snapshot
            else None,
            "monitoring_bundle_dir": (
                monitoring_bundle["bundle_dir"] if monitoring_bundle is not None else None
            ),
            "monitoring_metadata": (
                monitoring_bundle["metadata_path"] if monitoring_bundle is not None else None
            ),
        }
        return context

    def _run_subset_search_export(
        self,
        *,
        context: PipelineContext,
        output_root: Path,
        tables_dir: Path,
        figures_dir: Path,
        html_dir: Path,
        png_dir: Path,
        json_dir: Path,
        metrics_path: Path,
        input_snapshot_path: Path,
        report_path: Path,
        interactive_report_path: Path,
        config_path: Path,
        tests_path: Path,
        manifest_path: Path,
        step_manifest_path: Path,
    ) -> PipelineContext:
        if not context.diagnostics_tables:
            raise ValueError("Feature subset search requires comparison tables before export.")

        self._write_json(metrics_path, context.metrics)
        self._write_json(config_path, context.config.to_dict())
        self._write_json(tests_path, context.statistical_tests)
        if context.config.artifacts.export_input_snapshot and context.raw_data is not None:
            context.raw_data.to_csv(input_snapshot_path, index=False)

        for table_name, table in context.diagnostics_tables.items():
            table.to_csv(tables_dir / f"{self._sanitize_name(table_name)}.csv", index=False)

        visualization_manifest = self._export_visualizations(context, html_dir, png_dir)
        report_path.write_text(self._build_subset_search_report(context), encoding="utf-8")
        interactive_report_path.write_text(
            self._build_interactive_report(context),
            encoding="utf-8",
        )
        step_manifest = self._build_step_manifest(context)
        self._write_json(step_manifest_path, step_manifest)

        manifest = {
            "core_artifacts": {
                "output_root": str(output_root),
                "metrics": str(metrics_path),
                "report": str(report_path),
                "interactive_report": str(interactive_report_path),
                "config": str(config_path),
                "tests": str(tests_path),
                "step_manifest": str(step_manifest_path),
                "artifact_manifest": str(manifest_path),
            },
            "directories": {
                "tables": str(tables_dir),
                "figures": str(figures_dir),
                "figures_html": str(html_dir),
                "json": str(json_dir),
            },
            **visualization_manifest,
        }
        if context.config.artifacts.export_input_snapshot and input_snapshot_path.exists():
            manifest["core_artifacts"]["input_snapshot"] = str(input_snapshot_path)
        self._write_json(manifest_path, manifest)

        context.artifacts = {
            "output_root": output_root,
            "metrics": metrics_path,
            "input_snapshot": input_snapshot_path if input_snapshot_path.exists() else None,
            "report": report_path,
            "interactive_report": interactive_report_path,
            "config": config_path,
            "tests": tests_path,
            "tables_dir": tables_dir,
            "figures_dir": figures_dir,
            "manifest": manifest_path,
            "step_manifest": step_manifest_path,
        }
        return context

    def _export_regulatory_reports(
        self,
        *,
        context: PipelineContext,
        committee_report_docx_path: Path,
        validation_report_docx_path: Path,
        committee_report_pdf_path: Path,
        validation_report_pdf_path: Path,
    ) -> dict[str, str]:
        if not context.config.regulatory_reporting.enabled:
            return {}

        report_bundle = build_regulatory_report_bundle(context)
        manifest: dict[str, str] = {}
        committee_report = report_bundle["committee"]
        validation_report = report_bundle["validation"]

        if committee_report.docx_bytes is not None:
            committee_report_docx_path.write_bytes(committee_report.docx_bytes)
            manifest["committee_docx"] = str(committee_report_docx_path)
        if validation_report.docx_bytes is not None:
            validation_report_docx_path.write_bytes(validation_report.docx_bytes)
            manifest["validation_docx"] = str(validation_report_docx_path)
        if committee_report.pdf_bytes is not None:
            committee_report_pdf_path.write_bytes(committee_report.pdf_bytes)
            manifest["committee_pdf"] = str(committee_report_pdf_path)
        if validation_report.pdf_bytes is not None:
            validation_report_pdf_path.write_bytes(validation_report.pdf_bytes)
            manifest["validation_pdf"] = str(validation_report_pdf_path)
        return manifest

    def _export_monitoring_bundle(
        self,
        *,
        context: PipelineContext,
        output_root: Path,
        model_path: Path,
        config_path: Path,
        runner_script_path: Path,
        manifest_path: Path,
        input_snapshot_path: Path | None,
        predictions_path: Path,
        code_snapshot_dir: Path | None,
    ) -> dict[str, Any]:
        bundle_dir = output_root / self.MONITORING_BUNDLE_DIRECTORY_NAME
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        bundled_paths: dict[str, str | None] = {}
        missing_optional_artifacts: list[str] = []

        bundled_paths["quant_model.joblib"] = self._copy_file_into_bundle(
            source_path=model_path,
            destination_path=bundle_dir / model_path.name,
        )
        bundled_paths["run_config.json"] = self._copy_file_into_bundle(
            source_path=config_path,
            destination_path=bundle_dir / config_path.name,
        )
        bundled_paths["generated_run.py"] = self._copy_file_into_bundle(
            source_path=runner_script_path,
            destination_path=bundle_dir / runner_script_path.name,
        )
        bundled_paths["artifact_manifest.json"] = self._copy_file_into_bundle(
            source_path=manifest_path,
            destination_path=bundle_dir / manifest_path.name,
        )
        bundled_paths["predictions.csv"] = self._copy_file_into_bundle(
            source_path=predictions_path,
            destination_path=bundle_dir / predictions_path.name,
        )

        if input_snapshot_path is not None and input_snapshot_path.exists():
            bundled_paths["input_snapshot.csv"] = self._copy_file_into_bundle(
                source_path=input_snapshot_path,
                destination_path=bundle_dir / input_snapshot_path.name,
            )
        else:
            bundled_paths["input_snapshot.csv"] = None
            missing_optional_artifacts.append("input_snapshot.csv")

        if code_snapshot_dir is not None and code_snapshot_dir.exists():
            bundled_paths["code_snapshot"] = self._copy_directory_into_bundle(
                source_path=code_snapshot_dir,
                destination_path=bundle_dir / code_snapshot_dir.name,
            )
        else:
            bundled_paths["code_snapshot"] = None
            missing_optional_artifacts.append("code_snapshot")

        metadata_path = bundle_dir / self.MONITORING_METADATA_FILE_NAME
        bundled_paths[self.MONITORING_METADATA_FILE_NAME] = metadata_path.name
        monitoring_metadata = self._build_monitoring_metadata(
            context=context,
            bundled_paths=bundled_paths,
            missing_optional_artifacts=missing_optional_artifacts,
        )
        self._write_json(metadata_path, monitoring_metadata)

        return {
            "bundle_dir": bundle_dir,
            "metadata_path": metadata_path,
            "metadata": monitoring_metadata,
        }

    def _build_monitoring_metadata(
        self,
        *,
        context: PipelineContext,
        bundled_paths: dict[str, str | None],
        missing_optional_artifacts: list[str],
    ) -> dict[str, Any]:
        split_config = context.config.split
        if context.config.target.mode.value == "binary":
            score_column = "predicted_probability"
            prediction_column = "predicted_class"
            threshold = context.config.model.threshold
        else:
            score_column = "predicted_value"
            prediction_column = "predicted_value"
            threshold = None

        segment_columns = [
            column_name
            for column_name in [context.config.diagnostics.default_segment_column]
            if column_name
        ]

        return {
            "bundle_type": "quant_studio_model_bundle_for_monitoring",
            "bundle_version": self.MONITORING_BUNDLE_VERSION,
            "created_by_run_id": context.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_type": context.config.model.model_type.value,
            "target_mode": context.config.target.mode.value,
            "target_column": context.target_column,
            "score_column": score_column,
            "prediction_column": prediction_column,
            "threshold": threshold,
            "calibration_method": context.metadata.get("recommended_calibration_method"),
            "selected_features": list(context.feature_columns),
            "date_column": split_config.date_column,
            "entity_column": split_config.entity_column,
            "segment_columns": segment_columns,
            "bundled_artifacts": bundled_paths,
            "missing_optional_artifacts": missing_optional_artifacts,
        }

    def _copy_file_into_bundle(self, *, source_path: Path, destination_path: Path) -> str:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return destination_path.name

    def _copy_directory_into_bundle(self, *, source_path: Path, destination_path: Path) -> str:
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        return destination_path.name

    def _refresh_monitoring_bundle_manifest_copy(
        self,
        *,
        bundle_dir: Path,
        manifest_path: Path,
    ) -> None:
        destination_path = bundle_dir / manifest_path.name
        if destination_path.exists():
            destination_path.unlink()
        shutil.copy2(manifest_path, destination_path)

    def _write_json(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    def _build_report(self, context: PipelineContext) -> str:
        feature_summary = context.metadata.get("feature_summary", {})
        imputation_summary = context.metadata.get("imputation_summary", {})
        split_summary = context.metadata.get("split_summary", {})
        transformation_summary = context.metadata.get("transformation_summary", {})
        assumption_summary = context.metadata.get("assumption_check_summary", {})
        guardrail_summary = context.metadata.get("workflow_guardrail_summary", {})

        lines = [
            "# Quantitative Model Run Report",
            "",
            f"- Run ID: `{context.run_id}`",
            (
                f"- Preset: "
                f"`{context.config.preset_name.value if context.config.preset_name else 'custom'}`"
            ),
            f"- Input rows: `{context.metadata.get('input_shape', {}).get('rows', 'n/a')}`",
            f"- Input columns: `{context.metadata.get('input_shape', {}).get('columns', 'n/a')}`",
            f"- Feature count: `{feature_summary.get('feature_count', 0)}`",
            f"- Numeric features: `{feature_summary.get('numeric_feature_count', 0)}`",
            f"- Categorical features: `{feature_summary.get('categorical_feature_count', 0)}`",
            f"- Features with imputation rules: `{imputation_summary.get('feature_count', 0)}`",
            f"- Governed transformations: `{transformation_summary.get('count', 0)}`",
            f"- Assumption-check failures: `{assumption_summary.get('fail_count', 0)}`",
            f"- Workflow guardrail warnings: `{guardrail_summary.get('warning_count', 0)}`",
            f"- Execution mode: `{context.config.execution.mode.value}`",
            f"- Labels available: `{bool(context.metadata.get('labels_available', False))}`",
            f"- Model type: `{context.config.model.model_type.value}`",
            f"- Target mode: `{context.config.target.mode.value}`",
            "",
            "## Split Sizes",
            "",
        ]

        for split_name, summary in split_summary.items():
            lines.append(f"- {split_name.title()}: `{summary['rows']}` rows")

        lines.extend(["", "## Metrics", ""])
        for split_name, metrics in context.metrics.items():
            lines.append(f"### {split_name.title()}")
            lines.append("")
            for metric_name, metric_value in metrics.items():
                lines.append(f"- {metric_name}: `{metric_value}`")
            lines.append("")

        if context.comparison_results is not None:
            lines.extend(["## Model Comparison", ""])
            recommended_model = context.metadata.get("comparison_recommended_model")
            if recommended_model:
                lines.append(f"- Recommended model: `{recommended_model}`")
            lines.append(
                f"- Challenger rows exported: `{len(context.comparison_results)}`"
            )
            lines.append("")

        recommended_calibration_method = context.metadata.get("recommended_calibration_method")
        if recommended_calibration_method:
            recommended_score_column = context.metadata.get(
                "recommended_calibration_score_column",
                "predicted_probability",
            )
            lines.extend(["## Calibration", ""])
            lines.append(
                f"- Recommended method: `{recommended_calibration_method}`"
            )
            lines.append(
                f"- Ranking metric: `{context.metadata.get('calibration_ranking_metric', 'n/a')}`"
            )
            lines.append(f"- Recommended score column: `{recommended_score_column}`")
            lines.append("")

        lines.extend(["## Diagnostics Tables", ""])
        for table_name in sorted(context.diagnostics_tables):
            lines.append(f"- {table_name}")
        lines.append("")

        if context.statistical_tests:
            lines.extend(["## Statistical Tests", ""])
            for test_name, payload in context.statistical_tests.items():
                if isinstance(payload, list):
                    lines.append(f"- {test_name}: `{len(payload)}` rows")
                else:
                    lines.append(f"- {test_name}")
            lines.append("")

        step_manifest = context.metadata.get("step_manifest", [])
        if step_manifest:
            lines.extend(["## Step Stack", ""])
            for step in step_manifest:
                lines.append(
                    f"- {step['order']}. `{step['name']}` via "
                    f"`{step['module']}.{step['class_name']}`"
                )
            lines.append("")

        lines.extend(
            [
                "## Rerun Bundle",
                "",
                "- `generated_run.py` replays the run bundle outside the GUI.",
                "- `run_config.json` stores the resolved config for the run.",
                "- `input_snapshot.csv` stores the ingested dataset when input export is enabled.",
                "- `step_manifest.json` stores the exact ordered pipeline step stack.",
                (
                    "- `reproducibility_manifest.json` captures hashes, versions, "
                    "and environment metadata."
                ),
                (
                    "- `configuration_template.xlsx` exports the review workbook "
                    "used for offline edits."
                ),
                "- `committee_report.docx` and `committee_report.pdf` provide "
                "committee-ready packaging when regulatory reporting is enabled.",
                "- `validation_report.docx` and `validation_report.pdf` provide "
                "validator-ready packaging when regulatory reporting is enabled.",
                "- `code_snapshot/` stores a Python copy of the framework, GUI, "
                "tests, and examples for editing.",
                "",
            ]
        )

        if context.warnings:
            lines.extend(["## Warnings", ""])
            for warning in context.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.extend(["## Pipeline Events", ""])
        for event in context.events:
            lines.append(f"- {event}")

        return "\n".join(lines) + "\n"

    def _build_subset_search_report(self, context: PipelineContext) -> str:
        summary = context.metrics.get("subset_search", {})
        best_candidate = context.metadata.get("subset_search_best_candidate", {})
        selected_coefficients = context.diagnostics_tables.get(
            "subset_search_selected_coefficients",
            pd.DataFrame(),
        )
        nonwinning_candidates = context.diagnostics_tables.get(
            "subset_search_nonwinning_candidates",
            pd.DataFrame(),
        )
        lines = [
            "# Feature Subset Search Report",
            "",
            f"- Run ID: `{context.run_id}`",
            f"- Execution mode: `{context.config.execution.mode.value}`",
            f"- Model family: `{context.config.model.model_type.value}`",
            f"- Ranking split: `{context.config.subset_search.ranking_split}`",
            f"- Ranking metric: `{context.config.subset_search.ranking_metric}`",
            f"- Candidate feature count: `{summary.get('candidate_feature_count', 'n/a')}`",
            f"- Enumerated subsets: `{summary.get('enumerated_subsets', 'n/a')}`",
            f"- Successful subsets: `{summary.get('successful_subsets', 'n/a')}`",
            f"- Failed subsets: `{summary.get('failed_subsets', 'n/a')}`",
            "",
            "## Best Candidate",
            "",
            f"- Candidate ID: `{best_candidate.get('candidate_id', 'n/a')}`",
            f"- Feature count: `{best_candidate.get('feature_count', 'n/a')}`",
            f"- Feature set: `{best_candidate.get('feature_set', 'n/a')}`",
            f"- Validation ROC AUC: `{best_candidate.get('ranking_roc_auc', 'n/a')}`",
            f"- Validation KS: `{best_candidate.get('ranking_ks_statistic', 'n/a')}`",
            f"- Test ROC AUC: `{best_candidate.get('test_roc_auc', 'n/a')}`",
            f"- Test KS: `{best_candidate.get('test_ks_statistic', 'n/a')}`",
            "",
        ]
        if not selected_coefficients.empty:
            lines.extend(
                [
                    "## Selected Candidate Coefficients",
                    "",
                ]
            )
            for _, row in selected_coefficients.head(15).iterrows():
                magnitude = row.get("abs_coefficient", row.get("importance_value", "n/a"))
                lines.append(
                    f"- `{row.get('feature_name', 'n/a')}`: "
                    f"`{row.get('coefficient', row.get('importance_value', 'n/a'))}` "
                    f"(magnitude `{magnitude}`)"
                )
            lines.append("")
        if not nonwinning_candidates.empty:
            lines.extend(
                [
                    "## Ranked Non-Winning Candidates",
                    "",
                    f"- Exported non-winning candidates: `{len(nonwinning_candidates)}`",
                    "- Review `subset_search_nonwinning_candidates.csv` for the full ranked table.",
                    "",
                ]
            )
        lines.extend(
            [
            "## Comparison Assets",
            "",
            ]
        )
        for table_name in sorted(context.diagnostics_tables):
            lines.append(f"- Table: `{table_name}`")
        for figure_name in sorted(context.visualizations):
            lines.append(f"- Figure: `{figure_name}`")
        lines.extend(["", "## Pipeline Events", ""])
        for event in context.events:
            lines.append(f"- {event}")
        if context.warnings:
            lines.extend(["", "## Warnings", ""])
            for warning in context.warnings:
                lines.append(f"- {warning}")
        lines.append("")
        return "\n".join(lines)

    def _build_documentation_pack(self, context: PipelineContext) -> str:
        documentation = context.config.documentation
        feature_dictionary = context.diagnostics_tables.get("feature_dictionary", pd.DataFrame())
        variable_selection = context.diagnostics_tables.get("variable_selection", pd.DataFrame())
        calibration_summary = context.diagnostics_tables.get("calibration_summary", pd.DataFrame())
        comparison_table = context.comparison_results
        guardrail_table = context.diagnostics_tables.get("workflow_guardrails", pd.DataFrame())
        lines = [
            f"# {documentation.model_name}",
            "",
            "## Development Summary",
            "",
            f"- Run ID: `{context.run_id}`",
            f"- Model type: `{context.config.model.model_type.value}`",
            f"- Target mode: `{context.config.target.mode.value}`",
            f"- Execution mode: `{context.config.execution.mode.value}`",
            f"- Model owner: `{documentation.model_owner or 'n/a'}`",
            f"- Portfolio: `{documentation.portfolio_name or 'n/a'}`",
            f"- Segment: `{documentation.segment_name or 'n/a'}`",
            "",
            "## Purpose",
            "",
            documentation.business_purpose or "Not provided.",
            "",
            "## Target And Horizon",
            "",
            f"- Target definition: {documentation.target_definition or 'Not provided.'}",
            f"- Horizon definition: {documentation.horizon_definition or 'Not provided.'}",
            f"- Loss definition: {documentation.loss_definition or 'Not provided.'}",
            "",
            "## Data And Features",
            "",
            f"- Input rows: `{context.metadata.get('input_shape', {}).get('rows', 'n/a')}`",
            f"- Input columns: `{context.metadata.get('input_shape', {}).get('columns', 'n/a')}`",
            f"- Selected feature count: `{len(context.feature_columns)}`",
            "",
        ]

        if not feature_dictionary.empty:
            documented_count = int(feature_dictionary["documented"].fillna(False).sum())
            lines.extend(
                [
                    "### Feature Dictionary Coverage",
                    "",
                    f"- Documented modeled features: `{documented_count}`",
                    f"- Dictionary rows exported: `{len(feature_dictionary)}`",
                    "",
                ]
            )

        if not guardrail_table.empty:
            warning_count = int((guardrail_table["severity"] == "warning").sum())
            lines.extend(
                [
                    "### Workflow Guardrails",
                    "",
                    f"- Guardrail rows exported: `{len(guardrail_table)}`",
                    f"- Guardrail warnings: `{warning_count}`",
                    "",
                ]
            )

        if not variable_selection.empty:
            selected_rows = variable_selection.loc[variable_selection["selected"]].copy(deep=True)
            lines.extend(["### Variable Selection", ""])
            for _, row in selected_rows.head(25).iterrows():
                lines.append(
                    f"- `{row['feature_name']}` | score `{row['univariate_score']}` | "
                    f"{row['selection_reason']}"
                )
            lines.append("")

        lines.extend(["## Performance Summary", ""])
        for split_name, metrics in context.metrics.items():
            lines.append(f"### {split_name.title()}")
            lines.append("")
            for metric_name, metric_value in metrics.items():
                lines.append(f"- {metric_name}: `{metric_value}`")
            lines.append("")

        if comparison_table is not None:
            lines.extend(["## Challenger Review", ""])
            recommended_model = context.metadata.get("comparison_recommended_model", "n/a")
            lines.append(
                f"- Recommended model: `{recommended_model}`"
            )
            lines.append(
                f"- Ranking split: `{context.config.comparison.ranking_split}`"
            )
            lines.append("")

        if not calibration_summary.empty:
            lines.extend(["## Calibration Review", ""])
            recommended_score_column = context.metadata.get(
                "recommended_calibration_score_column",
                "predicted_probability",
            )
            lines.append(
                f"- Recommended calibration method: "
                f"`{context.metadata.get('recommended_calibration_method', 'n/a')}`"
            )
            lines.append(
                f"- Recommended score column: `{recommended_score_column}`"
            )
            lines.append("")

        lines.extend(["## Assumptions", ""])
        if documentation.assumptions:
            for item in documentation.assumptions:
                lines.append(f"- {item}")
        else:
            lines.append("- None recorded.")
        lines.append("")

        lines.extend(["## Exclusions", ""])
        if documentation.exclusions:
            for item in documentation.exclusions:
                lines.append(f"- {item}")
        else:
            lines.append("- None recorded.")
        lines.append("")

        lines.extend(["## Limitations", ""])
        if documentation.limitations:
            for item in documentation.limitations:
                lines.append(f"- {item}")
        else:
            lines.append("- None recorded.")
        lines.append("")

        lines.extend(["## Reviewer Notes", ""])
        lines.append(documentation.reviewer_notes or "Not provided.")
        lines.append("")

        if context.warnings:
            lines.extend(["## Warnings", ""])
            for warning in context.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.extend(["## Audit Assets", ""])
        lines.append("- `validation_pack.md` provides the validator-facing summary.")
        lines.append(
            "- `committee_report.docx` / `committee_report.pdf` provide "
            "committee-ready distribution assets."
        )
        lines.append(
            "- `validation_report.docx` / `validation_report.pdf` provide "
            "validator-ready distribution assets."
        )
        lines.append(
            "- `reproducibility_manifest.json` provides hashes, package versions, "
            "and run fingerprint metadata."
        )
        lines.append(
            "- `configuration_template.xlsx` exports the editable review workbook "
            "for offline governance."
        )
        lines.append("")

        return "\n".join(lines) + "\n"

    def _build_validation_pack(self, context: PipelineContext) -> str:
        documentation = context.config.documentation
        feature_dictionary = context.diagnostics_tables.get("feature_dictionary", pd.DataFrame())
        assumption_checks = context.diagnostics_tables.get("assumption_checks", pd.DataFrame())
        workflow_guardrails = context.diagnostics_tables.get("workflow_guardrails", pd.DataFrame())
        variable_selection = context.diagnostics_tables.get("variable_selection", pd.DataFrame())
        manual_review = context.diagnostics_tables.get(
            "manual_review_feature_decisions",
            pd.DataFrame(),
        )
        transformation_table = context.diagnostics_tables.get(
            "governed_transformations",
            pd.DataFrame(),
        )
        scorecard_overrides = context.diagnostics_tables.get(
            "scorecard_bin_overrides",
            pd.DataFrame(),
        )
        rows = [
            f"# Validation Pack: {documentation.model_name}",
            "",
            "## Run Overview",
            "",
            f"- Run ID: `{context.run_id}`",
            f"- Execution mode: `{context.config.execution.mode.value}`",
            f"- Model type: `{context.config.model.model_type.value}`",
            f"- Target mode: `{context.config.target.mode.value}`",
            f"- Model owner: `{documentation.model_owner or 'n/a'}`",
            "",
            "## Purpose And Scope",
            "",
            documentation.business_purpose or "Not provided.",
            "",
            "## Data Contract Summary",
            "",
            f"- Input rows: `{context.metadata.get('input_shape', {}).get('rows', 'n/a')}`",
            f"- Input columns: `{context.metadata.get('input_shape', {}).get('columns', 'n/a')}`",
            f"- Final feature count: `{len(context.feature_columns)}`",
            f"- Numeric features: `{len(context.numeric_features)}`",
            f"- Categorical features: `{len(context.categorical_features)}`",
            "",
        ]

        if not feature_dictionary.empty:
            documented_count = int(feature_dictionary["documented"].fillna(False).sum())
            rows.extend(
                [
                    "## Feature Dictionary Coverage",
                    "",
                    f"- Documented modeled features: `{documented_count}`",
                    f"- Modeled features in dictionary table: `{len(feature_dictionary)}`",
                    "",
                ]
            )

        if not transformation_table.empty:
            rows.extend(["## Governed Transformations", ""])
            for _, row in transformation_table.iterrows():
                rows.append(
                    f"- `{row['output_feature']}` via `{row['transform_type']}` from "
                    f"`{row['source_feature']}`"
                )
            rows.append("")

        if not assumption_checks.empty:
            fail_count = int((assumption_checks["status"] == "fail").sum())
            warn_count = int((assumption_checks["status"] == "warn").sum())
            rows.extend(
                [
                    "## Suitability Checks",
                    "",
                    f"- Failed checks: `{fail_count}`",
                    f"- Warning checks: `{warn_count}`",
                    "",
                ]
            )

        if not workflow_guardrails.empty:
            error_count = int((workflow_guardrails["severity"] == "error").sum())
            warning_count = int((workflow_guardrails["severity"] == "warning").sum())
            rows.extend(
                [
                    "## Workflow Guardrails",
                    "",
                    f"- Blocking findings: `{error_count}`",
                    f"- Warning findings: `{warning_count}`",
                    "",
                ]
            )

        if not variable_selection.empty:
            selected_count = int(variable_selection["selected"].fillna(False).sum())
            rows.extend(
                [
                    "## Variable Selection And Review",
                    "",
                    f"- Selected features after screening and review: `{selected_count}`",
                    "",
                ]
            )
        if not manual_review.empty:
            rows.append(f"- Manual feature review decisions: `{len(manual_review)}`")
            rows.append("")
        if not scorecard_overrides.empty:
            rows.append(f"- Scorecard bin overrides: `{len(scorecard_overrides)}`")
            rows.append("")

        rows.extend(["## Performance Snapshot", ""])
        for split_name, metrics in context.metrics.items():
            rows.append(f"### {split_name.title()}")
            rows.append("")
            for metric_name, metric_value in metrics.items():
                rows.append(f"- {metric_name}: `{metric_value}`")
            rows.append("")

        if context.comparison_results is not None:
            rows.extend(["## Challenger Review", ""])
            rows.append(
                f"- Recommended model: "
                f"`{context.metadata.get('comparison_recommended_model', 'n/a')}`"
            )
            rows.append("")

        if context.scenario_results:
            rows.extend(["## Scenario Testing", ""])
            for table_name, table in context.scenario_results.items():
                rows.append(f"- {table_name}: `{len(table)}` rows")
            rows.append("")

        rows.extend(["## Assumptions, Exclusions, And Limitations", ""])
        if documentation.assumptions:
            rows.append("### Assumptions")
            rows.append("")
            rows.extend(f"- {item}" for item in documentation.assumptions)
            rows.append("")
        if documentation.exclusions:
            rows.append("### Exclusions")
            rows.append("")
            rows.extend(f"- {item}" for item in documentation.exclusions)
            rows.append("")
        if documentation.limitations:
            rows.append("### Limitations")
            rows.append("")
            rows.extend(f"- {item}" for item in documentation.limitations)
            rows.append("")

        if context.warnings:
            rows.extend(["## Run Warnings", ""])
            rows.extend(f"- {warning}" for warning in context.warnings)
            rows.append("")

        rows.extend(
            [
                "## Artifact Index",
                "",
                "- `run_report.md` for the narrative run summary.",
                "- `model_documentation_pack.md` for the development-facing documentation.",
                "- `validation_pack.md` for validator-oriented review packaging.",
                (
                    "- `committee_report.docx` / `committee_report.pdf` "
                    "for committee-facing delivery."
                ),
                (
                    "- `validation_report.docx` / `validation_report.pdf` "
                    "for validator-facing delivery."
                ),
                "- `reproducibility_manifest.json` for run fingerprint metadata.",
                "- `configuration_template.xlsx` for offline governance editing.",
                "",
            ]
        )

        return "\n".join(rows) + "\n"

    def _build_reproducibility_manifest(
        self,
        *,
        context: PipelineContext,
        model_path: Path,
        config_path: Path,
        input_snapshot_path: Path | None,
    ) -> dict[str, Any]:
        package_versions = self._collect_package_versions(context)
        config_payload = self._build_export_config_payload(context, model_path)
        input_fingerprint = (
            self._hash_dataframe(context.raw_data) if context.raw_data is not None else None
        )
        rows = [
            {"field": "run_id", "value": context.run_id},
            {"field": "python_version", "value": platform.python_version()},
            {"field": "platform", "value": platform.platform()},
            {"field": "model_type", "value": context.config.model.model_type.value},
            {"field": "target_mode", "value": context.config.target.mode.value},
            {"field": "execution_mode", "value": context.config.execution.mode.value},
            {"field": "input_dataframe_sha256", "value": input_fingerprint},
            {"field": "input_snapshot_sha256", "value": self._hash_file(input_snapshot_path)},
            {"field": "model_artifact_sha256", "value": self._hash_file(model_path)},
            {
                "field": "resolved_config_sha256",
                "value": hashlib.sha256(
                    json.dumps(config_payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            {"field": "random_state", "value": context.config.split.random_state},
            {
                "field": "git_commit",
                "value": (
                    self._get_git_commit()
                    if context.config.reproducibility.capture_git_metadata
                    else ""
                ),
            },
        ]
        rows.extend(
            {
                "field": f"package_version::{package_name}",
                "value": version,
            }
            for package_name, version in package_versions.items()
        )
        return {
            "run_id": context.run_id,
            "rows": rows,
            "package_versions": package_versions,
        }

    def _build_template_workbook(self, context: PipelineContext) -> bytes:
        return build_template_workbook_bytes(
            schema_frame=build_column_editor_frame_from_schema(context.config.schema),
            feature_dictionary_frame=build_feature_dictionary_frame_from_config(
                context.config.feature_dictionary,
                context.feature_columns,
            ),
            transformation_frame=build_transformation_frame_from_config(
                context.config.transformations
            ),
            feature_review_frame=build_feature_review_frame_from_config(
                context.config.manual_review
            ),
            scorecard_override_frame=build_scorecard_override_frame_from_config(
                context.config.manual_review
            ),
        )

    def _build_export_config_payload(
        self, context: PipelineContext, model_path: Path
    ) -> dict[str, Any]:
        exported_config = deepcopy(context.config)
        if exported_config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
            exported_config.execution.existing_model_path = Path(model_path.name)
            exported_config.execution.existing_config_path = None
        return exported_config.to_dict()

    def _build_interactive_report(self, context: PipelineContext) -> str:
        input_shape = context.metadata.get("input_shape", {})
        feature_summary = context.metadata.get("feature_summary", {})
        split_summary = context.metadata.get("split_summary", {})
        performance = context.config.performance
        return build_interactive_report_html(
            run_id=context.run_id,
            model_type=context.config.model.model_type.value,
            execution_mode=context.config.execution.mode.value,
            target_mode=context.config.target.mode.value,
            labels_available=bool(context.metadata.get("labels_available", False)),
            warning_count=len(context.warnings),
            metrics=context.metrics,
            input_rows=input_shape.get("rows"),
            feature_count=int(feature_summary.get("feature_count", 0)),
            split_summary=split_summary,
            warnings=context.warnings,
            events=context.events,
            diagnostics_tables=context.diagnostics_tables,
            visualizations=context.visualizations,
            table_preview_rows=performance.html_table_preview_rows,
            max_figures_per_section=performance.html_max_figures_per_section,
            max_tables_per_section=performance.html_max_tables_per_section,
        )

    def _export_excel_workbook(
        self,
        context: PipelineContext,
        workbook_path: Path,
        predictions: pd.DataFrame,
    ) -> None:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            pd.DataFrame(context.metrics).T.to_excel(writer, sheet_name="metrics")
            predictions.to_excel(writer, sheet_name="predictions", index=False)
            context.feature_importance.to_excel(
                writer, sheet_name="feature_importance", index=False
            )
            context.backtest_summary.to_excel(writer, sheet_name="backtest_summary", index=False)
            for table_name, table in context.diagnostics_tables.items():
                safe_name = self._sanitize_sheet_name(table_name)
                table.to_excel(writer, sheet_name=safe_name, index=False)

    def _export_visualizations(
        self,
        context: PipelineContext,
        html_dir: Path,
        png_dir: Path,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {"figures": {}}
        for figure_name, figure in context.visualizations.items():
            safe_name = self._sanitize_name(figure_name)
            manifest["figures"][figure_name] = {}
            if context.config.diagnostics.interactive_visualizations:
                html_path = html_dir / f"{safe_name}.html"
                figure.write_html(html_path, include_plotlyjs=True, full_html=True)
                manifest["figures"][figure_name]["html"] = str(html_path)
            if context.config.diagnostics.static_image_exports:
                png_path = png_dir / f"{safe_name}.png"
                try:
                    pio.write_image(figure, png_path)
                    manifest["figures"][figure_name]["png"] = str(png_path)
                except Exception as exc:
                    context.warn(f"Could not export PNG for {figure_name}: {exc}")
        return manifest

    def _build_step_manifest(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "run_id": context.run_id,
            "steps": context.metadata.get("step_manifest", []),
            "pipeline_events": context.events,
        }

    def _build_generated_runner_script(self, context: PipelineContext) -> str:
        artifacts = context.config.artifacts
        return textwrap.dedent(
            f"""\
            from __future__ import annotations

            import sys
            from pathlib import Path

            THIS_DIR = Path(__file__).resolve().parent
            SNAPSHOT_SRC = THIS_DIR / "{artifacts.code_snapshot_directory_name}" / "src"
            if SNAPSHOT_SRC.exists():
                sys.path.insert(0, str(SNAPSHOT_SRC))

            from quant_pd_framework.run import main


            def _has_option(arguments: list[str], option: str) -> bool:
                return option in arguments or any(
                    argument.startswith(f"{{option}}=") for argument in arguments
                )


            def _build_args() -> list[str]:
                arguments = list(sys.argv[1:])
                if not _has_option(arguments, "--config"):
                    arguments = [
                        "--config",
                        str(THIS_DIR / "{artifacts.config_file_name}"),
                    ] + arguments
                if not _has_option(arguments, "--input"):
                    default_input = THIS_DIR / "{artifacts.input_snapshot_file_name}"
                    if default_input.exists():
                        arguments = ["--input", str(default_input)] + arguments
                if not _has_option(arguments, "--output-root"):
                    arguments = ["--output-root", str(THIS_DIR / "reruns")] + arguments
                return arguments


            if __name__ == "__main__":
                raise SystemExit(main(_build_args()))
            """
        )

    def _build_rerun_readme(self, context: PipelineContext) -> str:
        artifacts = context.config.artifacts
        return textwrap.dedent(
            f"""\
            # Rerun Bundle

            This run folder contains everything needed to replay the pipeline outside the GUI.

            ## Fastest Path

            From this directory:

            ```powershell
            python {artifacts.runner_script_file_name}
            ```

            The generated launcher automatically:

            - uses `{artifacts.config_file_name}` as the saved framework config
            - uses `{artifacts.input_snapshot_file_name}` as the default input when that file exists
            - writes new rerun artifacts under `reruns/`
            - imports the local `code_snapshot/src/` copy first, so step-level
              edits apply immediately

            ## Direct CLI Path

            You can also run the packaged CLI directly:

            ```powershell
            python -m quant_pd_framework.run ^
              --config {artifacts.config_file_name} ^
              --input {artifacts.input_snapshot_file_name} ^
              --output-root reruns
            ```

            If you want to rerun on a different dataset, change the input path:

            ```powershell
            python -m quant_pd_framework.run ^
              --config {artifacts.config_file_name} ^
              --input path\\to\\new_data.csv ^
              --output-root reruns
            ```

            ## Editing The Python Code

            The exported `code_snapshot/` directory is a copy of the framework
            at the time this run was produced.

            Common places to edit:

            - `code_snapshot/src/quant_pd_framework/steps/` for pipeline-step logic
            - `code_snapshot/src/quant_pd_framework/models.py` for model adapters
            - `code_snapshot/src/quant_pd_framework/orchestrator.py` for step ordering
            - `code_snapshot/tests/` for regression tests tied to your edits

            The generated launcher prepends `code_snapshot/src/` to `sys.path`,
            so edits there take effect when you rerun `generated_run.py`.

            ## Step Manifest

            The exact ordered step stack used for this run is recorded in
            `{artifacts.step_manifest_file_name}`.
            """
        )

    def _export_code_snapshot(self, output_dir: Path) -> None:
        project_root = Path(__file__).resolve().parents[3]
        ignore_patterns = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.egg-info")
        directories_to_copy = {
            ".streamlit": project_root / ".streamlit",
            "src/quant_pd_framework": project_root / "src" / "quant_pd_framework",
            "app": project_root / "app",
            "examples": project_root / "examples",
            "tests": project_root / "tests",
        }
        files_to_copy = [
            project_root / "README.md",
            project_root / "pyproject.toml",
            project_root / "launch_gui.bat",
        ]

        for relative_name, source_path in directories_to_copy.items():
            if source_path.exists():
                shutil.copytree(
                    source_path,
                    output_dir / relative_name,
                    dirs_exist_ok=True,
                    ignore=ignore_patterns,
                )

        for source_path in files_to_copy:
            if source_path.exists():
                destination = output_dir / source_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)

    def _collect_package_versions(self, context: PipelineContext) -> dict[str, str]:
        versions: dict[str, str] = {}
        if not context.config.reproducibility.enabled:
            return versions
        for package_name in context.config.reproducibility.package_names:
            try:
                versions[package_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                versions[package_name] = "not_installed"
        return versions

    def _hash_dataframe(self, dataframe: pd.DataFrame) -> str:
        hash_values = pd.util.hash_pandas_object(dataframe, index=True).to_numpy()
        digest = hashlib.sha256()
        digest.update(hash_values.tobytes())
        digest.update("|".join(dataframe.columns.astype(str)).encode("utf-8"))
        digest.update("|".join(dataframe.dtypes.astype(str)).encode("utf-8"))
        return digest.hexdigest()

    def _hash_file(self, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_git_commit(self) -> str:
        project_root = Path(__file__).resolve().parents[3]
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                capture_output=True,
                check=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip()

    def _sanitize_name(self, name: str) -> str:
        return "".join(
            character if character.isalnum() or character in {"_", "-"} else "_"
            for character in name
        )

    def _sanitize_sheet_name(self, name: str) -> str:
        return self._sanitize_name(name)[:31] or "sheet"
