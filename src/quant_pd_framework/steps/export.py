"""Writes model artifacts, scored output, and a human-readable run report."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import textwrap
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.io as pio

from ..base import BasePipelineStep
from ..config import ExecutionMode, LargeDataExportPolicy, TabularOutputFormat
from ..context import PipelineContext
from ..decision_summary import (
    build_decision_summary,
    build_decision_summary_markdown,
    build_decision_summary_snapshot_from_context,
)
from ..export_layout import ExportPathLayout, build_export_path_layout
from ..export_profiles import (
    code_snapshot_enabled,
    excel_workbook_enabled,
    input_snapshot_enabled,
    regulatory_reports_enabled,
    resolve_html_report_limits,
)
from ..gui_support import (
    build_column_editor_frame_from_schema,
    build_feature_dictionary_frame_from_config,
    build_feature_review_frame_from_config,
    build_scorecard_override_frame_from_config,
    build_template_workbook_bytes,
    build_transformation_frame_from_config,
)
from ..presentation import (
    apply_advanced_visual_analytics,
    build_interactive_report_html,
    enhance_report_visualizations,
    infer_asset_section,
)
from ..report_payload import optimize_report_visualizations
from ..reporting import build_regulatory_report_bundle
from ..tabular_policy import resolve_tabular_output_format
from ..validation_evidence import publish_validation_evidence_tables

ARTIFACT_LAYOUT_VERSION = "2.0"
TABLE_SECTION_DIRECTORIES = {
    "model_performance": "model_performance",
    "calibration_thresholds": "calibration",
    "stability_drift": "stability",
    "sample_segmentation": "segmentation",
    "feature_effects": "explainability",
    "statistical_tests": "statistical_tests",
    "feature_subset_search": "feature_subset_search",
    "scorecard_workbench": "scorecard",
    "credit_risk_development": "credit_risk",
    "data_quality": "diagnostics",
    "backtesting_time": "backtesting",
    "governance_export": "governance",
}


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
        paths = build_export_path_layout(context.config.artifacts, output_root)

        tables_dir = paths.tables_dir
        figures_dir = paths.figures_dir
        html_dir = paths.html_dir
        png_dir = paths.png_dir
        metadata_dir = paths.metadata_dir
        self._ensure_layout_directories(paths)
        if self._html_figure_exports_enabled(context):
            html_dir.mkdir(parents=True, exist_ok=True)
        if self._png_figure_exports_enabled(context):
            png_dir.mkdir(parents=True, exist_ok=True)

        model_path = paths.model_path
        metrics_path = paths.metrics_path
        input_snapshot_path = paths.input_snapshot_path
        input_snapshot_parquet_path = paths.input_snapshot_parquet_path
        predictions_path = paths.predictions_path
        predictions_parquet_path = paths.predictions_parquet_path
        feature_importance_path = paths.feature_importance_path
        backtest_path = paths.backtest_path
        report_path = paths.report_path
        interactive_report_path = paths.interactive_report_path
        config_path = paths.config_path
        tests_path = paths.tests_path
        workbook_path = paths.workbook_path
        model_summary_path = paths.model_summary_path
        manifest_path = paths.manifest_path
        step_manifest_path = paths.step_manifest_path
        decision_summary_path = paths.decision_summary_path
        documentation_pack_path = paths.documentation_pack_path
        validation_pack_path = paths.validation_pack_path
        committee_report_docx_path = paths.committee_report_docx_path
        validation_report_docx_path = paths.validation_report_docx_path
        committee_report_pdf_path = paths.committee_report_pdf_path
        validation_report_pdf_path = paths.validation_report_pdf_path
        reproducibility_manifest_path = paths.reproducibility_manifest_path
        run_debug_trace_path = paths.run_debug_trace_path
        template_workbook_path = paths.template_workbook_path
        runner_script_path = paths.runner_script_path
        rerun_readme_path = paths.rerun_readme_path
        code_snapshot_dir = paths.code_snapshot_dir

        if context.config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            return self._run_subset_search_export(
                context=context,
                output_root=output_root,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                html_dir=html_dir,
                png_dir=png_dir,
                metadata_dir=metadata_dir,
                metrics_path=metrics_path,
                input_snapshot_path=input_snapshot_path,
                input_snapshot_parquet_path=input_snapshot_parquet_path,
                report_path=report_path,
                interactive_report_path=interactive_report_path,
                config_path=config_path,
                tests_path=tests_path,
                decision_summary_path=decision_summary_path,
                manifest_path=manifest_path,
                step_manifest_path=step_manifest_path,
                run_debug_trace_path=run_debug_trace_path,
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

        input_snapshot_exports: dict[str, Any] = {}
        if self._input_snapshot_enabled(context) and context.raw_data is not None:
            input_snapshot_exports = self._export_tabular_dataframe(
                context=context,
                dataframe=context.raw_data,
                dataset_name="input_snapshot",
                csv_path=input_snapshot_path,
                parquet_path=input_snapshot_parquet_path,
            )

        prediction_exports = self._export_prediction_dataframes(
            context=context,
            csv_path=predictions_path,
            parquet_path=predictions_parquet_path,
            split_output_dir=paths.data_predictions_dir,
        )
        self._publish_tabular_export_policy_table(context)
        context.feature_importance.to_csv(feature_importance_path, index=False)
        context.backtest_summary.to_csv(backtest_path, index=False)
        report_visualizations = self._build_report_visualizations(context)
        publish_validation_evidence_tables(context)
        for table_name, table in context.diagnostics_tables.items():
            self._export_table(
                context=context,
                table=table,
                csv_path=self._table_export_path(tables_dir, table_name),
            )

        visualization_manifest = self._export_visualizations(
            context,
            html_dir,
            png_dir,
            visualizations=report_visualizations,
        )
        if self._excel_workbook_enabled(context):
            predictions = self._combine_prediction_frames_for_export(context)
            self._export_excel_workbook(context, workbook_path, predictions)
        if context.model_summary is not None:
            if isinstance(context.model_summary, pd.DataFrame):
                context.model_summary.to_csv(model_summary_path.with_suffix(".csv"), index=False)
            else:
                model_summary_path.write_text(str(context.model_summary), encoding="utf-8")

        report_path.write_text(self._build_report(context), encoding="utf-8")
        decision_summary_path.write_text(
            self._build_decision_summary(
                context,
                artifacts=self._decision_summary_artifacts(
                    model_path=model_path,
                    metrics_path=metrics_path,
                    predictions_path=self._primary_export_path(prediction_exports),
                    feature_importance_path=feature_importance_path,
                    report_path=report_path,
                    interactive_report_path=interactive_report_path,
                    decision_summary_path=decision_summary_path,
                    documentation_pack_path=documentation_pack_path,
                    validation_pack_path=validation_pack_path,
                    config_path=config_path,
                    tests_path=tests_path,
                    run_debug_trace_path=run_debug_trace_path,
                ),
            ),
            encoding="utf-8",
        )
        documentation_pack_path.write_text(
            self._build_documentation_pack(context),
            encoding="utf-8",
        )
        validation_pack_path.write_text(
            self._build_validation_pack(context),
            encoding="utf-8",
        )
        regulatory_report_manifest = (
            self._export_regulatory_reports(
                context=context,
                committee_report_docx_path=committee_report_docx_path,
                validation_report_docx_path=validation_report_docx_path,
                committee_report_pdf_path=committee_report_pdf_path,
                validation_report_pdf_path=validation_report_pdf_path,
            )
            if self._regulatory_reports_enabled(context)
            else {}
        )
        reproducibility_manifest = self._build_reproducibility_manifest(
            context=context,
            model_path=model_path,
            config_path=config_path,
            input_snapshot_path=self._primary_export_path(input_snapshot_exports),
        )
        self._write_json(reproducibility_manifest_path, reproducibility_manifest)
        context.diagnostics_tables["reproducibility_manifest"] = pd.DataFrame(
            reproducibility_manifest["rows"]
        )
        interactive_report_path.write_text(
            self._build_interactive_report(
                context,
                visualizations=report_visualizations,
            ),
            encoding="utf-8",
        )
        template_workbook_path.write_bytes(self._build_template_workbook(context))
        step_manifest = self._build_step_manifest(context)
        self._write_json(step_manifest_path, step_manifest)
        self._write_json(run_debug_trace_path, self._build_run_debug_trace(context))
        start_here_path = paths.start_here_path
        start_here_path.write_text(
            self._build_start_here_guide(context=context, paths=paths),
            encoding="utf-8",
        )
        manifest = {
            "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
            "core_artifacts": {
                "output_root": str(output_root),
                "start_here": str(start_here_path),
                "export_profile": context.config.artifacts.export_profile.value,
                "model": str(model_path),
                "metrics": str(metrics_path),
                "predictions": self._path_string(self._primary_export_path(prediction_exports)),
                "predictions_csv": self._path_string(predictions_path)
                if predictions_path.exists()
                else None,
                "predictions_parquet": self._path_string(predictions_parquet_path)
                if predictions_parquet_path.exists()
                else None,
                "feature_importance": str(feature_importance_path),
                "backtest": str(backtest_path),
                "report": str(report_path),
                "decision_summary": str(decision_summary_path),
                "config": str(config_path),
                "tests": str(tests_path),
                "step_manifest": str(step_manifest_path),
                "run_debug_trace": str(run_debug_trace_path),
                "artifact_manifest": str(manifest_path),
            },
            "directories": {
                "reports": str(paths.reports_dir),
                "model": str(paths.model_dir),
                "data": str(paths.data_dir),
                "data_input": str(paths.data_input_dir),
                "data_predictions": str(paths.data_predictions_dir),
                "tables": str(tables_dir),
                "table_groups": self._table_group_directories(tables_dir),
                "figures": str(figures_dir) if figures_dir.exists() else None,
                "figures_html": str(html_dir) if html_dir.exists() else None,
                "figures_png": str(png_dir) if png_dir.exists() else None,
                "config": str(paths.config_dir),
                "metadata": str(metadata_dir),
                "checkpoints": str(output_root / "checkpoints")
                if (output_root / "checkpoints").exists()
                else None,
                "workbooks": str(paths.workbooks_dir),
                "code": str(paths.code_dir),
                "sample_development": self._path_string(
                    context.artifacts.get("sample_development_dir")
                ),
                "full_data_scoring": self._path_string(
                    context.artifacts.get("full_data_scoring_dir")
                ),
                "large_data_metadata": self._path_string(
                    context.artifacts.get("large_data_metadata_dir")
                ),
            },
            **visualization_manifest,
            "interactive_report": str(interactive_report_path),
            "decision_summary": str(decision_summary_path),
            "documentation_pack": str(documentation_pack_path),
            "validation_pack": str(validation_pack_path),
            "regulatory_reports": regulatory_report_manifest,
            "reproducibility_manifest": str(reproducibility_manifest_path),
            "checkpoint_manifest": context.metadata.get("checkpointed_execution", {}).get(
                "checkpoint_manifest"
            ),
            "configuration_template": str(template_workbook_path),
            "rerun_bundle": {
                "step_manifest": str(step_manifest_path),
                "runner_script": str(runner_script_path),
                "rerun_readme": str(rerun_readme_path),
            },
        }
        if self._excel_workbook_enabled(context):
            manifest["core_artifacts"]["analysis_workbook"] = str(workbook_path)
        if self._input_snapshot_enabled(context) and input_snapshot_exports:
            manifest["rerun_bundle"]["input_snapshot"] = self._path_string(
                self._primary_export_path(input_snapshot_exports)
            )
            manifest["rerun_bundle"]["input_snapshot_csv"] = (
                str(input_snapshot_path) if input_snapshot_path.exists() else None
            )
            manifest["rerun_bundle"]["input_snapshot_parquet"] = (
                str(input_snapshot_parquet_path) if input_snapshot_parquet_path.exists() else None
            )
        self._write_manifest(manifest_path, manifest, output_root=output_root)

        runner_script_path.write_text(
            self._build_generated_runner_script(context), encoding="utf-8"
        )
        rerun_readme_path.write_text(self._build_rerun_readme(context), encoding="utf-8")
        start_here_path.write_text(
            self._build_start_here_guide(context=context, paths=paths),
            encoding="utf-8",
        )
        self._write_manifest(manifest_path, manifest, output_root=output_root)

        if self._code_snapshot_enabled(context):
            self._export_code_snapshot(code_snapshot_dir)
            manifest["rerun_bundle"]["code_snapshot"] = str(code_snapshot_dir)
            self._write_manifest(manifest_path, manifest, output_root=output_root)

        monitoring_bundle: dict[str, Any] | None = None
        if context.config.execution.mode == ExecutionMode.FIT_NEW_MODEL:
            monitoring_bundle = self._export_monitoring_bundle(
                context=context,
                output_root=output_root,
                model_path=model_path,
                config_path=config_path,
                runner_script_path=runner_script_path,
                manifest_path=manifest_path,
                input_snapshot_path=self._primary_export_path(input_snapshot_exports),
                predictions_path=self._primary_export_path(prediction_exports),
                code_snapshot_dir=code_snapshot_dir
                if self._code_snapshot_enabled(context)
                else None,
            )
            manifest["monitoring_bundle"] = {
                "directory": str(monitoring_bundle["bundle_dir"]),
                "metadata": str(monitoring_bundle["metadata_path"]),
                "bundle_version": monitoring_bundle["metadata"]["bundle_version"],
            }
            self._write_manifest(manifest_path, manifest, output_root=output_root)
            self._refresh_monitoring_bundle_manifest_copy(
                bundle_dir=monitoring_bundle["bundle_dir"],
                manifest_path=manifest_path,
            )

        context.artifacts = {
            "output_root": output_root,
            "start_here": start_here_path,
            "model": model_path,
            "metrics": metrics_path,
            "input_snapshot": self._primary_export_path(input_snapshot_exports),
            "input_snapshot_csv": input_snapshot_path if input_snapshot_path.exists() else None,
            "input_snapshot_parquet": input_snapshot_parquet_path
            if input_snapshot_parquet_path.exists()
            else None,
            "predictions": self._primary_export_path(prediction_exports),
            "predictions_csv": predictions_path if predictions_path.exists() else None,
            "predictions_parquet": predictions_parquet_path
            if predictions_parquet_path.exists()
            else None,
            "feature_importance": feature_importance_path,
            "backtest": backtest_path,
            "report": report_path,
            "decision_summary": decision_summary_path,
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
            "figures_dir": figures_dir if figures_dir.exists() else None,
            "workbook": workbook_path if self._excel_workbook_enabled(context) else None,
            "run_debug_trace": run_debug_trace_path,
            "manifest": manifest_path,
            "artifact_manifest": manifest_path,
            "step_manifest": step_manifest_path,
            "sample_development_dir": context.artifacts.get("sample_development_dir"),
            "large_data_training_sample": context.artifacts.get("large_data_training_sample"),
            "full_data_scoring_dir": context.artifacts.get("full_data_scoring_dir"),
            "full_data_predictions": context.artifacts.get("full_data_predictions"),
            "large_data_metadata_dir": context.artifacts.get("large_data_metadata_dir"),
            "large_data_full_scoring_progress": context.artifacts.get(
                "large_data_full_scoring_progress"
            ),
            "large_data_full_scoring_metadata": context.artifacts.get(
                "large_data_full_scoring_metadata"
            ),
            "runner_script": runner_script_path,
            "rerun_readme": rerun_readme_path,
            "code_snapshot_dir": code_snapshot_dir
            if self._code_snapshot_enabled(context)
            else None,
            "monitoring_bundle_dir": (
                monitoring_bundle["bundle_dir"] if monitoring_bundle is not None else None
            ),
            "monitoring_metadata": (
                monitoring_bundle["metadata_path"] if monitoring_bundle is not None else None
            ),
        }
        return context

    def _export_tabular_dataframe(
        self,
        *,
        context: PipelineContext,
        dataframe: pd.DataFrame,
        dataset_name: str,
        csv_path: Path,
        parquet_path: Path,
    ) -> dict[str, Any]:
        artifacts = context.config.artifacts
        output_format = resolve_tabular_output_format(context.metadata)
        export_policy = artifacts.large_data_export_policy
        sample_rows = min(len(dataframe), artifacts.large_data_sample_rows)
        metadata: dict[str, Any] = {
            "dataset_name": dataset_name,
            "row_count": int(len(dataframe)),
            "column_count": int(dataframe.shape[1]),
            "configured_output_format": artifacts.tabular_output_format.value,
            "output_format": output_format.value,
            "output_format_basis": "original_input_suffix",
            "export_policy": export_policy.value,
            "sample_rows": int(sample_rows),
            "csv_path": None,
            "parquet_path": None,
            "primary_path": None,
        }

        if export_policy == LargeDataExportPolicy.METADATA_ONLY:
            self._record_tabular_export(context, metadata)
            return metadata

        csv_frame = dataframe
        if export_policy == LargeDataExportPolicy.SAMPLED and len(dataframe) > sample_rows:
            csv_frame = dataframe.sample(
                sample_rows,
                random_state=context.config.split.random_state,
            ).reset_index(drop=True)

        if output_format in {TabularOutputFormat.CSV, TabularOutputFormat.BOTH}:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_frame.to_csv(csv_path, index=False)
            metadata["csv_path"] = str(csv_path)
            metadata["primary_path"] = str(csv_path)

        if output_format in {TabularOutputFormat.PARQUET, TabularOutputFormat.BOTH}:
            self._write_parquet(
                dataframe,
                parquet_path,
                compression=artifacts.parquet_compression,
            )
            metadata["parquet_path"] = str(parquet_path)
            if (
                output_format == TabularOutputFormat.PARQUET
                or export_policy == LargeDataExportPolicy.SAMPLED
            ):
                metadata["primary_path"] = str(parquet_path)

        self._record_tabular_export(context, metadata)
        return metadata

    def _export_prediction_dataframes(
        self,
        *,
        context: PipelineContext,
        csv_path: Path,
        parquet_path: Path,
        split_output_dir: Path,
    ) -> dict[str, Any]:
        artifacts = context.config.artifacts
        output_format = resolve_tabular_output_format(context.metadata)
        export_policy = artifacts.large_data_export_policy
        compact_frames = {
            split_name: self._compact_prediction_frame(context, split_frame)
            for split_name, split_frame in context.predictions.items()
        }
        row_count = sum(len(frame) for frame in compact_frames.values())
        sample_rows = min(row_count, artifacts.large_data_sample_rows)
        metadata: dict[str, Any] = {
            "dataset_name": "predictions",
            "row_count": int(row_count),
            "column_count": int(
                max((frame.shape[1] for frame in compact_frames.values()), default=0)
            ),
            "configured_output_format": artifacts.tabular_output_format.value,
            "output_format": output_format.value,
            "output_format_basis": "original_input_suffix",
            "export_policy": export_policy.value,
            "sample_rows": int(sample_rows),
            "compact_prediction_exports": artifacts.compact_prediction_exports,
            "csv_path": None,
            "parquet_path": None,
            "primary_path": None,
            "split_paths": {},
        }

        if export_policy == LargeDataExportPolicy.METADATA_ONLY:
            self._record_tabular_export(context, metadata)
            return metadata

        sampled_frames = self._sample_prediction_frames(
            compact_frames,
            sample_rows=sample_rows,
            random_state=context.config.split.random_state,
            sampled=export_policy == LargeDataExportPolicy.SAMPLED,
        )
        split_output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, frame in compact_frames.items():
            split_csv_path = split_output_dir / f"predictions_{split_name}.csv"
            if output_format in {TabularOutputFormat.CSV, TabularOutputFormat.BOTH}:
                split_csv_path.parent.mkdir(parents=True, exist_ok=True)
                sampled_frames[split_name].to_csv(split_csv_path, index=False)
            if output_format in {TabularOutputFormat.PARQUET, TabularOutputFormat.BOTH}:
                self._write_parquet(
                    frame,
                    split_csv_path.with_suffix(".parquet"),
                    compression=artifacts.parquet_compression,
                )
            metadata["split_paths"][split_name] = {
                "csv": str(split_csv_path)
                if output_format in {TabularOutputFormat.CSV, TabularOutputFormat.BOTH}
                else None,
                "parquet": str(split_csv_path.with_suffix(".parquet"))
                if output_format in {TabularOutputFormat.PARQUET, TabularOutputFormat.BOTH}
                else None,
            }

        if output_format in {TabularOutputFormat.CSV, TabularOutputFormat.BOTH}:
            self._write_csv_frames(sampled_frames.values(), csv_path)
            metadata["csv_path"] = str(csv_path)
            metadata["primary_path"] = str(csv_path)

        if output_format in {TabularOutputFormat.PARQUET, TabularOutputFormat.BOTH}:
            self._write_parquet_frames(
                compact_frames.values(),
                parquet_path,
                compression=artifacts.parquet_compression,
            )
            metadata["parquet_path"] = str(parquet_path)
            if (
                output_format == TabularOutputFormat.PARQUET
                or export_policy == LargeDataExportPolicy.SAMPLED
            ):
                metadata["primary_path"] = str(parquet_path)

        self._record_tabular_export(context, metadata)
        return metadata

    def _compact_prediction_frame(
        self,
        context: PipelineContext,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if not context.config.artifacts.compact_prediction_exports:
            return frame
        columns = self._prediction_export_columns(context, frame)
        if not columns:
            return frame
        return frame.loc[:, columns].copy(deep=False)

    def _prediction_export_columns(
        self,
        context: PipelineContext,
        frame: pd.DataFrame,
    ) -> list[str]:
        role_columns = [
            spec.name
            for spec in context.config.schema.column_specs
            if spec.role.value in {"identifier", "date"} and spec.name in frame.columns
        ]
        preferred_columns = [
            *role_columns,
            context.config.split.date_column,
            context.config.split.entity_column,
            context.target_column,
            context.config.diagnostics.default_segment_column,
            *context.metadata.get("hazard_time_features", []),
            *self._low_cardinality_segment_columns(context, frame),
            "split",
            "predicted_probability",
            "predicted_probability_recommended",
            "predicted_class",
            "predicted_value",
            "residual",
            "prediction_score",
            "scorecard_score",
            "scorecard_points",
        ]
        return list(
            dict.fromkeys(
                column_name
                for column_name in preferred_columns
                if column_name and column_name in frame.columns
            )
        )

    def _low_cardinality_segment_columns(
        self,
        context: PipelineContext,
        frame: pd.DataFrame,
    ) -> list[str]:
        return [
            feature_name
            for feature_name in context.categorical_features
            if feature_name in frame.columns
            and frame[feature_name].nunique(dropna=True)
            <= context.config.performance.max_categorical_cardinality
        ]

    def _sample_prediction_frames(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        sample_rows: int,
        random_state: int,
        sampled: bool,
    ) -> dict[str, pd.DataFrame]:
        if not sampled:
            return frames
        total_rows = sum(len(frame) for frame in frames.values())
        if total_rows <= sample_rows:
            return frames
        sampled_frames: dict[str, pd.DataFrame] = {}
        for split_name, frame in frames.items():
            split_sample_rows = max(1, round(sample_rows * len(frame) / total_rows))
            if len(frame) <= split_sample_rows:
                sampled_frames[split_name] = frame
            else:
                sampled_frames[split_name] = frame.sample(
                    split_sample_rows,
                    random_state=random_state,
                ).sort_index()
        return sampled_frames

    def _combine_prediction_frames_for_export(self, context: PipelineContext) -> pd.DataFrame:
        frames = [
            self._compact_prediction_frame(context, frame)
            for frame in context.predictions.values()
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _write_csv_frames(self, frames, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        wrote_header = False
        for frame in frames:
            frame.to_csv(
                path,
                index=False,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
            )
            wrote_header = True

    def _write_parquet_frames(self, frames, path: Path, *, compression: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            combined = pd.concat(list(frames), ignore_index=True)
            self._write_parquet(combined, path, compression=compression)
            return

        writer: pq.ParquetWriter | None = None
        try:
            for frame in frames:
                safe_frame = self._prepare_parquet_safe_frame(frame)
                table = pa.Table.from_pandas(safe_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression=compression)
                else:
                    table = table.cast(writer.schema, safe=False)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    def _export_table(
        self,
        *,
        context: PipelineContext,
        table: pd.DataFrame,
        csv_path: Path,
    ) -> None:
        output_format = resolve_tabular_output_format(context.metadata)
        if output_format in {TabularOutputFormat.CSV, TabularOutputFormat.BOTH}:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(csv_path, index=False)
        if output_format in {TabularOutputFormat.PARQUET, TabularOutputFormat.BOTH}:
            self._write_parquet(
                table,
                csv_path.with_suffix(".parquet"),
                compression=context.config.artifacts.parquet_compression,
            )

    def _record_tabular_export(
        self,
        context: PipelineContext,
        metadata: dict[str, Any],
    ) -> None:
        export_rows = context.metadata.setdefault("tabular_export_policy", [])
        if isinstance(export_rows, list):
            export_rows.append(metadata)

    def _publish_tabular_export_policy_table(self, context: PipelineContext) -> None:
        export_rows = context.metadata.get("tabular_export_policy")
        if isinstance(export_rows, list) and export_rows:
            context.diagnostics_tables["tabular_export_policy"] = pd.DataFrame(export_rows)

    def _primary_export_path(self, export_metadata: dict[str, Any]) -> Path | None:
        primary_path = export_metadata.get("primary_path")
        if not primary_path:
            return None
        return Path(primary_path)

    def _path_string(self, path: Path | None) -> str | None:
        return str(path) if path is not None else None

    def _write_parquet(
        self,
        dataframe: pd.DataFrame,
        path: Path,
        *,
        compression: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dataframe.to_parquet(path, index=False, compression=compression)
        except Exception:
            safe_frame = self._prepare_parquet_safe_frame(dataframe)
            safe_frame.to_parquet(path, index=False, compression=compression)

    def _prepare_parquet_safe_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        safe_frame = dataframe.copy(deep=False)
        for column_name in safe_frame.columns:
            series = safe_frame[column_name]
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or isinstance(series.dtype, pd.IntervalDtype)
            ):
                safe_frame[column_name] = series.map(self._parquet_safe_value).astype("string")
        return safe_frame

    def _parquet_safe_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Interval):
            return str(value)
        if isinstance(value, pd.Period):
            return str(value)
        if isinstance(value, (list, tuple, set, dict)):
            return json.dumps(value, default=str)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            return str(value)
        return str(value)

    def _run_subset_search_export(
        self,
        *,
        context: PipelineContext,
        output_root: Path,
        tables_dir: Path,
        figures_dir: Path,
        html_dir: Path,
        png_dir: Path,
        metadata_dir: Path,
        metrics_path: Path,
        input_snapshot_path: Path,
        input_snapshot_parquet_path: Path,
        report_path: Path,
        interactive_report_path: Path,
        config_path: Path,
        tests_path: Path,
        decision_summary_path: Path,
        manifest_path: Path,
        step_manifest_path: Path,
        run_debug_trace_path: Path,
    ) -> PipelineContext:
        if not context.diagnostics_tables:
            raise ValueError("Feature subset search requires comparison tables before export.")

        self._write_json(metrics_path, context.metrics)
        self._write_json(config_path, context.config.to_dict())
        self._write_json(tests_path, context.statistical_tests)
        input_snapshot_exports: dict[str, Any] = {}
        if self._input_snapshot_enabled(context) and context.raw_data is not None:
            input_snapshot_exports = self._export_tabular_dataframe(
                context=context,
                dataframe=context.raw_data,
                dataset_name="input_snapshot",
                csv_path=input_snapshot_path,
                parquet_path=input_snapshot_parquet_path,
            )
        self._publish_tabular_export_policy_table(context)
        report_visualizations = self._build_report_visualizations(context)
        publish_validation_evidence_tables(context)

        for table_name, table in context.diagnostics_tables.items():
            self._export_table(
                context=context,
                table=table,
                csv_path=self._table_export_path(tables_dir, table_name),
            )

        visualization_manifest = self._export_visualizations(
            context,
            html_dir,
            png_dir,
            visualizations=report_visualizations,
        )
        report_path.write_text(self._build_subset_search_report(context), encoding="utf-8")
        decision_summary_path.write_text(
            self._build_decision_summary(
                context,
                artifacts=self._decision_summary_artifacts(
                    model_path=None,
                    metrics_path=metrics_path,
                    predictions_path=None,
                    feature_importance_path=None,
                    report_path=report_path,
                    interactive_report_path=interactive_report_path,
                    decision_summary_path=decision_summary_path,
                    documentation_pack_path=None,
                    validation_pack_path=None,
                    config_path=config_path,
                    tests_path=tests_path,
                    run_debug_trace_path=run_debug_trace_path,
                ),
            ),
            encoding="utf-8",
        )
        interactive_report_path.write_text(
            self._build_interactive_report(
                context,
                visualizations=report_visualizations,
            ),
            encoding="utf-8",
        )
        step_manifest = self._build_step_manifest(context)
        self._write_json(step_manifest_path, step_manifest)
        self._write_json(run_debug_trace_path, self._build_run_debug_trace(context))
        paths = build_export_path_layout(context.config.artifacts, output_root)
        paths.start_here_path.write_text(
            self._build_start_here_guide(context=context, paths=paths),
            encoding="utf-8",
        )

        manifest = {
            "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
            "core_artifacts": {
                "output_root": str(output_root),
                "start_here": str(paths.start_here_path),
                "export_profile": context.config.artifacts.export_profile.value,
                "metrics": str(metrics_path),
                "report": str(report_path),
                "interactive_report": str(interactive_report_path),
                "decision_summary": str(decision_summary_path),
                "config": str(config_path),
                "tests": str(tests_path),
                "step_manifest": str(step_manifest_path),
                "run_debug_trace": str(run_debug_trace_path),
                "artifact_manifest": str(manifest_path),
            },
            "directories": {
                "reports": str(paths.reports_dir),
                "data": str(paths.data_dir),
                "data_input": str(paths.data_input_dir),
                "tables": str(tables_dir),
                "table_groups": self._table_group_directories(tables_dir),
                "figures": str(figures_dir) if figures_dir.exists() else None,
                "figures_html": str(html_dir) if html_dir.exists() else None,
                "figures_png": str(png_dir) if png_dir.exists() else None,
                "config": str(paths.config_dir),
                "metadata": str(metadata_dir),
                "checkpoints": str(output_root / "checkpoints")
                if (output_root / "checkpoints").exists()
                else None,
            },
            **visualization_manifest,
        }
        if self._input_snapshot_enabled(context) and input_snapshot_exports:
            manifest["core_artifacts"]["input_snapshot"] = self._path_string(
                self._primary_export_path(input_snapshot_exports)
            )
            manifest["core_artifacts"]["input_snapshot_csv"] = (
                str(input_snapshot_path) if input_snapshot_path.exists() else None
            )
            manifest["core_artifacts"]["input_snapshot_parquet"] = (
                str(input_snapshot_parquet_path) if input_snapshot_parquet_path.exists() else None
            )
        self._write_manifest(manifest_path, manifest, output_root=output_root)

        context.artifacts = {
            "output_root": output_root,
            "start_here": paths.start_here_path,
            "metrics": metrics_path,
            "input_snapshot": self._primary_export_path(input_snapshot_exports),
            "input_snapshot_csv": input_snapshot_path if input_snapshot_path.exists() else None,
            "input_snapshot_parquet": input_snapshot_parquet_path
            if input_snapshot_parquet_path.exists()
            else None,
            "report": report_path,
            "interactive_report": interactive_report_path,
            "decision_summary": decision_summary_path,
            "config": config_path,
            "tests": tests_path,
            "tables_dir": tables_dir,
            "figures_dir": figures_dir if figures_dir.exists() else None,
            "manifest": manifest_path,
            "artifact_manifest": manifest_path,
            "step_manifest": step_manifest_path,
            "run_debug_trace": run_debug_trace_path,
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
        predictions_path: Path | None,
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
        if predictions_path is not None and predictions_path.exists():
            prediction_bundle_name = context.config.artifacts.predictions_file_name
            bundled_paths[prediction_bundle_name] = self._copy_tabular_as_csv_into_bundle(
                source_path=predictions_path,
                destination_path=bundle_dir / prediction_bundle_name,
            )
        else:
            missing_name = context.config.artifacts.predictions_file_name
            bundled_paths[missing_name] = None
            missing_optional_artifacts.append(missing_name)

        if input_snapshot_path is not None and input_snapshot_path.exists():
            input_bundle_name = context.config.artifacts.input_snapshot_file_name
            bundled_paths[input_bundle_name] = self._copy_tabular_as_csv_into_bundle(
                source_path=input_snapshot_path,
                destination_path=bundle_dir / input_bundle_name,
            )
        else:
            missing_name = context.config.artifacts.input_snapshot_file_name
            bundled_paths[missing_name] = None
            missing_optional_artifacts.append(missing_name)

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

    def _copy_tabular_as_csv_into_bundle(
        self,
        *,
        source_path: Path,
        destination_path: Path,
    ) -> str:
        if source_path.suffix.lower() == ".csv":
            return self._copy_file_into_bundle(
                source_path=source_path,
                destination_path=destination_path,
            )
        if source_path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(source_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(destination_path, index=False)
            return destination_path.name
        return self._copy_file_into_bundle(
            source_path=source_path,
            destination_path=destination_path,
        )

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

    def _ensure_layout_directories(self, paths: ExportPathLayout) -> None:
        directories = [
            paths.reports_dir,
            paths.model_dir,
            paths.data_input_dir,
            paths.data_predictions_dir,
            paths.tables_dir,
            paths.config_dir,
            paths.metadata_dir,
            paths.workbooks_dir,
            paths.code_dir,
        ]
        directories.extend(paths.tables_dir / name for name in TABLE_SECTION_DIRECTORIES.values())
        for directory in dict.fromkeys(directories):
            directory.mkdir(parents=True, exist_ok=True)

    def _table_export_path(self, tables_dir: Path, table_name: str) -> Path:
        section = infer_asset_section(table_name, kind="table")
        directory_name = TABLE_SECTION_DIRECTORIES.get(section, "other")
        return tables_dir / directory_name / f"{self._sanitize_name(table_name)}.csv"

    def _table_group_directories(self, tables_dir: Path) -> dict[str, str | None]:
        return {
            group_name: str(tables_dir / group_name) if (tables_dir / group_name).exists() else None
            for group_name in sorted(set(TABLE_SECTION_DIRECTORIES.values()))
        }

    def _write_manifest(self, path: Path, manifest: dict[str, Any], *, output_root: Path) -> None:
        manifest["artifact_layout_version"] = ARTIFACT_LAYOUT_VERSION
        manifest.pop("artifact_index", None)
        manifest["artifact_index"] = self._build_artifact_index(manifest, output_root)
        self._write_json(path, manifest)

    def _build_artifact_index(
        self,
        manifest: dict[str, Any],
        output_root: Path,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for key_path, path in self._iter_manifest_file_paths(manifest):
            try:
                resolved = path.resolve()
                relative_path = resolved.relative_to(output_root.resolve()).as_posix()
            except ValueError:
                relative_path = str(path)
            if relative_path in seen:
                continue
            seen.add(relative_path)
            category = self._artifact_category(relative_path)
            rows.append(
                {
                    "key": key_path,
                    "category": category,
                    "purpose": self._artifact_purpose(key_path, relative_path),
                    "relative_path": relative_path,
                    "send_to": self._artifact_audience(category, relative_path),
                }
            )
        return sorted(rows, key=lambda row: row["relative_path"])

    def _iter_manifest_file_paths(
        self,
        payload: Any,
        *,
        prefix: str = "",
    ) -> list[tuple[str, Path]]:
        paths: list[tuple[str, Path]] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key == "artifact_index":
                    continue
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                paths.extend(self._iter_manifest_file_paths(value, prefix=next_prefix))
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                paths.extend(self._iter_manifest_file_paths(value, prefix=f"{prefix}[{index}]"))
        elif isinstance(payload, str) and (":" in payload or "\\" in payload or "/" in payload):
            path = Path(payload)
            if path.suffix or path.exists():
                paths.append((prefix, path))
        return paths

    def _artifact_category(self, relative_path: str) -> str:
        top_level = relative_path.split("/", 1)[0]
        return {
            "reports": "reports",
            "model": "model",
            "data": "data",
            "tables": "tables",
            "config": "configuration",
            "metadata": "metadata",
            "workbooks": "workbooks",
            "code": "rerun_code",
            "figures": "figures",
            self.MONITORING_BUNDLE_DIRECTORY_NAME: "monitoring_bundle",
            "artifact_manifest.json": "manifest",
            "START_HERE.md": "orientation",
        }.get(top_level, "other")

    def _artifact_audience(self, category: str, relative_path: str) -> str:
        if category == "reports":
            return "model builders, validators, business reviewers"
        if category == "model":
            return "model builders, scoring workflows"
        if category == "data":
            return "model builders, validators, downstream scoring users"
        if category == "tables":
            return "validators, auditors, technical reviewers"
        if category in {"configuration", "metadata", "manifest"}:
            return "auditors, technical reviewers"
        if category == "rerun_code":
            return "developers"
        if category == "monitoring_bundle":
            return "monitoring application"
        if relative_path == "START_HERE.md":
            return "all reviewers"
        return "technical reviewers"

    def _artifact_purpose(self, key_path: str, relative_path: str) -> str:
        file_name = Path(relative_path).name
        purpose_by_name = {
            "START_HERE.md": "Plain-English orientation guide for the run folder.",
            "artifact_manifest.json": "Machine-readable index of exported artifacts.",
            "interactive_report.html": "Standalone visual diagnostic report.",
            "decision_summary.md": "Decision-ready scorecard for model review.",
            "run_report.md": "Markdown summary of run metrics, warnings, and diagnostics.",
            "model_documentation_pack.md": "Development-facing model documentation summary.",
            "validation_pack.md": "Validator-facing evidence index and review summary.",
            "quant_model.joblib": "Serialized fitted model object.",
            "feature_importance.csv": "Feature-level coefficients or importance values.",
            "model_summary.txt": "Text model summary from the fitted estimator.",
            "run_config.json": "Resolved configuration used for the run.",
            "configuration_template.xlsx": "Offline review workbook for configuration edits.",
            "metrics.json": "Structured metrics by split.",
            "statistical_tests.json": "Structured statistical-test payloads.",
            "step_manifest.json": "Ordered pipeline step stack.",
            "run_debug_trace.json": "Per-step debug trace and timing metadata.",
            "reproducibility_manifest.json": "Hashes, package versions, and environment metadata.",
            "analysis_workbook.xlsx": (
                "Excel workbook containing metrics, predictions, and diagnostics."
            ),
            "generated_run.py": "Python entry point for rerunning the workflow without the GUI.",
            "HOW_TO_RERUN.md": "Plain-English rerun instructions.",
        }
        if file_name in purpose_by_name:
            return purpose_by_name[file_name]
        if "predictions" in file_name:
            return "Row-level model scoring output."
        if relative_path.startswith("tables/"):
            return "Diagnostic table exported from the model-development workflow."
        if relative_path.startswith("figures/"):
            return "Individual chart export."
        if relative_path.startswith(self.MONITORING_BUNDLE_DIRECTORY_NAME):
            return "Copied artifact for the separate monitoring application."
        return f"Exported artifact recorded from `{key_path}`."

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    def _build_decision_summary(
        self,
        context: PipelineContext,
        *,
        artifacts: Mapping[str, Any],
    ) -> str:
        snapshot = build_decision_summary_snapshot_from_context(context, artifacts=artifacts)
        return build_decision_summary_markdown(snapshot)

    def _decision_summary_artifacts(
        self,
        *,
        model_path: Path | None,
        metrics_path: Path,
        predictions_path: Path | None,
        feature_importance_path: Path | None,
        report_path: Path,
        interactive_report_path: Path,
        decision_summary_path: Path,
        documentation_pack_path: Path | None,
        validation_pack_path: Path | None,
        config_path: Path,
        tests_path: Path,
        run_debug_trace_path: Path,
    ) -> dict[str, Path]:
        artifacts: dict[str, Path] = {
            "metrics": metrics_path,
            "report": report_path,
            "interactive_report": interactive_report_path,
            "decision_summary": decision_summary_path,
            "config": config_path,
            "tests": tests_path,
            "run_debug_trace": run_debug_trace_path,
        }
        optional_paths = {
            "model": model_path,
            "predictions": predictions_path,
            "feature_importance": feature_importance_path,
            "documentation_pack": documentation_pack_path,
            "validation_pack": validation_pack_path,
        }
        artifacts.update({key: path for key, path in optional_paths.items() if path is not None})
        return artifacts

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
            lines.append(f"- Challenger rows exported: `{len(context.comparison_results)}`")
            lines.append("")

        recommended_calibration_method = context.metadata.get("recommended_calibration_method")
        if recommended_calibration_method:
            recommended_score_column = context.metadata.get(
                "recommended_calibration_score_column",
                "predicted_probability",
            )
            lines.extend(["## Calibration", ""])
            lines.append(f"- Recommended method: `{recommended_calibration_method}`")
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
                "- `code/generated_run.py` replays the run bundle outside the GUI.",
                "- `config/run_config.json` stores the resolved config for the run.",
                (
                    "- `data/input/input_snapshot.csv` stores the ingested dataset "
                    "when input export is enabled."
                ),
                "- `metadata/step_manifest.json` stores the exact ordered pipeline step stack.",
                (
                    "- `metadata/reproducibility_manifest.json` captures hashes, versions, "
                    "and environment metadata."
                ),
                (
                    "- `config/configuration_template.xlsx` exports the review workbook "
                    "used for offline edits."
                ),
                "- `reports/committee_report.docx` and `reports/committee_report.pdf` provide "
                "committee-ready packaging when regulatory reporting is enabled.",
                "- `reports/validation_report.docx` and `reports/validation_report.pdf` provide "
                "validator-ready packaging when regulatory reporting is enabled.",
                "- `code/code_snapshot/` stores a Python copy of the framework, GUI, "
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
        decision_summary = build_decision_summary(
            build_decision_summary_snapshot_from_context(context)
        )
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

        lines.extend(
            [
                "## Decision Summary",
                "",
                f"- Recommendation: `{decision_summary['recommendation']}`",
                f"- Decision level: `{decision_summary['level'].replace('_', ' ').title()}`",
            ]
        )
        for rationale in decision_summary["rationale"]:
            lines.append(f"- {rationale}")
        lines.append("")

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
            lines.append(f"- Recommended model: `{recommended_model}`")
            lines.append(f"- Ranking split: `{context.config.comparison.ranking_split}`")
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
            lines.append(f"- Recommended score column: `{recommended_score_column}`")
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
        lines.append("- `reports/decision_summary.md` provides the decision-ready scorecard.")
        lines.append("- `reports/validation_pack.md` provides the validator-facing summary.")
        lines.append(
            "- `reports/committee_report.docx` / `reports/committee_report.pdf` provide "
            "committee-ready distribution assets."
        )
        lines.append(
            "- `reports/validation_report.docx` / `reports/validation_report.pdf` provide "
            "validator-ready distribution assets."
        )
        lines.append(
            "- `metadata/reproducibility_manifest.json` provides hashes, package versions, "
            "and run fingerprint metadata."
        )
        lines.append(
            "- `config/configuration_template.xlsx` exports the editable review workbook "
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
        decision_summary = build_decision_summary(
            build_decision_summary_snapshot_from_context(context)
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

        rows.extend(
            [
                "## Decision Summary",
                "",
                f"- Recommendation: `{decision_summary['recommendation']}`",
                f"- Decision level: `{decision_summary['level'].replace('_', ' ').title()}`",
            ]
        )
        for rationale in decision_summary["rationale"]:
            rows.append(f"- {rationale}")
        rows.append("")

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
                "- `reports/run_report.md` for the narrative run summary.",
                "- `reports/decision_summary.md` for the decision-ready scorecard.",
                "- `reports/model_documentation_pack.md` for development-facing documentation.",
                "- `reports/validation_pack.md` for validator-oriented review packaging.",
                (
                    "- `reports/committee_report.docx` / `reports/committee_report.pdf` "
                    "for committee-facing delivery."
                ),
                (
                    "- `reports/validation_report.docx` / `reports/validation_report.pdf` "
                    "for validator-facing delivery."
                ),
                "- `metadata/reproducibility_manifest.json` for run fingerprint metadata.",
                "- `config/configuration_template.xlsx` for offline governance editing.",
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
        rows.extend(self._build_input_source_manifest_rows(context))
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

    def _build_input_source_manifest_rows(self, context: PipelineContext) -> list[dict[str, Any]]:
        source_metadata = context.metadata.get("input_source")
        if not isinstance(source_metadata, dict):
            return []

        tracked_fields = [
            "source_kind",
            "display_label",
            "file_name",
            "relative_path",
            "suffix",
            "size_bytes",
            "modified_at_utc",
        ]
        return [
            {
                "field": f"input_source::{field_name}",
                "value": source_metadata.get(field_name, ""),
            }
            for field_name in tracked_fields
        ]

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
            exported_config.execution.existing_model_path = Path("..") / "model" / model_path.name
            exported_config.execution.existing_config_path = None
        return exported_config.to_dict()

    def _build_start_here_guide(
        self,
        *,
        context: PipelineContext,
        paths: ExportPathLayout,
    ) -> str:
        mode = context.config.execution.mode.value
        model_type = context.config.model.model_type.value
        lines = [
            "# Quant Studio Run Folder",
            "",
            f"Run ID: `{context.run_id}`",
            f"Execution mode: `{mode}`",
            f"Model family: `{model_type}`",
            f"Artifact layout version: `{ARTIFACT_LAYOUT_VERSION}`",
            "",
            "## Open First",
            "",
            "- `reports/decision_summary.md` for the decision-ready model scorecard.",
            "- `reports/interactive_report.html` for the visual diagnostic report.",
            "- `reports/validation_pack.md` for validator-facing evidence.",
            "- `artifact_manifest.json` for the machine-readable artifact index.",
            "",
            "## Folder Map",
            "",
            "- `reports/` contains HTML, Markdown, DOCX, and PDF reports.",
            "- `model/` contains the fitted model object, model summary, and feature importance.",
            "- `data/input/` contains the exported input snapshot when enabled.",
            "- `data/predictions/` contains row-level predictions and split predictions.",
            "- `tables/` contains diagnostic CSV/Parquet tables grouped by review topic.",
            "- `config/` contains the resolved run configuration and offline template workbook.",
            "- `metadata/` contains metrics, test payloads, manifests, and debug traces.",
            "- `checkpoints/` contains the stage manifest and retained checkpoint contexts.",
            "- `workbooks/` contains the optional analysis workbook.",
            "- `code/` contains rerun instructions, generated Python launcher, and code snapshot.",
            "- `figures/` contains optional individual chart HTML/PNG exports.",
            "- `model_bundle_for_monitoring/` is the handoff bundle for the monitoring app.",
            "",
            "## Common Tasks",
            "",
            "- Review results: open `reports/interactive_report.html`.",
            "- Review the model decision: open `reports/decision_summary.md`.",
            "- Reuse the model: use `model/quant_model.joblib`.",
            "- Rerun outside the GUI: open `code/HOW_TO_RERUN.md`.",
            "- Audit exact values: inspect `tables/` and `metadata/`.",
            "- Review staged execution: inspect `checkpoints/checkpoint_manifest.json`.",
            "- Send to monitoring: use `model_bundle_for_monitoring/`.",
            "",
            "## Key Paths",
            "",
            f"- Manifest: `{paths.manifest_path.relative_to(paths.output_root).as_posix()}`",
            f"- Config: `{paths.config_path.relative_to(paths.output_root).as_posix()}`",
            f"- Metrics: `{paths.metrics_path.relative_to(paths.output_root).as_posix()}`",
            (
                "- Report: "
                f"`{paths.interactive_report_path.relative_to(paths.output_root).as_posix()}`"
            ),
        ]
        return "\n".join(lines) + "\n"

    def _build_report_visualizations(self, context: PipelineContext) -> dict[str, Any]:
        """Builds the report-grade visualization set used by HTML and optional files."""

        visualizations: Mapping[str, Any] = context.visualizations
        if context.config.artifacts.include_enhanced_report_visuals:
            visualizations = enhance_report_visualizations(
                metrics=context.metrics,
                diagnostics_tables=context.diagnostics_tables,
                visualizations=visualizations,
                target_mode=context.config.target.mode.value,
                labels_available=bool(context.metadata.get("labels_available", False)),
                predictions=context.predictions,
            )
        else:
            visualizations = dict(visualizations)
        if context.config.artifacts.include_advanced_visual_analytics:
            visualizations = apply_advanced_visual_analytics(
                metrics=context.metrics,
                diagnostics_tables=context.diagnostics_tables,
                visualizations=visualizations,
                target_mode=context.config.target.mode.value,
                labels_available=bool(context.metadata.get("labels_available", False)),
                predictions=context.predictions,
            )
        report_limits = resolve_html_report_limits(context)
        optimized_visualizations, payload_audit = optimize_report_visualizations(
            dict(visualizations),
            max_points_per_figure=report_limits["max_points_per_figure"],
            max_figure_payload_mb=report_limits["max_figure_payload_mb"],
            max_total_figure_payload_mb=report_limits["max_total_figure_payload_mb"],
        )
        if not payload_audit.empty:
            context.diagnostics_tables["report_payload_audit"] = payload_audit
            summary_row = payload_audit[payload_audit["figure_name"] == "__total__"]
            if not summary_row.empty:
                context.metadata["interactive_report_payload"] = {
                    "figure_count": int(len(optimized_visualizations)),
                    "original_payload_mb": float(summary_row.iloc[0]["original_payload_mb"]),
                    "report_payload_mb": float(summary_row.iloc[0]["report_payload_mb"]),
                    "max_points_per_figure": report_limits["max_points_per_figure"],
                    "max_figure_payload_mb": report_limits["max_figure_payload_mb"],
                    "max_total_figure_payload_mb": report_limits[
                        "max_total_figure_payload_mb"
                    ],
                }
            skipped_count = int((payload_audit["action"] == "skipped").sum())
            downsampled_count = int((payload_audit["action"] == "downsampled").sum())
            if skipped_count or downsampled_count:
                context.warn(
                    "Interactive report size controls "
                    f"downsampled {downsampled_count:,} chart(s) and skipped "
                    f"{skipped_count:,} chart(s). Review report_payload_audit."
                )
        return optimized_visualizations

    def _build_interactive_report(
        self,
        context: PipelineContext,
        *,
        visualizations: Mapping[str, Any] | None = None,
    ) -> str:
        input_shape = context.metadata.get("input_shape", {})
        feature_summary = context.metadata.get("feature_summary", {})
        split_summary = context.metadata.get("split_summary", {})
        report_limits = resolve_html_report_limits(context)
        report_visualizations = (
            self._build_report_visualizations(context)
            if visualizations is None
            else visualizations
        )
        should_expand_visuals = False
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
            visualizations=report_visualizations,
            table_preview_rows=report_limits["table_preview_rows"],
            max_figures_per_section=report_limits["max_figures_per_section"],
            max_tables_per_section=report_limits["max_tables_per_section"],
            include_enhanced_report_visuals=(
                should_expand_visuals and context.config.artifacts.include_enhanced_report_visuals
            ),
            include_advanced_visual_analytics=(
                should_expand_visuals and context.config.artifacts.include_advanced_visual_analytics
            ),
            predictions=context.predictions,
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
        *,
        visualizations: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        html_enabled = self._html_figure_exports_enabled(context)
        png_enabled = self._png_figure_exports_enabled(context)
        manifest: dict[str, Any] = {
            "figure_file_exports": {
                "enabled": context.config.artifacts.export_individual_figure_files,
                "html_enabled": html_enabled,
                "png_enabled": png_enabled,
                "enhanced_report_visuals_enabled": (
                    context.config.artifacts.include_enhanced_report_visuals
                ),
                "advanced_visual_analytics_enabled": (
                    context.config.artifacts.include_advanced_visual_analytics
                ),
            },
            "figures": {},
        }
        if not context.config.artifacts.export_individual_figure_files:
            return manifest
        export_visualizations = context.visualizations if visualizations is None else visualizations
        for figure_name, figure in export_visualizations.items():
            safe_name = self._sanitize_name(figure_name)
            manifest["figures"][figure_name] = {}
            if html_enabled:
                html_path = html_dir / f"{safe_name}.html"
                figure.write_html(html_path, include_plotlyjs=True, full_html=True)
                manifest["figures"][figure_name]["html"] = str(html_path)
            if png_enabled:
                png_path = png_dir / f"{safe_name}.png"
                try:
                    pio.write_image(figure, png_path)
                    manifest["figures"][figure_name]["png"] = str(png_path)
                except Exception as exc:
                    context.warn(f"Could not export PNG for {figure_name}: {exc}")
        return manifest

    def _html_figure_exports_enabled(self, context: PipelineContext) -> bool:
        return bool(
            context.config.artifacts.export_individual_figure_files
            and context.config.diagnostics.interactive_visualizations
        )

    def _png_figure_exports_enabled(self, context: PipelineContext) -> bool:
        return bool(
            context.config.artifacts.export_individual_figure_files
            and context.config.diagnostics.static_image_exports
        )

    def _excel_workbook_enabled(self, context: PipelineContext) -> bool:
        return excel_workbook_enabled(context)

    def _regulatory_reports_enabled(self, context: PipelineContext) -> bool:
        return regulatory_reports_enabled(context)

    def _code_snapshot_enabled(self, context: PipelineContext) -> bool:
        return code_snapshot_enabled(context)

    def _input_snapshot_enabled(self, context: PipelineContext) -> bool:
        return input_snapshot_enabled(context)

    def _build_step_manifest(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "run_id": context.run_id,
            "steps": context.metadata.get("step_manifest", []),
            "pipeline_events": context.events,
        }

    def _build_run_debug_trace(self, context: PipelineContext) -> dict[str, Any]:
        output_root = context.config.artifacts.output_root / context.run_id
        return {
            "run_id": context.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "execution_mode": context.config.execution.mode.value,
            "model_type": context.config.model.model_type.value,
            "export_profile": context.config.artifacts.export_profile.value,
            "run_started_at_utc": context.metadata.get("run_started_at_utc", ""),
            "run_completed_at_utc": context.metadata.get("run_completed_at_utc", ""),
            "step_count": len(context.debug_trace),
            "steps": context.debug_trace,
            "summary": {
                "total_step_seconds": round(
                    sum(float(row.get("elapsed_seconds", 0.0)) for row in context.debug_trace),
                    6,
                ),
                "total_run_seconds": context.metadata.get("run_elapsed_seconds"),
                "diagnostic_table_count": len(context.diagnostics_tables),
                "visualization_count": len(context.visualizations),
                "warning_count": len(context.warnings),
                "artifact_count": len(context.artifacts),
                "artifact_output_root": str(output_root),
            },
        }

    def _build_generated_runner_script(self, context: PipelineContext) -> str:
        artifacts = context.config.artifacts
        return textwrap.dedent(
            f"""\
            from __future__ import annotations

            import sys
            from pathlib import Path

            THIS_DIR = Path(__file__).resolve().parent
            RUN_ROOT = THIS_DIR.parent
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
                        str(RUN_ROOT / "config" / "{artifacts.config_file_name}"),
                    ] + arguments
                if not _has_option(arguments, "--input"):
                    for default_input_name in [
                        "{artifacts.input_snapshot_file_name}",
                        "{artifacts.input_snapshot_parquet_file_name}",
                    ]:
                        default_input = RUN_ROOT / "data" / "input" / default_input_name
                        if default_input.exists():
                            arguments = ["--input", str(default_input)] + arguments
                            break
                if not _has_option(arguments, "--output-root"):
                    arguments = ["--output-root", str(RUN_ROOT / "reruns")] + arguments
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

            - uses `../config/{artifacts.config_file_name}` as the saved framework config
            - uses `../data/input/{artifacts.input_snapshot_file_name}` or
              `../data/input/{artifacts.input_snapshot_parquet_file_name}` as the default input
            - writes new rerun artifacts under `../reruns/`
            - imports the local `code_snapshot/src/` copy first, so step-level
              edits apply immediately

            ## Direct CLI Path

            You can also run the packaged CLI directly:

            ```powershell
            python -m quant_pd_framework.run ^
              --config ..\\config\\{artifacts.config_file_name} ^
              --input ..\\data\\input\\{artifacts.input_snapshot_file_name} ^
              --output-root ..\\reruns
            ```

            If you want to rerun on a different dataset, change the input path:

            ```powershell
            python -m quant_pd_framework.run ^
              --config ..\\config\\{artifacts.config_file_name} ^
              --input path\\to\\new_data.csv ^
              --output-root ..\\reruns
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
            `../metadata/{artifacts.step_manifest_file_name}`.
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
