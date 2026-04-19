"""Writes model artifacts, scored output, and a human-readable run report."""

from __future__ import annotations

import json
import shutil
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.io as pio

from ..base import BasePipelineStep
from ..config import ExecutionMode
from ..context import PipelineContext
from ..presentation import build_interactive_report_html


class ArtifactExportStep(BasePipelineStep):
    """
    Persists the outputs a quant practitioner usually needs after a model run.

    The exported bundle includes the trained model, scored observations, metrics,
    feature importance, backtesting output, diagnostics, and a rerun-ready code bundle.
    """

    name = "artifact_export"

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
        runner_script_path = output_root / context.config.artifacts.runner_script_file_name
        rerun_readme_path = output_root / context.config.artifacts.rerun_readme_file_name
        code_snapshot_dir = output_root / context.config.artifacts.code_snapshot_directory_name

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
        interactive_report_path.write_text(
            self._build_interactive_report(context),
            encoding="utf-8",
        )
        step_manifest = self._build_step_manifest(context)
        self._write_json(step_manifest_path, step_manifest)
        manifest = {
            **visualization_manifest,
            "interactive_report": str(interactive_report_path),
            "rerun_bundle": {
                "step_manifest": str(step_manifest_path),
                "runner_script": str(runner_script_path),
                "rerun_readme": str(rerun_readme_path),
            },
        }
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

        context.artifacts = {
            "output_root": output_root,
            "model": model_path,
            "metrics": metrics_path,
            "input_snapshot": input_snapshot_path if input_snapshot_path.exists() else None,
            "predictions": predictions_path,
            "feature_importance": feature_importance_path,
            "backtest": backtest_path,
            "report": report_path,
            "interactive_report": interactive_report_path,
            "config": config_path,
            "tests": tests_path,
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
        }
        return context

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

    def _sanitize_name(self, name: str) -> str:
        return "".join(
            character if character.isalnum() or character in {"_", "-"} else "_"
            for character in name
        )

    def _sanitize_sheet_name(self, name: str) -> str:
        return self._sanitize_name(name)[:31] or "sheet"
