"""Regression tests for exported run-bundle contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.reference_workflows import get_reference_workflow_definition
from quant_pd_framework.steps.export import ArtifactExportStep
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def _build_artifact_contract_config(output_root: Path) -> FrameworkConfig:
    return FrameworkConfig(
        schema=build_common_schema("account_id", include_legacy_drop=True),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        target=TargetConfig(
            source_column="default_status",
            mode=TargetMode.BINARY,
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.CROSS_SECTIONAL,
            date_column="as_of_date",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
        ),
        diagnostics=DiagnosticConfig(
            interactive_visualizations=True,
            static_image_exports=False,
        ),
        artifacts=ArtifactConfig(
            output_root=output_root,
            export_individual_figure_files=True,
        ),
    )


def _iter_manifest_paths(payload: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "artifact_index":
                continue
            paths.extend(_iter_manifest_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            paths.extend(_iter_manifest_paths(value))
    elif isinstance(payload, str) and (":" in payload or "\\" in payload or "/" in payload):
        paths.append(Path(payload))
    return paths


def test_artifact_manifest_indexes_core_outputs_and_rerun_bundle() -> None:
    dataframe = build_binary_dataframe(row_count=220)

    with temporary_artifact_root("pytest_artifact_contracts") as artifact_root:
        config = _build_artifact_contract_config(artifact_root)
        context = QuantModelOrchestrator(config=config).run(dataframe)

        manifest_path = Path(context.artifacts["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["core_artifacts"]["artifact_manifest"] == str(manifest_path)
        assert manifest["core_artifacts"]["output_root"] == str(context.artifacts["output_root"])
        assert manifest["core_artifacts"]["decision_summary"] == str(
            context.artifacts["decision_summary"]
        )
        assert manifest["rerun_bundle"]["step_manifest"] == str(context.artifacts["step_manifest"])
        assert manifest["rerun_bundle"]["runner_script"] == str(context.artifacts["runner_script"])
        assert manifest["rerun_bundle"]["rerun_readme"] == str(context.artifacts["rerun_readme"])
        assert "input_snapshot" in manifest["rerun_bundle"]
        assert "code_snapshot" in manifest["rerun_bundle"]
        assert "analysis_workbook" in manifest["core_artifacts"]
        assert manifest["artifact_layout_version"] == "2.0"
        assert manifest["core_artifacts"]["start_here"] == str(context.artifacts["start_here"])
        assert any(
            row["relative_path"] == "START_HERE.md" and row["send_to"] == "all reviewers"
            for row in manifest["artifact_index"]
        )

        for path in _iter_manifest_paths(manifest):
            assert path.exists(), path

        tables_dir = Path(manifest["directories"]["tables"])
        html_dir = Path(manifest["directories"]["figures_html"])
        code_snapshot_dir = Path(manifest["rerun_bundle"]["code_snapshot"])
        assert any(tables_dir.rglob("*.csv"))
        assert any(html_dir.glob("*.html"))
        assert Path(manifest["directories"]["metadata"]).exists()
        assert Path(manifest["directories"]["config"]).exists()
        assert Path(context.artifacts["decision_summary"]).exists()
        assert (code_snapshot_dir / "src" / "quant_pd_framework" / "run.py").exists()
        assert (code_snapshot_dir / "app" / "streamlit_app.py").exists()


def test_run_artifact_folder_uses_readable_datetime_name() -> None:
    dataframe = build_binary_dataframe(row_count=120)

    with temporary_artifact_root("pytest_readable_run_id") as artifact_root:
        config = _build_artifact_contract_config(artifact_root)
        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert re.fullmatch(
            r"run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC",
            context.run_id,
        )
        assert Path(context.artifacts["output_root"]).name == context.run_id


def test_artifact_manifest_can_skip_individual_figure_exports() -> None:
    dataframe = build_binary_dataframe(row_count=220)

    with temporary_artifact_root("pytest_artifact_contracts_no_figure_files") as artifact_root:
        config = _build_artifact_contract_config(artifact_root)
        config.artifacts = ArtifactConfig(
            output_root=artifact_root,
            export_individual_figure_files=False,
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)

        manifest_path = Path(context.artifacts["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert Path(context.artifacts["interactive_report"]).exists()
        assert manifest["figure_file_exports"] == {
            "enabled": False,
            "html_enabled": False,
            "png_enabled": False,
            "enhanced_report_visuals_enabled": True,
            "advanced_visual_analytics_enabled": False,
        }
        assert manifest["figures"] == {}
        assert manifest["directories"]["figures"] is None
        assert manifest["directories"]["figures_html"] is None
        assert manifest["directories"]["figures_png"] is None


def test_advanced_visual_analytics_can_export_individual_figure_file() -> None:
    with temporary_artifact_root("pytest_advanced_visual_export") as artifact_root:
        config = SimpleNamespace(
            artifacts=ArtifactConfig(
                output_root=artifact_root,
                export_individual_figure_files=True,
                include_enhanced_report_visuals=False,
                include_advanced_visual_analytics=True,
            ),
            diagnostics=SimpleNamespace(
                interactive_visualizations=True,
                static_image_exports=False,
            ),
            target=SimpleNamespace(mode=TargetMode.BINARY),
        )
        context = SimpleNamespace(
            config=config,
            metadata={"labels_available": True},
            metrics={"test": {"roc_auc": 0.82}},
            diagnostics_tables={
                "feature_importance": pd.DataFrame(
                    {"feature_name": ["balance"], "coefficient": [0.55]}
                )
            },
            visualizations={},
            predictions={
                "test": pd.DataFrame(
                    {
                        "predicted_probability": [0.1, 0.2, 0.8, 0.9],
                        "default_status": [0, 0, 1, 1],
                        "balance": [100, 200, 300, 400],
                    }
                )
            },
            warn=lambda message: None,
        )
        step = ArtifactExportStep()
        visualizations = step._build_report_visualizations(context)
        html_dir = artifact_root / "figures" / "html"
        png_dir = artifact_root / "figures" / "png"
        html_dir.mkdir(parents=True)

        manifest = step._export_visualizations(
            context,
            html_dir,
            png_dir,
            visualizations=visualizations,
        )

        assert "advanced_contribution_beeswarm" in manifest["figures"]
        exported_html = Path(manifest["figures"]["advanced_contribution_beeswarm"]["html"])
        assert exported_html.exists()
        assert manifest["figure_file_exports"]["advanced_visual_analytics_enabled"] is True


def test_individual_figure_exports_default_to_disabled() -> None:
    assert ArtifactConfig().export_individual_figure_files is False
    assert ArtifactConfig().include_enhanced_report_visuals is True
    assert ArtifactConfig().include_advanced_visual_analytics is False


def test_reference_workflow_bundle_contract_contains_expected_sections() -> None:
    definition = get_reference_workflow_definition("pd_development")

    with temporary_artifact_root("pytest_reference_contract_bundle") as artifact_root:
        context = definition.run(output_root=artifact_root)

        interactive_report = Path(context.artifacts["interactive_report"]).read_text(
            encoding="utf-8"
        )
        validation_pack = Path(context.artifacts["validation_pack"]).read_text(encoding="utf-8")
        documentation_pack = Path(context.artifacts["documentation_pack"]).read_text(
            encoding="utf-8"
        )

        assert "Governance / Export Bundle" in interactive_report
        assert "Model Performance" in interactive_report
        assert "## Artifact Index" in validation_pack
        assert (
            "- `metadata/reproducibility_manifest.json` for run fingerprint metadata."
            in validation_pack
        )
        assert "## Development Summary" in documentation_pack
        assert "## Decision Summary" in documentation_pack
        assert "## Calibration Review" in documentation_pack
