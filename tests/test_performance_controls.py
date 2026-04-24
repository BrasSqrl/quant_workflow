"""Regression tests for performance and memory guardrails."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pd_framework import (
    AdvancedImputationConfig,
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    ExportProfile,
    FeatureEngineeringConfig,
    FrameworkConfig,
    PerformanceConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.diagnostics import predict_modified_frames
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


class _CountingModel:
    def __init__(self) -> None:
        self.call_count = 0

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        self.call_count += 1
        return frame["x"].to_numpy(dtype=float) + frame["y"].to_numpy(dtype=float)


def test_predict_modified_frames_scores_batch_in_one_model_call() -> None:
    model = _CountingModel()
    base_frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})

    scores = predict_modified_frames(
        model=model,
        base_frame=base_frame,
        feature_columns=["x", "y"],
        modifications=[{"x": 5.0}, {"y": pd.Series([1.0, 1.5, 2.0])}],
    )

    assert model.call_count == 1
    assert [score.tolist() for score in scores] == [
        [15.0, 25.0, 35.0],
        [2.0, 3.5, 5.0],
    ]


def test_multiple_imputation_row_cap_records_performance_actions() -> None:
    dataframe = build_binary_dataframe(row_count=320)
    for column_name in ("balance", "utilization", "tenure_months", "recent_inquiries"):
        dataframe.loc[dataframe.index % 4 == 0, column_name] = np.nan

    with temporary_artifact_root("pytest_performance_controls") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
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
            advanced_imputation=AdvancedImputationConfig(
                enabled=True,
                iterative_max_iter=5,
                minimum_complete_rows=20,
                multiple_imputation_enabled=True,
                multiple_imputation_datasets=2,
                multiple_imputation_evaluation_split="test",
                multiple_imputation_top_features=6,
            ),
            performance=PerformanceConfig(
                upload_warning_mb=250,
                dataframe_warning_rows=200_000,
                dataframe_warning_columns=150,
                ui_preview_rows=25,
                html_table_preview_rows=4,
                html_max_figures_per_section=3,
                html_max_tables_per_section=3,
                multiple_imputation_row_cap=24,
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        actions = context.diagnostics_tables["performance_hardening_actions"]
        assert set(actions["action_name"]) >= {
            "multiple_imputation_train_sample",
            "multiple_imputation_eval_sample",
        }

        interactive_report = Path(context.artifacts["interactive_report"]).read_text(
            encoding="utf-8"
        )
        assert "Showing the first 3 figures in this section in the HTML view." in interactive_report
        assert "Showing the first 3 tables in this section in the HTML view." in interactive_report


def test_fast_export_profile_skips_heavy_assets_and_writes_debug_trace() -> None:
    dataframe = build_binary_dataframe(row_count=180)

    with temporary_artifact_root("pytest_fast_export_profile") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
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
                output_root=artifact_root,
                export_profile=ExportProfile.FAST,
                export_input_snapshot=True,
                export_code_snapshot=True,
            ),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        manifest = json.loads(Path(context.artifacts["manifest"]).read_text(encoding="utf-8"))
        trace_path = Path(context.artifacts["run_debug_trace"])
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))

        assert manifest["core_artifacts"]["export_profile"] == ExportProfile.FAST.value
        assert "analysis_workbook" not in manifest["core_artifacts"]
        assert "input_snapshot" not in manifest["rerun_bundle"]
        assert "code_snapshot" not in manifest["rerun_bundle"]
        assert context.artifacts["workbook"] is None
        assert context.artifacts["input_snapshot"] is None
        assert context.artifacts["code_snapshot_dir"] is None
        assert trace_payload["step_count"] == len(context.debug_trace)
        assert trace_payload["steps"][-1]["step_name"] == "artifact_export"
