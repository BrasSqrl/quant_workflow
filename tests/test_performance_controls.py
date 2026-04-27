"""Regression tests for performance and memory guardrails."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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
from quant_pd_framework.diagnostic_frameworks import _align_comparison_prediction_frames
from quant_pd_framework.diagnostics import predict_modified_frames
from quant_pd_framework.steps.comparison import ModelComparisonStep
from quant_pd_framework.steps.diagnostics import DiagnosticsStep
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


def test_imputation_score_delta_uses_row_order_not_pandas_index_alignment() -> None:
    context = SimpleNamespace(
        config=SimpleNamespace(target=SimpleNamespace(mode=TargetMode.BINARY))
    )
    step = DiagnosticsStep()
    baseline_frame = pd.DataFrame(
        {"predicted_probability": [0.10, 0.20, 0.30]},
        index=[0, 0, 0],
    )
    alternative_frame = pd.DataFrame(
        {"predicted_probability": [0.20, 0.25]},
        index=[0, 0],
    )

    baseline_scores = step._prediction_score_array(baseline_frame, context)
    alternative_scores = step._prediction_score_array(alternative_frame, context)
    deltas = step._score_delta_statistics(
        baseline_scores=baseline_scores,
        alternative_scores=alternative_scores,
    )

    assert deltas["average_score_delta"] == pytest.approx(0.075)
    assert deltas["average_abs_score_delta"] == pytest.approx(0.075)
    assert deltas["max_abs_score_delta"] == pytest.approx(0.10)

    sampled_context = SimpleNamespace(
        config=SimpleNamespace(
            performance=PerformanceConfig(diagnostic_sample_rows=2),
            split=SimpleNamespace(random_state=42),
        ),
        diagnostics_tables={},
    )
    positions = step._diagnostic_row_positions(
        context=sampled_context,
        row_count=5,
        action_name="test_sample",
        detail="Test sampling.",
    )

    assert len(positions) == 2
    assert "test_sample" in set(
        sampled_context.diagnostics_tables["performance_hardening_actions"]["action_name"]
    )


def test_model_comparison_alignment_avoids_many_to_many_date_merge() -> None:
    context = SimpleNamespace(
        target_column="target",
        config=SimpleNamespace(
            target=SimpleNamespace(mode=TargetMode.BINARY),
            split=SimpleNamespace(date_column="as_of_date", entity_column=None),
        ),
    )
    primary_frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2024-01-01"] * 3),
            "target": [0, 1, 0],
            "predicted_probability": [0.1, 0.8, 0.2],
            "predicted_class": [0, 1, 0],
        }
    )
    challenger_frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2024-01-01"] * 2),
            "predicted_probability": [0.2, 0.7],
            "predicted_class": [0, 1],
        }
    )

    aligned = _align_comparison_prediction_frames(
        primary_frame=primary_frame,
        challenger_frame=challenger_frame,
        context=context,
    )

    assert aligned is not None
    assert len(aligned) == 2
    assert "predicted_probability_primary" in aligned.columns
    assert "predicted_probability_challenger" in aligned.columns


def test_model_comparison_snapshots_are_sampled_by_position() -> None:
    step = ModelComparisonStep()
    context = SimpleNamespace(
        config=SimpleNamespace(
            performance=PerformanceConfig(diagnostic_sample_rows=3),
            split=SimpleNamespace(random_state=7),
        ),
        metadata={},
    )
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2024-01-01"] * 10),
            "target": [0, 1] * 5,
            "predicted_probability": np.linspace(0.1, 0.9, 10),
        },
        index=[0] * 10,
    )
    sampled_positions: dict[str, np.ndarray] = {}

    first = step._comparison_prediction_snapshot(
        context=context,
        frame=frame,
        split_name="test",
        sampled_positions_by_split=sampled_positions,
    )
    second = step._comparison_prediction_snapshot(
        context=context,
        frame=frame.assign(predicted_probability=np.linspace(0.2, 0.8, 10)),
        split_name="test",
        sampled_positions_by_split=sampled_positions,
    )

    assert len(first) == 3
    assert len(second) == 3
    assert first.index.tolist() == [0, 1, 2]
    assert context.metadata["comparison_prediction_snapshot_sample"]["snapshot_rows"] == 3


def test_transition_outputs_skip_high_cardinality_state_candidates() -> None:
    step = DiagnosticsStep()
    context = SimpleNamespace(
        target_column="target",
        metadata={},
        diagnostics_tables={},
        visualizations={},
        warnings=[],
        config=SimpleNamespace(
            performance=PerformanceConfig(max_categorical_cardinality=20),
            credit_risk=SimpleNamespace(top_segments=4, migration_state_column="rating_grade"),
            split=SimpleNamespace(date_column="as_of_date", entity_column="account_id"),
        ),
        warn=lambda message: context.warnings.append(message),
    )
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 60,
            "account_id": np.repeat(np.arange(30), 2),
            "as_of_date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "rating_grade": [f"unique_grade_{index}" for index in range(60)],
            "target": [0, 1] * 30,
            "predicted_probability": np.linspace(0.05, 0.95, 60),
        }
    )

    step._add_transition_outputs(
        context=context,
        predictions=predictions,
        date_column="as_of_date",
        entity_column="account_id",
    )

    assert "migration_matrix" not in context.diagnostics_tables
    assert "migration_heatmap" not in context.visualizations
    assert context.metadata["skipped_transition_state_candidates"][0]["column_name"] == (
        "rating_grade"
    )
    assert any("too high-cardinality" in warning for warning in context.warnings)


def test_transition_outputs_do_not_auto_detect_state_columns_by_default() -> None:
    step = DiagnosticsStep()
    context = SimpleNamespace(
        target_column="target",
        metadata={},
        diagnostics_tables={},
        visualizations={},
        warnings=[],
        config=SimpleNamespace(
            performance=PerformanceConfig(max_categorical_cardinality=20),
            credit_risk=SimpleNamespace(top_segments=4, migration_state_column=None),
            split=SimpleNamespace(date_column="as_of_date", entity_column="account_id"),
        ),
        warn=lambda message: context.warnings.append(message),
    )
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 6,
            "account_id": [1, 1, 2, 2, 3, 3],
            "as_of_date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "rating_grade": ["A", "B", "A", "C", "B", "C"],
            "target": [0, 1, 0, 0, 1, 0],
            "predicted_probability": np.linspace(0.05, 0.95, 6),
        }
    )

    step._add_transition_outputs(
        context=context,
        predictions=predictions,
        date_column="as_of_date",
        entity_column="account_id",
    )

    assert "migration_matrix" not in context.diagnostics_tables
    assert context.warnings == []


def test_transition_heatmap_uses_bounded_observed_state_matrix() -> None:
    step = DiagnosticsStep()
    context = SimpleNamespace(
        warnings=[],
        config=SimpleNamespace(
            performance=PerformanceConfig(max_categorical_cardinality=20),
            credit_risk=SimpleNamespace(top_segments=4),
        ),
        warn=lambda message: context.warnings.append(message),
    )
    grouped = pd.DataFrame(
        {
            "current_state": ["A", "A", "B", "C"],
            "next_state": ["B", "C", "C", "A"],
            "transition_count": [10, 2, 3, 1],
        }
    )

    heatmap = step._build_transition_heatmap_table(
        context=context,
        grouped=grouped,
        state_column="rating_grade",
    )

    assert heatmap.loc["A", "B"] == 10
    assert heatmap.loc["B", "C"] == 3
    assert heatmap.shape == (3, 3)


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
                upload_warning_mb=5_120,
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

        progress_events: list[dict[str, object]] = []
        context = QuantModelOrchestrator(
            config=config,
            progress_callback=progress_events.append,
        ).run(dataframe)

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
        assert context.metadata["run_elapsed_seconds"] >= 0
        assert trace_payload["run_started_at_utc"]
        assert trace_payload["run_completed_at_utc"]
        assert trace_payload["summary"]["total_run_seconds"] >= 0
        assert trace_payload["step_count"] == len(context.debug_trace)
        assert trace_payload["steps"][-1]["step_name"] == "artifact_export"
        assert progress_events[0]["event_type"] == "run_started"
        assert progress_events[-1]["event_type"] == "run_completed"
        assert any(event["event_type"] == "step_started" for event in progress_events)


def test_compact_predictions_and_diagnostic_working_snapshot_reduce_retained_rows() -> None:
    dataframe = build_binary_dataframe(row_count=180)

    with temporary_artifact_root("pytest_memory_compaction") as artifact_root:
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
            performance=PerformanceConfig(
                diagnostic_sample_rows=30,
                retain_full_working_data=False,
                capture_memory_profile=True,
            ),
            artifacts=ArtifactConfig(
                output_root=artifact_root,
                compact_prediction_exports=True,
                export_code_snapshot=False,
                export_individual_figure_files=False,
            ),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        snapshot_metadata = context.metadata["working_data_snapshot"]
        prediction_frame = next(iter(context.predictions.values()))

        assert snapshot_metadata["sample_strategy"] == "stratified_split_sample"
        assert "working_data_snapshot" in context.diagnostics_tables
        assert snapshot_metadata["snapshot_rows"] <= 30
        assert len(context.working_data) <= 30
        assert "predicted_probability" in prediction_frame.columns
        assert "balance" not in prediction_frame.columns
        assert context.debug_trace[-1]["after"]["prediction_memory_bytes"] > 0
        hardening_actions = context.diagnostics_tables["performance_hardening_actions"]
        assert "model_specification_sample" in set(hardening_actions["action_name"])
        assert "specification_framework_sample" in set(hardening_actions["action_name"])


def test_high_cardinality_categorical_features_are_blocked_by_default() -> None:
    dataframe = build_binary_dataframe(row_count=80)
    dataframe["high_cardinality_feature"] = [f"category_{index}" for index in range(len(dataframe))]

    with temporary_artifact_root("pytest_high_cardinality_guardrail") as artifact_root:
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
            performance=PerformanceConfig(
                max_categorical_cardinality=20,
                max_categorical_cardinality_ratio=0.2,
                allow_high_cardinality_categoricals=False,
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        with pytest.raises(ValueError, match="High-cardinality categorical model features"):
            QuantModelOrchestrator(config=config).run(dataframe)
