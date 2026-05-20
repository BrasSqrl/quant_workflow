"""Canonical small-data workflow reliability tests.

These tests define the lightweight product smoke contract for normal in-memory
runs. They intentionally validate artifact contents, not just file existence.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SchemaConfig,
    ScorecardConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.safe_serialization import load_joblib_verified
from quant_pd_framework.sample_data import build_sample_pd_dataframe
from tests.support import temporary_artifact_root

CANONICAL_FEATURES: tuple[tuple[str, str], ...] = (
    ("annual_income", "float"),
    ("utilization", "float"),
    ("debt_to_income", "float"),
    ("delinquency_count", "int"),
    ("days_past_due", "int"),
    ("inquiries", "int"),
    ("region", "string"),
    ("industry", "string"),
    ("risk_rating", "string"),
    ("statement_quality", "string"),
    ("current_ratio", "float"),
    ("debt_to_assets", "float"),
    ("dscr", "float"),
    ("interest_coverage", "float"),
    ("macro_unemployment_rate", "float"),
    ("gdp_growth_rate", "float"),
)
NOISY_WARNING_PATTERNS = (
    "use_container_width",
    "Serialization of dataframe to Arrow",
    "Data Validation extension is not supported",
    "invalid value encountered",
    "divide by zero encountered",
)


def test_canonical_bundled_logistic_run_exports_reusable_evidence() -> None:
    with temporary_artifact_root("pytest_canonical_logistic") as artifact_root:
        context, recorded_warnings = _run_canonical_workflow(
            model_type=ModelType.LOGISTIC_REGRESSION,
            artifact_root=artifact_root,
        )

        _assert_no_noisy_warnings(recorded_warnings)
        _assert_common_canonical_outputs(context, expected_model_type="logistic_regression")
        _assert_binary_prediction_contract(
            context.predictions["test"],
            target_column=context.target_column,
        )

        loaded_model = load_joblib_verified(Path(context.artifacts["model"]))
        assert loaded_model.predict_score(
            context.split_frames["test"][context.feature_columns]
        ).shape[0] == len(context.split_frames["test"])


def test_canonical_bundled_scorecard_run_exports_scorecard_evidence() -> None:
    with temporary_artifact_root("pytest_canonical_scorecard") as artifact_root:
        context, recorded_warnings = _run_canonical_workflow(
            model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION,
            artifact_root=artifact_root,
        )

        _assert_no_noisy_warnings(recorded_warnings)
        _assert_common_canonical_outputs(
            context,
            expected_model_type="scorecard_logistic_regression",
        )
        _assert_binary_prediction_contract(
            context.predictions["test"],
            target_column=context.target_column,
        )
        assert {"scorecard_points", "reason_code_1", "reason_code_2", "reason_code_3"}.issubset(
            context.predictions["test"].columns
        )
        for table_name in (
            "scorecard_woe_table",
            "scorecard_points_table",
            "scorecard_scaling_summary",
            "scorecard_feature_summary",
        ):
            assert table_name in context.diagnostics_tables
            assert not context.diagnostics_tables[table_name].empty


def _run_canonical_workflow(
    *,
    model_type: ModelType,
    artifact_root: Path,
) -> tuple[object, list[warnings.WarningMessage]]:
    dataframe = build_sample_pd_dataframe(row_count=1000, random_state=42)
    config = _canonical_config(artifact_root=artifact_root, model_type=model_type)
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        context = QuantModelOrchestrator(config=config).run(dataframe)
    _assert_artifact_root_is_temporary(context, artifact_root)
    return context, list(recorded_warnings)


def _canonical_config(*, artifact_root: Path, model_type: ModelType) -> FrameworkConfig:
    column_specs = [
        ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
        ColumnSpec(name="loan_id", dtype="string", role=ColumnRole.IDENTIFIER),
        ColumnSpec(name="default_status", dtype="int", role=ColumnRole.TARGET_SOURCE),
        ColumnSpec(name="legacy_text_field", enabled=False),
        *[
            ColumnSpec(name=feature_name, dtype=dtype, role=ColumnRole.FEATURE)
            for feature_name, dtype in CANONICAL_FEATURES
        ],
    ]
    return FrameworkConfig(
        schema=SchemaConfig(
            column_specs=column_specs,
            pass_through_unconfigured_columns=False,
        ),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(
            derive_date_parts=False,
            drop_raw_date_columns=True,
        ),
        target=TargetConfig(
            source_column="default_status",
            mode=TargetMode.BINARY,
            positive_values=[1],
        ),
        split=SplitConfig(
            data_structure=DataStructure.PANEL,
            date_column="as_of_date",
            entity_column="loan_id",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_state=42,
        ),
        model=ModelConfig(
            model_type=model_type,
            max_iter=1000,
            scorecard_bins=5,
            threshold=0.5,
        ),
        scorecard=ScorecardConfig(reason_code_count=3),
        diagnostics=DiagnosticConfig(
            adf_analysis=False,
            model_specification_tests=False,
            forecasting_statistical_tests=False,
            interactive_visualizations=True,
            static_image_exports=False,
            export_excel_workbook=False,
            quantile_bucket_count=6,
            default_segment_column="risk_rating",
            max_plot_rows=5000,
        ),
        artifacts=ArtifactConfig(
            output_root=artifact_root,
            export_code_snapshot=False,
            export_individual_figure_files=False,
            include_enhanced_report_visuals=True,
            include_advanced_visual_analytics=False,
        ),
    )


def _assert_common_canonical_outputs(context: object, *, expected_model_type: str) -> None:
    artifacts = context.artifacts
    manifest = json.loads(Path(artifacts["manifest"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(artifacts["metrics"]).read_text(encoding="utf-8"))
    run_config = json.loads(Path(artifacts["config"]).read_text(encoding="utf-8"))
    decision_summary = Path(artifacts["decision_summary"]).read_text(encoding="utf-8")
    interactive_report = Path(artifacts["interactive_report"]).read_text(encoding="utf-8")

    assert Path(artifacts["model"]).exists()
    assert Path(str(artifacts["model"]) + ".sha256").exists()
    assert run_config["model"]["model_type"] == expected_model_type
    assert run_config["schema"]["pass_through_unconfigured_columns"] is False
    assert manifest["core_artifacts"]["model"] == str(artifacts["model"])
    assert manifest["core_artifacts"]["decision_summary"] == str(artifacts["decision_summary"])
    assert manifest["figure_file_exports"]["enabled"] is False
    assert set(metrics).issuperset({"train", "validation", "test"})
    assert metrics["test"]["labels_available"] is True
    assert metrics["test"]["roc_auc"] is not None
    assert metrics["test"]["ks_statistic"] is not None

    required_tables = [
        "split_metrics",
        "feature_importance",
        "backtest_summary",
        "calibration",
        "threshold_analysis",
        "lift_gain",
        "validation_checklist",
        "evidence_traceability_map",
        "report_payload_audit",
    ]
    for table_name in required_tables:
        table = context.diagnostics_tables.get(table_name, pd.DataFrame())
        assert not table.empty, table_name

    for split_name in ("train", "validation", "test"):
        assert split_name in context.predictions
        assert not context.predictions[split_name].empty

    assert "# Decision Summary" in decision_summary
    assert f"- Model family: `{expected_model_type}`" in decision_summary
    assert "- Target mode: `binary`" in decision_summary
    assert "## Primary Metrics" in decision_summary
    assert "## Evidence Index" in decision_summary
    assert "Model Performance" in interactive_report
    assert "Calibration / Thresholds" in interactive_report
    assert "Governance / Export Bundle" in interactive_report
    assert "Traceback" not in interactive_report
    assert "&lt;div" not in interactive_report


def _assert_binary_prediction_contract(predictions: pd.DataFrame, *, target_column: str) -> None:
    required_columns = {
        "as_of_date",
        "loan_id",
        target_column,
        "predicted_probability",
        "predicted_class",
    }
    assert required_columns.issubset(predictions.columns)
    assert predictions["predicted_probability"].between(0, 1).all()
    assert set(predictions["predicted_class"].dropna().unique()).issubset({0, 1})


def _assert_no_noisy_warnings(recorded_warnings: list[warnings.WarningMessage]) -> None:
    messages = [str(warning.message) for warning in recorded_warnings]
    noisy = [
        message
        for message in messages
        if any(pattern in message for pattern in NOISY_WARNING_PATTERNS)
    ]
    assert not noisy


def _assert_artifact_root_is_temporary(context: object, artifact_root: Path) -> None:
    assert Path(context.artifacts["output_root"]).is_relative_to(artifact_root)
