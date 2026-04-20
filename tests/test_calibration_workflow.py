"""Regression tests for the development calibration workflow."""

from __future__ import annotations

from quant_pd_framework import (
    ArtifactConfig,
    CalibrationConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
)
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


def test_binary_run_exports_calibration_method_comparison_outputs() -> None:
    dataframe = build_binary_dataframe(row_count=280)
    with temporary_artifact_root("pytest_calibration_workflow") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="default_status",
                output_column="default_flag",
                positive_values=[1],
            ),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                random_state=17,
                stratify=True,
            ),
            calibration=CalibrationConfig(
                bin_count=8,
                platt_scaling=True,
                isotonic_calibration=True,
            ),
            diagnostics=DiagnosticConfig(
                interactive_visualizations=False,
                static_image_exports=False,
                export_excel_workbook=False,
                quantile_bucket_count=8,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "calibration" in context.diagnostics_tables
        assert "calibration_summary" in context.diagnostics_tables
        assert "calibration_curve" in context.visualizations
        assert "calibration_method_comparison" in context.visualizations
        assert "calibration_methods" in context.statistical_tests
        summary = context.diagnostics_tables["calibration_summary"]
        assert {"base", "platt", "isotonic"}.issubset(set(summary["method_name"]))
        assert context.metadata["recommended_calibration_method"] in set(summary["method_name"])
        assert context.metadata["calibration_ranking_metric"] == "brier_score"
        assert "predicted_probability_recommended" in context.predictions["test"].columns
        assert "predicted_probability_platt" in context.predictions["test"].columns
        assert "predicted_probability_isotonic" in context.predictions["test"].columns
