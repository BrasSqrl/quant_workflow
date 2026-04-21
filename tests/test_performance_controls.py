"""Regression tests for performance and memory guardrails."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_pd_framework import (
    AdvancedImputationConfig,
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    DiagnosticConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    PerformanceConfig,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from tests.support import build_binary_dataframe, build_common_schema, temporary_artifact_root


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
