"""Regression tests for validation checklist and evidence traceability outputs."""

from __future__ import annotations

import pandas as pd

from quant_pd_framework import (
    CleaningConfig,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.context import PipelineContext
from quant_pd_framework.validation_evidence import publish_validation_evidence_tables
from tests.support import build_binary_dataframe, build_common_schema


def test_validation_evidence_tables_are_published_from_context() -> None:
    dataframe = build_binary_dataframe(row_count=80)
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
    )
    context = PipelineContext(
        config=config,
        run_id="run_validation_evidence",
        raw_input=dataframe,
        raw_data=dataframe,
    )
    context.target_column = "default_status"
    context.feature_columns = ["annual_income", "utilization"]
    context.split_frames = {"train": dataframe.head(40), "test": dataframe.tail(20)}
    context.model = object()
    context.metrics = {"test": {"roc_auc": 0.82, "ks_statistic": 0.35}}
    context.feature_importance = pd.DataFrame(
        {"feature_name": ["annual_income"], "importance_value": [0.42]}
    )
    context.backtest_summary = pd.DataFrame({"split": ["test"], "rows": [20]})
    context.metadata = {
        "labels_available": True,
        "input_shape": {"rows": len(dataframe), "columns": len(dataframe.columns)},
        "split_summary": {"test": {"rows": 20}},
    }
    context.diagnostics_tables = {
        "calibration": pd.DataFrame({"bucket": [1], "observed_rate": [0.1]}),
        "report_payload_audit": pd.DataFrame(
            {"figure_name": ["__total__"], "action": ["summary"]}
        ),
    }

    publish_validation_evidence_tables(context)

    checklist = context.diagnostics_tables["validation_checklist"]
    traceability = context.diagnostics_tables["evidence_traceability_map"]
    assert "Performance metrics" in checklist["review_area"].tolist()
    assert "Report size and payload controls" in checklist["review_area"].tolist()
    assert "Attention needed" not in checklist.loc[
        checklist["review_area"] == "Performance metrics",
        "status",
    ].tolist()
    assert "report_payload_audit" in traceability["table_name"].tolist()
    assert "reports/interactive_report.html" in traceability["artifact_location"].tolist()
