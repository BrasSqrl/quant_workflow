"""Smoke test for the end-to-end probability of default pipeline."""

from __future__ import annotations

import numpy as np
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
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
)
from tests.support import temporary_artifact_root


def build_synthetic_dataframe(row_count: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    fico_bucket = rng.choice(["A", "B", "C", "D"], size=row_count, p=[0.15, 0.35, 0.35, 0.15])
    employment_status = rng.choice(["full_time", "part_time", "contract"], size=row_count)
    balance = rng.normal(12000, 3500, size=row_count).clip(1000, None)
    utilization = rng.uniform(0.1, 0.95, size=row_count)
    inquiries = rng.poisson(1.2, size=row_count)
    probability = 1 / (
        1
        + np.exp(
            -(
                -4.5
                + 0.00012 * balance
                + 2.5 * utilization
                + 0.35 * inquiries
                + np.where(fico_bucket == "D", 1.25, 0.0)
                + np.where(fico_bucket == "C", 0.55, 0.0)
            )
        )
    )
    default_status = (rng.uniform(size=row_count) < probability).astype(int)

    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=row_count, freq="D"),
            "account_id": [f"A{i:05d}" for i in range(row_count)],
            "fico_bucket": fico_bucket,
            "employment_status": employment_status,
            "balance": balance,
            "utilization": utilization,
            "inquiries": inquiries,
            "legacy_unused_column": "remove",
            "default_status": default_status,
        }
    )


def test_pipeline_end_to_end() -> None:
    dataframe = build_synthetic_dataframe()
    with temporary_artifact_root("pytest_smoke") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                    ColumnSpec(name="legacy_unused_column", enabled=False),
                    ColumnSpec(
                        name="portfolio_name",
                        create_if_missing=True,
                        default_value="consumer",
                        dtype="string",
                    ),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            diagnostics=DiagnosticConfig(quantile_bucket_count=6),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        orchestrator = QuantModelOrchestrator(config=config)
        context = orchestrator.run(dataframe)

        assert context.model is not None
        assert "test" in context.metrics
        assert context.feature_importance is not None
        assert context.backtest_summary is not None
        assert context.metrics["test"]["roc_auc"] is not None
        assert len(context.backtest_summary) == 6
        assert "adf_tests" not in context.diagnostics_tables
        assert context.artifacts["model"].exists()
        assert context.artifacts["report"].exists()
