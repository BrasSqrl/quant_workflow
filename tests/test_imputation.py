"""Tests for governed missing-value handling in the modeling pipeline."""

from __future__ import annotations

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
    MissingValuePolicy,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
)
from tests.support import temporary_artifact_root


def build_imputation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "account_id": [f"I{i:04d}" for i in range(10)],
            "balance": [1.0, 2.0, None, 4.0, 5.0, None, 100.0, 100.0, 100.0, None],
            "channel": [
                "branch",
                "digital",
                "branch",
                "broker",
                None,
                "digital",
                None,
                "broker",
                "branch",
                None,
            ],
            "utilization": [0.1, 0.2, 0.3, 0.25, 0.4, 0.35, 0.6, 0.55, 0.65, 0.7],
            "default_status": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_imputation_uses_training_split_rules_and_exports_contract() -> None:
    dataframe = build_imputation_dataframe()
    with temporary_artifact_root("pytest_imputation") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                    ColumnSpec(
                        name="balance",
                        missing_value_policy=MissingValuePolicy.MEAN,
                    ),
                    ColumnSpec(
                        name="channel",
                        missing_value_policy=MissingValuePolicy.CONSTANT,
                        missing_value_fill_value="unknown",
                    ),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.TIME_SERIES,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            diagnostics=DiagnosticConfig(quantile_bucket_count=2),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.diagnostics_tables["imputation_rules"].shape[0] >= 3
        balance_rule = context.diagnostics_tables["imputation_rules"].loc[
            context.diagnostics_tables["imputation_rules"]["feature_name"] == "balance"
        ].iloc[0]
        assert balance_rule["applied_policy"] == MissingValuePolicy.MEAN.value
        assert float(balance_rule["fill_value"]) == 3.0

        channel_rule = context.diagnostics_tables["imputation_rules"].loc[
            context.diagnostics_tables["imputation_rules"]["feature_name"] == "channel"
        ].iloc[0]
        assert channel_rule["applied_policy"] == MissingValuePolicy.CONSTANT.value
        assert channel_rule["fill_value"] == "unknown"

        test_balance = context.split_frames["test"]["balance"].iloc[-1]
        assert float(test_balance) == 3.0
        assert "unknown" in context.split_frames["validation"]["channel"].tolist()
        assert not context.split_frames["train"][context.feature_columns].isna().any().any()
        assert not context.split_frames["validation"][context.feature_columns].isna().any().any()
        assert not context.split_frames["test"][context.feature_columns].isna().any().any()
