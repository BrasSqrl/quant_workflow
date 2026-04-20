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


def build_advanced_imputation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "account_id": [f"A{i:04d}" for i in range(12)],
            "portfolio": [
                "retail",
                "retail",
                "retail",
                "retail",
                "retail",
                "retail",
                "commercial",
                "commercial",
                "commercial",
                "commercial",
                "commercial",
                "commercial",
            ],
            "balance": [10.0, None, 14.0, None, 16.0, 18.0, 100.0, None, 110.0, 120.0, None, 130.0],
            "utilization": [0.1, 0.2, 0.22, 0.24, 0.25, 0.27, 0.6, 0.62, 0.64, 0.65, 0.67, 0.7],
            "default_status": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
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


def test_advanced_imputation_supports_group_rules_and_missing_indicators() -> None:
    dataframe = build_advanced_imputation_dataframe()
    with temporary_artifact_root("pytest_advanced_imputation") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                    ColumnSpec(
                        name="balance",
                        missing_value_policy=MissingValuePolicy.MEDIAN,
                        missing_value_group_columns=["portfolio"],
                        create_missing_indicator=True,
                    ),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.TIME_SERIES,
                date_column="as_of_date",
                train_size=0.5,
                validation_size=0.25,
                test_size=0.25,
            ),
            diagnostics=DiagnosticConfig(quantile_bucket_count=2),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        imputation_rules = context.diagnostics_tables["imputation_rules"]
        balance_rule = imputation_rules.loc[
            imputation_rules["feature_name"] == "balance"
        ].iloc[0]
        assert balance_rule["group_columns"] == "portfolio"
        assert balance_rule["group_rule_count"] >= 1
        assert balance_rule["missing_indicator_column"] == "balance__missing_indicator"
        assert "imputation_group_rules" in context.diagnostics_tables
        assert "balance__missing_indicator" in context.feature_columns
        assert "balance__missing_indicator" in context.numeric_features
        assert (
            context.metadata["imputation_summary"]["generated_missing_indicator_count"] == 1
        )

        train_frame = context.split_frames["train"]
        retail_imputed_value = train_frame.loc[
            train_frame["portfolio"] == "retail", "balance"
        ].iloc[1]
        assert float(retail_imputed_value) == 15.0
        assert set(train_frame["balance__missing_indicator"].unique()) == {0, 1}
