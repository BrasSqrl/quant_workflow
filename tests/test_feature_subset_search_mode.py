"""Regression coverage for the feature-subset-search execution mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    DataStructure,
    ExecutionConfig,
    ExecutionMode,
    FeatureEngineeringConfig,
    FeatureSubsetSearchConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from tests.support import temporary_artifact_root
from tests.test_pipeline_smoke import build_synthetic_dataframe


def test_feature_subset_search_mode_exports_comparison_only_bundle() -> None:
    dataframe = build_synthetic_dataframe(row_count=260)
    with temporary_artifact_root("pytest_subset_search") as output_root:
        config = FrameworkConfig(
            schema=SchemaConfig(
                column_specs=[
                    ColumnSpec(name="as_of_date", dtype="datetime", role=ColumnRole.DATE),
                    ColumnSpec(name="account_id", dtype="string", role=ColumnRole.IDENTIFIER),
                ]
            ),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                date_column="as_of_date",
            ),
            execution=ExecutionConfig(mode=ExecutionMode.SEARCH_FEATURE_SUBSETS),
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
            subset_search=FeatureSubsetSearchConfig(
                enabled=True,
                candidate_feature_names=["balance", "utilization", "inquiries"],
                min_subset_size=1,
                max_subset_size=2,
                max_candidate_features=6,
                top_candidate_count=10,
                top_curve_count=3,
                include_significance_tests=True,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "subset_search_candidates" in context.diagnostics_tables
        assert "subset_search_frontier" in context.diagnostics_tables
        assert "subset_search_feature_frequency" in context.diagnostics_tables
        assert "subset_search_selected_candidate" in context.diagnostics_tables
        assert "subset_search_selected_coefficients" in context.diagnostics_tables
        assert "subset_search_nonwinning_candidates" in context.diagnostics_tables
        candidate_table = context.diagnostics_tables["subset_search_candidates"]
        assert {"ranking_roc_auc", "ranking_ks_statistic", "rank"}.issubset(
            candidate_table.columns
        )
        assert "subset_search_selected_roc_curve" in context.visualizations
        assert "subset_search_selected_ks_curve" in context.visualizations
        assert context.artifacts["report"].exists()
        assert context.artifacts["interactive_report"].exists()
        assert "model" not in context.artifacts

        selected_candidate_table = context.diagnostics_tables["subset_search_selected_candidate"]
        assert list(selected_candidate_table.columns[:4]) == [
            "candidate_id",
            "feature_set",
            "feature_count",
            "ranking_roc_auc",
        ]
        nonwinning_table = context.diagnostics_tables["subset_search_nonwinning_candidates"]
        assert {"rank", "candidate_id", "ranking_roc_auc", "ranking_ks_statistic"}.issubset(
            nonwinning_table.columns
        )

        manifest = json.loads(Path(context.artifacts["manifest"]).read_text(encoding="utf-8"))
        assert "model" not in manifest["core_artifacts"]
        assert "interactive_report" in manifest["core_artifacts"]
        interactive_html = context.artifacts["interactive_report"].read_text(encoding="utf-8")
        assert "Selected Candidate Coefficients" in interactive_html
        assert "Non-Winning Candidate Ranking" in interactive_html


def test_feature_subset_search_requires_binary_target() -> None:
    with pytest.raises(ValueError, match="only supported for binary targets"):
        FrameworkConfig(
            schema=SchemaConfig(column_specs=[]),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="target",
                mode=TargetMode.CONTINUOUS,
                output_column="target",
            ),
            split=SplitConfig(),
            execution=ExecutionConfig(mode=ExecutionMode.SEARCH_FEATURE_SUBSETS),
            model=ModelConfig(model_type=ModelType.LINEAR_REGRESSION),
            subset_search=FeatureSubsetSearchConfig(enabled=True),
        ).validate()
