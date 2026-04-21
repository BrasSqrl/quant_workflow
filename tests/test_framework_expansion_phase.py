"""Regression tests for the expanded roadmap frameworks."""

from __future__ import annotations

import numpy as np

from quant_pd_framework import (
    AdvancedImputationConfig,
    ArtifactConfig,
    CleaningConfig,
    ColumnSpec,
    ComparisonConfig,
    DataStructure,
    DocumentationConfig,
    FeatureEngineeringConfig,
    FrameworkConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    PresetName,
    QuantModelOrchestrator,
    RobustnessConfig,
    SplitConfig,
    StructuralBreakConfig,
    TargetConfig,
    TargetMode,
    TimeSeriesDiagnosticConfig,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
)
from tests.support import (
    build_binary_dataframe,
    build_common_schema,
    build_continuous_dataframe,
    build_panel_forecast_dataframe,
    temporary_artifact_root,
)


def test_advanced_imputation_and_transformation_extensions() -> None:
    dataframe = build_binary_dataframe(row_count=220)
    dataframe.loc[dataframe.index[::7], "balance"] = np.nan
    dataframe.loc[dataframe.index[::9], "utilization"] = np.nan
    dataframe["positive_exposure"] = dataframe["balance"].fillna(12_000) + 1_000
    dataframe.loc[dataframe.index[::11], "positive_exposure"] = np.nan

    with temporary_artifact_root("framework_phase_imputation") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
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
            advanced_imputation=AdvancedImputationConfig(
                enabled=True,
                knn_neighbors=3,
                iterative_max_iter=5,
                minimum_complete_rows=10,
            ),
            transformations=TransformationConfig(
                enabled=True,
                transformations=[
                    TransformationSpec(
                        transform_type=TransformationType.BOX_COX,
                        source_feature="positive_exposure",
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.PIECEWISE_LINEAR,
                        source_feature="utilization",
                        parameter_value=0.55,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.ROLLING_STD,
                        source_feature="balance",
                        window_size=4,
                    ),
                ],
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        config.schema.column_specs.extend(
            [
                ColumnSpec(name="balance", missing_value_policy=MissingValuePolicy.KNN),
                ColumnSpec(
                    name="utilization",
                    missing_value_policy=MissingValuePolicy.ITERATIVE,
                    create_missing_indicator=True,
                ),
                ColumnSpec(
                    name="positive_exposure",
                    missing_value_policy=MissingValuePolicy.KNN,
                ),
            ]
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "advanced_imputation_summary" in context.diagnostics_tables
        assert "positive_exposure_box_cox" in context.feature_columns
        assert "utilization_piecewise_0_55" in context.feature_columns
        assert "balance_rollstd_4" in context.feature_columns
        assert "utilization__missing_indicator" in context.feature_columns


def test_expanded_diagnostic_framework_outputs_for_ccar_workflow() -> None:
    dataframe = build_panel_forecast_dataframe(entity_count=10, periods_per_entity=18)

    with temporary_artifact_root("framework_phase_diagnostics") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("segment_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="forecast_value",
                mode=TargetMode.CONTINUOUS,
                output_column="forecast_value",
            ),
            split=SplitConfig(
                data_structure=DataStructure.PANEL,
                date_column="as_of_date",
                entity_column="segment_id",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            preset_name=PresetName.CCAR_FORECASTING,
            model=ModelConfig(model_type=ModelType.PANEL_REGRESSION),
            documentation=DocumentationConfig(
                enabled=True,
                model_name="CCAR Framework Expansion Test",
                business_purpose="Regression test for expanded CCAR diagnostics.",
                horizon_definition="Quarterly panel forecast horizon.",
                target_definition="Continuous forecast target.",
            ),
            transformations=TransformationConfig(
                enabled=True,
                auto_interactions_enabled=True,
                include_numeric_numeric_interactions=True,
                transformations=[
                    TransformationSpec(
                        transform_type=TransformationType.LAG,
                        source_feature="unemployment_rate",
                        lag_periods=1,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.ROLLING_STD,
                        source_feature="gdp_gap",
                        window_size=3,
                    ),
                ],
            ),
            time_series_diagnostics=TimeSeriesDiagnosticConfig(
                enabled=True,
                minimum_series_length=8,
                maximum_lag=3,
            ),
            structural_breaks=StructuralBreakConfig(
                enabled=True,
                minimum_segment_size=4,
                candidate_break_count=3,
                rolling_window_fraction=0.3,
            ),
            robustness=RobustnessConfig(
                enabled=True,
                resample_count=3,
                sample_fraction=0.7,
                evaluation_split="test",
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "distribution_tests" in context.diagnostics_tables
        assert "distribution_shift_tests" in context.diagnostics_tables
        assert "dependency_cluster_summary" in context.diagnostics_tables
        assert "time_series_extension_tests" in context.diagnostics_tables
        assert "structural_break_tests" in context.diagnostics_tables
        assert "feature_construction_workbench" in context.diagnostics_tables
        assert "preset_test_recommendations" in context.diagnostics_tables
        assert "robustness_framework_summary" in context.diagnostics_tables
        assert "structural_break_profile" in context.visualizations


def test_binning_framework_outputs_with_scorecard_and_manual_bins() -> None:
    dataframe = build_binary_dataframe(row_count=240)

    with temporary_artifact_root("framework_phase_binning") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
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
            model=ModelConfig(model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION),
            transformations=TransformationConfig(
                enabled=True,
                transformations=[
                    TransformationSpec(
                        transform_type=TransformationType.MANUAL_BINS,
                        source_feature="utilization",
                        bin_edges=[0.2, 0.45, 0.7],
                    )
                ],
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "binning_framework_summary" in context.diagnostics_tables
        assert "manual_binning_profile" in context.diagnostics_tables
        assert "manual_binning_distribution" in context.visualizations


def test_priority_batch_binary_outputs_include_pooling_mcar_and_comparison_tests() -> None:
    dataframe = build_binary_dataframe(row_count=280)
    dataframe.loc[dataframe.index[::5], "balance"] = np.nan
    dataframe.loc[dataframe.index[::7], "utilization"] = np.nan
    dataframe.loc[dataframe.index[::9], "delinquencies"] = np.nan

    with temporary_artifact_root("framework_priority_binary") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(source_column="default_status", positive_values=[1]),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                random_state=7,
            ),
            model=ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
            comparison=ComparisonConfig(
                enabled=True,
                challenger_model_types=[ModelType.ELASTIC_NET_LOGISTIC_REGRESSION],
            ),
            advanced_imputation=AdvancedImputationConfig(
                enabled=True,
                iterative_max_iter=5,
                minimum_complete_rows=10,
                multiple_imputation_enabled=True,
                multiple_imputation_datasets=3,
                multiple_imputation_evaluation_split="test",
                multiple_imputation_top_features=3,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        config.schema.column_specs.extend(
            [
                ColumnSpec(name="balance", missing_value_policy=MissingValuePolicy.ITERATIVE),
                ColumnSpec(name="utilization", missing_value_policy=MissingValuePolicy.ITERATIVE),
                ColumnSpec(name="delinquencies", missing_value_policy=MissingValuePolicy.MEDIAN),
            ]
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert "multiple_imputation_pooling_summary" in context.diagnostics_tables
        assert "multiple_imputation_pooled_coefficients" in context.diagnostics_tables
        assert "littles_mcar_test" in context.diagnostics_tables
        assert "model_comparison_significance_tests" in context.diagnostics_tables
        significance_tests = set(
            context.diagnostics_tables["model_comparison_significance_tests"]["test_name"]
            .astype(str)
            .tolist()
        )
        assert "delong_auc_difference" in significance_tests
        assert "mcnemar_threshold_difference" in significance_tests
        assert "diebold_mariano" in significance_tests


def test_priority_batch_time_series_outputs_include_new_transforms_and_stability_tests() -> None:
    dataframe = build_panel_forecast_dataframe(entity_count=10, periods_per_entity=18)

    with temporary_artifact_root("framework_priority_timeseries") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("segment_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="forecast_value",
                mode=TargetMode.CONTINUOUS,
                output_column="forecast_value",
            ),
            split=SplitConfig(
                data_structure=DataStructure.PANEL,
                date_column="as_of_date",
                entity_column="segment_id",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=ModelConfig(model_type=ModelType.PANEL_REGRESSION),
            transformations=TransformationConfig(
                enabled=True,
                transformations=[
                    TransformationSpec(
                        transform_type=TransformationType.NATURAL_SPLINE,
                        source_feature="utilization",
                        parameter_value=4,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.EWMA,
                        source_feature="unemployment_rate",
                        window_size=3,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.DIFFERENCE,
                        source_feature="gdp_gap",
                        lag_periods=1,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.ROLLING_MEDIAN,
                        source_feature="delinquency_rate",
                        window_size=3,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.ROLLING_MIN,
                        source_feature="utilization",
                        window_size=3,
                    ),
                    TransformationSpec(
                        transform_type=TransformationType.ROLLING_MAX,
                        source_feature="utilization",
                        window_size=3,
                    ),
                ],
            ),
            time_series_diagnostics=TimeSeriesDiagnosticConfig(
                enabled=True,
                minimum_series_length=8,
                maximum_lag=3,
            ),
            structural_breaks=StructuralBreakConfig(
                enabled=True,
                minimum_segment_size=4,
                candidate_break_count=3,
                rolling_window_fraction=0.3,
            ),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert any(
            feature_name.startswith("utilization_spline_df_4_basis_")
            for feature_name in context.feature_columns
        )
        assert "unemployment_rate_ewma_3" in context.feature_columns
        assert "gdp_gap_diff_1" in context.feature_columns
        assert "delinquency_rate_rollmedian_3" in context.feature_columns
        assert "utilization_rollmin_3" in context.feature_columns
        assert "utilization_rollmax_3" in context.feature_columns

        time_series_tests = set(
            context.diagnostics_tables["time_series_extension_tests"]["test_name"]
            .astype(str)
            .tolist()
        )
        structural_break_tests = set(
            context.diagnostics_tables["structural_break_tests"]["test_name"]
            .astype(str)
            .tolist()
        )
        assert "kpss" in time_series_tests
        assert "phillips_perron" in time_series_tests
        assert "cusum" in structural_break_tests
        assert "cusum_squares" in structural_break_tests


def test_priority_batch_specification_outputs_include_reset_white_dfbetas_and_dffits() -> None:
    dataframe = build_continuous_dataframe(row_count=220)

    with temporary_artifact_root("framework_priority_specification") as output_root:
        config = FrameworkConfig(
            schema=build_common_schema("loan_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="censored_target",
                mode=TargetMode.CONTINUOUS,
                output_column="censored_target",
            ),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=ModelConfig(model_type=ModelType.LINEAR_REGRESSION),
            artifacts=ArtifactConfig(output_root=output_root),
        )
        context = QuantModelOrchestrator(config=config).run(dataframe)

        specification_tests = set(
            context.diagnostics_tables["model_specification_tests"]["test_name"]
            .astype(str)
            .tolist()
        )
        assert "ramsey_reset" in specification_tests
        assert "white_test" in specification_tests
        assert "dfbetas_flag_count" in specification_tests
        assert "dffits_flag_count" in specification_tests
        assert "model_dfbetas_summary" in context.diagnostics_tables
        assert "model_dffits_summary" in context.diagnostics_tables
