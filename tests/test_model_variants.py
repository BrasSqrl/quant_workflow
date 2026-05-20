"""Smoke tests for additional supported model families."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from quant_pd_framework import (
    ArtifactConfig,
    CleaningConfig,
    DataStructure,
    FeatureEngineeringConfig,
    FrameworkConfig,
    ModelConfig,
    ModelType,
    QuantModelOrchestrator,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.models import build_model_adapter
from tests.support import (
    build_binary_dataframe,
    build_common_schema,
    build_continuous_dataframe,
    build_panel_forecast_dataframe,
    temporary_artifact_root,
)


def _skip_if_xgboost_unavailable(model_type: ModelType) -> None:
    if model_type != ModelType.XGBOOST:
        return
    try:
        importlib.import_module("xgboost")
    except Exception as exc:  # noqa: BLE001 - xgboost can fail on native-library loading.
        pytest.skip(f"XGBoost runtime is unavailable in this environment: {exc}")


def test_xgboost_adapter_failure_is_lazy_and_model_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_pd_framework import optional_dependencies

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "xgboost":
            raise RuntimeError("native library missing")
        return real_import_module(name, package)

    monkeypatch.setattr(optional_dependencies.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="XGBoost could not be loaded"):
        build_model_adapter(
            ModelConfig(model_type=ModelType.XGBOOST),
            TargetMode.BINARY,
        )

    non_xgboost_adapter = build_model_adapter(
        ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION),
        TargetMode.BINARY,
    )
    assert non_xgboost_adapter.model_type == ModelType.LOGISTIC_REGRESSION


@pytest.mark.parametrize(
    ("model_type", "model_config"),
    [
        (ModelType.LOGISTIC_REGRESSION, ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)),
        (
            ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
            ModelConfig(
                model_type=ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                l1_ratio=0.35,
                max_iter=400,
            ),
        ),
        (
            ModelType.SCORECARD_LOGISTIC_REGRESSION,
            ModelConfig(
                model_type=ModelType.SCORECARD_LOGISTIC_REGRESSION,
                scorecard_bins=4,
                max_iter=400,
            ),
        ),
        (
            ModelType.PROBIT_REGRESSION,
            ModelConfig(model_type=ModelType.PROBIT_REGRESSION, max_iter=300),
        ),
        (
            ModelType.GEE_LOGISTIC_REGRESSION,
            ModelConfig(
                model_type=ModelType.GEE_LOGISTIC_REGRESSION,
                max_iter=100,
                gee_group_column="account_id",
            ),
        ),
        (
            ModelType.RANDOM_FOREST,
            ModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                tree_n_estimators=20,
                tree_max_depth=3,
            ),
        ),
        (
            ModelType.EXTRA_TREES,
            ModelConfig(
                model_type=ModelType.EXTRA_TREES,
                tree_n_estimators=20,
                tree_max_depth=3,
            ),
        ),
        (
            ModelType.EXPLAINABLE_BOOSTING_MACHINE,
            ModelConfig(
                model_type=ModelType.EXPLAINABLE_BOOSTING_MACHINE,
                tree_n_estimators=20,
                tree_max_depth=2,
                xgboost_learning_rate=0.05,
            ),
        ),
        (
            ModelType.XGBOOST,
            ModelConfig(
                model_type=ModelType.XGBOOST,
                xgboost_n_estimators=20,
                xgboost_max_depth=3,
            ),
        ),
    ],
)
def test_binary_model_variants_run(model_type: ModelType, model_config: ModelConfig) -> None:
    _skip_if_xgboost_unavailable(model_type)
    dataframe = build_binary_dataframe()
    with temporary_artifact_root(f"pytest_{model_type.value}") as artifact_root:
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
            model=model_config,
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["roc_auc"] is not None
        assert context.feature_importance is not None
        assert "quantile_summary" in context.diagnostics_tables
        assert context.artifacts["interactive_report"] is not None
        assert "package_zip" not in context.artifacts


@pytest.mark.parametrize(
    ("model_type", "model_config"),
    [
        (ModelType.LINEAR_REGRESSION, ModelConfig(model_type=ModelType.LINEAR_REGRESSION)),
        (
            ModelType.BETA_REGRESSION,
            ModelConfig(model_type=ModelType.BETA_REGRESSION, max_iter=300),
        ),
        (
            ModelType.FRACTIONAL_LOGIT,
            ModelConfig(model_type=ModelType.FRACTIONAL_LOGIT, max_iter=300),
        ),
        (
            ModelType.ZERO_ONE_INFLATED_BETA,
            ModelConfig(model_type=ModelType.ZERO_ONE_INFLATED_BETA, max_iter=300),
        ),
        (
            ModelType.TWO_STAGE_LGD_MODEL,
            ModelConfig(model_type=ModelType.TWO_STAGE_LGD_MODEL, max_iter=300),
        ),
        (
            ModelType.RIDGE_REGRESSION,
            ModelConfig(model_type=ModelType.RIDGE_REGRESSION, regularization_alpha=0.5),
        ),
        (
            ModelType.LASSO_REGRESSION,
            ModelConfig(
                model_type=ModelType.LASSO_REGRESSION,
                regularization_alpha=0.01,
                max_iter=500,
            ),
        ),
        (
            ModelType.ELASTIC_NET_REGRESSION,
            ModelConfig(
                model_type=ModelType.ELASTIC_NET_REGRESSION,
                regularization_alpha=0.01,
                l1_ratio=0.35,
                max_iter=500,
            ),
        ),
        (
            ModelType.QUANTILE_REGRESSION,
            ModelConfig(model_type=ModelType.QUANTILE_REGRESSION, quantile_alpha=0.65),
        ),
        (
            ModelType.TOBIT_REGRESSION,
            ModelConfig(
                model_type=ModelType.TOBIT_REGRESSION,
                max_iter=250,
                tobit_left_censoring=0.0,
                tobit_right_censoring=1.0,
            ),
        ),
        (
            ModelType.COX_PROPORTIONAL_HAZARDS,
            ModelConfig(model_type=ModelType.COX_PROPORTIONAL_HAZARDS, max_iter=100),
        ),
        (
            ModelType.AFT_SURVIVAL_MODEL,
            ModelConfig(model_type=ModelType.AFT_SURVIVAL_MODEL),
        ),
    ],
)
def test_continuous_model_variants_run(model_type: ModelType, model_config: ModelConfig) -> None:
    dataframe = build_continuous_dataframe()
    with temporary_artifact_root(f"pytest_{model_type.value}") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("loan_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="censored_target",
                mode=TargetMode.CONTINUOUS,
                output_column="target_value",
            ),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=model_config,
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["rmse"] is not None
        assert context.feature_importance is not None
        assert (
            "residual_summary" in context.diagnostics_tables
            or model_type == ModelType.LINEAR_REGRESSION
        )
        assert context.artifacts["workbook"] is not None


def test_panel_regression_variant_runs_on_panel_data() -> None:
    dataframe = build_panel_forecast_dataframe()
    with temporary_artifact_root("pytest_panel_regression") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("segment_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(drop_raw_date_columns=False),
            target=TargetConfig(
                source_column="forecast_value",
                mode=TargetMode.CONTINUOUS,
                output_column="target_value",
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
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["rmse"] is not None
        assert context.feature_importance is not None
        assert "segment_id" in context.feature_columns


@pytest.mark.parametrize(
    "model_type",
    [
        ModelType.MULTINOMIAL_LOGISTIC_REGRESSION,
        ModelType.ORDINAL_LOGISTIC_REGRESSION,
        ModelType.DECISION_TREE,
    ],
)
def test_multiclass_model_variants_run(model_type: ModelType) -> None:
    dataframe = build_binary_dataframe(row_count=180)
    dataframe["risk_grade"] = pd.qcut(
        dataframe["utilization"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)
    with temporary_artifact_root(f"pytest_{model_type.value}") as artifact_root:
        config = FrameworkConfig(
            schema=build_common_schema("account_id"),
            cleaning=CleaningConfig(),
            feature_engineering=FeatureEngineeringConfig(),
            target=TargetConfig(
                source_column="risk_grade",
                mode=TargetMode.MULTICLASS,
                output_column="target_class",
            ),
            split=SplitConfig(
                data_structure=DataStructure.CROSS_SECTIONAL,
                date_column="as_of_date",
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
            ),
            model=ModelConfig(
                model_type=model_type,
                max_iter=250,
                tree_max_depth=3,
            ),
            artifacts=ArtifactConfig(output_root=artifact_root),
        )

        context = QuantModelOrchestrator(config=config).run(dataframe)

        assert context.metrics["test"]["accuracy"] is not None
        assert context.feature_importance is not None
        assert "predicted_class" in context.predictions["test"].columns


@pytest.mark.parametrize(
    ("model_type", "target_mode", "model_config"),
    [
        (
            ModelType.POISSON_REGRESSION,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.POISSON_REGRESSION, max_iter=100),
        ),
        (
            ModelType.NEGATIVE_BINOMIAL_REGRESSION,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.NEGATIVE_BINOMIAL_REGRESSION, max_iter=100),
        ),
        (
            ModelType.GAMMA_REGRESSION,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.GAMMA_REGRESSION, max_iter=100),
        ),
        (
            ModelType.TWEEDIE_REGRESSION,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.TWEEDIE_REGRESSION, max_iter=100),
        ),
        (
            ModelType.GAM_SPLINE_REGRESSION,
            TargetMode.CONTINUOUS,
            ModelConfig(
                model_type=ModelType.GAM_SPLINE_REGRESSION,
                spline_n_knots=4,
                spline_degree=2,
            ),
        ),
        (
            ModelType.GAM_SPLINE_LOGISTIC,
            TargetMode.BINARY,
            ModelConfig(
                model_type=ModelType.GAM_SPLINE_LOGISTIC,
                spline_n_knots=4,
                spline_degree=2,
                max_iter=250,
            ),
        ),
        (
            ModelType.MIXED_EFFECTS_REGRESSION,
            TargetMode.CONTINUOUS,
            ModelConfig(
                model_type=ModelType.MIXED_EFFECTS_REGRESSION,
                mixed_effects_group_column="segment",
                max_iter=100,
            ),
        ),
        (
            ModelType.SARIMAX_FORECAST,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.SARIMAX_FORECAST, max_iter=50),
        ),
        (
            ModelType.EXPONENTIAL_SMOOTHING_FORECAST,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.EXPONENTIAL_SMOOTHING_FORECAST),
        ),
        (
            ModelType.UNOBSERVED_COMPONENTS_FORECAST,
            TargetMode.CONTINUOUS,
            ModelConfig(model_type=ModelType.UNOBSERVED_COMPONENTS_FORECAST, max_iter=50),
        ),
    ],
)
def test_sas_equivalent_adapters_fit_and_predict(
    model_type: ModelType,
    target_mode: TargetMode,
    model_config: ModelConfig,
) -> None:
    rng = np.random.default_rng(123)
    row_count = 72
    frame = pd.DataFrame(
        {
            "feature_a": rng.normal(size=row_count),
            "feature_b": rng.uniform(0.1, 1.0, size=row_count),
            "segment": np.repeat(["a", "b", "c", "d"], row_count // 4),
        }
    )
    if target_mode == TargetMode.BINARY:
        target = ((frame["feature_a"] + frame["feature_b"]) > 0.8).astype(int)
    elif model_type in {
        ModelType.POISSON_REGRESSION,
        ModelType.NEGATIVE_BINOMIAL_REGRESSION,
    }:
        target = pd.Series(rng.poisson(2.0, size=row_count), dtype=float)
    elif model_type in {
        ModelType.GAMMA_REGRESSION,
        ModelType.TWEEDIE_REGRESSION,
    }:
        target = pd.Series(np.exp(0.2 + frame["feature_b"] + rng.normal(0, 0.1, row_count)))
    else:
        target = pd.Series(
            10 + np.arange(row_count) * 0.05 + frame["feature_a"] * 0.2,
            dtype=float,
        )

    adapter = build_model_adapter(model_config, target_mode)
    adapter.fit(
        frame,
        target,
        numeric_features=["feature_a", "feature_b"],
        categorical_features=["segment"],
    )
    predictions = adapter.predict_score(frame.head(10))

    assert len(predictions) == 10
    assert adapter.get_feature_importance() is not None
