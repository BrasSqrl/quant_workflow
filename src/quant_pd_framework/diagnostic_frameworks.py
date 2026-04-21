"""Expanded statistical-framework helpers layered on top of the core diagnostics step."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import chi2, ks_2samp, kurtosis, norm, normaltest, skew
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    breaks_cusumolsresid,
    het_breuschpagan,
    het_white,
    linear_reset,
    recursive_olsresiduals,
)
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss, mackinnonp

from .config import DataStructure, ExecutionMode, PresetName, TargetMode
from .context import PipelineContext


def add_expanded_framework_outputs(
    *,
    context: PipelineContext,
    top_features: list[str],
    labels_available: bool,
) -> None:
    """Adds the deeper roadmap frameworks on top of the base diagnostics output."""

    _add_imputation_framework_extensions(context)
    _add_missingness_framework_extensions(context)
    _add_distribution_framework_outputs(context, top_features)
    _add_binning_framework_outputs(context)
    _add_residual_framework_outputs(context, top_features, labels_available)
    _add_outlier_framework_outputs(context, labels_available)
    _add_dependency_framework_outputs(context, top_features)
    _add_model_comparison_framework_outputs(context)
    _add_specification_framework_extensions(context, top_features, labels_available)
    _add_time_series_framework_outputs(context, top_features, labels_available)
    _add_structural_break_framework_outputs(context, top_features, labels_available)
    _add_robustness_framework_extensions(context)
    _add_feature_workbench_outputs(context, labels_available)
    _add_preset_recommendation_outputs(context)


def score_column_name(context: PipelineContext) -> str:
    return (
        "predicted_probability"
        if context.config.target.mode == TargetMode.BINARY
        else "predicted_value"
    )


def labels_available(context: PipelineContext) -> bool:
    return bool(context.metadata.get("labels_available", False))


def default_segment_column(context: PipelineContext, frame: pd.DataFrame) -> str | None:
    configured = context.config.diagnostics.default_segment_column
    if configured and configured in frame.columns:
        return configured
    for column_name in context.categorical_features:
        if column_name in frame.columns:
            return column_name
    return None


def _prediction_frame_with_labels(
    context: PipelineContext,
) -> tuple[pd.DataFrame | None, str | None]:
    if context.target_column is None:
        return None, None
    for split_name in ("test", "validation", "train"):
        prediction_frame = context.predictions.get(split_name)
        if prediction_frame is None or context.target_column not in prediction_frame.columns:
            continue
        return prediction_frame, split_name
    return None, None


def _numeric_features_for_framework(
    context: PipelineContext,
    top_features: list[str],
    *,
    maximum: int,
    frame: pd.DataFrame | None = None,
) -> list[str]:
    available_frame = frame if frame is not None else context.working_data
    candidates = [
        feature_name
        for feature_name in top_features
        if feature_name in context.numeric_features
        and available_frame is not None
        and feature_name in available_frame.columns
    ]
    return candidates[:maximum]


def _add_imputation_framework_extensions(context: PipelineContext) -> None:
    config = context.config.advanced_imputation
    if (
        not config.enabled
        or not config.multiple_imputation_enabled
        or context.config.execution.mode != ExecutionMode.FIT_NEW_MODEL
        or context.target_column is None
    ):
        return
    pre_imputation_frames = context.metadata.get("pre_imputation_split_frames")
    if not isinstance(pre_imputation_frames, dict):
        return
    train_frame = pre_imputation_frames.get("train")
    evaluation_split = config.multiple_imputation_evaluation_split
    evaluation_frame = pre_imputation_frames.get(evaluation_split)
    if train_frame is None or evaluation_frame is None or context.target_column not in train_frame:
        return

    candidate_features = _select_multiple_imputation_features(context, train_frame)
    if not candidate_features:
        return

    surrogate_feature_limit = min(
        len(candidate_features),
        max(3, config.multiple_imputation_top_features),
    )
    candidate_features = candidate_features[:surrogate_feature_limit]
    target_train = pd.to_numeric(train_frame[context.target_column], errors="coerce")
    target_eval = pd.to_numeric(evaluation_frame[context.target_column], errors="coerce")
    train_numeric = (
        train_frame[candidate_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    eval_numeric = (
        evaluation_frame[candidate_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    aligned_train = pd.concat([target_train.rename("target"), train_numeric], axis=1).dropna(
        subset=["target"]
    )
    aligned_eval = pd.concat([target_eval.rename("target"), eval_numeric], axis=1).dropna(
        subset=["target"]
    )
    if len(aligned_train) < max(config.minimum_complete_rows, 40) or len(aligned_eval) < 20:
        return
    performance = context.config.performance
    mi_row_cap = performance.multiple_imputation_row_cap
    if performance.enabled and len(aligned_train) > mi_row_cap:
        aligned_train = aligned_train.sample(
            mi_row_cap,
            random_state=context.config.split.random_state,
        )
        _append_performance_action(
            context,
            action_name="multiple_imputation_train_sample",
            detail=(
                "Sampled the surrogate training frame used for multiply-imputed "
                "pooling diagnostics to keep the workflow tractable on a large run."
            ),
            original_rows=len(train_frame),
            effective_rows=len(aligned_train),
        )
    if performance.enabled and len(aligned_eval) > mi_row_cap:
        aligned_eval = aligned_eval.sample(
            mi_row_cap,
            random_state=context.config.split.random_state,
        )
        _append_performance_action(
            context,
            action_name="multiple_imputation_eval_sample",
            detail=(
                "Sampled the surrogate evaluation frame used for multiply-imputed "
                "pooling diagnostics to keep the workflow tractable on a large run."
            ),
            original_rows=len(evaluation_frame),
            effective_rows=len(aligned_eval),
        )

    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for dataset_id in range(1, config.multiple_imputation_datasets + 1):
        imputer = IterativeImputer(
            max_iter=config.iterative_max_iter,
            random_state=config.iterative_random_state + dataset_id,
            sample_posterior=True,
            skip_complete=True,
        )
        train_matrix = imputer.fit_transform(aligned_train[candidate_features].astype(float))
        eval_matrix = imputer.transform(aligned_eval[candidate_features].astype(float))
        train_imputed = pd.DataFrame(
            train_matrix,
            columns=candidate_features,
            index=aligned_train.index,
        )
        eval_imputed = pd.DataFrame(
            eval_matrix,
            columns=candidate_features,
            index=aligned_eval.index,
        )
        fitted = _fit_multiple_imputation_surrogate(
            context=context,
            train_features=train_imputed,
            train_target=aligned_train["target"],
        )
        if fitted is None:
            continue
        coefficient_rows.extend(
            _extract_multiple_imputation_coefficients(
                fitted=fitted,
                dataset_id=dataset_id,
                feature_names=candidate_features,
            )
        )
        prediction_output = _predict_multiple_imputation_surrogate(
            fitted=fitted,
            evaluation_features=eval_imputed,
            target_mode=context.config.target.mode,
        )
        if prediction_output is None:
            continue
        prediction_frame = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "target": aligned_eval["target"].astype(float).to_numpy(),
                "prediction": prediction_output,
            }
        )
        prediction_rows.append(prediction_frame)
        metric_rows.append(
            _score_multiple_imputation_prediction_frame(
                prediction_frame,
                dataset_id=dataset_id,
                target_mode=context.config.target.mode,
            )
        )

    if not metric_rows or not coefficient_rows or not prediction_rows:
        return

    metric_table = pd.DataFrame(metric_rows)
    pooled_coefficients = _pool_multiple_imputation_coefficients(pd.DataFrame(coefficient_rows))
    pooled_metrics = _pool_multiple_imputation_metrics(metric_table, context.config.target.mode)
    context.diagnostics_tables["multiple_imputation_metric_paths"] = metric_table
    context.diagnostics_tables["multiple_imputation_pooled_coefficients"] = pooled_coefficients
    context.diagnostics_tables["multiple_imputation_pooling_summary"] = pooled_metrics


def _add_missingness_framework_extensions(context: PipelineContext) -> None:
    association = context.diagnostics_tables.get("missingness_target_association", pd.DataFrame())
    if association.empty:
        predictive = pd.DataFrame()
    else:
        predictive = association.copy(deep=True)
        predictive["association_rank"] = np.arange(1, len(predictive) + 1)
        predictive["predictive_band"] = pd.cut(
            predictive["association_score"].fillna(0.0),
            bins=[-np.inf, 0.05, 0.15, np.inf],
            labels=["low", "moderate", "high"],
        ).astype("string")
        context.diagnostics_tables["missingness_predictive_power"] = predictive

    littles_mcar = _build_littles_mcar_output(context)
    if littles_mcar is not None:
        context.diagnostics_tables["littles_mcar_test"] = pd.DataFrame([littles_mcar])


def _add_distribution_framework_outputs(
    context: PipelineContext,
    top_features: list[str],
) -> None:
    config = context.config.distribution_diagnostics
    if not config.enabled:
        return
    numeric_features = _numeric_features_for_framework(
        context,
        top_features,
        maximum=config.top_features,
    )
    if not numeric_features:
        return

    distribution_rows: list[dict[str, Any]] = []
    for split_name, split_frame in context.split_frames.items():
        for feature_name in numeric_features:
            if feature_name not in split_frame.columns:
                continue
            values = pd.to_numeric(split_frame[feature_name], errors="coerce").dropna()
            if len(values) < config.minimum_rows:
                continue
            normaltest_statistic = np.nan
            normaltest_p_value = np.nan
            skewness = np.nan
            excess_kurtosis = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if config.include_normality_tests and len(values) >= 8:
                    try:
                        normaltest_statistic, normaltest_p_value = normaltest(
                            values.to_numpy(dtype=float)
                        )
                    except Exception:
                        pass
                try:
                    skewness = float(skew(values.to_numpy(dtype=float), bias=False))
                except Exception:
                    skewness = np.nan
                try:
                    excess_kurtosis = float(
                        kurtosis(values.to_numpy(dtype=float), fisher=True, bias=False)
                    )
                except Exception:
                    excess_kurtosis = np.nan
            distribution_rows.append(
                {
                    "split": split_name,
                    "feature_name": feature_name,
                    "observation_count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "skewness": skewness,
                    "excess_kurtosis": excess_kurtosis,
                    "normaltest_statistic": float(normaltest_statistic)
                    if pd.notna(normaltest_statistic)
                    else np.nan,
                    "normaltest_p_value": float(normaltest_p_value)
                    if pd.notna(normaltest_p_value)
                    else np.nan,
                }
            )
    if distribution_rows:
        context.diagnostics_tables["distribution_tests"] = pd.DataFrame(distribution_rows)

    if config.include_shift_tests:
        shift_rows: list[dict[str, Any]] = []
        train_frame = context.split_frames.get("train")
        if train_frame is not None:
            for feature_name in numeric_features:
                if feature_name not in train_frame.columns:
                    continue
                train_values = pd.to_numeric(train_frame[feature_name], errors="coerce").dropna()
                if len(train_values) < config.minimum_rows:
                    continue
                for split_name in ("validation", "test"):
                    candidate_frame = context.split_frames.get(split_name)
                    if candidate_frame is None or feature_name not in candidate_frame.columns:
                        continue
                    candidate_values = (
                        pd.to_numeric(candidate_frame[feature_name], errors="coerce").dropna()
                    )
                    if len(candidate_values) < config.minimum_rows:
                        continue
                    try:
                        statistic, p_value = ks_2samp(
                            train_values.to_numpy(dtype=float),
                            candidate_values.to_numpy(dtype=float),
                        )
                    except Exception:
                        continue
                    shift_rows.append(
                        {
                            "feature_name": feature_name,
                            "comparison_split": split_name,
                            "ks_statistic": float(statistic),
                            "p_value": float(p_value),
                            "status": "warning" if p_value < 0.05 else "pass",
                        }
                    )
        if shift_rows:
            context.diagnostics_tables["distribution_shift_tests"] = pd.DataFrame(shift_rows)

    sampled_rows: list[pd.DataFrame] = []
    for split_name in ("train", "test"):
        split_frame = context.split_frames.get(split_name)
        if split_frame is None:
            continue
        sample_columns = [
            feature for feature in numeric_features if feature in split_frame.columns
        ]
        if not sample_columns:
            continue
        sample = split_frame.loc[:, sample_columns].head(context.config.diagnostics.max_plot_rows)
        sample = sample.copy(deep=True)
        sample["split"] = split_name
        sampled_rows.append(sample)
    if sampled_rows:
        sampled_frame = pd.concat(sampled_rows, ignore_index=True)
        long_frame = sampled_frame.melt(
            id_vars=["split"],
            value_vars=[
                feature for feature in numeric_features if feature in sampled_frame.columns
            ],
            var_name="feature_name",
            value_name="value",
        ).dropna()
        if not long_frame.empty:
            context.visualizations["distribution_shift_overview"] = px.box(
                long_frame,
                x="feature_name",
                y="value",
                color="split",
                title="Distribution Shift Overview",
                labels={"feature_name": "Feature", "value": "Value", "split": "Split"},
            )


def _add_binning_framework_outputs(context: PipelineContext) -> None:
    scorecard_summary = context.diagnostics_tables.get("scorecard_feature_summary", pd.DataFrame())
    if not scorecard_summary.empty:
        binning_summary = scorecard_summary.copy(deep=True)
        binning_summary["framework_type"] = "scorecard"
        binning_summary["monotonic_review"] = np.where(
            binning_summary["bad_rate_trend"].astype("string").str.contains("flat|mixed"),
            "review",
            "pass",
        )
        context.diagnostics_tables["binning_framework_summary"] = binning_summary

    manual_binning_rows: list[dict[str, Any]] = []
    governed_transformations = context.diagnostics_tables.get(
        "governed_transformations", pd.DataFrame()
    )
    if context.working_data is None or governed_transformations.empty:
        return

    manual_bins = governed_transformations.loc[
        governed_transformations["transform_type"] == "manual_bins"
    ].copy(deep=True)
    if manual_bins.empty:
        return
    target_column = context.target_column
    for _, row in manual_bins.iterrows():
        output_feature = str(row["output_feature"])
        if output_feature not in context.working_data.columns:
            continue
        grouped = (
            context.working_data.groupby(output_feature, dropna=False)
            .size()
            .rename("observation_count")
            .reset_index()
            .rename(columns={output_feature: "bucket_label"})
        )
        if (
            labels_available(context)
            and target_column
            and target_column in context.working_data.columns
        ):
            target_means = (
                context.working_data.groupby(output_feature, dropna=False)[target_column]
                .mean()
                .rename("target_mean")
                .reset_index()
                .rename(columns={output_feature: "bucket_label"})
            )
            grouped = grouped.merge(target_means, on="bucket_label", how="left")
        grouped["feature_name"] = output_feature
        manual_binning_rows.extend(grouped.to_dict(orient="records"))
    if manual_binning_rows:
        manual_binning_profile = pd.DataFrame(manual_binning_rows)
        context.diagnostics_tables["manual_binning_profile"] = manual_binning_profile
        context.visualizations["manual_binning_distribution"] = px.bar(
            manual_binning_profile,
            x="bucket_label",
            y="observation_count",
            color="feature_name",
            barmode="group",
            title="Manual Binning Distribution",
            labels={"bucket_label": "Bucket", "observation_count": "Observations"},
        )


def _add_residual_framework_outputs(
    context: PipelineContext,
    top_features: list[str],
    labels_available_flag: bool,
) -> None:
    config = context.config.residual_diagnostics
    if not config.enabled or not labels_available_flag or context.target_column is None:
        return
    scored_frame, split_name = _prediction_frame_with_labels(context)
    if scored_frame is None or split_name is None:
        return
    residual_frame = scored_frame.copy(deep=True)
    score_column = score_column_name(context)
    residual_frame["residual"] = (
        pd.to_numeric(residual_frame[context.target_column], errors="coerce")
        - pd.to_numeric(residual_frame[score_column], errors="coerce")
    )
    residuals = residual_frame["residual"].dropna()
    if len(residuals) < config.minimum_rows:
        return

    residual_std = float(residuals.std(ddof=0))
    rows = [
        {
            "test_name": "residual_bias",
            "scope": split_name,
            "statistic": float(residuals.mean()),
            "p_value": np.nan,
            "status": "review"
            if residual_std > 0 and abs(float(residuals.mean())) > residual_std * 0.1
            else "pass",
            "detail": "Residual mean close to zero suggests limited overall bias.",
        },
        {
            "test_name": "residual_rmse",
            "scope": split_name,
            "statistic": float(np.sqrt(np.mean(np.square(residuals)))),
            "p_value": np.nan,
            "status": "review",
            "detail": "Root-mean-square residual magnitude on the scored split.",
        },
    ]
    if config.heteroskedasticity_tests:
        numeric_features = _numeric_features_for_framework(
            context,
            top_features,
            maximum=min(5, len(top_features)),
            frame=residual_frame,
        )
        design_columns = [score_column, *numeric_features]
        design = (
            residual_frame.loc[
                :,
                [column for column in design_columns if column in residual_frame.columns],
            ]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        aligned = pd.concat([residual_frame["residual"], design], axis=1).dropna()
        if len(aligned) >= config.minimum_rows and design.shape[1] >= 1:
            try:
                statistic, p_value, _, _ = het_breuschpagan(
                    aligned["residual"].to_numpy(dtype=float),
                    sm.add_constant(aligned.drop(columns=["residual"]), has_constant="add"),
                )
                rows.append(
                    {
                        "test_name": "breusch_pagan",
                        "scope": split_name,
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "status": "warning" if p_value < 0.05 else "pass",
                        "detail": (
                            "Low p-values suggest residual variance changes across "
                            "the design."
                        ),
                    }
                )
            except Exception:
                pass

    if (
        config.autocorrelation_tests
        and context.config.split.data_structure in {DataStructure.TIME_SERIES, DataStructure.PANEL}
    ):
        date_column = context.config.split.date_column
        if date_column and date_column in residual_frame.columns:
            ordered_residuals = (
                residual_frame.sort_values(date_column, kind="mergesort")["residual"].dropna()
            )
            if len(ordered_residuals) >= config.minimum_rows:
                lag1_autocorr = ordered_residuals.autocorr(lag=1)
                rows.append(
                    {
                        "test_name": "residual_lag1_autocorrelation",
                        "scope": split_name,
                        "statistic": float(lag1_autocorr) if pd.notna(lag1_autocorr) else np.nan,
                        "p_value": np.nan,
                        "status": "warning"
                        if pd.notna(lag1_autocorr) and abs(float(lag1_autocorr)) >= 0.3
                        else "pass",
                        "detail": (
                            "Large lag-1 residual autocorrelation suggests time "
                            "structure remains."
                        ),
                    }
                )
    if rows:
        context.diagnostics_tables["residual_diagnostics"] = pd.DataFrame(rows)

    if config.segment_bias_analysis:
        segment_column = default_segment_column(context, residual_frame)
        if segment_column and segment_column in residual_frame.columns:
            grouped = (
                residual_frame.groupby(segment_column, dropna=False)["residual"]
                .agg(["mean", "median", "std", "count"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "average_residual",
                        "median": "median_residual",
                        "std": "residual_std",
                        "count": "observation_count",
                    }
                )
            )
            grouped["absolute_average_residual"] = grouped["average_residual"].abs()
            context.diagnostics_tables["residual_segment_bias"] = grouped.sort_values(
                "absolute_average_residual",
                ascending=False,
            )
            context.visualizations["residual_segment_bias"] = px.bar(
                grouped.sort_values("absolute_average_residual", ascending=False).head(12),
                x=segment_column,
                y="average_residual",
                color="observation_count",
                title="Residual Bias by Segment",
                labels={segment_column: "Segment", "average_residual": "Average Residual"},
            )


def _add_outlier_framework_outputs(
    context: PipelineContext,
    labels_available_flag: bool,
) -> None:
    config = context.config.outlier_diagnostics
    if not config.enabled:
        return

    flagged_rows: list[dict[str, Any]] = []
    influence_table = context.diagnostics_tables.get("model_influence_summary", pd.DataFrame())
    if not influence_table.empty:
        leverage_threshold = float(influence_table["leverage"].mean()) * config.leverage_multiplier
        cooks_threshold = config.cooks_distance_multiplier / max(len(influence_table), 1)
        flagged_influence = influence_table.loc[
            (influence_table["leverage"] >= leverage_threshold)
            | (influence_table["cooks_distance"] >= cooks_threshold)
        ].copy(deep=True)
        if not flagged_influence.empty:
            flagged_influence["flag_source"] = "model_influence"
            flagged_influence["leverage_threshold"] = leverage_threshold
            flagged_influence["cooks_distance_threshold"] = cooks_threshold
            flagged_rows.extend(flagged_influence.head(config.max_rows).to_dict(orient="records"))
            context.visualizations["outlier_influence_map"] = px.scatter(
                influence_table.head(max(config.max_rows, 200)),
                x="leverage",
                y="cooks_distance",
                size="absolute_residual",
                hover_name="observation_id",
                title="Outlier / Influence Map",
                labels={
                    "leverage": "Leverage",
                    "cooks_distance": "Cook's Distance",
                    "absolute_residual": "Absolute Residual",
                },
            )

    scored_frame, split_name = _prediction_frame_with_labels(context)
    if labels_available_flag and scored_frame is not None and context.target_column is not None:
        residuals = (
            pd.to_numeric(scored_frame[context.target_column], errors="coerce")
            - pd.to_numeric(scored_frame[score_column_name(context)], errors="coerce")
        )
        residual_mean = float(residuals.mean())
        residual_std = float(residuals.std(ddof=0))
        if residual_std > 0:
            z_scores = ((residuals - residual_mean) / residual_std).abs()
            flagged_eval = scored_frame.loc[z_scores >= config.zscore_threshold].copy(deep=True)
            if not flagged_eval.empty:
                flagged_eval = flagged_eval.head(config.max_rows)
                flagged_eval["flag_source"] = f"{split_name}_residual_zscore"
                flagged_eval["absolute_residual_zscore"] = z_scores.loc[flagged_eval.index]
                flagged_rows.extend(
                    flagged_eval[
                        ["flag_source", "absolute_residual_zscore"]
                    ].reset_index(names="observation_id").to_dict(orient="records")
                )

    if flagged_rows:
        context.diagnostics_tables["outlier_flags"] = pd.DataFrame(flagged_rows)


def _add_dependency_framework_outputs(
    context: PipelineContext,
    top_features: list[str],
) -> None:
    config = context.config.dependency_diagnostics
    if not config.enabled:
        return
    train_frame = context.split_frames.get("train")
    if train_frame is None:
        return
    numeric_features = _numeric_features_for_framework(
        context,
        top_features,
        maximum=config.maximum_features,
        frame=train_frame,
    )
    if len(numeric_features) < 2:
        return
    numeric_frame = (
        train_frame[numeric_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(numeric_frame) < 10:
        return
    corr_matrix = numeric_frame.corr()
    context.visualizations["dependency_cluster_heatmap"] = px.imshow(
        corr_matrix,
        title="Dependency / Correlation Cluster Heatmap",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )

    condition_index_detail = _compute_condition_index_detail(numeric_frame.to_numpy(dtype=float))
    if condition_index_detail:
        context.diagnostics_tables["condition_index_detail"] = pd.DataFrame(
            condition_index_detail
        )

    cluster_rows = _build_dependency_clusters(
        corr_matrix,
        threshold=config.clustering_correlation_threshold,
    )
    if cluster_rows:
        context.diagnostics_tables["dependency_cluster_summary"] = pd.DataFrame(cluster_rows)


def _add_time_series_framework_outputs(
    context: PipelineContext,
    top_features: list[str],
    labels_available_flag: bool,
) -> None:
    config = context.config.time_series_diagnostics
    if not config.enabled:
        return
    if context.config.split.data_structure not in {DataStructure.TIME_SERIES, DataStructure.PANEL}:
        return
    series_frame = build_forecasting_series_frame(
        context,
        top_features=top_features,
        labels_available_flag=labels_available_flag,
        minimum_rows=config.minimum_series_length,
    )
    if series_frame is None or len(series_frame) < config.minimum_series_length:
        return

    rows: list[dict[str, Any]] = []
    stationarity_series_candidates: list[tuple[str, pd.Series]] = []
    if "residual" in series_frame.columns:
        residual_series = pd.to_numeric(series_frame["residual"], errors="coerce").dropna()
        if len(residual_series) >= config.minimum_series_length:
            stationarity_series_candidates.append(("aggregated_residuals", residual_series))
            autocorr = residual_series.autocorr(lag=1)
            rows.append(
                {
                    "test_name": "residual_lag1_autocorrelation",
                    "scope": "aggregated_residuals",
                    "statistic": float(autocorr) if pd.notna(autocorr) else np.nan,
                    "p_value": np.nan,
                    "status": "warning"
                    if pd.notna(autocorr) and abs(float(autocorr)) >= 0.3
                    else "pass",
                    "detail": (
                        "Large lag-1 residual autocorrelation suggests remaining "
                        "time dependence."
                    ),
                }
            )
            seasonal_strength = _seasonal_strength(
                residual_series.reset_index(drop=True),
                period=config.seasonal_period,
            )
            rows.append(
                {
                    "test_name": "seasonal_strength",
                    "scope": "aggregated_residuals",
                    "statistic": float(seasonal_strength),
                    "p_value": np.nan,
                    "status": "review",
                    "detail": "Higher values indicate a stronger repeating seasonal pattern.",
                }
            )
            context.visualizations["seasonality_profile"] = px.line(
                _build_seasonality_profile(
                    residual_series.reset_index(drop=True),
                    period=config.seasonal_period,
                ),
                x="seasonal_bucket",
                y="average_value",
                title="Residual Seasonality Profile",
                labels={
                    "seasonal_bucket": "Seasonal Bucket",
                    "average_value": "Average Residual",
                },
            )

    if "target_mean" in series_frame.columns and "prediction_mean" in series_frame.columns:
        regression_frame = series_frame.dropna(subset=["target_mean", "prediction_mean"]).copy(
            deep=True
        )
        if len(regression_frame) >= config.minimum_series_length:
            stationarity_series_candidates.extend(
                [
                    (
                        "target_mean",
                        pd.to_numeric(regression_frame["target_mean"], errors="coerce").dropna(),
                    ),
                    (
                        "prediction_mean",
                        pd.to_numeric(
                            regression_frame["prediction_mean"],
                            errors="coerce",
                        ).dropna(),
                    ),
                ]
            )
            exog_columns = ["prediction_mean"]
            for feature_name in top_features[:2]:
                if feature_name in regression_frame.columns:
                    exog_columns.append(feature_name)
            design = sm.add_constant(
                regression_frame[exog_columns].apply(pd.to_numeric, errors="coerce"),
                has_constant="add",
            )
            target = pd.to_numeric(regression_frame["target_mean"], errors="coerce")
            aligned = pd.concat([target.rename("target_mean"), design], axis=1).dropna()
            if len(aligned) >= config.minimum_series_length:
                try:
                    fitted = sm.OLS(
                        aligned["target_mean"].astype(float),
                        aligned.drop(columns=["target_mean"]),
                    ).fit()
                    bg_rows = acorr_breusch_godfrey(
                        fitted,
                        nlags=min(config.maximum_lag, max(1, len(aligned) // 5)),
                    )
                    rows.append(
                        {
                            "test_name": "breusch_godfrey",
                            "scope": "aggregated_target_model",
                            "statistic": float(bg_rows[0]),
                            "p_value": float(bg_rows[1]),
                            "status": "warning" if float(bg_rows[1]) < 0.05 else "pass",
                            "detail": (
                                "Low p-values suggest residual autocorrelation "
                                "remains in the aggregate time model."
                            ),
                        }
                    )
                except Exception:
                    pass

    for series_name, series in stationarity_series_candidates[:4]:
        rows.extend(
            _run_extended_stationarity_tests(
                series.reset_index(drop=True),
                scope=series_name,
                maximum_lag=config.maximum_lag,
            )
        )

    if rows:
        context.diagnostics_tables["time_series_extension_tests"] = pd.DataFrame(rows)


def _add_structural_break_framework_outputs(
    context: PipelineContext,
    top_features: list[str],
    labels_available_flag: bool,
) -> None:
    config = context.config.structural_breaks
    if not config.enabled:
        return
    if context.config.split.data_structure not in {DataStructure.TIME_SERIES, DataStructure.PANEL}:
        return
    minimum_rows = max(config.minimum_segment_size * 2, 8)
    series_frame = build_forecasting_series_frame(
        context,
        top_features=top_features,
        labels_available_flag=labels_available_flag,
        minimum_rows=minimum_rows,
    )
    if series_frame is None or len(series_frame) < minimum_rows:
        return

    candidate_breaks = _candidate_break_indices(
        len(series_frame),
        minimum_segment_size=config.minimum_segment_size,
        candidate_break_count=config.candidate_break_count,
    )
    if not candidate_breaks:
        return

    target_series_name = (
        "target_mean" if "target_mean" in series_frame.columns else "prediction_mean"
    )
    rows: list[dict[str, Any]] = []
    date_column = context.config.split.date_column
    for break_index in candidate_breaks:
        result = _chow_style_break_test(
            series_frame=series_frame,
            break_index=break_index,
            target_column=target_series_name,
        )
        breakpoint_label = (
            series_frame.iloc[break_index][date_column]
            if date_column and date_column in series_frame.columns
            else break_index
        )
        if result is None:
            left_values = pd.to_numeric(
                series_frame[target_series_name].iloc[:break_index],
                errors="coerce",
            ).dropna()
            right_values = pd.to_numeric(
                series_frame[target_series_name].iloc[break_index:],
                errors="coerce",
            ).dropna()
            if left_values.empty or right_values.empty:
                continue
            result = {
                "f_statistic": float(abs(right_values.mean() - left_values.mean())),
                "pooled_sse": float("nan"),
                "split_sse": float("nan"),
                "status": "review",
            }
        rows.append(
            {
                "break_index": int(break_index),
                "breakpoint": str(breakpoint_label),
                **result,
            }
        )
    cusum_rows = _run_cusum_stability_tests(
        series_frame=series_frame,
        top_features=top_features,
        minimum_rows=config.minimum_segment_size * 2,
    )
    if rows or cusum_rows:
        structural_breaks = pd.DataFrame(rows if rows else [])
        if not structural_breaks.empty and "f_statistic" in structural_breaks.columns:
            structural_breaks = structural_breaks.sort_values("f_statistic", ascending=False)
        if cusum_rows:
            structural_breaks = pd.concat(
                [structural_breaks, pd.DataFrame(cusum_rows)],
                ignore_index=True,
                sort=False,
            )
        context.diagnostics_tables["structural_break_tests"] = structural_breaks

        series_with_signal = series_frame.copy(deep=True)
        signal_column = (
            "residual" if "residual" in series_with_signal.columns else "prediction_mean"
        )
        window_size = max(
            config.minimum_segment_size,
            int(round(len(series_with_signal) * config.rolling_window_fraction)),
        )
        series_with_signal["rolling_signal"] = (
            pd.to_numeric(series_with_signal[signal_column], errors="coerce")
            .abs()
            .rolling(window_size, min_periods=1)
            .mean()
        )
        x_axis = (
            date_column
            if date_column and date_column in series_with_signal.columns
            else series_with_signal.index
        )
        context.visualizations["structural_break_profile"] = px.line(
            series_with_signal,
            x=x_axis,
            y="rolling_signal",
            title="Structural Break / Regime Profile",
            labels={"rolling_signal": "Rolling Signal"},
        )


def _add_model_comparison_framework_outputs(context: PipelineContext) -> None:
    comparison_table = context.comparison_results
    prediction_snapshots = context.metadata.get("comparison_prediction_snapshots")
    if comparison_table is None or comparison_table.empty or not isinstance(
        prediction_snapshots,
        dict,
    ):
        return

    primary_model = context.config.model.model_type.value
    primary_frame = prediction_snapshots.get(primary_model)
    if primary_frame is None or context.target_column is None:
        return

    score_column = score_column_name(context)
    rows: list[dict[str, Any]] = []
    for challenger_model, challenger_frame in prediction_snapshots.items():
        if challenger_model == primary_model or challenger_frame is None:
            continue
        aligned = _align_comparison_prediction_frames(
            primary_frame=primary_frame,
            challenger_frame=challenger_frame,
            context=context,
        )
        if aligned is None or len(aligned) < 20:
            continue
        if context.config.target.mode == TargetMode.BINARY:
            delong_result = _run_delong_test(
                y_true=aligned[context.target_column].astype(int).to_numpy(),
                baseline_scores=aligned[f"{score_column}_primary"].to_numpy(dtype=float),
                challenger_scores=aligned[f"{score_column}_challenger"].to_numpy(dtype=float),
            )
            if delong_result is not None:
                rows.append(
                    {
                        "comparison_model": challenger_model,
                        "test_name": "delong_auc_difference",
                        **delong_result,
                    }
                )
            if {
                "predicted_class_primary",
                "predicted_class_challenger",
            }.issubset(aligned.columns):
                rows.append(
                    {
                        "comparison_model": challenger_model,
                        "test_name": "mcnemar_threshold_difference",
                        **_run_mcnemar_test(
                            y_true=aligned[context.target_column].astype(int).to_numpy(),
                            baseline_class=aligned["predicted_class_primary"].astype(int).to_numpy(),
                            challenger_class=aligned["predicted_class_challenger"]
                            .astype(int)
                            .to_numpy(),
                        ),
                    }
                )
        dm_result = _run_diebold_mariano_test(
            y_true=aligned[context.target_column].to_numpy(dtype=float),
            baseline_scores=aligned[f"{score_column}_primary"].to_numpy(dtype=float),
            challenger_scores=aligned[f"{score_column}_challenger"].to_numpy(dtype=float),
            target_mode=context.config.target.mode,
        )
        if dm_result is not None:
            rows.append(
                {
                    "comparison_model": challenger_model,
                    "test_name": "diebold_mariano",
                    **dm_result,
                }
            )

    if rows:
        significance_table = pd.DataFrame(rows)
        context.diagnostics_tables["model_comparison_significance_tests"] = significance_table
        context.statistical_tests["model_comparison_significance"] = significance_table.to_dict(
            orient="records"
        )


def _add_specification_framework_extensions(
    context: PipelineContext,
    top_features: list[str],
    labels_available_flag: bool,
) -> None:
    if not labels_available_flag or context.target_column is None:
        return
    train_frame = context.split_frames.get("train")
    if train_frame is None:
        return
    numeric_features = _numeric_features_for_framework(
        context,
        top_features,
        maximum=6,
        frame=train_frame,
    )
    if len(numeric_features) < 2:
        numeric_features = [
            feature_name
            for feature_name in context.numeric_features
            if feature_name in train_frame.columns
        ][:6]
    if len(numeric_features) < 2:
        return

    target_series = pd.to_numeric(train_frame[context.target_column], errors="coerce")
    design = (
        train_frame[numeric_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    aligned = pd.concat([target_series.rename("target"), design], axis=1).dropna()
    if len(aligned) < 30:
        return

    fitted = None
    specification_rows: list[dict[str, Any]] = []
    try:
        exog = sm.add_constant(
            aligned[numeric_features].astype(float),
            has_constant="add",
        )
        fitted = sm.OLS(
            aligned["target"].astype(float),
            exog,
        ).fit()
        reset_test = linear_reset(fitted, power=2, use_f=True)
        specification_rows.append(
            {
                "test_name": "ramsey_reset",
                "scope": "surrogate_ols_specification",
                "statistic": float(getattr(reset_test, "fvalue", np.nan)),
                "p_value": float(getattr(reset_test, "pvalue", np.nan)),
                "status": (
                    "warning"
                    if pd.notna(getattr(reset_test, "pvalue", np.nan))
                    and float(reset_test.pvalue) < 0.05
                    else "pass"
                ),
                "detail": (
                    "RESET on a numeric surrogate regression. Low p-values suggest "
                    "functional-form misspecification."
                ),
            }
        )
        white_stat, white_p_value, _, _ = het_white(fitted.resid, fitted.model.exog)
        specification_rows.append(
            {
                "test_name": "white_test",
                "scope": "surrogate_ols_specification",
                "statistic": float(white_stat),
                "p_value": float(white_p_value),
                "status": "warning" if float(white_p_value) < 0.05 else "pass",
                "detail": "Low p-values suggest heteroskedasticity in the surrogate model.",
            }
        )
    except Exception:
        fitted = None

    if specification_rows:
        existing = context.diagnostics_tables.get("model_specification_tests", pd.DataFrame())
        context.diagnostics_tables["model_specification_tests"] = pd.concat(
            [existing, pd.DataFrame(specification_rows)],
            ignore_index=True,
            sort=False,
        )

    if fitted is None:
        return
    influence = fitted.get_influence()
    dfbetas_frame = pd.DataFrame(
        influence.dfbetas,
        columns=fitted.model.exog_names,
        index=aligned.index,
    )
    dfbeta_columns = [column for column in dfbetas_frame.columns if column != "const"]
    if dfbeta_columns:
        dfbetas_frame["max_abs_dfbeta"] = dfbetas_frame[dfbeta_columns].abs().max(axis=1)
        dfbetas_frame["driver_parameter"] = (
            dfbetas_frame[dfbeta_columns].abs().idxmax(axis=1).astype("string")
        )
        dfbetas_frame["observation_id"] = dfbetas_frame.index.astype(str)
        threshold = 2.0 / np.sqrt(max(len(dfbetas_frame), 1))
        flagged = dfbetas_frame.loc[
            dfbetas_frame["max_abs_dfbeta"] >= threshold,
            ["observation_id", "max_abs_dfbeta", "driver_parameter", *dfbeta_columns[:4]],
        ].sort_values("max_abs_dfbeta", ascending=False)
        context.diagnostics_tables["model_dfbetas_summary"] = flagged.head(50).reset_index(
            drop=True
        )

    dffits_values = np.asarray(influence.dffits[0], dtype=float)
    dffits_threshold = 2.0 * np.sqrt(fitted.model.exog.shape[1] / max(len(aligned), 1))
    dffits_frame = pd.DataFrame(
        {
            "observation_id": aligned.index.astype(str),
            "dffits": dffits_values,
            "absolute_dffits": np.abs(dffits_values),
        }
    ).sort_values("absolute_dffits", ascending=False)
    context.diagnostics_tables["model_dffits_summary"] = dffits_frame.head(50).reset_index(
        drop=True
    )

    if "model_influence_summary" in context.diagnostics_tables:
        influence_summary = context.diagnostics_tables["model_influence_summary"].copy(deep=True)
        influence_summary = influence_summary.merge(
            dffits_frame,
            on="observation_id",
            how="left",
        )
        if dfbeta_columns:
            influence_summary = influence_summary.merge(
                dfbetas_frame.loc[:, ["observation_id", "max_abs_dfbeta", "driver_parameter"]],
                on="observation_id",
                how="left",
            )
        context.diagnostics_tables["model_influence_summary"] = influence_summary

    extension_rows = [
        {
            "test_name": "dfbetas_flag_count",
            "scope": "surrogate_ols_influence",
            "statistic": int(
                context.diagnostics_tables.get(
                    "model_dfbetas_summary",
                    pd.DataFrame(),
                ).shape[0]
            ),
            "p_value": np.nan,
            "status": (
                "warning"
                if context.diagnostics_tables.get("model_dfbetas_summary", pd.DataFrame()).shape[0]
                > 0
                else "pass"
            ),
            "detail": f"Flagged using |DFBETA| >= {threshold:.4f}.",
        },
        {
            "test_name": "dffits_flag_count",
            "scope": "surrogate_ols_influence",
            "statistic": int((dffits_frame["absolute_dffits"] >= dffits_threshold).sum()),
            "p_value": np.nan,
            "status": "warning"
            if int((dffits_frame["absolute_dffits"] >= dffits_threshold).sum()) > 0
            else "pass",
            "detail": f"Flagged using |DFFITS| >= {dffits_threshold:.4f}.",
        },
    ]
    existing_specification = context.diagnostics_tables.get(
        "model_specification_tests",
        pd.DataFrame(),
    )
    context.diagnostics_tables["model_specification_tests"] = pd.concat(
        [existing_specification, pd.DataFrame(extension_rows)],
        ignore_index=True,
        sort=False,
    )


def _add_robustness_framework_extensions(context: PipelineContext) -> None:
    metric_summary = context.diagnostics_tables.get("robustness_metric_summary", pd.DataFrame())
    if metric_summary.empty:
        return
    summary = metric_summary.copy(deep=True)
    if {"mean", "std"}.issubset(summary.columns):
        summary["coefficient_of_variation"] = np.where(
            pd.to_numeric(summary["mean"], errors="coerce").abs() > 0,
            pd.to_numeric(summary["std"], errors="coerce")
            / pd.to_numeric(summary["mean"], errors="coerce").abs(),
            np.nan,
        )
        summary["stability_band"] = pd.cut(
            summary["coefficient_of_variation"].fillna(0.0),
            bins=[-np.inf, 0.05, 0.15, np.inf],
            labels=["stable", "monitor", "volatile"],
        ).astype("string")
    context.diagnostics_tables["robustness_framework_summary"] = summary


def _add_feature_workbench_outputs(
    context: PipelineContext,
    labels_available_flag: bool,
) -> None:
    config = context.config.feature_workbench
    if not config.enabled or context.working_data is None:
        return

    working_data = context.working_data
    transformation_table = context.diagnostics_tables.get(
        "governed_transformations", pd.DataFrame()
    )
    transformation_lookup = {
        str(row["output_feature"]): {
            "construction_type": str(row["transform_type"]),
            "source_feature": str(row["source_feature"]),
            "secondary_feature": str(row.get("secondary_feature", "")),
        }
        for _, row in transformation_table.iterrows()
        if str(row.get("status", "")).startswith("applied")
    }
    generated_interactions = set(context.metadata.get("generated_interaction_features", []))
    generated_indicators = set(context.metadata.get("generated_missing_indicator_columns", []))

    rows: list[dict[str, Any]] = []
    for feature_name in context.feature_columns:
        if feature_name not in working_data.columns:
            continue
        if feature_name in generated_indicators:
            construction_type = "missing_indicator"
            source_feature = feature_name.rsplit("__missing_indicator", 1)[0]
            secondary_feature = ""
        elif feature_name in transformation_lookup:
            construction_type = transformation_lookup[feature_name]["construction_type"]
            source_feature = transformation_lookup[feature_name]["source_feature"]
            secondary_feature = transformation_lookup[feature_name]["secondary_feature"]
        elif feature_name in generated_interactions:
            construction_type = "interaction"
            source_feature = feature_name
            secondary_feature = ""
        else:
            continue
        series = working_data[feature_name]
        numeric_series = pd.to_numeric(series, errors="coerce")
        row = {
            "feature_name": feature_name,
            "construction_type": construction_type,
            "source_feature": source_feature,
            "secondary_feature": secondary_feature,
            "non_null_pct": float(series.notna().mean() * 100),
            "unique_count": int(series.nunique(dropna=True)),
            "mean": float(numeric_series.mean()) if numeric_series.notna().any() else np.nan,
            "std": float(numeric_series.std(ddof=0)) if numeric_series.notna().any() else np.nan,
        }
        if (
            config.include_target_association
            and labels_available_flag
            and context.target_column
            and context.target_column in working_data.columns
        ):
            aligned = pd.DataFrame(
                {
                    "feature": numeric_series,
                    "target": pd.to_numeric(
                        working_data[context.target_column], errors="coerce"
                    ),
                }
            ).dropna()
            row["target_association"] = (
                float(abs(aligned["feature"].corr(aligned["target"])))
                if len(aligned) >= 10 and aligned["feature"].nunique() > 1
                else np.nan
            )
        rows.append(row)
    if not rows:
        return
    workbench = pd.DataFrame(rows).sort_values(
        ["target_association", "non_null_pct"],
        ascending=[False, False],
        na_position="last",
    )
    context.diagnostics_tables["feature_construction_workbench"] = workbench
    if "target_association" in workbench.columns:
        chart_data = workbench.dropna(subset=["target_association"]).head(config.max_features)
        if not chart_data.empty:
            context.visualizations["feature_construction_association"] = px.bar(
                chart_data.sort_values("target_association", ascending=True),
                x="target_association",
                y="feature_name",
                color="construction_type",
                orientation="h",
                title="Constructed Feature Association",
                labels={
                    "target_association": "Absolute Target Association",
                    "feature_name": "Feature",
                },
            )


def _add_preset_recommendation_outputs(context: PipelineContext) -> None:
    config = context.config.preset_recommendations
    preset_name = context.config.preset_name
    if not config.enabled or preset_name is None:
        return
    preset_recommendations = PRESET_FRAMEWORK_RECOMMENDATIONS.get(preset_name)
    if preset_recommendations is None:
        return

    if config.include_imputation_recommendations:
        context.diagnostics_tables["preset_imputation_recommendations"] = pd.DataFrame(
            _render_preset_recommendation_rows(
                recommendations=preset_recommendations["imputation"],
                current_state=build_current_imputation_state(context),
                category="imputation",
            )
        )
    if config.include_transformation_recommendations:
        context.diagnostics_tables["preset_transformation_recommendations"] = pd.DataFrame(
            _render_preset_recommendation_rows(
                recommendations=preset_recommendations["transformations"],
                current_state=build_current_transformation_state(context),
                category="transformation",
            )
        )
    if config.include_test_recommendations:
        context.diagnostics_tables["preset_test_recommendations"] = pd.DataFrame(
            _render_preset_recommendation_rows(
                recommendations=preset_recommendations["tests"],
                current_state=build_current_test_state(context),
                category="test",
            )
        )


def _select_multiple_imputation_features(
    context: PipelineContext,
    train_frame: pd.DataFrame,
) -> list[str]:
    numeric_features = list(
        context.metadata.get("pre_imputation_numeric_features", context.numeric_features)
    )
    rows: list[tuple[int, str]] = []
    for feature_name in numeric_features:
        if feature_name not in train_frame.columns:
            continue
        missing_count = int(train_frame[feature_name].isna().sum())
        if missing_count <= 0:
            continue
        rows.append((missing_count, feature_name))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [feature_name for _, feature_name in rows]


def _fit_multiple_imputation_surrogate(
    *,
    context: PipelineContext,
    train_features: pd.DataFrame,
    train_target: pd.Series,
):
    design = sm.add_constant(train_features.astype(float), has_constant="add")
    aligned = pd.concat([train_target.rename("target"), design], axis=1).dropna()
    if len(aligned) < 30:
        return None
    try:
        if context.config.target.mode == TargetMode.BINARY:
            if aligned["target"].nunique() < 2:
                return None
            return sm.Logit(
                aligned["target"].astype(float),
                aligned.drop(columns=["target"]),
            ).fit(disp=False)
        return sm.OLS(
            aligned["target"].astype(float),
            aligned.drop(columns=["target"]),
        ).fit()
    except Exception:
        return None


def _predict_multiple_imputation_surrogate(
    *,
    fitted,
    evaluation_features: pd.DataFrame,
    target_mode: TargetMode,
) -> np.ndarray | None:
    try:
        design = sm.add_constant(evaluation_features.astype(float), has_constant="add")
        predictions = np.asarray(fitted.predict(design), dtype=float)
    except Exception:
        return None
    if target_mode == TargetMode.BINARY:
        return np.clip(predictions, 1e-6, 1 - 1e-6)
    return predictions


def _extract_multiple_imputation_coefficients(
    *,
    fitted,
    dataset_id: int,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    params = pd.Series(fitted.params).drop(labels=["const"], errors="ignore")
    standard_errors = pd.Series(getattr(fitted, "bse", np.nan))
    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        if feature_name not in params.index:
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "feature_name": feature_name,
                "coefficient": float(params.loc[feature_name]),
                "std_error": float(standard_errors.get(feature_name, np.nan)),
            }
        )
    return rows


def _score_multiple_imputation_prediction_frame(
    prediction_frame: pd.DataFrame,
    *,
    dataset_id: int,
    target_mode: TargetMode,
) -> dict[str, Any]:
    y_true = prediction_frame["target"].to_numpy(dtype=float)
    y_pred = prediction_frame["prediction"].to_numpy(dtype=float)
    row: dict[str, Any] = {"dataset_id": dataset_id}
    if target_mode == TargetMode.BINARY:
        row.update(
            {
                "roc_auc": float(roc_auc_score(y_true.astype(int), y_pred)),
                "brier_score": float(brier_score_loss(y_true.astype(int), y_pred)),
                "log_loss": float(log_loss(y_true.astype(int), y_pred, labels=[0, 1])),
            }
        )
    else:
        row.update(
            {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        )
    return row


def _pool_multiple_imputation_coefficients(coefficient_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, feature_frame in coefficient_table.groupby("feature_name", dropna=False):
        coefficient_values = pd.to_numeric(feature_frame["coefficient"], errors="coerce").dropna()
        if len(coefficient_values) < 2:
            continue
        standard_errors = pd.to_numeric(feature_frame["std_error"], errors="coerce")
        within_variance = (
            float(np.nanmean(np.square(standard_errors)))
            if standard_errors.notna().any()
            else 0.0
        )
        between_variance = float(coefficient_values.var(ddof=1))
        total_variance = (
            within_variance
            + (1.0 + 1.0 / len(coefficient_values)) * between_variance
        )
        pooled_std_error = float(np.sqrt(max(total_variance, 0.0)))
        z_statistic = (
            coefficient_values.mean() / pooled_std_error
            if pooled_std_error > 0
            else np.nan
        )
        p_value = 2 * (1 - norm.cdf(abs(z_statistic))) if pd.notna(z_statistic) else np.nan
        rows.append(
            {
                "feature_name": feature_name,
                "pooled_coefficient": float(coefficient_values.mean()),
                "pooled_std_error": pooled_std_error,
                "within_variance": within_variance,
                "between_variance": between_variance,
                "total_variance": total_variance,
                "z_statistic": float(z_statistic) if pd.notna(z_statistic) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "dataset_count": int(len(coefficient_values)),
            }
        )
    pooled = pd.DataFrame(rows)
    if pooled.empty:
        return pooled
    return pooled.sort_values("dataset_count", ascending=False).reset_index(drop=True)


def _pool_multiple_imputation_metrics(
    metric_table: pd.DataFrame,
    target_mode: TargetMode,
) -> pd.DataFrame:
    metric_columns = (
        ["roc_auc", "brier_score", "log_loss"]
        if target_mode == TargetMode.BINARY
        else ["rmse", "mae"]
    )
    rows: list[dict[str, Any]] = []
    for metric_name in metric_columns:
        if metric_name not in metric_table.columns:
            continue
        values = pd.to_numeric(metric_table[metric_name], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric_name": metric_name,
                "mean_value": float(values.mean()),
                "std_value": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "dataset_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _build_littles_mcar_output(context: PipelineContext) -> dict[str, Any] | None:
    pre_imputation_frames = context.metadata.get("pre_imputation_split_frames")
    if not isinstance(pre_imputation_frames, dict):
        return None
    combined = pd.concat(pre_imputation_frames.values(), ignore_index=True)
    numeric_features = _select_multiple_imputation_features(context, combined)
    if len(numeric_features) < 2:
        return None
    numeric_features = numeric_features[: min(8, len(numeric_features))]
    frame = (
        combined[numeric_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    if frame.isna().sum().sum() == 0:
        return None

    overall_mean = frame.mean()
    overall_cov = frame.cov(min_periods=2)
    patterns = frame.isna().astype(int).astype(str).agg("".join, axis=1)
    statistic = 0.0
    df_total = 0
    usable_patterns = 0
    for pattern_name, index_values in patterns.groupby(patterns).groups.items():
        observed_features = [
            feature_name
            for feature_name, is_missing in zip(numeric_features, pattern_name, strict=False)
            if is_missing == "0"
        ]
        if not observed_features:
            continue
        group_frame = frame.loc[index_values, observed_features]
        if len(group_frame) < 2:
            continue
        group_mean = group_frame.mean()
        covariance_slice = overall_cov.loc[observed_features, observed_features]
        if covariance_slice.isna().any().any():
            continue
        diff = (group_mean - overall_mean.loc[observed_features]).to_numpy(dtype=float)
        precision = np.linalg.pinv(covariance_slice.to_numpy(dtype=float))
        statistic += float(len(group_frame) * diff.T @ precision @ diff)
        df_total += len(observed_features)
        usable_patterns += 1
    degrees_of_freedom = max(df_total - len(numeric_features), 1)
    p_value = 1.0 - chi2.cdf(statistic, degrees_of_freedom)
    return {
        "test_name": "littles_mcar",
        "feature_count": int(len(numeric_features)),
        "pattern_count": int(usable_patterns),
        "statistic": float(statistic),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": float(p_value),
        "status": "warning" if float(p_value) < 0.05 else "pass",
        "detail": (
            "Approximate Little's MCAR test using pairwise covariance moments. "
            "Low p-values suggest missingness is unlikely to be completely random."
        ),
    }


def _align_comparison_prediction_frames(
    *,
    primary_frame: pd.DataFrame,
    challenger_frame: pd.DataFrame,
    context: PipelineContext,
) -> pd.DataFrame | None:
    target_column = context.target_column
    if target_column is None:
        return None
    score_column = score_column_name(context)
    base_columns = [target_column, score_column]
    challenger_columns = [score_column]
    if "predicted_class" in primary_frame.columns:
        base_columns.append("predicted_class")
    if "predicted_class" in challenger_frame.columns:
        challenger_columns.append("predicted_class")
    date_column = context.config.split.date_column
    entity_column = context.config.split.entity_column
    key_columns = [
        column_name
        for column_name in (date_column, entity_column)
        if (
            column_name
            and column_name in primary_frame.columns
            and column_name in challenger_frame.columns
        )
    ]
    if key_columns:
        left_columns = list(dict.fromkeys([*key_columns, *base_columns]))
        right_columns = list(dict.fromkeys([*key_columns, *challenger_columns]))
        left = primary_frame.loc[:, left_columns].copy(deep=True)
        right = challenger_frame.loc[:, right_columns].copy(deep=True)
        aligned = left.merge(
            right,
            on=key_columns,
            how="inner",
            suffixes=("_primary", "_challenger"),
        )
    else:
        row_count = min(len(primary_frame), len(challenger_frame))
        left = primary_frame.loc[:, base_columns].head(row_count).reset_index(drop=True)
        right = challenger_frame.loc[:, challenger_columns].head(row_count).reset_index(drop=True)
        aligned = pd.concat([left, right.add_suffix("_challenger")], axis=1)
        aligned = aligned.rename(columns={f"{score_column}": f"{score_column}_primary"})
        if "predicted_class" in aligned.columns:
            aligned = aligned.rename(columns={"predicted_class": "predicted_class_primary"})
    if f"{score_column}_primary" not in aligned.columns:
        aligned = aligned.rename(columns={score_column: f"{score_column}_primary"})
    return aligned.dropna(
        subset=[
            target_column,
            f"{score_column}_primary",
            f"{score_column}_challenger",
        ]
    )


def _run_mcnemar_test(
    *,
    y_true: np.ndarray,
    baseline_class: np.ndarray,
    challenger_class: np.ndarray,
) -> dict[str, Any]:
    baseline_correct = baseline_class == y_true
    challenger_correct = challenger_class == y_true
    table = np.array(
        [
            [
                int(np.sum(baseline_correct & challenger_correct)),
                int(np.sum(baseline_correct & ~challenger_correct)),
            ],
            [
                int(np.sum(~baseline_correct & challenger_correct)),
                int(np.sum(~baseline_correct & ~challenger_correct)),
            ],
        ],
        dtype=int,
    )
    discordant_pairs = int(table[0, 1] + table[1, 0])
    if discordant_pairs == 0:
        return {
            "statistic": 0.0,
            "p_value": np.nan,
            "status": "review",
            "detail": (
                "McNemar testing is uninformative because the paired predictions have "
                "no discordant correctness outcomes on the evaluated sample."
            ),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mcnemar(table, exact=False, correction=True)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "status": "warning" if float(result.pvalue) < 0.05 else "pass",
        "detail": "Low p-values suggest materially different thresholded classifications.",
    }


def _run_diebold_mariano_test(
    *,
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    challenger_scores: np.ndarray,
    target_mode: TargetMode,
) -> dict[str, Any] | None:
    if len(y_true) < 20:
        return None
    if target_mode == TargetMode.BINARY:
        baseline_loss = np.square(y_true - baseline_scores)
        challenger_loss = np.square(y_true - challenger_scores)
    else:
        baseline_loss = np.square(y_true - baseline_scores)
        challenger_loss = np.square(y_true - challenger_scores)
    differential = baseline_loss - challenger_loss
    statistic, p_value = _diebold_mariano_statistic(differential)
    if statistic is None:
        return None
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "status": "warning" if float(p_value) < 0.05 else "pass",
        "detail": "Low p-values suggest the challenger materially changes forecast error.",
    }


def _diebold_mariano_statistic(differential: np.ndarray) -> tuple[float | None, float | None]:
    differential = np.asarray(differential, dtype=float)
    differential = differential[np.isfinite(differential)]
    if len(differential) < 20:
        return None, None
    bandwidth = min(3, max(1, len(differential) // 8))
    centered = differential - differential.mean()
    gamma0 = float(np.dot(centered, centered) / len(centered))
    long_run_variance = gamma0
    for lag in range(1, bandwidth + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / len(centered))
        weight = 1.0 - lag / (bandwidth + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return None, None
    statistic = float(differential.mean() / np.sqrt(long_run_variance / len(differential)))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(statistic))))
    return statistic, p_value


def _run_delong_test(
    *,
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    challenger_scores: np.ndarray,
) -> dict[str, Any] | None:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        aucs, covariance = _fast_delong(
            np.vstack([baseline_scores, challenger_scores])[:, np.argsort(-y_true)],
            int(y_true.sum()),
        )
    except Exception:
        return None
    variance = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    if variance <= 0:
        return None
    z_statistic = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_statistic))))
    return {
        "statistic": z_statistic,
        "p_value": p_value,
        "status": "warning" if p_value < 0.05 else "pass",
        "detail": "DeLong test for paired ROC AUC difference on the held-out ranking split.",
    }


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    sorted_index = np.argsort(values)
    sorted_values = values[sorted_index]
    midranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(len(values), dtype=float)
    result[sorted_index] = midranks
    return result


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    positive_examples = predictions_sorted_transposed[:, :label_1_count]
    negative_examples = predictions_sorted_transposed[:, label_1_count:]
    classifier_count = predictions_sorted_transposed.shape[0]
    m = positive_examples.shape[1]
    n = negative_examples.shape[1]
    tx = np.empty((classifier_count, m), dtype=float)
    ty = np.empty((classifier_count, n), dtype=float)
    tz = np.empty((classifier_count, m + n), dtype=float)
    for classifier_index in range(classifier_count):
        tx[classifier_index] = _compute_midrank(positive_examples[classifier_index])
        ty[classifier_index] = _compute_midrank(negative_examples[classifier_index])
        tz[classifier_index] = _compute_midrank(predictions_sorted_transposed[classifier_index])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def _run_extended_stationarity_tests(
    series: pd.Series,
    *,
    scope: str,
    maximum_lag: int,
) -> list[dict[str, Any]]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean_series) < 8:
        return []
    rows: list[dict[str, Any]] = []
    try:
        kpss_detail = "Low p-values suggest the series is not level-stationary."
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", InterpolationWarning)
            kpss_statistic, kpss_p_value, _, _ = kpss(
                clean_series.to_numpy(dtype=float),
                regression="c",
                nlags=min(maximum_lag, max(1, len(clean_series) // 5)),
            )
        if any(issubclass(record.category, InterpolationWarning) for record in captured):
            kpss_detail += (
                " Statsmodels reported that the p-value lies outside its lookup-table "
                "range, so treat the direction of the result as more informative than "
                "the exact boundary value."
            )
        rows.append(
            {
                "test_name": "kpss",
                "scope": scope,
                "statistic": float(kpss_statistic),
                "p_value": float(kpss_p_value),
                "status": "warning" if float(kpss_p_value) < 0.05 else "pass",
                "detail": kpss_detail,
            }
        )
    except Exception:
        pass
    phillips_perron = _phillips_perron_test(clean_series, maximum_lag=maximum_lag)
    if phillips_perron is not None:
        rows.append(
            {
                "test_name": "phillips_perron",
                "scope": scope,
                **phillips_perron,
            }
        )
    return rows


def _phillips_perron_test(
    series: pd.Series,
    *,
    maximum_lag: int,
) -> dict[str, Any] | None:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 8:
        return None
    delta_y = np.diff(clean)
    lagged_level = clean[:-1]
    design = sm.add_constant(lagged_level, has_constant="add")
    try:
        fitted = sm.OLS(delta_y, design).fit()
    except Exception:
        return None
    residuals = np.asarray(fitted.resid, dtype=float)
    sample_size = len(residuals)
    bandwidth = min(maximum_lag, max(1, int(round(4 * (sample_size / 100) ** (2 / 9)))))
    gamma0 = float(np.dot(residuals, residuals) / sample_size)
    long_run_variance = gamma0
    for lag in range(1, bandwidth + 1):
        covariance = float(np.dot(residuals[lag:], residuals[:-lag]) / sample_size)
        weight = 1.0 - lag / (bandwidth + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0 or gamma0 <= 0:
        return None
    base_t = float(fitted.tvalues[1])
    slope_std_error = float(fitted.bse[1])
    scale = float(np.sqrt(gamma0))
    adjustment = 0.5 * ((long_run_variance - gamma0) / long_run_variance) * (
        sample_size * slope_std_error / max(scale, 1e-12)
    )
    statistic = float(np.sqrt(gamma0 / long_run_variance) * base_t - adjustment)
    try:
        p_value = float(mackinnonp(statistic, regression="c", N=1))
    except Exception:
        p_value = np.nan
    return {
        "statistic": statistic,
        "p_value": p_value,
        "status": "warning" if pd.notna(p_value) and p_value >= 0.05 else "pass",
        "detail": "Phillips-Perron-style unit-root test with a Newey-West variance correction.",
    }


def _run_cusum_stability_tests(
    *,
    series_frame: pd.DataFrame,
    top_features: list[str],
    minimum_rows: int,
) -> list[dict[str, Any]]:
    if "target_mean" not in series_frame.columns or "prediction_mean" not in series_frame.columns:
        return []
    exog_columns = [
        "prediction_mean",
        *[feature for feature in top_features[:2] if feature in series_frame.columns],
    ]
    design = sm.add_constant(
        series_frame[exog_columns].apply(pd.to_numeric, errors="coerce"),
        has_constant="add",
    )
    aligned = pd.concat(
        [
            pd.to_numeric(series_frame["target_mean"], errors="coerce").rename("target_mean"),
            design,
        ],
        axis=1,
    ).dropna()
    if len(aligned) < minimum_rows:
        return []
    try:
        fitted = sm.OLS(
            aligned["target_mean"].astype(float),
            aligned.drop(columns=["target_mean"]),
        ).fit()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    try:
        cusum_statistic, cusum_p_value, _ = breaks_cusumolsresid(
            fitted.resid,
            ddof=int(fitted.df_model) + 1,
        )
        rows.append(
            {
                "break_index": np.nan,
                "breakpoint": "global",
                "test_name": "cusum",
                "f_statistic": float(cusum_statistic),
                "pooled_sse": np.nan,
                "split_sse": np.nan,
                "p_value": float(cusum_p_value),
                "status": "warning" if float(cusum_p_value) < 0.05 else "pass",
            }
        )
    except Exception:
        pass
    try:
        recursive = recursive_olsresiduals(fitted)
        standardized_residuals = np.asarray(recursive[4], dtype=float)
        standardized_residuals = standardized_residuals[np.isfinite(standardized_residuals)]
        if len(standardized_residuals) >= 5:
            cusum_sq = np.cumsum(np.square(standardized_residuals))
            cusum_sq = cusum_sq / cusum_sq[-1]
            benchmark = np.arange(1, len(cusum_sq) + 1, dtype=float) / len(cusum_sq)
            max_deviation = float(np.max(np.abs(cusum_sq - benchmark)))
            rows.append(
                {
                    "break_index": np.nan,
                    "breakpoint": "global",
                    "test_name": "cusum_squares",
                    "f_statistic": max_deviation,
                    "pooled_sse": np.nan,
                    "split_sse": np.nan,
                    "p_value": np.nan,
                    "status": "warning" if max_deviation >= 0.15 else "pass",
                }
            )
    except Exception:
        pass
    return rows


def build_forecasting_series_frame(
    context: PipelineContext,
    *,
    top_features: list[str],
    labels_available_flag: bool,
    minimum_rows: int,
) -> pd.DataFrame | None:
    date_column = context.config.split.date_column
    if not date_column:
        return None
    for split_name in ("test", "validation", "train"):
        prediction_frame = context.predictions.get(split_name)
        if prediction_frame is None or date_column not in prediction_frame.columns:
            continue
        aggregation_frame = prediction_frame.copy(deep=True)
        aggregations: dict[str, tuple[str, str]] = {
            "prediction_mean": (score_column_name(context), "mean"),
        }
        if (
            labels_available_flag
            and context.target_column
            and context.target_column in aggregation_frame.columns
        ):
            aggregations["target_mean"] = (context.target_column, "mean")
        for feature_name in top_features[:3]:
            if feature_name in aggregation_frame.columns:
                aggregations[feature_name] = (feature_name, "mean")
        series_frame = (
            aggregation_frame.groupby(date_column, dropna=False)
            .agg(**aggregations)
            .sort_index()
            .reset_index()
        )
        if "target_mean" in series_frame.columns:
            series_frame["residual"] = series_frame["target_mean"] - series_frame["prediction_mean"]
        if len(series_frame) >= minimum_rows:
            return series_frame
    return None


def _compute_condition_index_detail(matrix: np.ndarray) -> list[dict[str, Any]]:
    centered = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    column_std = np.nanstd(centered, axis=0, keepdims=True)
    column_std[column_std == 0] = 1.0
    standardized = centered / column_std
    _, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    if singular_values.size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for index, singular_value in enumerate(singular_values, start=1):
        rows.append(
            {
                "component": index,
                "singular_value": float(singular_value),
                "condition_index": float(np.max(singular_values) / singular_value)
                if singular_value != 0.0
                else np.inf,
            }
        )
    return rows


def _build_dependency_clusters(
    corr_matrix: pd.DataFrame,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    remaining = set(corr_matrix.columns)
    rows: list[dict[str, Any]] = []
    cluster_id = 1
    while remaining:
        seed = remaining.pop()
        cluster = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            correlated = {
                candidate
                for candidate in remaining
                if abs(float(corr_matrix.loc[current, candidate])) >= threshold
            }
            frontier.extend(sorted(correlated))
            cluster.update(correlated)
            remaining -= correlated
        if len(cluster) <= 1:
            continue
        cluster_features = sorted(cluster)
        max_corr = max(
            abs(float(corr_matrix.loc[left, right]))
            for left in cluster_features
            for right in cluster_features
            if left != right
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "feature_count": len(cluster_features),
                "features": ", ".join(cluster_features),
                "max_absolute_correlation": max_corr,
            }
        )
        cluster_id += 1
    return rows


def _seasonal_strength(series: pd.Series, *, period: int) -> float:
    if len(series) < period * 2:
        return float("nan")
    buckets = series.groupby(series.index % period).mean()
    variance = float(series.var(ddof=0))
    if variance == 0:
        return 0.0
    return float(buckets.var(ddof=0) / variance)


def _build_seasonality_profile(series: pd.Series, *, period: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "seasonal_bucket": series.index % period,
            "value": series.to_numpy(dtype=float),
        }
    )
    return (
        frame.groupby("seasonal_bucket", dropna=False)["value"]
        .mean()
        .rename("average_value")
        .reset_index()
    )


def _candidate_break_indices(
    total_length: int,
    *,
    minimum_segment_size: int,
    candidate_break_count: int,
) -> list[int]:
    start = minimum_segment_size
    stop = total_length - minimum_segment_size
    if stop <= start:
        return []
    candidate_space = np.linspace(start, stop, num=candidate_break_count + 2, dtype=int)[1:-1]
    return sorted({int(value) for value in candidate_space if start <= int(value) <= stop})


def _chow_style_break_test(
    *,
    series_frame: pd.DataFrame,
    break_index: int,
    target_column: str,
) -> dict[str, Any] | None:
    if "prediction_mean" not in series_frame.columns:
        return None
    design = sm.add_constant(
        series_frame[["prediction_mean"]].apply(pd.to_numeric, errors="coerce"),
        has_constant="add",
    )
    target = pd.to_numeric(series_frame[target_column], errors="coerce")
    aligned = pd.concat([target.rename(target_column), design], axis=1).dropna()
    if len(aligned) <= 6 or break_index <= 2 or break_index >= len(aligned) - 2:
        return None
    left = aligned.iloc[:break_index]
    right = aligned.iloc[break_index:]
    if len(left) < 3 or len(right) < 3:
        return None
    try:
        pooled_fit = sm.OLS(
            aligned[target_column].astype(float),
            aligned.drop(columns=[target_column]),
        ).fit()
        left_fit = sm.OLS(
            left[target_column].astype(float),
            left.drop(columns=[target_column]),
        ).fit()
        right_fit = sm.OLS(
            right[target_column].astype(float),
            right.drop(columns=[target_column]),
        ).fit()
    except Exception:
        return None
    sse_pooled = float(np.sum(np.square(pooled_fit.resid)))
    sse_split = float(np.sum(np.square(left_fit.resid)) + np.sum(np.square(right_fit.resid)))
    parameter_count = int(aligned.drop(columns=[target_column]).shape[1])
    denominator_df = len(left) + len(right) - (2 * parameter_count)
    if denominator_df <= 0 or sse_split <= 0:
        return None
    f_statistic = ((sse_pooled - sse_split) / parameter_count) / (sse_split / denominator_df)
    return {
        "f_statistic": float(f_statistic),
        "pooled_sse": sse_pooled,
        "split_sse": sse_split,
        "status": "warning" if f_statistic > 2.0 else "pass",
    }


def build_current_imputation_state(context: PipelineContext) -> dict[str, bool]:
    rules = context.diagnostics_tables.get("imputation_rules", pd.DataFrame())
    if rules.empty:
        return {}
    applied_policies = {
        str(policy)
        for policy in rules.get("applied_policy", pd.Series(dtype="object")).astype(str).tolist()
    }
    return {
        "grouped_scalar": bool(
            context.diagnostics_tables.get("imputation_group_rules", pd.DataFrame()).shape[0]
        ),
        "model_based_imputation": bool({"knn", "iterative"} & applied_policies),
        "multiple_imputation_pooling": bool(
            "multiple_imputation_pooling_summary" in context.diagnostics_tables
        ),
        "missingness_indicators": bool(
            context.metadata.get("generated_missing_indicator_columns", [])
        ),
        "imputation_sensitivity": context.config.imputation_sensitivity.enabled,
    }


def build_current_transformation_state(context: PipelineContext) -> dict[str, bool]:
    transformations = context.diagnostics_tables.get("governed_transformations", pd.DataFrame())
    transform_types = {
        str(transform_type)
        for transform_type in transformations.get("transform_type", pd.Series(dtype="object"))
        .astype(str)
        .tolist()
    }
    return {
        "governed_transformations": context.config.transformations.enabled,
        "manual_bins": "manual_bins" in transform_types,
        "interactions": bool(context.metadata.get("generated_interaction_features", []))
        or "interaction" in transform_types,
        "spline_transforms": "natural_spline" in transform_types,
        "time_series_transforms": bool(
            {
                "lag",
                "difference",
                "ewma",
                "rolling_mean",
                "rolling_median",
                "rolling_min",
                "rolling_max",
                "rolling_std",
                "pct_change",
            }
            & transform_types
        ),
    }


def build_current_test_state(context: PipelineContext) -> dict[str, bool]:
    return {
        "distribution_tests": "distribution_tests" in context.diagnostics_tables,
        "missingness_framework": "missingness_predictive_power" in context.diagnostics_tables,
        "missingness_mcar": "littles_mcar_test" in context.diagnostics_tables,
        "specification_tests": "model_specification_tests" in context.diagnostics_tables,
        "comparison_significance": (
            "model_comparison_significance_tests" in context.diagnostics_tables
        ),
        "dependency_framework": "dependency_cluster_summary" in context.diagnostics_tables,
        "residual_framework": "residual_diagnostics" in context.diagnostics_tables,
        "time_series_framework": "time_series_extension_tests" in context.diagnostics_tables,
        "structural_break_framework": "structural_break_tests" in context.diagnostics_tables,
        "robustness_framework": "robustness_framework_summary" in context.diagnostics_tables,
    }


def _append_performance_action(
    context: PipelineContext,
    *,
    action_name: str,
    detail: str,
    original_rows: int,
    effective_rows: int,
) -> None:
    row = pd.DataFrame(
        [
            {
                "action_name": action_name,
                "detail": detail,
                "original_rows": int(original_rows),
                "effective_rows": int(effective_rows),
            }
        ]
    )
    existing = context.diagnostics_tables.get("performance_hardening_actions")
    if existing is None or existing.empty:
        context.diagnostics_tables["performance_hardening_actions"] = row
    else:
        context.diagnostics_tables["performance_hardening_actions"] = pd.concat(
            [existing, row],
            ignore_index=True,
        )


def _render_preset_recommendation_rows(
    *,
    recommendations: list[tuple[str, str]],
    current_state: dict[str, bool],
    category: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, rationale in recommendations:
        rows.append(
            {
                "category": category,
                "recommendation_key": key,
                "currently_enabled": bool(current_state.get(key, False)),
                "rationale": rationale,
            }
        )
    return rows


PRESET_FRAMEWORK_RECOMMENDATIONS: dict[
    PresetName,
    dict[str, list[tuple[str, str]]],
] = {
    PresetName.PD_DEVELOPMENT: {
        "imputation": [
            (
                "missingness_indicators",
                "Missingness often carries signal in borrower-level PD data.",
            ),
            (
                "grouped_scalar",
                "Portfolio or segment-aware fills improve auditability for retail segments.",
            ),
            (
                "imputation_sensitivity",
                "PD development should show whether fill choices move discrimination materially.",
            ),
        ],
        "transformations": [
            (
                "governed_transformations",
                "PD builds often need documented non-linear or capped features.",
            ),
            (
                "interactions",
                "Selected interactions can capture combined utilization and "
                "capacity stress effects.",
            ),
            (
                "manual_bins",
                "Scorecard-aligned binning remains useful for challenger development.",
            ),
        ],
        "tests": [
            (
                "distribution_tests",
                "Distribution and shift tests help validate feature stability.",
            ),
            (
                "missingness_framework",
                "Missingness-to-target evidence is important for variable justification.",
            ),
            (
                "specification_tests",
                "Logit form checks are part of interpretable PD development.",
            ),
            (
                "robustness_framework",
                "Repeated-resample stability is helpful for challenger review.",
            ),
        ],
    },
    PresetName.LIFETIME_PD_CECL: {
        "imputation": [
            (
                "missingness_indicators",
                "Lifetime panel data often benefits from explicit missingness indicators.",
            ),
            (
                "model_based_imputation",
                "Panel structures can support richer numeric imputation on macro "
                "and account features.",
            ),
            (
                "imputation_sensitivity",
                "CECL development should show whether expected-loss outputs are fill-sensitive.",
            ),
        ],
        "transformations": [
            (
                "governed_transformations",
                "Lag and rolling transforms are natural for lifetime PD term structures.",
            ),
            (
                "time_series_transforms",
                "Temporal transforms align with forward-looking CECL panels.",
            ),
            (
                "interactions",
                "Macro-by-account interactions help encode scenario sensitivity.",
            ),
        ],
        "tests": [
            (
                "time_series_framework",
                "Lifetime PD workflows benefit from deeper residual and econometric checks.",
            ),
            (
                "structural_break_framework",
                "Regime testing helps identify timing shifts in stressed panels.",
            ),
            (
                "distribution_tests",
                "Panel feature distributions should be checked across time.",
            ),
        ],
    },
    PresetName.LGD_SEVERITY: {
        "imputation": [
            (
                "grouped_scalar",
                "Workout or collateral segments often justify grouped severity imputation.",
            ),
            (
                "imputation_sensitivity",
                "LGD severity can be sensitive to fill assumptions in sparse recovery fields.",
            ),
        ],
        "transformations": [
            (
                "governed_transformations",
                "Bounded LGD models often need transformed severity drivers.",
            ),
            (
                "manual_bins",
                "Documented bins can simplify severity segmentation.",
            ),
        ],
        "tests": [
            (
                "residual_framework",
                "Continuous severity models need residual-bias and heteroskedasticity review.",
            ),
            (
                "outlier_framework",
                "Recovery and severity tails frequently create influential observations.",
            ),
            (
                "distribution_tests",
                "LGD drivers often show heavy tails that should be documented.",
            ),
        ],
    },
    PresetName.CCAR_FORECASTING: {
        "imputation": [
            (
                "model_based_imputation",
                "Macro-linked forecasting panels often support richer numeric fills.",
            ),
            (
                "grouped_scalar",
                "Segment-aware fills keep macro forecasting rules transparent.",
            ),
        ],
        "transformations": [
            (
                "governed_transformations",
                "Forecasting workflows benefit from documented lag, rolling, and "
                "volatility transforms.",
            ),
            (
                "time_series_transforms",
                "Temporal transforms are central to CCAR development.",
            ),
            (
                "interactions",
                "Macro-by-portfolio interactions can improve stress sensitivity.",
            ),
        ],
        "tests": [
            (
                "time_series_framework",
                "Econometric checks are central to forecasting development.",
            ),
            (
                "structural_break_framework",
                "Forecasting runs should identify regime changes and structural breaks.",
            ),
            (
                "dependency_framework",
                "Macro designs often carry multicollinearity that needs explicit review.",
            ),
        ],
    },
}
