"""Enterprise large-data planning, contracts, and advisory screening helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    LargeDataExecutionStageStatus,
    LargeDataPartitionStrategy,
    LargeDataWorkerMode,
    TargetMode,
    TransformationType,
)
from .context import PipelineContext, PipelineMetadataKey
from .export_layout import build_export_path_layout
from .large_data_policy import LargeDataCertification
from .large_data_runtime import (
    LargeDataExecutionPlan,
    LargeDataExecutionPlanStage,
    LargeDataFeatureScreenManifest,
    LargeDataTransformationContract,
    PartitionedDatasetManifest,
)

Meta = PipelineMetadataKey

COMPILED_TRANSFORMATIONS = {
    TransformationType.WINSORIZE,
    TransformationType.STANDARD_SCALE,
    TransformationType.ROBUST_SCALE,
    TransformationType.MIN_MAX_SCALE,
    TransformationType.EQUAL_WIDTH_BINS,
    TransformationType.QUANTILE_BINS,
    TransformationType.WOE_ENCODING,
    TransformationType.BAD_RATE_ENCODING,
    TransformationType.FREQUENCY_ENCODING,
    TransformationType.DATE_YEAR,
    TransformationType.DATE_MONTH,
    TransformationType.DATE_QUARTER,
    TransformationType.DATE_MONTH_END_FLAG,
    TransformationType.DATE_FISCAL_QUARTER,
    TransformationType.DATE_AGE_DAYS,
    TransformationType.DATE_AGE_MONTHS,
    TransformationType.RATIO,
    TransformationType.SAFE_RATIO,
    TransformationType.MARGIN_RATIO,
    TransformationType.DEBT_SERVICE_RATIO,
    TransformationType.ADD,
    TransformationType.SUBTRACT,
    TransformationType.PRODUCT,
    TransformationType.INTERACTION,
    TransformationType.LAG,
    TransformationType.ROLLING_MEAN,
    TransformationType.ROLLING_MIN,
    TransformationType.ROLLING_MAX,
    TransformationType.ROLLING_SUM,
    TransformationType.ROW_MISSING_COUNT,
    TransformationType.ROW_MISSING_SHARE,
    TransformationType.ANY_MISSING_FLAG,
}

SAMPLE_ONLY_TRANSFORMATIONS = {
    TransformationType.TARGET_ENCODING,
    TransformationType.ORDINAL_ENCODING,
    TransformationType.RARE_CATEGORY_COLLAPSE,
    TransformationType.MONOTONIC_BINS,
    TransformationType.PERCENTILE_RANK,
    TransformationType.NORMAL_SCORE,
    TransformationType.EWMA,
    TransformationType.ROLLING_MEDIAN,
    TransformationType.ROLLING_STD,
    TransformationType.ROLLING_RANGE,
    TransformationType.ROLLING_CV,
    TransformationType.ROLLING_SLOPE,
    TransformationType.EXPANDING_MEAN,
    TransformationType.CUMULATIVE_SUM,
    TransformationType.CUMULATIVE_COUNT,
    TransformationType.CHANGE_FROM_BASELINE,
    TransformationType.PCT_CHANGE,
}


def record_large_data_transformation_contract(context: PipelineContext) -> dict[str, Any]:
    """Writes a contract explaining how enabled transformations run on large files."""

    rows: list[dict[str, Any]] = []
    for spec in context.config.transformations.transformations:
        if not spec.enabled:
            continue
        status, reason, sql_template = _classify_transformation(spec.transform_type)
        rows.append(
            {
                "output_feature": spec.output_feature or _default_output_feature(spec),
                "source_feature": spec.source_feature,
                "secondary_feature": spec.secondary_feature or "",
                "transform_type": spec.transform_type.value,
                "large_data_status": status,
                "reason": reason,
                "sql_template": sql_template,
                "window_size": spec.window_size,
                "lag_periods": spec.lag_periods,
                "parameter_value": spec.parameter_value,
            }
        )
    contract = LargeDataTransformationContract(
        run_id=context.run_id,
        rows=rows,
        supported_count=sum(1 for row in rows if row["large_data_status"] == "compiled"),
        fallback_count=sum(1 for row in rows if row["large_data_status"] == "sample_only"),
        unsupported_count=sum(
            1 for row in rows if row["large_data_status"] == "unsupported"
        ),
        blocked_count=sum(1 for row in rows if row["large_data_status"] == "blocked"),
    )
    payload = contract.to_dict()
    metadata_dir = _large_data_metadata_dir(context)
    path = metadata_dir / "large_data_transformation_contract.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    context.set_metadata(Meta.LARGE_DATA_TRANSFORMATION_CONTRACT, payload)
    context.artifacts["large_data_transformation_contract"] = path
    context.diagnostics_tables["large_data_transformation_contract"] = pd.DataFrame(rows)
    if contract.fallback_count:
        context.warn(
            "Some configured transformations are sample-only in Large Data Mode. "
            "Review `large_data_transformation_contract` before relying on full-file replay."
        )
    return payload


def record_large_data_execution_plan(
    context: PipelineContext,
    certification: LargeDataCertification,
    *,
    source_identifier: str,
) -> dict[str, Any]:
    """Writes the per-stage large-data execution plan used by readiness and exports."""

    profile = context.get_metadata_dict(Meta.LARGE_DATA_PROFILE)
    contract = context.get_metadata_dict(Meta.LARGE_DATA_TRANSFORMATION_CONTRACT)
    rows = contract.get("rows", []) if isinstance(contract, dict) else []
    blocked_transforms = [
        row for row in rows if isinstance(row, dict) and row.get("large_data_status") == "blocked"
    ]
    fallback_transforms = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("large_data_status") in {"sample_only", "unsupported"}
    ]
    status = certification.status.value
    fit_status = certification.fit_capability.status.value
    model_stage_status = (
        LargeDataExecutionStageStatus.FULL_DATA
        if status == "full_data_certified"
        else (
            LargeDataExecutionStageStatus.REQUIRES_OVERRIDE
            if status == "experimental_full_data_override"
            else (
                LargeDataExecutionStageStatus.BLOCKED
                if status == "blocked"
                else LargeDataExecutionStageStatus.SAMPLE_BASED
            )
        )
    )
    stages = [
        LargeDataExecutionPlanStage(
            stage_name="source_profile",
            status=LargeDataExecutionStageStatus.FILE_BACKED,
            basis="profile_cache" if profile.get("profile_cache_hit") else "fresh_profile",
            description=(
                "Schema, preview, and source metadata are read without loading the "
                "full file into Streamlit."
            ),
            details={
                "row_count": profile.get("row_count"),
                "column_count": profile.get("column_count"),
                "cache_key": profile.get("profile_cache_key"),
                "cache_hit": bool(profile.get("profile_cache_hit")),
            },
        ),
        LargeDataExecutionPlanStage(
            stage_name="parquet_staging",
            status=LargeDataExecutionStageStatus.FILE_BACKED,
            basis="projected_parquet",
            description=(
                "The source is staged or reused as Parquet and projected to required "
                "columns where possible."
            ),
            details=context.get_metadata_dict(Meta.LARGE_DATA_PROJECTED_DATASET),
        ),
        LargeDataExecutionPlanStage(
            stage_name="transformation_replay",
            status=(
                LargeDataExecutionStageStatus.BLOCKED
                if blocked_transforms
                else (
                    LargeDataExecutionStageStatus.SAMPLE_BASED
                    if fallback_transforms
                    else LargeDataExecutionStageStatus.FILE_BACKED
                )
            ),
            basis="duckdb_contract",
            description=(
                "Supported transformations are documented as SQL/aggregate replay; "
                "unsupported items use governed-sample fallback."
            ),
            blocking=bool(blocked_transforms),
            details={
                "compiled_count": contract.get("supported_count", 0)
                if isinstance(contract, dict)
                else 0,
                "sample_only_count": len(fallback_transforms),
                "unsupported_count": contract.get("unsupported_count", 0)
                if isinstance(contract, dict)
                else 0,
                "blocked_count": len(blocked_transforms),
            },
        ),
        LargeDataExecutionPlanStage(
            stage_name="feature_prescreening",
            status=LargeDataExecutionStageStatus.SAMPLE_BASED,
            basis="governed_sample",
            description=(
                "Feature pre-screening is advisory by default and only auto-excludes "
                "features when explicitly enabled."
            ),
            details={
                "enabled": bool(context.config.performance.large_data_prescreen_enabled),
                "auto_apply": bool(context.config.performance.large_data_auto_apply_prescreen),
            },
        ),
        LargeDataExecutionPlanStage(
            stage_name="model_fit",
            status=model_stage_status,
            basis=certification.execution_strategy,
            description=certification.fit_capability.explanation,
            blocking=status == "blocked",
            details=certification.to_metadata(),
        ),
        LargeDataExecutionPlanStage(
            stage_name="full_file_scoring",
            status=LargeDataExecutionStageStatus.FILE_BACKED,
            basis="chunked_batches",
            description="Full-file predictions are written chunk-by-chunk from the staged source.",
            details={
                "chunk_rows": context.config.performance.large_data_score_chunk_rows,
                "result_page_rows": context.config.performance.large_data_result_page_rows,
            },
        ),
        LargeDataExecutionPlanStage(
            stage_name="exports",
            status=LargeDataExecutionStageStatus.FILE_BACKED,
            basis=context.config.artifacts.large_data_export_policy.value,
            description=(
                "Large tabular artifacts follow the configured full, sampled, or "
                "metadata-only policy."
            ),
            details={
                "tabular_output_format": context.config.artifacts.tabular_output_format.value,
                "large_data_export_policy": context.config.artifacts.large_data_export_policy.value,
            },
        ),
    ]
    worker_mode = context.config.performance.large_data_worker_mode
    if not isinstance(worker_mode, LargeDataWorkerMode):
        worker_mode = LargeDataWorkerMode(worker_mode)
    plan = LargeDataExecutionPlan(
        run_id=context.run_id,
        source_identifier=source_identifier,
        model_type=context.config.model.model_type.value,
        backend=context.config.performance.large_data_backend.value,
        policy=context.config.performance.large_data_model_policy.value,
        certification_status=status,
        fit_capability_status=fit_status,
        profile_cache_key=str(profile.get("profile_cache_key") or ""),
        worker_mode=worker_mode.value,
        stages=stages,
    )
    payload = plan.to_dict()
    metadata_dir = _large_data_metadata_dir(context)
    path = metadata_dir / "execution_plan.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    context.set_metadata(Meta.LARGE_DATA_EXECUTION_PLAN, payload)
    context.artifacts["large_data_execution_plan"] = path
    context.diagnostics_tables["large_data_execution_plan"] = pd.DataFrame(
        [stage.to_dict() for stage in stages]
    )
    return payload


def record_partitioned_sample_manifest(context: PipelineContext) -> None:
    """Writes split partition Parquet files for the governed development sample."""

    if not context.config.performance.large_data_mode or not context.split_frames:
        return
    strategy = context.config.performance.large_data_partition_strategy
    if not isinstance(strategy, LargeDataPartitionStrategy):
        strategy = LargeDataPartitionStrategy(strategy)
    if strategy == LargeDataPartitionStrategy.NONE:
        return

    output_root = context.config.artifacts.output_root / context.run_id
    layout = build_export_path_layout(context.config.artifacts, output_root)
    base_dir = layout.data_dir / "sample_development" / "partitioned"
    base_dir.mkdir(parents=True, exist_ok=True)
    partition_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for split_name, frame in context.split_frames.items():
        split_dir = base_dir / f"split={split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_path = split_dir / "sample.parquet"
        frame.to_parquet(
            split_path,
            index=False,
            compression=context.config.artifacts.parquet_compression,
        )
        partition_paths[split_name] = str(split_path)
        row_counts[split_name] = int(len(frame))

    manifest = PartitionedDatasetManifest(
        run_id=context.run_id,
        dataset_role="governed_sample_development",
        partition_strategy=(
            "split"
            if strategy in {LargeDataPartitionStrategy.AUTO, LargeDataPartitionStrategy.SPLIT}
            else strategy.value
        ),
        partition_columns=["split"],
        partition_paths=partition_paths,
        row_counts=row_counts,
        base_path=str(base_dir),
    )
    metadata_dir = _large_data_metadata_dir(context)
    path = metadata_dir / "partitioned_dataset_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    context.set_metadata(Meta.PARTITIONED_DATASET_MANIFEST, manifest.to_dict())
    context.artifacts["partitioned_dataset_manifest"] = path
    context.artifacts["partitioned_sample_development_dir"] = base_dir
    context.diagnostics_tables["partitioned_dataset_manifest"] = pd.DataFrame(
        [manifest.to_dict()]
    )


def record_large_data_feature_screening(context: PipelineContext) -> None:
    """Computes advisory feature screening from the governed train sample."""

    performance = context.config.performance
    if not performance.large_data_mode or not performance.large_data_prescreen_enabled:
        return
    if context.get_metadata(Meta.LARGE_DATA_FEATURE_SCREENING_RECORDED):
        return
    train_frame = context.split_frames.get("train")
    if train_frame is None or train_frame.empty or not context.feature_columns:
        return
    target_column = context.target_column
    if not target_column or target_column not in train_frame.columns:
        return

    y = train_frame[target_column]
    rows: list[dict[str, Any]] = []
    for feature_name in context.feature_columns:
        if feature_name not in train_frame.columns:
            continue
        series = train_frame[feature_name]
        row = _feature_screen_row(context, feature_name, series, y)
        rows.append(row)
    if not rows:
        return

    table = pd.DataFrame(rows).sort_values(
        ["recommended_action", "missing_rate", "feature_name"],
        ascending=[True, False, True],
        kind="stable",
    )
    output_root = context.config.artifacts.output_root / context.run_id
    layout = build_export_path_layout(context.config.artifacts, output_root)
    table_dir = layout.tables_dir / "feature_screening"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "large_data_feature_screening.csv"
    parquet_path = table_dir / "large_data_feature_screening.parquet"
    table.to_csv(table_path, index=False)
    try:
        table.to_parquet(
            parquet_path,
            index=False,
            compression=context.config.artifacts.parquet_compression,
        )
    except Exception:
        parquet_path = Path("")

    excluded = table.loc[
        table["recommended_action"].eq("exclude"),
        "feature_name",
    ].astype(str).tolist()
    auto_apply = bool(performance.large_data_auto_apply_prescreen)
    manifest = LargeDataFeatureScreenManifest(
        run_id=context.run_id,
        basis="governed_train_sample",
        row_count=int(len(train_frame)),
        feature_count=int(len(table)),
        auto_apply=auto_apply,
        excluded_features=excluded if auto_apply else [],
        table_path=str(table_path),
        parquet_path=str(parquet_path) if str(parquet_path) else "",
    )
    metadata_dir = _large_data_metadata_dir(context)
    manifest_path = metadata_dir / "feature_screening_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    context.diagnostics_tables["large_data_feature_screening"] = table
    context.set_metadata(Meta.LARGE_DATA_FEATURE_SCREENING_MANIFEST, manifest.to_dict())
    context.set_metadata(Meta.LARGE_DATA_FEATURE_SCREENING_RECORDED, True)
    context.artifacts["large_data_feature_screening"] = table_path
    if str(parquet_path):
        context.artifacts["large_data_feature_screening_parquet"] = parquet_path
    context.artifacts["large_data_feature_screening_manifest"] = manifest_path

    if auto_apply and excluded:
        retained = [feature for feature in context.feature_columns if feature not in set(excluded)]
        if not retained:
            context.warn(
                "Large-data pre-screening recommended excluding all features; recommendations "
                "were recorded but not auto-applied."
            )
            return
        _apply_feature_exclusions(context, excluded)
        context.warn(
            "Large-data pre-screening auto-excluded "
            f"{len(excluded)} feature(s): {', '.join(excluded[:10])}."
        )


def _feature_screen_row(
    context: PipelineContext,
    feature_name: str,
    series: pd.Series,
    y: pd.Series,
) -> dict[str, Any]:
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    missing_rate = float(series.isna().mean())
    unique_ratio = unique_count / max(1, int(series.notna().sum()))
    numeric = pd.to_numeric(series, errors="coerce")
    is_numeric = bool(pd.api.types.is_numeric_dtype(series) or numeric.notna().mean() > 0.9)
    variance = float(numeric.var()) if is_numeric and numeric.notna().any() else None
    low_variance = bool((variance is not None and variance <= 1e-12) or unique_count <= 1)
    high_cardinality = bool(
        not is_numeric
        and (
            unique_count > context.config.performance.max_categorical_cardinality
            or unique_ratio > context.config.performance.max_categorical_cardinality_ratio
        )
    )
    target_relationship = _target_relationship(
        numeric if is_numeric else series,
        y,
        context.config.target.mode,
        is_numeric=is_numeric,
    )
    iv_value = (
        _information_value(series, y)
        if context.config.target.mode == TargetMode.BINARY and y.nunique(dropna=True) == 2
        else None
    )
    psi_value = _split_psi(context, feature_name)
    reasons: list[str] = []
    if missing_rate >= 0.95:
        reasons.append("very_high_missingness")
    if low_variance:
        reasons.append("low_variance")
    if high_cardinality:
        reasons.append("high_cardinality_categorical")
    action = "exclude" if reasons else "retain"
    return {
        "feature_name": feature_name,
        "feature_type": "numeric" if is_numeric else "categorical",
        "row_count": int(len(series)),
        "missing_rate": missing_rate,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "variance": variance,
        "low_variance": low_variance,
        "high_cardinality": high_cardinality,
        "target_relationship": target_relationship,
        "information_value": iv_value,
        "psi_vs_train": psi_value,
        "recommended_action": action,
        "recommendation_reason": ",".join(reasons) if reasons else "screen_passed",
    }


def _target_relationship(
    series: pd.Series,
    y: pd.Series,
    target_mode: TargetMode,
    *,
    is_numeric: bool,
) -> float | None:
    try:
        target = pd.to_numeric(y, errors="coerce")
        if target_mode == TargetMode.BINARY and target.nunique(dropna=True) == 2:
            if is_numeric:
                predictor = pd.to_numeric(series, errors="coerce").fillna(series.median())
            else:
                bucket = series.astype("object").fillna("Missing").astype(str)
                predictor = bucket.map(target.groupby(bucket).mean()).fillna(target.mean())
            auc = roc_auc_score(target.astype(int), predictor.to_numpy(dtype=float))
            return float(abs(auc - 0.5) * 2.0)
        predictor = pd.to_numeric(series, errors="coerce")
        corr = predictor.corr(target)
        return None if pd.isna(corr) else float(abs(corr))
    except Exception:
        return None


def _information_value(series: pd.Series, y: pd.Series) -> float | None:
    try:
        target = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        if pd.api.types.is_numeric_dtype(series):
            bucket = pd.qcut(pd.to_numeric(series, errors="coerce"), q=10, duplicates="drop")
        else:
            bucket = series.astype("object").fillna("Missing").astype(str)
            top_values = bucket.value_counts().head(50).index
            bucket = bucket.where(bucket.isin(top_values), "Other")
        grouped = pd.DataFrame({"bucket": bucket, "target": target}).groupby(
            "bucket",
            observed=False,
        )
        bad = grouped["target"].sum().astype(float) + 0.5
        total = grouped["target"].count().astype(float) + 1.0
        good = total - bad
        bad_dist = bad / bad.sum()
        good_dist = good / good.sum()
        iv = ((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum()
        return float(iv)
    except Exception:
        return None


def _split_psi(context: PipelineContext, feature_name: str) -> float | None:
    train = context.split_frames.get("train")
    if train is None or feature_name not in train.columns:
        return None
    challengers = [
        frame[feature_name]
        for split_name, frame in context.split_frames.items()
        if split_name != "train" and feature_name in frame.columns and not frame.empty
    ]
    if not challengers:
        return None
    compare = pd.concat(challengers, ignore_index=True)
    return _psi(train[feature_name], compare)


def _psi(expected: pd.Series, actual: pd.Series) -> float | None:
    try:
        expected_numeric = pd.to_numeric(expected, errors="coerce")
        actual_numeric = pd.to_numeric(actual, errors="coerce")
        if expected_numeric.notna().mean() > 0.9 and actual_numeric.notna().mean() > 0.9:
            edges = np.unique(np.nanquantile(expected_numeric.dropna(), np.linspace(0, 1, 11)))
            if len(edges) < 3:
                return None
            expected_bins = pd.cut(expected_numeric, bins=edges, include_lowest=True)
            actual_bins = pd.cut(actual_numeric, bins=edges, include_lowest=True)
        else:
            expected_bins = expected.astype("object").fillna("Missing").astype(str)
            actual_bins = actual.astype("object").fillna("Missing").astype(str)
        expected_dist = expected_bins.value_counts(normalize=True, dropna=False)
        actual_dist = actual_bins.value_counts(normalize=True, dropna=False)
        all_bins = expected_dist.index.union(actual_dist.index)
        expected_values = expected_dist.reindex(all_bins, fill_value=0.0001).clip(lower=0.0001)
        actual_values = actual_dist.reindex(all_bins, fill_value=0.0001).clip(lower=0.0001)
        return float(
            ((actual_values - expected_values) * np.log(actual_values / expected_values)).sum()
        )
    except Exception:
        return None


def _apply_feature_exclusions(context: PipelineContext, excluded: list[str]) -> None:
    excluded_set = set(excluded)
    context.feature_columns = [
        feature for feature in context.feature_columns if feature not in excluded_set
    ]
    context.numeric_features = [
        feature for feature in context.numeric_features if feature not in excluded_set
    ]
    context.categorical_features = [
        feature for feature in context.categorical_features if feature not in excluded_set
    ]
    context.set_metadata(Meta.LARGE_DATA_PRESCREEN_EXCLUDED_FEATURES, sorted(excluded_set))


def _classify_transformation(transform_type: TransformationType) -> tuple[str, str, str]:
    if transform_type in COMPILED_TRANSFORMATIONS:
        return (
            "compiled",
            "Can be represented as DuckDB SQL or aggregate-fit-plus-SQL replay.",
            _sql_template(transform_type),
        )
    if transform_type in SAMPLE_ONLY_TRANSFORMATIONS:
        return (
            "sample_only",
            "Requires ordered/sample-fitted logic that remains governed-sample in this phase.",
            "",
        )
    return (
        "unsupported",
        "Available in normal mode, but not certified for disk-backed large-data replay.",
        "",
    )


def _sql_template(transform_type: TransformationType) -> str:
    templates = {
        TransformationType.SAFE_RATIO: (
            "CASE WHEN denominator IS NULL OR denominator = 0 "
            "THEN NULL ELSE numerator / denominator END"
        ),
        TransformationType.RATIO: "numerator / denominator",
        TransformationType.ADD: "feature_a + feature_b",
        TransformationType.SUBTRACT: "feature_a - feature_b",
        TransformationType.PRODUCT: "feature_a * feature_b",
        TransformationType.INTERACTION: "feature_a * feature_b",
        TransformationType.DATE_YEAR: "EXTRACT(year FROM date_column)",
        TransformationType.DATE_MONTH: "EXTRACT(month FROM date_column)",
        TransformationType.DATE_QUARTER: "EXTRACT(quarter FROM date_column)",
        TransformationType.FREQUENCY_ENCODING: "LEFT JOIN fitted_frequency_table",
        TransformationType.WOE_ENCODING: "LEFT JOIN fitted_woe_table",
        TransformationType.BAD_RATE_ENCODING: "LEFT JOIN fitted_bad_rate_table",
    }
    return templates.get(transform_type, "aggregate_fit_then_sql_replay")


def _default_output_feature(spec: Any) -> str:
    return f"{spec.source_feature}_{spec.transform_type.value}"


def _large_data_metadata_dir(context: PipelineContext) -> Path:
    output_root = context.config.artifacts.output_root / context.run_id
    metadata_dir = build_export_path_layout(
        context.config.artifacts,
        output_root,
    ).metadata_dir / "large_data"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    context.artifacts["large_data_metadata_dir"] = metadata_dir
    return metadata_dir
