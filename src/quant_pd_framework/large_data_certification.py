"""CLI-first certification harness for Large Data Mode acceptance evidence."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    CreditRiskDiagnosticConfig,
    DataStructure,
    DiagnosticConfig,
    DocumentationConfig,
    ExportProfile,
    FeatureEngineeringConfig,
    FrameworkConfig,
    LargeDataBackend,
    LargeDataExportPolicy,
    LargeDataModelPolicy,
    ModelConfig,
    ModelType,
    PerformanceConfig,
    RegulatoryReportConfig,
    SchemaConfig,
    SplitConfig,
    TabularOutputFormat,
    TargetConfig,
    TargetMode,
)
from .large_data import (
    build_dataset_handle,
    build_s3_dataset_handle,
    describe_s3_uri,
    is_s3_uri,
)
from .large_data_policy import (
    resolve_large_data_certification,
    resolve_large_data_fit_capability,
)
from .logging import configure_cli_logging
from .orchestrator import QuantModelOrchestrator

CERTIFICATION_STATUSES = (
    "certified_pass",
    "certified_fail",
    "fallback_pass",
    "fallback_fail",
    "blocked_expected",
    "blocked_unexpected",
    "override_recorded",
    "benchmark_error",
)

DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "large_data_certification"
    / "default_acceptance_suite.json"
)

BINARY_MODELS = frozenset(
    {
        ModelType.LOGISTIC_REGRESSION,
        ModelType.DISCRETE_TIME_HAZARD_MODEL,
        ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
        ModelType.SCORECARD_LOGISTIC_REGRESSION,
        ModelType.PROBIT_REGRESSION,
        ModelType.GEE_LOGISTIC_REGRESSION,
        ModelType.GAM_SPLINE_LOGISTIC,
        ModelType.RANDOM_FOREST,
        ModelType.EXTRA_TREES,
        ModelType.EXPLAINABLE_BOOSTING_MACHINE,
        ModelType.XGBOOST,
    }
)
MULTICLASS_MODELS = frozenset(
    {
        ModelType.MULTINOMIAL_LOGISTIC_REGRESSION,
        ModelType.ORDINAL_LOGISTIC_REGRESSION,
        ModelType.DECISION_TREE,
    }
)
COUNT_MODELS = frozenset(
    {
        ModelType.POISSON_REGRESSION,
        ModelType.NEGATIVE_BINOMIAL_REGRESSION,
    }
)
POSITIVE_CONTINUOUS_MODELS = frozenset(
    {
        ModelType.GAMMA_REGRESSION,
        ModelType.TWEEDIE_REGRESSION,
    }
)
BOUNDED_CONTINUOUS_MODELS = frozenset(
    {
        ModelType.BETA_REGRESSION,
        ModelType.FRACTIONAL_LOGIT,
        ModelType.ZERO_ONE_INFLATED_BETA,
        ModelType.TWO_STAGE_LGD_MODEL,
        ModelType.TOBIT_REGRESSION,
    }
)
SURVIVAL_MODELS = frozenset(
    {
        ModelType.COX_PROPORTIONAL_HAZARDS,
        ModelType.AFT_SURVIVAL_MODEL,
    }
)
FORECAST_MODELS = frozenset(
    {
        ModelType.SARIMAX_FORECAST,
        ModelType.EXPONENTIAL_SMOOTHING_FORECAST,
        ModelType.UNOBSERVED_COMPONENTS_FORECAST,
    }
)


@dataclass(frozen=True, slots=True)
class CertificationScenario:
    """One model/source/format certification benchmark scenario."""

    scenario_id: str
    model_type: ModelType
    target_mode: TargetMode
    target_column: str
    source_kind: str
    source_format: str
    preset: str
    rows: int
    projected_rows: int
    features: int
    sample_rows: int
    chunk_rows: int
    seed: int
    expected_capability_status: str
    expected_certification_status: str
    force_override: bool = False
    certified_only_block: bool = False
    s3_uri: str = ""

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_type"] = self.model_type.value
        payload["target_mode"] = self.target_mode.value
        return payload


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Acceptance thresholds applied to benchmark payloads."""

    max_wall_time_seconds: float
    max_peak_memory_gb: float
    max_artifact_size_gb: float
    required_completion_status: str = "completed"
    expected_scoring_behavior: str = "policy_consistent"
    expected_export_policy_behavior: str = "compact_large_data_exports"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main(argv: list[str] | None = None) -> int:
    """Runs the large-data certification CLI."""

    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_certification_suite(
        suite_path=args.config,
        preset=args.preset,
        model_scope=args.model_scope,
        models=_split_csv_values(args.models),
        source_formats=args.source_format,
        source_kinds=args.source_kind,
        output_root=Path(args.output_root),
        seed=args.seed,
        s3_uri=args.s3_uri or "",
        include_force_overrides=args.include_force_overrides,
        include_blocked_scenarios=args.include_blocked_scenarios,
        materialize_projected=args.materialize_50gb,
        threshold_overrides={
            "max_wall_time_seconds": args.max_wall_time_seconds,
            "max_peak_memory_gb": args.max_peak_memory_gb,
            "max_artifact_size_gb": args.max_artifact_size_gb,
        },
        size_overrides={
            "rows": args.rows,
            "features": args.features,
            "sample_rows": args.sample_rows,
            "chunk_rows": args.chunk_rows,
        },
    )
    sys.stdout.write(
        json.dumps({"certification_root": str(result["certification_root"])}, indent=2)
        + "\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line interface for acceptance certification."""

    parser = argparse.ArgumentParser(
        description="Run Quant Studio Large Data Mode certification scenarios."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SUITE_PATH),
        help="Acceptance suite JSON file.",
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        help="Suite size tier, such as smoke, 1gb, 5gb, 10gb, or 50gb_projected.",
    )
    parser.add_argument(
        "--model-scope",
        choices=["small", "certified", "fallback", "all", "model-list"],
        default="small",
        help="Which model families to certify.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model_type values. Requires --model-scope model-list.",
    )
    parser.add_argument(
        "--source-format",
        action="append",
        choices=["parquet", "csv"],
        help="Input format to generate. Can be repeated.",
    )
    parser.add_argument(
        "--source-kind",
        action="append",
        choices=["local", "data_load", "s3"],
        help="Input source kind to simulate. Can be repeated.",
    )
    parser.add_argument("--s3-uri", default="", help="Optional s3:// input URI.")
    parser.add_argument(
        "--output-root",
        default="artifacts/certification",
        help="Certification output root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Synthetic data seed.")
    parser.add_argument("--rows", type=int, help="Override suite row count.")
    parser.add_argument("--features", type=int, help="Override suite feature count.")
    parser.add_argument("--sample-rows", type=int, help="Override governed sample rows.")
    parser.add_argument("--chunk-rows", type=int, help="Override full-score chunk rows.")
    parser.add_argument("--max-wall-time-seconds", type=float, help="Override wall-time limit.")
    parser.add_argument("--max-peak-memory-gb", type=float, help="Override peak-memory limit.")
    parser.add_argument("--max-artifact-size-gb", type=float, help="Override artifact-size limit.")
    parser.add_argument(
        "--include-force-overrides",
        action="store_true",
        help="Add experimental forced override scenarios for uncertified model families.",
    )
    parser.add_argument(
        "--include-blocked-scenarios",
        action="store_true",
        help="Add certified-only blocked scenarios for uncertified model families.",
    )
    parser.add_argument(
        "--materialize-50gb",
        action="store_true",
        help="Materialize the 50gb_projected tier instead of using projected row evidence.",
    )
    return parser


def run_certification_suite(
    *,
    suite_path: str | Path | None = None,
    preset: str = "smoke",
    model_scope: str = "small",
    models: list[str] | None = None,
    source_formats: list[str] | None = None,
    source_kinds: list[str] | None = None,
    output_root: Path = Path("artifacts/certification"),
    seed: int = 42,
    s3_uri: str = "",
    include_force_overrides: bool = False,
    include_blocked_scenarios: bool = False,
    materialize_projected: bool = False,
    threshold_overrides: dict[str, Any] | None = None,
    size_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expands, runs, evaluates, and reports a certification suite."""

    suite = load_suite_config(suite_path)
    thresholds = resolve_thresholds(suite, preset, threshold_overrides or {})
    scenarios = expand_scenarios(
        suite,
        preset=preset,
        model_scope=model_scope,
        models=models or [],
        source_formats=source_formats,
        source_kinds=source_kinds,
        seed=seed,
        s3_uri=s3_uri,
        include_force_overrides=include_force_overrides,
        include_blocked_scenarios=include_blocked_scenarios,
        materialize_projected=materialize_projected,
        size_overrides=size_overrides or {},
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    certification_root = output_root / timestamp
    certification_root.mkdir(parents=True, exist_ok=True)

    environment_profile = build_environment_profile()
    rows: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []
    for scenario in scenarios:
        benchmark = run_large_data_benchmark(
            scenario=scenario,
            output_root=certification_root / "benchmark_runs",
        )
        evaluation = evaluate_benchmark_result(
            scenario=scenario,
            benchmark=benchmark,
            thresholds=thresholds,
        )
        rows.append({**scenario.to_metadata(), **benchmark, **evaluation})
        run_index.append(
            {
                "scenario_id": scenario.scenario_id,
                "model_type": scenario.model_type.value,
                "benchmark_json": benchmark.get("benchmark_json", ""),
                "benchmark_markdown": benchmark.get("benchmark_markdown", ""),
                "artifact_output_root": benchmark.get("output_root", ""),
                "result_status": evaluation["result_status"],
            }
        )

    matrix = build_model_capability_matrix()
    write_certification_reports(
        certification_root=certification_root,
        suite=suite,
        preset=preset,
        thresholds=thresholds,
        environment_profile=environment_profile,
        scenario_rows=rows,
        run_index=run_index,
        capability_matrix=matrix,
    )
    return {
        "certification_root": certification_root,
        "scenario_count": len(rows),
        "scenarios": rows,
    }


def load_suite_config(path: str | Path | None = None) -> dict[str, Any]:
    """Loads a certification suite config."""

    suite_path = Path(path or DEFAULT_SUITE_PATH)
    if not suite_path.exists():
        return _fallback_suite_config()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if "size_tiers" not in payload:
        raise ValueError("Certification suite config must include size_tiers.")
    return payload


def resolve_thresholds(
    suite: dict[str, Any],
    preset: str,
    overrides: dict[str, Any],
) -> Thresholds:
    """Resolves acceptance thresholds for a preset."""

    tier = _get_size_tier(suite, preset)
    threshold_payload = dict(tier.get("thresholds", {}))
    for key, value in overrides.items():
        if value is not None:
            threshold_payload[key] = value
    return Thresholds(
        max_wall_time_seconds=float(threshold_payload.get("max_wall_time_seconds", 600)),
        max_peak_memory_gb=float(threshold_payload.get("max_peak_memory_gb", 8)),
        max_artifact_size_gb=float(threshold_payload.get("max_artifact_size_gb", 2)),
        required_completion_status=str(
            threshold_payload.get("required_completion_status", "completed")
        ),
        expected_scoring_behavior=str(
            threshold_payload.get("expected_scoring_behavior", "policy_consistent")
        ),
        expected_export_policy_behavior=str(
            threshold_payload.get("expected_export_policy_behavior", "compact_large_data_exports")
        ),
    )


def expand_scenarios(
    suite: dict[str, Any],
    *,
    preset: str,
    model_scope: str,
    models: list[str],
    source_formats: list[str] | None,
    source_kinds: list[str] | None,
    seed: int,
    s3_uri: str,
    include_force_overrides: bool,
    include_blocked_scenarios: bool,
    materialize_projected: bool,
    size_overrides: dict[str, Any],
) -> list[CertificationScenario]:
    """Builds all certification scenarios requested by the suite and CLI."""

    tier = _get_size_tier(suite, preset)
    rows = int(size_overrides.get("rows") or tier.get("rows", 1_000))
    projected_rows = int(tier.get("projected_rows", rows))
    if preset == "50gb_projected" and materialize_projected:
        rows = projected_rows
    features = int(size_overrides.get("features") or tier.get("features", 6))
    sample_rows = int(size_overrides.get("sample_rows") or tier.get("sample_rows", rows))
    chunk_rows = int(size_overrides.get("chunk_rows") or tier.get("chunk_rows", rows))
    formats = source_formats or list(suite.get("default_source_formats", ["parquet"]))
    kinds = source_kinds or list(suite.get("default_source_kinds", ["local"]))
    if "s3" in kinds and not s3_uri:
        raise ValueError("--source-kind s3 requires --s3-uri.")
    model_types = _resolve_model_scope(suite, model_scope, models)

    scenarios: list[CertificationScenario] = []
    for model_type in model_types:
        target_mode, target_column = model_target_profile(model_type)
        certification = resolve_large_data_certification(model_type, _large_data_performance())
        for source_kind in kinds:
            for source_format in formats:
                scenario_id = _scenario_id(
                    preset=preset,
                    model_type=model_type,
                    source_kind=source_kind,
                    source_format=source_format,
                    force_override=False,
                )
                scenarios.append(
                    CertificationScenario(
                        scenario_id=scenario_id,
                        model_type=model_type,
                        target_mode=target_mode,
                        target_column=target_column,
                        source_kind=source_kind,
                        source_format=source_format,
                        preset=preset,
                        rows=rows,
                        projected_rows=projected_rows,
                        features=features,
                        sample_rows=sample_rows,
                        chunk_rows=chunk_rows,
                        seed=seed,
                        expected_capability_status=certification.fit_capability.status.value,
                        expected_certification_status=certification.status.value,
                        s3_uri=s3_uri if source_kind == "s3" else "",
                    )
                )
                if include_blocked_scenarios and not certification.fit_capability.certified:
                    blocked_cert = resolve_large_data_certification(
                        model_type,
                        _large_data_performance(
                            model_policy=LargeDataModelPolicy.CERTIFIED_ONLY,
                        ),
                    )
                    scenarios.append(
                        CertificationScenario(
                            scenario_id=_scenario_id(
                                preset=preset,
                                model_type=model_type,
                                source_kind=source_kind,
                                source_format=source_format,
                                force_override=False,
                                blocked=True,
                            ),
                            model_type=model_type,
                            target_mode=target_mode,
                            target_column=target_column,
                            source_kind=source_kind,
                            source_format=source_format,
                            preset=preset,
                            rows=rows,
                            projected_rows=projected_rows,
                            features=features,
                            sample_rows=sample_rows,
                            chunk_rows=chunk_rows,
                            seed=seed,
                            expected_capability_status=blocked_cert.fit_capability.status.value,
                            expected_certification_status=blocked_cert.status.value,
                            certified_only_block=True,
                            s3_uri=s3_uri if source_kind == "s3" else "",
                        )
                    )
                if include_force_overrides and not certification.fit_capability.certified:
                    override_cert = resolve_large_data_certification(
                        model_type,
                        _large_data_performance(
                            model_policy=LargeDataModelPolicy.FORCE_FULL_DATA_OVERRIDE,
                            override_confirmed=True,
                            override_reason="Certification harness forced override evidence.",
                        ),
                    )
                    scenarios.append(
                        CertificationScenario(
                            scenario_id=_scenario_id(
                                preset=preset,
                                model_type=model_type,
                                source_kind=source_kind,
                                source_format=source_format,
                                force_override=True,
                                blocked=False,
                            ),
                            model_type=model_type,
                            target_mode=target_mode,
                            target_column=target_column,
                            source_kind=source_kind,
                            source_format=source_format,
                            preset=preset,
                            rows=rows,
                            projected_rows=projected_rows,
                            features=features,
                            sample_rows=sample_rows,
                            chunk_rows=chunk_rows,
                            seed=seed,
                            expected_capability_status=override_cert.fit_capability.status.value,
                            expected_certification_status=override_cert.status.value,
                            force_override=True,
                            s3_uri=s3_uri if source_kind == "s3" else "",
                        )
                    )
    return scenarios


def run_large_data_benchmark(
    *,
    scenario: CertificationScenario,
    output_root: Path,
) -> dict[str, Any]:
    """Runs one synthetic large-data workflow and returns structured evidence."""

    output_root.mkdir(parents=True, exist_ok=True)
    scenario_root = output_root / scenario.scenario_id
    scenario_root.mkdir(parents=True, exist_ok=True)
    phase_timings: dict[str, float] = {}

    dataset_path = Path("")
    if scenario.source_kind == "s3":
        if not is_s3_uri(scenario.s3_uri):
            raise ValueError("S3 certification scenarios require a valid s3:// URI.")
        handle = build_s3_dataset_handle(scenario.s3_uri, describe_s3_uri(scenario.s3_uri))
        dataset_size_mb = _safe_size_mb(handle.metadata.get("size_bytes", 0))
    else:
        directory_name = "Data_Load" if scenario.source_kind == "data_load" else "input"
        dataset_dir = scenario_root / directory_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / f"{scenario.scenario_id}.{scenario.source_format}"
        phase_start = perf_counter()
        write_synthetic_dataset(
            dataset_path,
            rows=scenario.rows,
            features=scenario.features,
            model_type=scenario.model_type,
            seed=scenario.seed,
        )
        phase_timings["dataset_write_seconds"] = round(perf_counter() - phase_start, 6)
        stat_result = dataset_path.stat()
        dataset_size_mb = round(stat_result.st_size / 1024 / 1024, 3)
        handle = build_dataset_handle(
            dataset_path,
            {
                "source_kind": "data_load_simulation"
                if scenario.source_kind == "data_load"
                else "certification_local",
                "size_bytes": stat_result.st_size,
                "modified_ns": stat_result.st_mtime_ns,
                "suffix": dataset_path.suffix.lower(),
                "preset": scenario.preset,
                "scenario_id": scenario.scenario_id,
            },
        )

    config = build_benchmark_config(
        output_root=scenario_root / "artifacts",
        scenario=scenario,
    )

    tracemalloc.start()
    started = perf_counter()
    failure = ""
    failure_point = ""
    context = None
    try:
        context = QuantModelOrchestrator(config=config).run(handle)
    except Exception as exc:  # noqa: BLE001 - benchmark evidence must capture all failures.
        failure = str(exc)
        failure_point = "workflow"
    elapsed_seconds = perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    phase_timings["workflow_seconds"] = round(elapsed_seconds, 6)

    artifact_output_root = (
        Path(str(context.artifacts.get("output_root")))
        if context is not None and context.artifacts.get("output_root")
        else scenario_root / "artifacts"
    )
    debug_trace = (
        str(context.artifacts.get("run_debug_trace", ""))
        if context is not None and context.artifacts.get("run_debug_trace")
        else ""
    )
    peak_memory_mb = max(
        round(peak / 1024 / 1024, 3),
        _peak_memory_from_debug_trace(Path(debug_trace)) if debug_trace else 0.0,
    )
    payload: dict[str, Any] = {
        "benchmark_preset": scenario.preset,
        "scenario_id": scenario.scenario_id,
        "rows": scenario.rows,
        "projected_rows": scenario.projected_rows,
        "features": scenario.features,
        "sample_rows": scenario.sample_rows,
        "chunk_rows": scenario.chunk_rows,
        "source_kind": scenario.source_kind,
        "source_format": scenario.source_format,
        "dataset_path": str(dataset_path) if dataset_path else scenario.s3_uri,
        "dataset_size_mb": dataset_size_mb,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "peak_tracemalloc_mb": round(peak / 1024 / 1024, 3),
        "peak_memory_mb": peak_memory_mb,
        "artifact_size_mb": _directory_size_mb(artifact_output_root),
        "phase_timings": phase_timings,
        "model_family": config.model.model_type.value,
        "target_mode": config.target.mode.value,
        "large_data_backend": config.performance.large_data_backend.value,
        "large_data_model_policy": config.performance.large_data_model_policy.value,
        "profile_cache_enabled": config.performance.large_data_profile_cache_enabled,
        "failure": failure,
        "failure_point": failure_point,
        "run_id": context.run_id if context is not None else "",
        "output_root": str(artifact_output_root),
        "debug_trace": debug_trace,
    }
    run_label = payload["run_id"] or f"failed_{scenario.scenario_id}"
    benchmark_path = scenario_root / f"benchmark_{run_label}.json"
    markdown_path = scenario_root / f"benchmark_{run_label}.md"
    benchmark_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(benchmark_markdown(payload), encoding="utf-8")
    payload["benchmark_json"] = str(benchmark_path)
    payload["benchmark_markdown"] = str(markdown_path)
    return payload


def evaluate_benchmark_result(
    *,
    scenario: CertificationScenario,
    benchmark: dict[str, Any],
    thresholds: Thresholds,
) -> dict[str, Any]:
    """Classifies benchmark evidence against expected capability and thresholds."""

    failure = str(benchmark.get("failure", "") or "")
    elapsed = float(benchmark.get("elapsed_seconds", 0.0) or 0.0)
    peak_gb = float(benchmark.get("peak_memory_mb", 0.0) or 0.0) / 1024
    artifact_gb = float(benchmark.get("artifact_size_mb", 0.0) or 0.0) / 1024
    within_thresholds = (
        not failure
        and elapsed <= thresholds.max_wall_time_seconds
        and peak_gb <= thresholds.max_peak_memory_gb
        and artifact_gb <= thresholds.max_artifact_size_gb
    )
    threshold_failures = []
    if elapsed > thresholds.max_wall_time_seconds:
        threshold_failures.append("max_wall_time_seconds")
    if peak_gb > thresholds.max_peak_memory_gb:
        threshold_failures.append("max_peak_memory_gb")
    if artifact_gb > thresholds.max_artifact_size_gb:
        threshold_failures.append("max_artifact_size_gb")

    expected_status = scenario.expected_certification_status
    if scenario.force_override:
        result_status = "override_recorded"
    elif failure:
        if expected_status == "blocked" and _looks_like_expected_block(failure):
            result_status = "blocked_expected"
        elif expected_status == "blocked":
            result_status = "blocked_unexpected"
        elif expected_status == "full_data_certified":
            result_status = "certified_fail"
        elif expected_status == "sample_fit_full_score":
            result_status = "fallback_fail"
        else:
            result_status = "benchmark_error"
    elif expected_status == "full_data_certified":
        result_status = "certified_pass" if within_thresholds else "certified_fail"
    elif expected_status == "sample_fit_full_score":
        result_status = "fallback_pass" if within_thresholds else "fallback_fail"
    elif expected_status == "experimental_full_data_override":
        result_status = "override_recorded"
    elif expected_status == "blocked":
        result_status = "blocked_unexpected"
    else:
        result_status = "benchmark_error"

    return {
        "result_status": result_status,
        "passed": result_status in {
            "certified_pass",
            "fallback_pass",
            "blocked_expected",
            "override_recorded",
        },
        "threshold_failures": ";".join(threshold_failures),
        "peak_memory_gb": round(peak_gb, 6),
        "artifact_size_gb": round(artifact_gb, 6),
    }


def build_benchmark_config(
    *,
    output_root: Path,
    scenario: CertificationScenario,
) -> FrameworkConfig:
    """Builds a low-overhead framework config for one certification scenario."""

    specs = _schema_specs(scenario)
    model_config = model_config_for_certification(
        scenario.model_type,
        force_override=scenario.force_override,
    )
    data_structure = (
        DataStructure.TIME_SERIES
        if scenario.model_type in FORECAST_MODELS
        else DataStructure.PANEL
        if scenario.model_type == ModelType.PANEL_REGRESSION
        else DataStructure.CROSS_SECTIONAL
    )
    split = SplitConfig(
        data_structure=data_structure,
        date_column="as_of_date",
        entity_column="account_id" if data_structure == DataStructure.PANEL else None,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        stratify=scenario.target_mode == TargetMode.BINARY,
    )
    return FrameworkConfig(
        schema=SchemaConfig(column_specs=specs),
        cleaning=CleaningConfig(),
        feature_engineering=FeatureEngineeringConfig(
            derive_date_parts=False,
            drop_raw_date_columns=True,
        ),
        target=TargetConfig(
            source_column=scenario.target_column,
            mode=scenario.target_mode,
            output_column=_target_output_column(scenario.target_mode),
            positive_values=[1] if scenario.target_mode == TargetMode.BINARY else None,
        ),
        split=split,
        model=model_config,
        diagnostics=DiagnosticConfig(
            data_quality=False,
            descriptive_statistics=False,
            missingness_analysis=False,
            correlation_analysis=False,
            vif_analysis=False,
            woe_iv_analysis=False,
            psi_analysis=False,
            adf_analysis=False,
            model_specification_tests=False,
            forecasting_statistical_tests=False,
            calibration_analysis=False,
            threshold_analysis=False,
            lift_gain_analysis=False,
            segment_analysis=False,
            residual_analysis=False,
            quantile_analysis=False,
            qq_analysis=False,
            interactive_visualizations=False,
            static_image_exports=False,
            export_excel_workbook=False,
            max_plot_rows=1_000,
        ),
        credit_risk=CreditRiskDiagnosticConfig(
            enabled=False,
            vintage_analysis=False,
            migration_analysis=False,
            delinquency_transition_analysis=False,
            cohort_pd_analysis=False,
            lgd_segment_analysis=False,
            recovery_analysis=False,
            macro_sensitivity_analysis=False,
        ),
        documentation=DocumentationConfig(enabled=False),
        regulatory_reporting=RegulatoryReportConfig(enabled=False),
        performance=_large_data_performance(
            model_policy=LargeDataModelPolicy.FORCE_FULL_DATA_OVERRIDE
            if scenario.force_override
            else LargeDataModelPolicy.CERTIFIED_ONLY
            if scenario.certified_only_block
            else LargeDataModelPolicy.ALLOW_SAMPLE_FALLBACK,
            override_confirmed=scenario.force_override,
            override_reason="Certification harness forced override evidence."
            if scenario.force_override
            else "",
            sample_rows=scenario.sample_rows,
            chunk_rows=scenario.chunk_rows,
        ),
        artifacts=ArtifactConfig(
            output_root=output_root,
            export_code_snapshot=False,
            export_input_snapshot=False,
            export_individual_figure_files=False,
            include_enhanced_report_visuals=False,
            include_advanced_visual_analytics=False,
            keep_all_checkpoints=False,
            compact_prediction_exports=True,
            export_profile=ExportProfile.FAST,
            tabular_output_format=TabularOutputFormat.PARQUET,
            large_data_export_policy=LargeDataExportPolicy.SAMPLED,
            large_data_sample_rows=min(max(100, scenario.sample_rows), 50_000),
        ),
    )


def write_synthetic_dataset(
    path: Path,
    *,
    rows: int,
    features: int,
    model_type: ModelType,
    seed: int,
) -> None:
    """Writes a target-compatible synthetic dataset for certification."""

    frame = build_synthetic_certification_frame(
        rows=rows,
        features=features,
        seed=seed,
        model_type=model_type,
    )
    _target_mode, target_column = model_target_profile(model_type)
    selected_columns = [
        *(f"feature_{index}" for index in range(max(features, 3))),
        target_column,
        "account_id",
        "as_of_date",
    ]
    if model_type == ModelType.MIXED_EFFECTS_REGRESSION:
        selected_columns.append("segment")
    frame = frame.loc[:, selected_columns]
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def build_synthetic_certification_frame(
    *,
    rows: int,
    features: int,
    seed: int,
    model_type: ModelType,
) -> pd.DataFrame:
    """Builds one synthetic frame that carries all target variants."""

    rng = np.random.default_rng(seed)
    feature_count = max(features, 3)
    feature_data = {f"feature_{index}": rng.normal(size=rows) for index in range(feature_count)}
    frame = pd.DataFrame(feature_data)
    signal = (
        frame["feature_0"] * 0.9
        - frame["feature_1"] * 0.45
        + frame["feature_2"] * 0.25
        + rng.normal(0, 0.35, rows)
    )
    probability = 1 / (1 + np.exp(-signal))
    frame["target_binary"] = (rng.uniform(size=rows) < probability).astype(int)
    frame["target_class"] = pd.qcut(
        pd.Series(signal).rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)
    frame["target_continuous"] = 10 + signal + rng.normal(0, 0.5, rows)
    fraction = np.clip(probability + rng.normal(0, 0.03, rows), 1e-4, 1 - 1e-4)
    if model_type == ModelType.ZERO_ONE_INFLATED_BETA and rows >= 10:
        fraction[: max(1, rows // 20)] = 0.0
        fraction[max(1, rows // 20) : max(2, rows // 10)] = 1.0
    frame["target_fraction"] = fraction
    frame["target_count"] = rng.poisson(np.clip(np.exp(0.3 + probability), 0.1, 8.0), size=rows)
    frame["target_positive"] = np.exp(0.2 + probability + rng.normal(0, 0.08, rows))
    frame["target_survival"] = np.clip(12 + signal + rng.normal(0, 1.0, rows), 0.5, None)
    frame["account_id"] = [f"A{index % max(10, min(rows, 200)):05d}" for index in range(rows)]
    frame["segment"] = np.array(["retail", "small_business", "middle_market", "commercial"])[
        np.arange(rows) % 4
    ]
    frame["as_of_date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        np.arange(rows),
        unit="D",
    )
    return frame


def model_target_profile(model_type: ModelType) -> tuple[TargetMode, str]:
    """Returns the target mode and source column for a model family."""

    if model_type in BINARY_MODELS:
        return TargetMode.BINARY, "target_binary"
    if model_type in MULTICLASS_MODELS:
        return TargetMode.MULTICLASS, "target_class"
    if model_type in COUNT_MODELS:
        return TargetMode.CONTINUOUS, "target_count"
    if model_type in POSITIVE_CONTINUOUS_MODELS:
        return TargetMode.CONTINUOUS, "target_positive"
    if model_type in BOUNDED_CONTINUOUS_MODELS:
        return TargetMode.CONTINUOUS, "target_fraction"
    if model_type in SURVIVAL_MODELS:
        return TargetMode.CONTINUOUS, "target_survival"
    return TargetMode.CONTINUOUS, "target_continuous"


def model_config_for_certification(
    model_type: ModelType,
    *,
    force_override: bool = False,
) -> ModelConfig:
    """Returns a fast, target-compatible model config for certification runs."""

    common = {
        "model_type": model_type,
        "max_iter": 100,
        "tree_n_estimators": 20,
        "tree_max_depth": 3,
        "xgboost_n_estimators": 20,
        "xgboost_max_depth": 3,
        "n_jobs": 0,
    }
    if model_type == ModelType.ELASTIC_NET_LOGISTIC_REGRESSION:
        common.update({"l1_ratio": 0.35, "max_iter": 250})
    elif model_type == ModelType.SCORECARD_LOGISTIC_REGRESSION:
        common.update({"scorecard_bins": 4, "max_iter": 250})
    elif model_type == ModelType.GEE_LOGISTIC_REGRESSION:
        common.update({"gee_group_column": "account_id", "max_iter": 75})
    elif model_type == ModelType.MIXED_EFFECTS_REGRESSION:
        common.update({"mixed_effects_group_column": "segment", "max_iter": 75})
    elif model_type in {ModelType.GAM_SPLINE_REGRESSION, ModelType.GAM_SPLINE_LOGISTIC}:
        common.update({"spline_n_knots": 4, "spline_degree": 2})
    elif model_type == ModelType.TOBIT_REGRESSION:
        common.update({"tobit_left_censoring": 0.0, "tobit_right_censoring": 1.0})
    elif model_type == ModelType.QUANTILE_REGRESSION:
        common.update({"quantile_alpha": 0.65})
    elif model_type in {ModelType.LASSO_REGRESSION, ModelType.ELASTIC_NET_REGRESSION}:
        common.update({"regularization_alpha": 0.01, "l1_ratio": 0.35, "max_iter": 250})
    elif model_type == ModelType.RIDGE_REGRESSION:
        common.update({"regularization_alpha": 0.5})
    elif model_type in {ModelType.SARIMAX_FORECAST, ModelType.UNOBSERVED_COMPONENTS_FORECAST}:
        common.update({"max_iter": 50})
    if force_override:
        common["max_iter"] = min(int(common["max_iter"]), 75)
    return ModelConfig(**common)


def build_model_capability_matrix() -> list[dict[str, Any]]:
    """Builds static model capability evidence for certification reports."""

    rows: list[dict[str, Any]] = []
    performance = _large_data_performance()
    for model_type in ModelType:
        certification = resolve_large_data_certification(model_type, performance)
        capability = resolve_large_data_fit_capability(model_type)
        rows.append(
            {
                "model_type": model_type.value,
                "fit_capability_status": capability.status.value,
                "certified": capability.certified,
                "large_data_certification_status": certification.status.value,
                "fit_strategy": capability.fit_strategy,
                "scoring_strategy": capability.scoring_strategy,
                "explanation": capability.explanation,
                "recommendation": certification.recommendation,
            }
        )
    return rows


def write_certification_reports(
    *,
    certification_root: Path,
    suite: dict[str, Any],
    preset: str,
    thresholds: Thresholds,
    environment_profile: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    run_index: list[dict[str, Any]],
    capability_matrix: list[dict[str, Any]],
) -> None:
    """Writes JSON, Markdown, HTML, and CSV certification outputs."""

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "preset": preset,
        "thresholds": thresholds.to_dict(),
        "scenario_count": len(scenario_rows),
        "status_counts": _status_counts(scenario_rows),
        "passed": all(bool(row.get("passed")) for row in scenario_rows) if scenario_rows else False,
        "certification_statuses": list(CERTIFICATION_STATUSES),
    }
    (certification_root / "certification_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (certification_root / "effective_thresholds.json").write_text(
        json.dumps(thresholds.to_dict(), indent=2),
        encoding="utf-8",
    )
    (certification_root / "environment_profile.json").write_text(
        json.dumps(environment_profile, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(certification_root / "scenario_results.csv", scenario_rows)
    _write_csv(certification_root / "run_index.csv", run_index)
    _write_csv(certification_root / "model_capability_matrix.csv", capability_matrix)
    markdown = certification_summary_markdown(
        summary=summary,
        suite=suite,
        scenario_rows=scenario_rows,
        capability_matrix=capability_matrix,
    )
    (certification_root / "certification_summary.md").write_text(markdown, encoding="utf-8")
    (certification_root / "certification_report.html").write_text(
        certification_report_html(markdown, scenario_rows, capability_matrix),
        encoding="utf-8",
    )


def certification_summary_markdown(
    *,
    summary: dict[str, Any],
    suite: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    capability_matrix: list[dict[str, Any]],
) -> str:
    """Builds the Markdown certification summary."""

    lines = [
        "# Large Data Certification Summary",
        "",
        f"- Created UTC: `{summary['created_at_utc']}`",
        f"- Preset: `{summary['preset']}`",
        f"- Scenario count: `{summary['scenario_count']}`",
        f"- Overall passed: `{summary['passed']}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Acceptance Thresholds",
            "",
        ]
    )
    for key, value in summary["thresholds"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Scenario Results", ""])
    if scenario_rows:
        lines.append("| Scenario | Model | Capability | Expected | Result | Elapsed | Peak GB |")
        lines.append("|---|---|---|---|---|---:|---:|")
        for row in scenario_rows:
            lines.append(
                "| "
                f"`{row.get('scenario_id', '')}` | "
                f"`{row.get('model_type', '')}` | "
                f"`{row.get('expected_capability_status', '')}` | "
                f"`{row.get('expected_certification_status', '')}` | "
                f"`{row.get('result_status', '')}` | "
                f"{row.get('elapsed_seconds', '')} | "
                f"{row.get('peak_memory_gb', '')} |"
            )
    lines.extend(["", "## Model Capability Matrix", ""])
    lines.append("| Model | Fit Capability | Certified | Certification Status |")
    lines.append("|---|---|---:|---|")
    for row in capability_matrix:
        lines.append(
            "| "
            f"`{row['model_type']}` | "
            f"`{row['fit_capability_status']}` | "
            f"`{row['certified']}` | "
            f"`{row['large_data_certification_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Suite Notes",
            "",
            str(suite.get("description", "No suite description supplied.")),
        ]
    )
    return "\n".join(lines) + "\n"


def certification_report_html(
    markdown: str,
    scenario_rows: list[dict[str, Any]],
    capability_matrix: list[dict[str, Any]],
) -> str:
    """Builds a simple self-contained HTML report without extra dependencies."""

    status_cards = "".join(
        f"<div class='card'><span>{_html(row.get('result_status', ''))}</span>"
        f"<strong>{_html(row.get('model_type', ''))}</strong>"
        f"<small>{_html(row.get('scenario_id', ''))}</small></div>"
        for row in scenario_rows
    )
    scenario_table = _html_table(scenario_rows)
    capability_table = _html_table(capability_matrix)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Large Data Certification Report</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f3f7fb;
      color: #17223b;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px; }}
    h1, h2 {{ color: #10213f; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin: 20px 0;
    }}
    .card {{
      background: white;
      border: 1px solid #dbe5f2;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 12px 28px rgba(16, 33, 63, 0.08);
    }}
    .card span {{
      display: block;
      color: #1f6feb;
      font-weight: 800;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: .05em;
    }}
    .card strong {{ display: block; margin-top: 8px; }}
    .card small {{ color: #60708a; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 14px;
      overflow: hidden;
      margin: 16px 0 32px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e6edf7;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{ background: #eaf2ff; color: #173866; }}
    pre {{
      white-space: pre-wrap;
      background: white;
      border: 1px solid #dbe5f2;
      border-radius: 16px;
      padding: 18px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Large Data Certification Report</h1>
  <section class="grid">{status_cards}</section>
  <h2>Scenario Results</h2>
  {scenario_table}
  <h2>Model Capability Matrix</h2>
  {capability_table}
  <h2>Markdown Summary</h2>
  <pre>{_html(markdown)}</pre>
</main>
</body>
</html>
"""


def build_environment_profile() -> dict[str, Any]:
    """Captures standard-library environment evidence for certification."""

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_memory_gb": _total_memory_gb(),
        "cwd": str(Path.cwd()),
    }


def benchmark_markdown(payload: dict[str, Any]) -> str:
    """Builds Markdown for one benchmark payload."""

    lines = [
        "# Large Data Benchmark",
        "",
        f"- Scenario: `{payload['scenario_id']}`",
        f"- Preset: `{payload['benchmark_preset']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Projected rows: `{payload['projected_rows']}`",
        f"- Features: `{payload['features']}`",
        f"- Source: `{payload['source_kind']} / {payload['source_format']}`",
        f"- Dataset size MB: `{payload['dataset_size_mb']}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']}`",
        f"- Peak memory MB: `{payload['peak_memory_mb']}`",
        f"- Artifact size MB: `{payload['artifact_size_mb']}`",
        f"- Model family: `{payload['model_family']}`",
        (
            f"- Backend policy: `{payload['large_data_backend']} / "
            f"{payload['large_data_model_policy']}`"
        ),
    ]
    if payload.get("failure"):
        lines.append(f"- Failure point: `{payload['failure_point']}`")
        lines.append(f"- Failure: `{payload['failure']}`")
    lines.extend(["", "## Phase Timings", ""])
    timings = payload.get("phase_timings", {})
    if isinstance(timings, dict):
        for key, value in timings.items():
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _schema_specs(scenario: CertificationScenario) -> list[ColumnSpec]:
    specs = [
        ColumnSpec(name="account_id", role=ColumnRole.IDENTIFIER),
        ColumnSpec(name="as_of_date", role=ColumnRole.DATE),
        ColumnSpec(name=scenario.target_column, role=ColumnRole.TARGET_SOURCE),
    ]
    if scenario.model_type == ModelType.MIXED_EFFECTS_REGRESSION:
        specs.append(ColumnSpec(name="segment", role=ColumnRole.IGNORE))
    specs.extend(
        ColumnSpec(name=f"feature_{index}", role=ColumnRole.FEATURE)
        for index in range(max(scenario.features, 3))
    )
    return specs


def _target_output_column(target_mode: TargetMode) -> str:
    if target_mode == TargetMode.BINARY:
        return "default_flag"
    if target_mode == TargetMode.MULTICLASS:
        return "target_class_prepared"
    return "target_value"


def _large_data_performance(
    *,
    model_policy: LargeDataModelPolicy = LargeDataModelPolicy.ALLOW_SAMPLE_FALLBACK,
    override_confirmed: bool = False,
    override_reason: str = "",
    sample_rows: int = 25_000,
    chunk_rows: int = 50_000,
) -> PerformanceConfig:
    return PerformanceConfig(
        large_data_mode=True,
        large_data_backend=LargeDataBackend.AUTO,
        large_data_model_policy=model_policy,
        large_data_override_confirmed=override_confirmed,
        large_data_override_reason=override_reason,
        optimize_dtypes=True,
        capture_memory_profile=True,
        deep_memory_profile=False,
        retain_full_working_data=False,
        convert_csv_to_parquet=True,
        large_data_training_sample_rows=sample_rows,
        large_data_score_chunk_rows=chunk_rows,
        large_data_max_in_memory_rows=max(1_000, sample_rows),
        large_data_profile_cache_enabled=True,
        large_data_prescreen_enabled=True,
        large_data_auto_apply_prescreen=False,
        large_data_certified_fit_enabled=True,
        memory_limit_gb=512.0,
    )


def default_large_data_performance() -> PerformanceConfig:
    """Returns the default Large Data Mode performance policy for certification."""

    return _large_data_performance()


def _resolve_model_scope(
    suite: dict[str, Any],
    model_scope: str,
    models: list[str],
) -> list[ModelType]:
    if model_scope == "model-list":
        if not models:
            raise ValueError("--model-scope model-list requires --models.")
        return [ModelType(value) for value in models]
    if model_scope == "small":
        return [ModelType(value) for value in suite.get("small_model_types", [])]
    if model_scope == "certified":
        return [
            model_type
            for model_type in ModelType
            if resolve_large_data_fit_capability(model_type).certified
        ]
    if model_scope == "fallback":
        return [
            model_type
            for model_type in ModelType
            if not resolve_large_data_fit_capability(model_type).certified
        ]
    if model_scope == "all":
        return list(ModelType)
    raise ValueError(f"Unsupported model scope: {model_scope}")


def _get_size_tier(suite: dict[str, Any], preset: str) -> dict[str, Any]:
    tiers = suite.get("size_tiers", {})
    if preset not in tiers:
        raise ValueError(f"Unknown certification preset '{preset}'.")
    return dict(tiers[preset])


def _scenario_id(
    *,
    preset: str,
    model_type: ModelType,
    source_kind: str,
    source_format: str,
    force_override: bool,
    blocked: bool = False,
) -> str:
    suffix = "_override" if force_override else "_blocked" if blocked else ""
    return f"{preset}_{model_type.value}_{source_kind}_{source_format}{suffix}"


def _fallback_suite_config() -> dict[str, Any]:
    return {
        "description": "Fallback Quant Studio Large Data Mode acceptance suite.",
        "default_source_formats": ["parquet"],
        "default_source_kinds": ["local"],
        "small_model_types": ["logistic_regression", "linear_regression"],
        "size_tiers": {
            "smoke": {
                "rows": 200,
                "features": 4,
                "sample_rows": 100,
                "chunk_rows": 100,
                "thresholds": {
                    "max_wall_time_seconds": 600,
                    "max_peak_memory_gb": 4,
                    "max_artifact_size_gb": 1,
                },
            }
        },
    }


def _split_csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _directory_size_mb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return round(total / 1024 / 1024, 3)


def _safe_size_mb(value: Any) -> float:
    try:
        return round(float(value) / 1024 / 1024, 3)
    except (TypeError, ValueError):
        return 0.0


def _peak_memory_from_debug_trace(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    values: list[float] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = str(key).lower()
                if key_lower.endswith("_memory_bytes") or key_lower in {
                    "raw_memory_bytes",
                    "working_memory_bytes",
                    "split_memory_bytes",
                    "prediction_memory_bytes",
                    "diagnostics_table_memory_bytes",
                    "tracked_dataframe_memory_bytes",
                }:
                    try:
                        values.append(float(value) / 1024 / 1024)
                    except (TypeError, ValueError):
                        visit(value)
                elif key_lower in {
                    "peak_memory_mb",
                    "peak_rss_mb",
                    "rss_mb",
                    "process_memory_mb",
                    "resident_memory_mb",
                }:
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        visit(value)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(payload)
    if not values:
        return 0.0
    return max(values)


def _looks_like_expected_block(failure: str) -> bool:
    lowered = failure.lower()
    return any(
        marker in lowered
        for marker in [
            "blocked",
            "override",
            "certified",
            "large-data",
            "large data",
        ]
    )


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in CERTIFICATION_STATUSES}
    for row in rows:
        status = str(row.get("result_status", "benchmark_error"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str)
    return value


def _html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = sorted({key for row in rows for key in row})
    head = "".join(f"<th>{_html(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            + "".join(f"<td>{_html(row.get(column, ''))}</td>" for column in columns)
            + "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _html(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _total_memory_gb() -> float | None:
    if platform.system().lower() == "windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        try:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except Exception:  # noqa: BLE001 - environment evidence is best effort.
            return None
        return round(status.ullTotalPhys / 1024**3, 3)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return round(float(pages * page_size) / 1024**3, 3)
