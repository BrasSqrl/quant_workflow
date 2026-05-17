"""Tests for the Large Data Mode certification harness."""

from __future__ import annotations

import json

import pytest

from quant_pd_framework import ModelType
from quant_pd_framework.large_data_certification import (
    CertificationScenario,
    Thresholds,
    evaluate_benchmark_result,
    expand_scenarios,
    load_suite_config,
    model_target_profile,
    run_certification_suite,
    write_certification_reports,
)


def test_default_acceptance_suite_loads() -> None:
    suite = load_suite_config()

    assert "smoke" in suite["size_tiers"]
    assert "logistic_regression" in suite["small_model_types"]


def test_scenario_expansion_covers_every_model_type() -> None:
    suite = load_suite_config()
    scenarios = expand_scenarios(
        suite,
        preset="smoke",
        model_scope="all",
        models=[],
        source_formats=["parquet"],
        source_kinds=["local"],
        seed=123,
        s3_uri="",
        include_force_overrides=False,
        include_blocked_scenarios=False,
        materialize_projected=False,
        size_overrides={"rows": 20, "features": 3, "sample_rows": 10, "chunk_rows": 10},
    )

    assert {scenario.model_type for scenario in scenarios} == set(ModelType)
    assert all(scenario.expected_capability_status for scenario in scenarios)


def test_s3_scenarios_require_uri() -> None:
    suite = load_suite_config()

    with pytest.raises(ValueError, match="--source-kind s3 requires --s3-uri"):
        expand_scenarios(
            suite,
            preset="smoke",
            model_scope="small",
            models=[],
            source_formats=["parquet"],
            source_kinds=["s3"],
            seed=123,
            s3_uri="",
            include_force_overrides=False,
            include_blocked_scenarios=False,
            materialize_projected=False,
            size_overrides={},
        )


def test_threshold_evaluation_distinguishes_certified_and_fallback() -> None:
    target_mode, target_column = model_target_profile(ModelType.LOGISTIC_REGRESSION)
    scenario = CertificationScenario(
        scenario_id="smoke_logistic_regression_local_parquet",
        model_type=ModelType.LOGISTIC_REGRESSION,
        target_mode=target_mode,
        target_column=target_column,
        source_kind="local",
        source_format="parquet",
        preset="smoke",
        rows=100,
        projected_rows=100,
        features=3,
        sample_rows=50,
        chunk_rows=50,
        seed=42,
        expected_capability_status="full_data_exact",
        expected_certification_status="full_data_certified",
    )
    benchmark = {
        "failure": "",
        "elapsed_seconds": 1,
        "peak_memory_mb": 128,
        "artifact_size_mb": 10,
    }

    result = evaluate_benchmark_result(
        scenario=scenario,
        benchmark=benchmark,
        thresholds=Thresholds(
            max_wall_time_seconds=10,
            max_peak_memory_gb=1,
            max_artifact_size_gb=1,
        ),
    )

    assert result["result_status"] == "certified_pass"
    assert result["passed"] is True


def test_threshold_evaluation_records_expected_block() -> None:
    target_mode, target_column = model_target_profile(ModelType.RANDOM_FOREST)
    scenario = CertificationScenario(
        scenario_id="smoke_random_forest_local_parquet_blocked",
        model_type=ModelType.RANDOM_FOREST,
        target_mode=target_mode,
        target_column=target_column,
        source_kind="local",
        source_format="parquet",
        preset="smoke",
        rows=100,
        projected_rows=100,
        features=3,
        sample_rows=50,
        chunk_rows=50,
        seed=42,
        expected_capability_status="in_memory_only",
        expected_certification_status="blocked",
        certified_only_block=True,
    )

    result = evaluate_benchmark_result(
        scenario=scenario,
        benchmark={
            "failure": "Large-data certified-only policy blocked this model.",
            "elapsed_seconds": 1,
            "peak_memory_mb": 1,
            "artifact_size_mb": 1,
        },
        thresholds=Thresholds(
            max_wall_time_seconds=10,
            max_peak_memory_gb=1,
            max_artifact_size_gb=1,
        ),
    )

    assert result["result_status"] == "blocked_expected"
    assert result["passed"] is True


def test_report_generation_writes_expected_files(tmp_path) -> None:
    scenario_rows = [
        {
            "scenario_id": "smoke_logistic_regression_local_parquet",
            "model_type": "logistic_regression",
            "expected_capability_status": "full_data_exact",
            "expected_certification_status": "full_data_certified",
            "result_status": "certified_pass",
            "passed": True,
            "elapsed_seconds": 1.2,
            "peak_memory_gb": 0.2,
        }
    ]

    write_certification_reports(
        certification_root=tmp_path,
        suite={"description": "test suite"},
        preset="smoke",
        thresholds=Thresholds(
            max_wall_time_seconds=10,
            max_peak_memory_gb=1,
            max_artifact_size_gb=1,
        ),
        environment_profile={"python_version": "test"},
        scenario_rows=scenario_rows,
        run_index=[
            {
                "scenario_id": "smoke_logistic_regression_local_parquet",
                "model_type": "logistic_regression",
                "result_status": "certified_pass",
            }
        ],
        capability_matrix=[
            {
                "model_type": "logistic_regression",
                "fit_capability_status": "full_data_exact",
                "certified": True,
                "large_data_certification_status": "full_data_certified",
            }
        ],
    )

    assert (tmp_path / "certification_summary.json").exists()
    assert (tmp_path / "certification_summary.md").exists()
    assert (tmp_path / "certification_report.html").exists()
    assert (tmp_path / "scenario_results.csv").exists()
    summary = json.loads((tmp_path / "certification_summary.json").read_text())
    assert summary["status_counts"]["certified_pass"] == 1


def test_smoke_certification_cli_path_runs_small_scope(tmp_path) -> None:
    result = run_certification_suite(
        preset="smoke",
        model_scope="model-list",
        models=[ModelType.LOGISTIC_REGRESSION.value],
        source_formats=["parquet"],
        source_kinds=["local"],
        output_root=tmp_path,
        seed=42,
        size_overrides={"rows": 80, "features": 3, "sample_rows": 40, "chunk_rows": 40},
        threshold_overrides={
            "max_wall_time_seconds": 600,
            "max_peak_memory_gb": 4,
            "max_artifact_size_gb": 1,
        },
    )

    certification_root = result["certification_root"]
    assert (certification_root / "certification_summary.json").exists()
    assert (certification_root / "scenario_results.csv").exists()
