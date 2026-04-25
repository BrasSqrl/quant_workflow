"""Run a repeatable synthetic Large Data Mode benchmark."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from pathlib import Path
from time import perf_counter

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
    PerformanceConfig,
    QuantModelOrchestrator,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
)
from quant_pd_framework.large_data import build_dataset_handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Large Data Mode on synthetic data.")
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic row count.")
    parser.add_argument("--features", type=int, default=12, help="Numeric feature count.")
    parser.add_argument("--output-root", default="artifacts/benchmarks", help="Benchmark root.")
    parser.add_argument("--sample-rows", type=int, default=25_000, help="Training sample rows.")
    parser.add_argument("--chunk-rows", type=int, default=50_000, help="Scoring chunk rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / f"synthetic_large_data_{args.rows}_{args.features}.parquet"
    _write_synthetic_dataset(
        dataset_path,
        rows=args.rows,
        features=args.features,
        seed=args.seed,
    )
    config = _build_benchmark_config(
        output_root=output_root,
        feature_count=args.features,
        sample_rows=args.sample_rows,
        chunk_rows=args.chunk_rows,
    )

    tracemalloc.start()
    started = perf_counter()
    context = QuantModelOrchestrator(config=config).run(
        build_dataset_handle(dataset_path, {"source_kind": "benchmark"})
    )
    elapsed_seconds = perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "rows": args.rows,
        "features": args.features,
        "sample_rows": args.sample_rows,
        "chunk_rows": args.chunk_rows,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "peak_tracemalloc_mb": round(peak / 1024 / 1024, 3),
        "run_id": context.run_id,
        "output_root": str(context.artifacts.get("output_root", "")),
        "debug_trace": str(context.artifacts.get("run_debug_trace", "")),
    }
    benchmark_path = output_root / f"benchmark_{context.run_id}.json"
    benchmark_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _write_synthetic_dataset(path: Path, *, rows: int, features: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            f"feature_{index}": rng.normal(size=rows)
            for index in range(features)
        }
    )
    logit = frame["feature_0"] * 0.9 - frame["feature_1"] * 0.5 + rng.normal(0, 0.5, rows)
    probability = 1 / (1 + np.exp(-logit))
    frame["default_status"] = (rng.uniform(size=rows) < probability).astype(int)
    frame["account_id"] = np.arange(rows)
    frame["as_of_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        np.arange(rows) % 365,
        unit="D",
    )
    frame.to_parquet(path, index=False)


def _build_benchmark_config(
    *,
    output_root: Path,
    feature_count: int,
    sample_rows: int,
    chunk_rows: int,
) -> FrameworkConfig:
    specs = [
        ColumnSpec(name="account_id", role=ColumnRole.IDENTIFIER),
        ColumnSpec(name="as_of_date", role=ColumnRole.DATE),
        ColumnSpec(name="default_status", role=ColumnRole.TARGET_SOURCE),
        *[
            ColumnSpec(name=f"feature_{index}", role=ColumnRole.FEATURE)
            for index in range(feature_count)
        ],
    ]
    return FrameworkConfig(
        schema=SchemaConfig(column_specs=specs),
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
        diagnostics=DiagnosticConfig(
            interactive_visualizations=False,
            static_image_exports=False,
            export_excel_workbook=False,
            correlation_analysis=False,
            vif_analysis=False,
            woe_iv_analysis=False,
            psi_analysis=False,
            adf_analysis=False,
            model_specification_tests=False,
            forecasting_statistical_tests=False,
        ),
        performance=PerformanceConfig(
            large_data_mode=True,
            optimize_dtypes=True,
            large_data_training_sample_rows=sample_rows,
            large_data_score_chunk_rows=chunk_rows,
            memory_limit_gb=512.0,
        ),
        artifacts=ArtifactConfig(
            output_root=output_root,
            export_code_snapshot=False,
            export_input_snapshot=False,
            export_individual_figure_files=False,
        ),
    )


if __name__ == "__main__":
    main()
