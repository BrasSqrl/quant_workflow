"""Run a repeatable synthetic Large Data Mode benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pd_framework.config import ModelType
from quant_pd_framework.large_data_certification import (
    CertificationScenario,
    default_large_data_performance,
    model_target_profile,
    run_large_data_benchmark,
)
from quant_pd_framework.large_data_policy import resolve_large_data_certification

BENCHMARK_PRESETS = {
    "custom": None,
    "1gb": {"rows": 1_000_000, "features": 20, "sample_rows": 100_000, "chunk_rows": 100_000},
    "5gb": {"rows": 5_000_000, "features": 30, "sample_rows": 250_000, "chunk_rows": 250_000},
    "10gb": {"rows": 10_000_000, "features": 40, "sample_rows": 500_000, "chunk_rows": 500_000},
    "50gb_projected": {
        "rows": 1_000_000,
        "features": 50,
        "sample_rows": 500_000,
        "chunk_rows": 500_000,
        "projected_rows": 50_000_000,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Large Data Mode on synthetic data.")
    parser.add_argument(
        "--preset",
        choices=sorted(BENCHMARK_PRESETS),
        default="custom",
        help=(
            "Named benchmark size preset. `50gb_projected` records projected scale "
            "without writing 50 GB by default."
        ),
    )
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic row count.")
    parser.add_argument("--features", type=int, default=12, help="Numeric feature count.")
    parser.add_argument("--output-root", default="artifacts/benchmarks", help="Benchmark root.")
    parser.add_argument("--sample-rows", type=int, default=25_000, help="Training sample rows.")
    parser.add_argument("--chunk-rows", type=int, default=50_000, help="Scoring chunk rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-type",
        default=ModelType.LOGISTIC_REGRESSION.value,
        choices=[model_type.value for model_type in ModelType],
        help="Model type to benchmark.",
    )
    parser.add_argument(
        "--source-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Synthetic source format.",
    )
    parser.add_argument(
        "--source-kind",
        choices=["local", "data_load"],
        default="local",
        help="Local source behavior to simulate.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(
        preset=args.preset,
        rows=args.rows,
        features=args.features,
        output_root=Path(args.output_root),
        sample_rows=args.sample_rows,
        chunk_rows=args.chunk_rows,
        seed=args.seed,
        model_type=ModelType(args.model_type),
        source_format=args.source_format,
        source_kind=args.source_kind,
    )
    print(json.dumps(payload, indent=2))


def run_benchmark(
    *,
    preset: str = "custom",
    rows: int = 100_000,
    features: int = 12,
    output_root: Path = Path("artifacts/benchmarks"),
    sample_rows: int = 25_000,
    chunk_rows: int = 50_000,
    seed: int = 42,
    model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
    source_format: str = "parquet",
    source_kind: str = "local",
) -> dict[str, object]:
    """Runs one benchmark and returns the structured payload."""

    resolved = _resolve_preset(
        preset=preset,
        rows=rows,
        features=features,
        sample_rows=sample_rows,
        chunk_rows=chunk_rows,
    )
    target_mode, target_column = model_target_profile(model_type)
    certification = resolve_large_data_certification(model_type, default_large_data_performance())
    scenario = CertificationScenario(
        scenario_id=f"{preset}_{model_type.value}_{source_kind}_{source_format}",
        model_type=model_type,
        target_mode=target_mode,
        target_column=target_column,
        source_kind=source_kind,
        source_format=source_format,
        preset=preset,
        rows=int(resolved["rows"]),
        projected_rows=int(resolved["projected_rows"]),
        features=int(resolved["features"]),
        sample_rows=int(resolved["sample_rows"]),
        chunk_rows=int(resolved["chunk_rows"]),
        seed=seed,
        expected_capability_status=certification.fit_capability.status.value,
        expected_certification_status=certification.status.value,
    )
    return run_large_data_benchmark(scenario=scenario, output_root=output_root)


def _resolve_preset(
    *,
    preset: str,
    rows: int,
    features: int,
    sample_rows: int,
    chunk_rows: int,
) -> dict[str, int]:
    preset_values = BENCHMARK_PRESETS.get(preset)
    if not preset_values:
        return {
            "rows": rows,
            "projected_rows": rows,
            "features": features,
            "sample_rows": sample_rows,
            "chunk_rows": chunk_rows,
        }
    return {
        "rows": int(preset_values["rows"]),
        "projected_rows": int(preset_values.get("projected_rows", preset_values["rows"])),
        "features": int(preset_values["features"]),
        "sample_rows": int(preset_values["sample_rows"]),
        "chunk_rows": int(preset_values["chunk_rows"]),
    }


if __name__ == "__main__":
    main()
