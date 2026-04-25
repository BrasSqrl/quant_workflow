"""Profile a Quant Studio saved workflow run.

Example:
    python scripts/profile_workflow.py --config artifacts/run_x/run_config.json \
        --input Data_Load/file.parquet
"""

from __future__ import annotations

import argparse
import json
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

from quant_pd_framework.run import run_saved_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a Quant Studio saved workflow.")
    parser.add_argument("--config", required=True, help="Path to run_config.json.")
    parser.add_argument("--input", help="Optional CSV, Excel, or Parquet input path.")
    parser.add_argument("--output-root", help="Optional output root for profiled artifacts.")
    parser.add_argument(
        "--profile-output",
        default="workflow_profile.json",
        help="Path for the generated profile JSON.",
    )
    parser.add_argument(
        "--top-steps",
        type=int,
        default=10,
        help="Number of slowest pipeline steps to include.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tracemalloc.start()
    started = perf_counter()
    context = run_saved_config(
        config_path=Path(args.config),
        input_path=Path(args.input) if args.input else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    elapsed_seconds = perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profile = build_profile_payload(
        context=context,
        elapsed_seconds=elapsed_seconds,
        current_bytes=current_bytes,
        peak_bytes=peak_bytes,
        top_steps=args.top_steps,
    )
    output_path = Path(args.profile_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(f"Wrote workflow profile to {output_path}")


def build_profile_payload(
    *,
    context: Any,
    elapsed_seconds: float,
    current_bytes: int,
    peak_bytes: int,
    top_steps: int,
) -> dict[str, Any]:
    debug_trace = list(context.debug_trace)
    slowest_steps = sorted(
        debug_trace,
        key=lambda row: float(row.get("elapsed_seconds", 0.0)),
        reverse=True,
    )[:top_steps]
    return {
        "run_id": context.run_id,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "tracemalloc_current_mb": round(current_bytes / 1024 / 1024, 3),
        "tracemalloc_peak_mb": round(peak_bytes / 1024 / 1024, 3),
        "step_count": len(debug_trace),
        "slowest_steps": slowest_steps,
        "artifact_count": len(context.artifacts),
        "diagnostic_table_count": len(context.diagnostics_tables),
        "visualization_count": len(context.visualizations),
        "output_root": str(context.artifacts.get("output_root", "")),
        "run_debug_trace": str(context.artifacts.get("run_debug_trace", "")),
    }


if __name__ == "__main__":
    main()
