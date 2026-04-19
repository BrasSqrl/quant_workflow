"""CLI runner for executing the framework from a saved config bundle."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .config_io import load_framework_config
from .orchestrator import QuantModelOrchestrator


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the command-line interface for saved-run execution."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the quant modeling framework from an exported run_config.json. "
            "If --input is omitted, the runner looks for an input snapshot beside the config."
        )
    )
    parser.add_argument("--config", required=True, help="Path to an exported run_config.json file.")
    parser.add_argument(
        "--input",
        help=(
            "Path to the input CSV or Excel file. If omitted, the runner uses the bundle's "
            "input snapshot when one is available."
        ),
    )
    parser.add_argument(
        "--output-root",
        help="Optional override for the artifact output root used by the rerun.",
    )
    return parser


def run_saved_config(
    config_path: str | Path,
    input_path: str | Path | None = None,
    output_root: str | Path | None = None,
):
    """Loads a saved config, resolves the input source, and executes the orchestrator."""

    config_path = Path(config_path).resolve()
    config = deepcopy(load_framework_config(config_path))
    if output_root is not None:
        config.artifacts.output_root = Path(output_root)

    resolved_input = _resolve_input_path(
        config_path, config.artifacts.input_snapshot_file_name, input_path
    )
    orchestrator = QuantModelOrchestrator(config=config)
    return orchestrator.run(resolved_input)


def _resolve_input_path(
    config_path: Path,
    default_input_name: str,
    input_path: str | Path | None,
) -> Path:
    if input_path is not None:
        return Path(input_path)

    default_input = config_path.parent / default_input_name
    if default_input.exists():
        return default_input

    raise ValueError(
        "No input data was supplied. Pass --input or place the exported "
        "input snapshot beside the config."
    )


def main(argv: list[str] | None = None) -> int:
    """Executes a saved-run pipeline and prints the main artifact location."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    context = run_saved_config(
        config_path=args.config,
        input_path=args.input,
        output_root=args.output_root,
    )
    output_root = context.artifacts.get("output_root")
    if output_root is not None:
        print(f"Run completed. Artifacts written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
