"""CLI for executing one checkpointed workflow stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .stage_runner import run_checkpoint_stage


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the stage-runner command line interface."""

    parser = argparse.ArgumentParser(
        description="Run one Quant Studio checkpoint stage from checkpoint_manifest.json."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the run's checkpoints/checkpoint_manifest.json file.",
    )
    parser.add_argument(
        "--stage-id",
        required=True,
        help="Stage identifier to execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return run_checkpoint_stage(Path(args.manifest), args.stage_id)


if __name__ == "__main__":
    raise SystemExit(main())
