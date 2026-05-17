"""CLI entrypoint for a Quant Studio background job manifest."""

from __future__ import annotations

import argparse

from .background_jobs import run_background_manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Quant Studio background job.")
    parser.add_argument("--manifest", required=True, help="Path to job_manifest.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    return run_background_manifest(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
