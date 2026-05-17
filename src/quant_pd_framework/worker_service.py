"""Local file-backed worker service for queued Quant Studio large-data jobs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from .background_jobs import (
    BACKGROUND_JOB_MANIFEST,
    read_background_manifest,
    run_background_manifest,
)
from .logging import configure_cli_logging, get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run queued Quant Studio large-data jobs.")
    parser.add_argument(
        "--queue-dir",
        default="artifacts/_job_queue",
        help="Directory containing queued job subfolders.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallel workers. Current implementation runs serially.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait between queue scans.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Process available queued jobs once and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_cli_logging()
    args = build_parser().parse_args(argv)
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    if args.workers != 1:
        LOGGER.warning(
            "quant-pd-worker currently processes jobs serially; --workers is recorded only."
        )
    while True:
        _write_heartbeat(queue_dir)
        processed = _process_queued_jobs(queue_dir)
        if args.run_once:
            return 0 if processed >= 0 else 1
        if processed == 0:
            time.sleep(max(0.2, float(args.poll_seconds)))


def _process_queued_jobs(queue_dir: Path) -> int:
    processed = 0
    for manifest_path in sorted(queue_dir.glob(f"*/{BACKGROUND_JOB_MANIFEST}")):
        try:
            manifest = read_background_manifest(manifest_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _quarantine_invalid_manifest(queue_dir, manifest_path, exc)
            continue
        if manifest.status != "queued":
            continue
        run_background_manifest(manifest_path)
        processed += 1
    return processed


def _write_heartbeat(queue_dir: Path) -> None:
    heartbeat_path = queue_dir / "worker_heartbeat.json"
    heartbeat_path.write_text(
        json.dumps({"updated_at_utc": datetime.now(UTC).isoformat()}, indent=2),
        encoding="utf-8",
    )


def _quarantine_invalid_manifest(
    queue_dir: Path,
    manifest_path: Path,
    exc: Exception,
) -> None:
    """Moves malformed queued jobs out of the active queue with an error record."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    failed_dir = queue_dir / "_failed_manifests" / f"{manifest_path.parent.name}_{timestamp}"
    failed_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(manifest_path.parent), str(failed_dir))
    except (OSError, shutil.Error):
        failed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, failed_dir / manifest_path.name)
    failure_record = {
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_path": str(manifest_path),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    (failed_dir / "manifest_parse_failure.json").write_text(
        json.dumps(failure_record, indent=2),
        encoding="utf-8",
    )
    LOGGER.exception("Quarantined invalid worker manifest: %s", manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
