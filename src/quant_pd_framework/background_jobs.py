"""Background execution helpers for large-data Streamlit workflows."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .config import FrameworkConfig
from .config_io import load_framework_config
from .context import PipelineMetadataKey
from .large_data import (
    DatasetHandle,
    build_dataset_handle,
    build_s3_dataset_handle,
    describe_s3_uri,
    is_s3_uri,
)
from .stage_runner import CheckpointedWorkflowRunner

BACKGROUND_JOB_MANIFEST = "job_manifest.json"
BACKGROUND_JOB_SNAPSHOT = "streamlit_snapshot.joblib"
Meta = PipelineMetadataKey


@dataclass(slots=True)
class BackgroundJobManifest:
    """File-backed status record for a detached large-data workflow."""

    job_id: str
    status: str
    job_dir: str
    config_path: str
    input_kind: str
    input_identifier: str
    pid: int | None = None
    run_id: str = ""
    current_stage: str = ""
    progress: str = ""
    checkpoint_manifest_path: str = ""
    snapshot_path: str = ""
    output_root: str = ""
    dispatch_mode: str = "detached_process"
    queue_dir: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    source_profile_key: str = ""
    prepared_manifest_path: str = ""
    memory_trace_path: str = ""
    cancel_requested: bool = False
    error_message: str = ""
    technical_details: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "job_dir": self.job_dir,
            "config_path": self.config_path,
            "input_kind": self.input_kind,
            "input_identifier": self.input_identifier,
            "pid": self.pid,
            "run_id": self.run_id,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "checkpoint_manifest_path": self.checkpoint_manifest_path,
            "snapshot_path": self.snapshot_path,
            "output_root": self.output_root,
            "dispatch_mode": self.dispatch_mode,
            "queue_dir": self.queue_dir,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "source_profile_key": self.source_profile_key,
            "prepared_manifest_path": self.prepared_manifest_path,
            "memory_trace_path": self.memory_trace_path,
            "cancel_requested": self.cancel_requested,
            "error_message": self.error_message,
            "technical_details": self.technical_details,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "completed_at_utc": self.completed_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackgroundJobManifest:
        return cls(
            job_id=str(payload.get("job_id") or ""),
            status=str(payload.get("status") or "unknown"),
            job_dir=str(payload.get("job_dir") or ""),
            config_path=str(payload.get("config_path") or ""),
            input_kind=str(payload.get("input_kind") or ""),
            input_identifier=str(payload.get("input_identifier") or ""),
            pid=payload.get("pid"),
            run_id=str(payload.get("run_id") or ""),
            current_stage=str(payload.get("current_stage") or ""),
            progress=str(payload.get("progress") or ""),
            checkpoint_manifest_path=str(payload.get("checkpoint_manifest_path") or ""),
            snapshot_path=str(payload.get("snapshot_path") or ""),
            output_root=str(payload.get("output_root") or ""),
            dispatch_mode=str(payload.get("dispatch_mode") or "detached_process"),
            queue_dir=str(payload.get("queue_dir") or ""),
            stdout_path=str(payload.get("stdout_path") or ""),
            stderr_path=str(payload.get("stderr_path") or ""),
            source_profile_key=str(payload.get("source_profile_key") or ""),
            prepared_manifest_path=str(payload.get("prepared_manifest_path") or ""),
            memory_trace_path=str(payload.get("memory_trace_path") or ""),
            cancel_requested=bool(payload.get("cancel_requested", False)),
            error_message=str(payload.get("error_message") or ""),
            technical_details=str(payload.get("technical_details") or ""),
            created_at_utc=str(payload.get("created_at_utc") or datetime.now(UTC).isoformat()),
            updated_at_utc=str(payload.get("updated_at_utc") or datetime.now(UTC).isoformat()),
            completed_at_utc=str(payload.get("completed_at_utc") or ""),
        )


def start_background_workflow(
    *,
    config: FrameworkConfig,
    input_data: Any,
) -> Path:
    """Starts a detached workflow process and returns the job manifest path."""

    job_id = datetime.now(UTC).strftime("large_data_job_%Y%m%d_%H%M%S_%f")
    job_dir = config.artifacts.output_root / "_background_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "run_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
    input_kind, input_identifier = _serialize_input(input_data)
    manifest = BackgroundJobManifest(
        job_id=job_id,
        status="starting",
        job_dir=str(job_dir),
        config_path=str(config_path),
        input_kind=input_kind,
        input_identifier=input_identifier,
        output_root=str(config.artifacts.output_root),
        dispatch_mode="detached_process",
    )
    manifest_path = job_dir / BACKGROUND_JOB_MANIFEST
    write_background_manifest(manifest_path, manifest)

    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    command = [
        sys.executable,
        "-m",
        "quant_pd_framework.background_job_runner",
        "--manifest",
        str(manifest_path),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            cwd=str(Path.cwd()),
            creationflags=creationflags,
        )
    manifest.pid = process.pid
    manifest.status = "running"
    manifest.progress = "Background process started."
    manifest.stdout_path = str(stdout_path)
    manifest.stderr_path = str(stderr_path)
    write_background_manifest(manifest_path, manifest)
    return manifest_path


def queue_background_workflow(
    *,
    config: FrameworkConfig,
    input_data: Any,
    queue_dir: str | Path | None = None,
) -> Path:
    """Queues a workflow manifest for a local `quant-pd-worker` service."""

    job_id = datetime.now(UTC).strftime("large_data_job_%Y%m%d_%H%M%S_%f")
    resolved_queue_dir = Path(queue_dir or config.artifacts.output_root / "_job_queue")
    job_dir = resolved_queue_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "run_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
    input_kind, input_identifier = _serialize_input(input_data)
    manifest = BackgroundJobManifest(
        job_id=job_id,
        status="queued",
        job_dir=str(job_dir),
        config_path=str(config_path),
        input_kind=input_kind,
        input_identifier=input_identifier,
        output_root=str(config.artifacts.output_root),
        dispatch_mode="worker_service",
        queue_dir=str(resolved_queue_dir),
        stdout_path=str(job_dir / "stdout.log"),
        stderr_path=str(job_dir / "stderr.log"),
        progress="Queued for local worker service.",
    )
    manifest_path = job_dir / BACKGROUND_JOB_MANIFEST
    write_background_manifest(manifest_path, manifest)
    return manifest_path


def read_background_manifest(manifest_path: str | Path) -> BackgroundJobManifest:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        return BackgroundJobManifest.from_dict(json.load(handle))


def write_background_manifest(
    manifest_path: str | Path,
    manifest: BackgroundJobManifest,
) -> None:
    manifest.updated_at_utc = datetime.now(UTC).isoformat()
    Path(manifest_path).write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )


def request_background_cancel(manifest_path: str | Path) -> None:
    manifest = read_background_manifest(manifest_path)
    manifest.cancel_requested = True
    manifest.progress = "Cancel requested. The worker will stop at the next safe boundary."
    write_background_manifest(manifest_path, manifest)


def load_background_snapshot(manifest: BackgroundJobManifest) -> dict[str, Any] | None:
    if not manifest.snapshot_path:
        return None
    snapshot_path = Path(manifest.snapshot_path)
    if not snapshot_path.exists():
        return None
    snapshot = joblib.load(snapshot_path)
    if not isinstance(snapshot, dict):
        return None
    return snapshot


def run_background_manifest(manifest_path: str | Path) -> int:
    """Worker entrypoint used by `quant_pd_framework.background_job_runner`."""

    manifest_path = Path(manifest_path)
    manifest = read_background_manifest(manifest_path)
    try:
        manifest.status = "running"
        manifest.progress = "Loading configuration."
        write_background_manifest(manifest_path, manifest)
        config = load_framework_config(manifest.config_path)
        input_data = _deserialize_input(manifest.input_kind, manifest.input_identifier)

        def progress_callback(event: dict[str, Any]) -> None:
            active_manifest = read_background_manifest(manifest_path)
            if active_manifest.cancel_requested:
                raise RuntimeError("Background large-data run was cancelled by the user.")
            active_manifest.status = "running"
            active_manifest.run_id = str(event.get("run_id") or active_manifest.run_id)
            active_manifest.current_stage = str(
                event.get("step_name") or event.get("stage_id") or active_manifest.current_stage
            )
            step_order = event.get("step_order")
            total_steps = event.get("total_steps")
            active_manifest.progress = (
                f"{step_order}/{total_steps}" if step_order and total_steps else ""
            )
            checkpoint_manifest = event.get("checkpoint_manifest")
            if checkpoint_manifest:
                active_manifest.checkpoint_manifest_path = str(checkpoint_manifest)
            active_manifest.source_profile_key = str(
                event.get("source_profile_key") or active_manifest.source_profile_key
            )
            active_manifest.prepared_manifest_path = str(
                event.get("prepared_manifest_path") or active_manifest.prepared_manifest_path
            )
            write_background_manifest(manifest_path, active_manifest)

        runner = CheckpointedWorkflowRunner(
            config=config,
            progress_callback=progress_callback,
            use_subprocess=True,
        )
        context = runner.run_all(input_data)
        snapshot_path = Path(manifest.job_dir) / BACKGROUND_JOB_SNAPSHOT
        from quant_pd_framework.streamlit_ui.state import build_run_snapshot

        snapshot = build_run_snapshot(context, config.to_dict())
        joblib.dump(snapshot, snapshot_path, compress=3)
        manifest = read_background_manifest(manifest_path)
        manifest.status = "completed"
        manifest.run_id = context.run_id
        manifest.snapshot_path = str(snapshot_path)
        manifest.output_root = str(
            context.artifacts.get("output_root") or config.artifacts.output_root
        )
        manifest.source_profile_key = str(
            context.get_metadata_dict(Meta.LARGE_DATA_PROFILE).get("profile_cache_key", "")
        )
        manifest.prepared_manifest_path = str(
            context.artifacts.get("prepared_dataset_manifest", "")
        )
        manifest.memory_trace_path = str(context.artifacts.get("run_debug_trace", ""))
        manifest.progress = "Completed."
        manifest.completed_at_utc = datetime.now(UTC).isoformat()
        write_background_manifest(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest = read_background_manifest(manifest_path)
        manifest.status = "failed"
        manifest.error_message = str(exc)
        manifest.technical_details = traceback.format_exc()
        manifest.completed_at_utc = datetime.now(UTC).isoformat()
        write_background_manifest(manifest_path, manifest)
        return 1


def _serialize_input(input_data: Any) -> tuple[str, str]:
    if isinstance(input_data, DatasetHandle):
        return ("s3" if input_data.is_s3 else "dataset_handle", input_data.source_identifier)
    if is_s3_uri(str(input_data)):
        return "s3", str(input_data)
    return "path", str(input_data)


def _deserialize_input(input_kind: str, input_identifier: str) -> Any:
    if input_kind == "s3" or is_s3_uri(input_identifier):
        return build_s3_dataset_handle(input_identifier, describe_s3_uri(input_identifier))
    path = Path(input_identifier)
    metadata = _describe_input_path(path)
    return build_dataset_handle(path, metadata)


def _describe_input_path(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.stat()
        return {
            "source_kind": "background_job_file",
            "display_label": str(path),
            "file_name": path.name,
            "relative_path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": int(stat_result.st_size),
            "modified_ns": int(stat_result.st_mtime_ns),
        }
    except OSError:
        return {
            "source_kind": "background_job_file",
            "display_label": str(path),
            "file_name": path.name,
            "relative_path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": 0,
            "modified_ns": 0,
        }
