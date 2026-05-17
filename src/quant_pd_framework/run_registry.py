"""File-backed run registry and audit event helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from .logging import get_logger

RUN_REGISTRY_SCHEMA_VERSION = "1.0"
AUDIT_EVENT_SCHEMA_VERSION = "1.0"
RUN_REGISTRY_DIRECTORY_NAME = "_run_registry"
RUN_REGISTRY_FILE_NAME = "run_registry.json"
AUDIT_EVENT_FILE_NAME = "audit_events.jsonl"
PER_RUN_AUDIT_EVENT_FILE_NAME = "audit_events.jsonl"
SENSITIVE_KEY_FRAGMENTS = (
    "access_key",
    "api_key",
    "credential",
    "password",
    "secret",
    "session_token",
    "token",
)
LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RunRegistryEntry:
    """Compact index record for one workflow run."""

    run_id: str
    status: str
    artifact_root: str
    started_at_utc: str = ""
    completed_at_utc: str = ""
    elapsed_seconds: float | None = None
    execution_mode: str = ""
    model_type: str = ""
    target_mode: str = ""
    dataset_source_label: str = ""
    dataset_source_kind: str = ""
    large_data_mode: bool = False
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    warning_count: int = 0
    reviewer_status: str = "Not reviewed"
    reviewer_name: str = ""
    review_updated_at_utc: str = ""
    error_message: str = ""
    updated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return json_safe(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRegistryEntry:
        fields = cls.__dataclass_fields__
        kwargs = {name: payload.get(name) for name in fields}
        kwargs["metrics_summary"] = dict(kwargs.get("metrics_summary") or {})
        kwargs["artifact_paths"] = {
            str(key): str(value)
            for key, value in dict(kwargs.get("artifact_paths") or {}).items()
            if value is not None
        }
        kwargs["large_data_mode"] = bool(kwargs.get("large_data_mode"))
        kwargs["warning_count"] = int(kwargs.get("warning_count") or 0)
        kwargs["elapsed_seconds"] = _optional_float(kwargs.get("elapsed_seconds"))
        for name in fields:
            if kwargs.get(name) is None and name not in {
                "elapsed_seconds",
                "large_data_mode",
                "warning_count",
                "metrics_summary",
                "artifact_paths",
            }:
                kwargs[name] = ""
        return cls(**kwargs)


@dataclass(slots=True)
class AuditEvent:
    """Append-only audit event written to JSONL."""

    event_id: str
    timestamp_utc: str
    event_type: str
    source: str
    session_id: str = ""
    run_id: str = ""
    actor: str = ""
    artifact_root: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = AUDIT_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuditEvent:
        return cls(
            event_id=str(payload.get("event_id") or ""),
            timestamp_utc=str(payload.get("timestamp_utc") or ""),
            event_type=str(payload.get("event_type") or ""),
            source=str(payload.get("source") or ""),
            session_id=str(payload.get("session_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            actor=str(payload.get("actor") or ""),
            artifact_root=str(payload.get("artifact_root") or ""),
            metadata=dict(payload.get("metadata") or {}),
            schema_version=str(payload.get("schema_version") or AUDIT_EVENT_SCHEMA_VERSION),
        )


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def registry_directory(output_root: str | Path) -> Path:
    return Path(output_root) / RUN_REGISTRY_DIRECTORY_NAME


def run_registry_path(output_root: str | Path) -> Path:
    return registry_directory(output_root) / RUN_REGISTRY_FILE_NAME


def audit_event_log_path(output_root: str | Path) -> Path:
    return registry_directory(output_root) / AUDIT_EVENT_FILE_NAME


def per_run_audit_event_path(run_root: str | Path) -> Path:
    return Path(run_root) / "metadata" / PER_RUN_AUDIT_EVENT_FILE_NAME


def json_safe(value: Any) -> Any:
    """Converts common Python objects into JSON-safe values with secret redaction."""

    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                safe[key_text] = "[redacted]"
            else:
                safe[key_text] = json_safe(item)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, AttributeError) as exc:
            LOGGER.debug("Unable to coerce registry scalar value: %s", exc)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def fingerprint_payload(payload: Any) -> str:
    text = json.dumps(json_safe(payload), sort_keys=True, default=str)
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_run_registry(
    output_root: str | Path,
    *,
    refresh_from_artifacts: bool = False,
) -> list[RunRegistryEntry]:
    """Loads the registry, optionally backfilling it from existing run folders."""

    if refresh_from_artifacts:
        sync_run_registry_from_artifacts(output_root)
    payload = _read_registry_payload(output_root)
    entries = payload.get("runs", [])
    if not isinstance(entries, list):
        return []
    result: list[RunRegistryEntry] = []
    for entry_payload in entries:
        if isinstance(entry_payload, dict):
            try:
                result.append(RunRegistryEntry.from_dict(entry_payload))
            except (TypeError, ValueError) as exc:
                LOGGER.warning("Skipping invalid run registry entry: %s", exc)
                continue
    return sorted(result, key=_entry_sort_key, reverse=True)


def upsert_run_registry_entry(
    output_root: str | Path,
    entry: RunRegistryEntry | dict[str, Any],
) -> RunRegistryEntry:
    """Inserts or updates one run registry entry."""

    resolved_entry = (
        entry if isinstance(entry, RunRegistryEntry) else RunRegistryEntry.from_dict(entry)
    )
    if not resolved_entry.updated_at_utc:
        resolved_entry.updated_at_utc = utc_timestamp()
    payload = _read_registry_payload(output_root)
    runs = [
        RunRegistryEntry.from_dict(row)
        for row in payload.get("runs", [])
        if isinstance(row, dict) and row.get("run_id") != resolved_entry.run_id
    ]
    runs.append(resolved_entry)
    payload = {
        "schema_version": RUN_REGISTRY_SCHEMA_VERSION,
        "updated_at_utc": utc_timestamp(),
        "runs": [entry.to_dict() for entry in sorted(runs, key=_entry_sort_key, reverse=True)],
    }
    _write_json_atomic(run_registry_path(output_root), payload)
    return resolved_entry


def append_audit_event(
    output_root: str | Path,
    event_type: str,
    *,
    source: str,
    run_id: str = "",
    session_id: str = "",
    actor: str = "",
    artifact_root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Appends a JSONL event to the global audit log and run folder when available."""

    event = AuditEvent(
        event_id=uuid4().hex,
        timestamp_utc=utc_timestamp(),
        event_type=event_type,
        source=source,
        session_id=session_id,
        run_id=run_id,
        actor=actor,
        artifact_root=str(artifact_root or ""),
        metadata=json_safe(metadata or {}),
    )
    _append_jsonl(audit_event_log_path(output_root), event.to_dict())
    if run_id:
        run_root = Path(artifact_root) if artifact_root else Path(output_root) / run_id
        if run_root.exists():
            _append_jsonl(per_run_audit_event_path(run_root), event.to_dict())
    return event


def load_audit_events(
    output_root: str | Path,
    *,
    run_id: str | None = None,
    event_type: str | None = None,
    source: str | None = None,
    limit: int | None = 1000,
) -> list[AuditEvent]:
    """Loads recent audit events from the global JSONL log."""

    path = audit_event_log_path(output_root)
    if not path.exists():
        return []
    events: list[AuditEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            event = AuditEvent.from_dict(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping invalid audit event row: %s", exc)
            continue
        if run_id is not None and event.run_id != run_id:
            continue
        if event_type is not None and event.event_type != event_type:
            continue
        if source is not None and event.source != source:
            continue
        events.append(event)
    events = sorted(events, key=lambda event: event.timestamp_utc, reverse=True)
    if limit is not None:
        return events[: max(0, int(limit))]
    return events


def write_per_run_audit_log(
    output_root: str | Path,
    run_id: str,
    run_root: str | Path,
) -> Path:
    """Writes the run-scoped audit log from global events into the run folder."""

    destination = per_run_audit_event_path(run_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events = load_audit_events(output_root, run_id=run_id, limit=None)
    text = "\n".join(json.dumps(event.to_dict(), sort_keys=True) for event in reversed(events))
    destination.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return destination


def sync_run_registry_from_artifacts(output_root: str | Path) -> list[RunRegistryEntry]:
    """Backfills or refreshes the registry from existing run folders."""

    root = Path(output_root)
    if not root.exists():
        return load_run_registry(root)
    entries: list[RunRegistryEntry] = []
    for run_root in sorted(root.glob("run_*")):
        if not run_root.is_dir():
            continue
        entry = build_run_registry_entry_from_run_folder(run_root)
        if entry is None:
            continue
        entries.append(upsert_run_registry_entry(root, entry))
    return entries


def update_run_registry_from_context(
    context: Any,
    *,
    status: str = "completed",
) -> RunRegistryEntry:
    """Builds and upserts a run registry entry from a pipeline context."""

    entry = build_run_registry_entry_from_context(context, status=status)
    return upsert_run_registry_entry(context.config.artifacts.output_root, entry)


def build_run_registry_entry_from_context(
    context: Any,
    *,
    status: str = "completed",
    error_message: str = "",
) -> RunRegistryEntry:
    metadata = getattr(context, "metadata", {}) or {}
    artifacts = getattr(context, "artifacts", {}) or {}
    input_source = metadata.get("input_source", {})
    if not isinstance(input_source, dict):
        input_source = {}
    run_root = (
        artifacts.get("output_root")
        or Path(context.config.artifacts.output_root) / context.run_id
    )
    return RunRegistryEntry(
        run_id=str(context.run_id),
        status=status,
        artifact_root=str(run_root),
        started_at_utc=str(metadata.get("run_started_at_utc") or ""),
        completed_at_utc=str(metadata.get("run_completed_at_utc") or ""),
        elapsed_seconds=_optional_float(metadata.get("run_elapsed_seconds")),
        execution_mode=str(context.config.execution.mode.value),
        model_type=str(context.config.model.model_type.value),
        target_mode=str(context.config.target.mode.value),
        dataset_source_label=_source_label(input_source, metadata),
        dataset_source_kind=str(
            input_source.get("source_kind") or metadata.get("input_type") or ""
        ),
        large_data_mode=bool(context.config.performance.large_data_mode),
        metrics_summary=_summarize_metrics(getattr(context, "metrics", {}) or {}),
        artifact_paths=_compact_artifact_paths(artifacts),
        warning_count=len(getattr(context, "warnings", []) or []),
        reviewer_status=_reviewer_status(Path(run_root)),
        reviewer_name=_reviewer_name(Path(run_root)),
        review_updated_at_utc=_review_updated_at(Path(run_root)),
        error_message=error_message,
        updated_at_utc=utc_timestamp(),
    )


def build_failed_run_registry_entry(
    *,
    output_root: str | Path,
    run_id: str,
    error_message: str,
    execution_mode: str = "",
    model_type: str = "",
    target_mode: str = "",
    dataset_source_label: str = "",
    dataset_source_kind: str = "",
    large_data_mode: bool = False,
    started_at_utc: str = "",
    artifact_root: str | Path | None = None,
) -> RunRegistryEntry:
    run_root = Path(artifact_root) if artifact_root else Path(output_root) / run_id
    return RunRegistryEntry(
        run_id=run_id,
        status="failed",
        artifact_root=str(run_root),
        started_at_utc=started_at_utc,
        completed_at_utc=utc_timestamp(),
        execution_mode=execution_mode,
        model_type=model_type,
        target_mode=target_mode,
        dataset_source_label=dataset_source_label,
        dataset_source_kind=dataset_source_kind,
        large_data_mode=large_data_mode,
        error_message=error_message,
        updated_at_utc=utc_timestamp(),
    )


def build_run_registry_entry_from_run_folder(run_root: str | Path) -> RunRegistryEntry | None:
    """Builds a registry entry from a previously exported run folder."""

    run_path = Path(run_root)
    manifest = _read_json_if_exists(run_path / "artifact_manifest.json")
    config = _read_json_if_exists(run_path / "config" / "run_config.json")
    metrics = _read_json_if_exists(run_path / "metadata" / "metrics.json")
    trace = _read_json_if_exists(run_path / "metadata" / "run_debug_trace.json")
    if not any([manifest, config, metrics, trace]):
        return None
    review = _read_json_if_exists(run_path / "review_workspace.json")
    source = _nested(config, "metadata", "input_source")
    if not isinstance(source, dict):
        source = {}
    run_id = str(trace.get("run_id") or run_path.name)
    summary = trace.get("summary", {}) if isinstance(trace.get("summary"), dict) else {}
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    execution_config = (
        config.get("execution", {}) if isinstance(config.get("execution"), dict) else {}
    )
    target_config = config.get("target", {}) if isinstance(config.get("target"), dict) else {}
    performance_config = (
        config.get("performance", {}) if isinstance(config.get("performance"), dict) else {}
    )
    reviewer_status = str(review.get("approval_status") or "Not reviewed")
    reviewer_name = str(review.get("reviewer_name") or "")
    return RunRegistryEntry(
        run_id=run_id,
        status="completed" if manifest else "unknown",
        artifact_root=str(run_path),
        started_at_utc=str(trace.get("run_started_at_utc") or ""),
        completed_at_utc=str(trace.get("run_completed_at_utc") or ""),
        elapsed_seconds=_optional_float(summary.get("total_run_seconds")),
        execution_mode=str(execution_config.get("mode") or trace.get("execution_mode") or ""),
        model_type=str(model_config.get("model_type") or trace.get("model_type") or ""),
        target_mode=str(target_config.get("mode") or ""),
        dataset_source_label=_source_label(source, {}),
        dataset_source_kind=str(source.get("source_kind") or ""),
        large_data_mode=bool(performance_config.get("large_data_mode") or False),
        metrics_summary=_summarize_metrics(metrics),
        artifact_paths=_compact_manifest_paths(manifest, run_path),
        warning_count=int(summary.get("warning_count") or 0),
        reviewer_status=reviewer_status,
        reviewer_name=reviewer_name,
        review_updated_at_utc=_file_mtime_iso(run_path / "review_workspace.json"),
        updated_at_utc=utc_timestamp(),
    )


def _read_registry_payload(output_root: str | Path) -> dict[str, Any]:
    path = run_registry_path(output_root)
    if not path.exists():
        return {"schema_version": RUN_REGISTRY_SCHEMA_VERSION, "runs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Unable to read run registry file %s: %s", path, exc)
        return {"schema_version": RUN_REGISTRY_SCHEMA_VERSION, "runs": []}
    if not isinstance(payload, dict):
        return {"schema_version": RUN_REGISTRY_SCHEMA_VERSION, "runs": []}
    payload.setdefault("schema_version", RUN_REGISTRY_SCHEMA_VERSION)
    payload.setdefault("runs", [])
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Unable to read JSON file %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _entry_sort_key(entry: RunRegistryEntry) -> str:
    return entry.completed_at_utc or entry.started_at_utc or entry.updated_at_utc or entry.run_id


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def _optional_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_label(source: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(
        source.get("display_label")
        or source.get("relative_path")
        or source.get("source_identifier")
        or source.get("path")
        or metadata.get("input_type")
        or ""
    )


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    preferred = (
        "auc",
        "roc_auc",
        "ks",
        "accuracy",
        "f1_score",
        "brier_score",
        "log_loss",
        "rmse",
        "mae",
        "r2",
    )
    rows: dict[str, Any] = {}
    for split_name, split_metrics in metrics.items():
        if not isinstance(split_metrics, dict):
            continue
        for metric_name in preferred:
            if metric_name in split_metrics and _is_scalar(split_metrics[metric_name]):
                rows[f"{split_name}_{metric_name}"] = split_metrics[metric_name]
        if len(rows) >= 12:
            break
    return json_safe(rows)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _compact_artifact_paths(artifacts: dict[str, Any]) -> dict[str, str]:
    keys = (
        "output_root",
        "decision_summary",
        "interactive_report",
        "model",
        "config",
        "metrics",
        "run_debug_trace",
        "artifact_manifest",
        "audit_events",
    )
    return {
        key: str(artifacts[key])
        for key in keys
        if artifacts.get(key) not in {"", None}
    }


def _compact_manifest_paths(manifest: dict[str, Any], run_path: Path) -> dict[str, str]:
    core = (
        manifest.get("core_artifacts", {})
        if isinstance(manifest.get("core_artifacts"), dict)
        else {}
    )
    paths = _compact_artifact_paths(core)
    paths.setdefault("output_root", str(run_path))
    for key in ("interactive_report", "decision_summary", "config", "metrics", "run_debug_trace"):
        if key in manifest and key not in paths:
            paths[key] = str(manifest[key])
    return paths


def _reviewer_status(run_root: Path) -> str:
    payload = _read_json_if_exists(run_root / "review_workspace.json")
    return str(payload.get("approval_status") or "Not reviewed")


def _reviewer_name(run_root: Path) -> str:
    payload = _read_json_if_exists(run_root / "review_workspace.json")
    return str(payload.get("reviewer_name") or "")


def _review_updated_at(run_root: Path) -> str:
    return _file_mtime_iso(run_root / "review_workspace.json")


def _file_mtime_iso(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return ""


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
