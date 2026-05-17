"""Checkpoint persistence for memory-isolated workflow execution."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .context import PipelineContext

CHECKPOINT_MANIFEST_FILE_NAME = "checkpoint_manifest.json"
CHECKPOINT_CONTEXT_SUFFIX = ".joblib"
CHECKPOINT_TABLE_REF_MARKER = "__quant_studio_checkpoint_table_ref__"


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Filesystem locations used by a checkpointed run."""

    run_root: Path
    checkpoints_dir: Path
    manifest_path: Path


def build_checkpoint_paths(output_root: Path, run_id: str) -> CheckpointPaths:
    """Builds the auditable checkpoint folder under a run artifact directory."""

    run_root = output_root / run_id
    checkpoints_dir = run_root / "checkpoints"
    return CheckpointPaths(
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        manifest_path=checkpoints_dir / CHECKPOINT_MANIFEST_FILE_NAME,
    )


def initialize_checkpoint_manifest(
    *,
    paths: CheckpointPaths,
    run_id: str,
    stages: list[dict[str, Any]],
    initial_context_path: Path,
    execution_strategy: str,
    keep_all_checkpoints: bool = True,
) -> dict[str, Any]:
    """Creates the initial checkpoint manifest."""

    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    created_at = _utc_now()
    manifest = {
        "manifest_version": "1.0",
        "run_id": run_id,
        "execution_strategy": execution_strategy,
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "run_root": str(paths.run_root),
        "checkpoints_dir": str(paths.checkpoints_dir),
        "initial_context_path": str(initial_context_path),
        "initial_context_pruned_at_utc": "",
        "latest_context_path": str(initial_context_path),
        "checkpoint_retention": {
            "keep_all_checkpoints": bool(keep_all_checkpoints),
            "policy": "keep_all" if keep_all_checkpoints else "prune_stale_contexts",
            "last_pruned_at_utc": "",
            "pruned_context_count": 0,
            "latest_context_retained": str(initial_context_path),
        },
        "status": "initialized",
        "stages": [
            {
                **stage,
                "status": "pending",
                "started_at_utc": "",
                "completed_at_utc": "",
                "elapsed_seconds": None,
                "context_path": "",
                "context_pruned_at_utc": "",
                "error_message": "",
            }
            for stage in stages
        ],
    }
    write_checkpoint_manifest(paths.manifest_path, manifest)
    return manifest


def read_checkpoint_manifest(manifest_path: Path) -> dict[str, Any]:
    """Reads a checkpoint manifest from disk."""

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint manifest is not a JSON object: {manifest_path}")
    return payload


def write_checkpoint_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Writes the manifest with stable formatting."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at_utc"] = _utc_now()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def save_context_checkpoint(context: PipelineContext, path: Path) -> Path:
    """Persists a pipeline context without compression to avoid extra CPU overhead."""

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_context = _spill_large_data_context_tables(context, path)
    joblib.dump(checkpoint_context, path, compress=0)
    return path


def load_context_checkpoint(path: Path) -> PipelineContext:
    """Loads a previously persisted pipeline context."""

    context = joblib.load(path)
    if not isinstance(context, PipelineContext):
        raise TypeError(f"Checkpoint did not contain a PipelineContext: {path}")
    _rehydrate_checkpoint_table_refs(context)
    return context


def _spill_large_data_context_tables(
    context: PipelineContext,
    checkpoint_path: Path,
) -> PipelineContext:
    """
    Writes large dataframe fields beside the checkpoint and stores lightweight refs.

    This keeps Large Data Mode checkpoint joblib files from duplicating large pandas
    frames. The loader rehydrates the refs before the next stage executes, preserving
    the existing stage API while reducing checkpoint serialization pressure.
    """

    if not getattr(context.config.performance, "large_data_mode", False):
        return context
    row_cap = int(getattr(context.config.performance, "large_data_max_in_memory_rows", 0) or 0)
    if row_cap <= 0:
        return context

    spill_dir = checkpoint_path.parent / "tables" / checkpoint_path.stem
    checkpoint_context = dataclass_replace(context)
    checkpoint_context.metadata = dict(context.metadata)
    spill_records: list[dict[str, Any]] = []

    for field_name in ("raw_data", "working_data", "feature_importance", "backtest_summary"):
        value = getattr(context, field_name)
        replacement, record = _spill_frame_if_needed(
            value,
            spill_dir=spill_dir,
            field_name=field_name,
            table_name=field_name,
            row_cap=row_cap,
        )
        setattr(checkpoint_context, field_name, replacement)
        if record:
            spill_records.append(record)

    for field_name in ("split_frames", "predictions", "scenario_results", "diagnostics_tables"):
        value = getattr(context, field_name)
        if isinstance(value, dict):
            replacement, records = _spill_frame_mapping_if_needed(
                value,
                spill_dir=spill_dir,
                field_name=field_name,
                row_cap=row_cap,
            )
            setattr(checkpoint_context, field_name, replacement)
            spill_records.extend(records)

    if isinstance(context.comparison_results, pd.DataFrame):
        replacement, record = _spill_frame_if_needed(
            context.comparison_results,
            spill_dir=spill_dir,
            field_name="comparison_results",
            table_name="comparison_results",
            row_cap=row_cap,
        )
        checkpoint_context.comparison_results = replacement
        if record:
            spill_records.append(record)

    if spill_records:
        existing_records = checkpoint_context.metadata.get("checkpoint_table_refs", [])
        if not isinstance(existing_records, list):
            existing_records = []
        checkpoint_context.metadata["checkpoint_table_refs"] = [
            *existing_records,
            *spill_records,
        ]
    return checkpoint_context


def _spill_frame_mapping_if_needed(
    value: dict[str, Any],
    *,
    spill_dir: Path,
    field_name: str,
    row_cap: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replacement = dict(value)
    records: list[dict[str, Any]] = []
    for key, frame in value.items():
        table_name = f"{field_name}_{_safe_table_name(str(key))}"
        spilled, record = _spill_frame_if_needed(
            frame,
            spill_dir=spill_dir,
            field_name=field_name,
            table_name=table_name,
            row_cap=row_cap,
            mapping_key=str(key),
        )
        replacement[key] = spilled
        if record:
            records.append(record)
    return replacement, records


def _spill_frame_if_needed(
    value: Any,
    *,
    spill_dir: Path,
    field_name: str,
    table_name: str,
    row_cap: int,
    mapping_key: str = "",
) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(value, pd.DataFrame) or len(value) <= row_cap:
        return value, None
    spill_dir.mkdir(parents=True, exist_ok=True)
    table_path = spill_dir / f"{_safe_table_name(table_name)}.parquet"
    try:
        value.to_parquet(table_path, index=False)
    except Exception:
        return value, None
    record = {
        CHECKPOINT_TABLE_REF_MARKER: True,
        "field_name": field_name,
        "mapping_key": mapping_key,
        "path": str(table_path),
        "format": "parquet",
        "rows": int(len(value)),
        "columns": list(map(str, value.columns)),
        "memory_bytes": int(value.memory_usage(deep=True).sum()),
    }
    return dict(record), record


def _rehydrate_checkpoint_table_refs(context: PipelineContext) -> None:
    for field_name in ("raw_data", "working_data", "feature_importance", "backtest_summary"):
        value = getattr(context, field_name)
        if _is_checkpoint_table_ref(value):
            setattr(context, field_name, _read_checkpoint_table_ref(value))

    for field_name in ("split_frames", "predictions", "scenario_results", "diagnostics_tables"):
        value = getattr(context, field_name)
        if isinstance(value, dict):
            setattr(
                context,
                field_name,
                {
                    key: _read_checkpoint_table_ref(item)
                    if _is_checkpoint_table_ref(item)
                    else item
                    for key, item in value.items()
                },
            )

    if _is_checkpoint_table_ref(context.comparison_results):
        context.comparison_results = _read_checkpoint_table_ref(context.comparison_results)


def _is_checkpoint_table_ref(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get(CHECKPOINT_TABLE_REF_MARKER))


def _read_checkpoint_table_ref(value: dict[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(Path(str(value["path"])))


def _safe_table_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )


def checkpoint_file_name(stage_order: int, stage_id: str) -> str:
    """Returns a stable context checkpoint filename for a completed stage."""

    safe_stage = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in stage_id
    )
    return f"{stage_order:02d}_{safe_stage}{CHECKPOINT_CONTEXT_SUFFIX}"


def prune_context_checkpoints(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    keep_latest: bool,
) -> list[Path]:
    """
    Deletes stale checkpoint context files while preserving the manifest.

    The latest context is kept during an active run because the next stage needs
    it as input. After a completed run, callers may pass `keep_latest=False` to
    remove the final context file as well.
    """

    checkpoints_dir = _safe_checkpoint_dir(manifest_path, manifest)
    latest_context_path = _safe_context_path(
        manifest.get("latest_context_path"),
        checkpoints_dir,
    )
    retained_path = latest_context_path if keep_latest else None
    candidates = _context_checkpoint_candidates(checkpoints_dir, manifest)
    pruned_at = _utc_now()
    deleted_paths: list[Path] = []

    for candidate in sorted(candidates):
        if retained_path is not None and candidate == retained_path:
            continue
        if not candidate.exists() or not candidate.is_file():
            continue
        candidate.unlink()
        deleted_paths.append(candidate)

    if not deleted_paths:
        _update_retention_manifest(
            manifest,
            retained_path=retained_path,
            pruned_at="",
            deleted_paths=[],
        )
        return []

    _mark_pruned_context_references(manifest, deleted_paths, pruned_at, checkpoints_dir)
    _update_retention_manifest(
        manifest,
        retained_path=retained_path,
        pruned_at=pruned_at,
        deleted_paths=deleted_paths,
    )
    return deleted_paths


def find_next_pending_stage(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Returns the next stage record that is not complete."""

    for stage in manifest.get("stages", []):
        if stage.get("status") not in {"completed", "skipped", "failed_optional"}:
            return stage
    return None


def mark_stage_started(manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
    """Marks a stage as running in the manifest."""

    stage = _stage_by_id(manifest, stage_id)
    stage["status"] = "running"
    stage["started_at_utc"] = _utc_now()
    stage["completed_at_utc"] = ""
    stage["elapsed_seconds"] = None
    stage["error_message"] = ""
    manifest["status"] = "running"
    return stage


def mark_stage_completed(
    manifest: dict[str, Any],
    stage_id: str,
    *,
    context_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Marks a stage as complete and advances the latest-context pointer."""

    stage = _stage_by_id(manifest, stage_id)
    stage["status"] = "completed"
    stage["completed_at_utc"] = _utc_now()
    stage["elapsed_seconds"] = round(float(elapsed_seconds), 6)
    stage["context_path"] = str(context_path)
    stage["error_message"] = ""
    manifest["latest_context_path"] = str(context_path)
    if find_next_pending_stage(manifest) is None:
        manifest["status"] = "completed"
    else:
        manifest["status"] = "running"
    return stage


def mark_stage_failed(
    manifest: dict[str, Any],
    stage_id: str,
    *,
    error_message: str,
    elapsed_seconds: float,
    optional: bool,
) -> dict[str, Any]:
    """Marks a stage failure, using a non-terminal status for optional diagnostics."""

    stage = _stage_by_id(manifest, stage_id)
    stage["status"] = "failed_optional" if optional else "failed"
    stage["completed_at_utc"] = _utc_now()
    stage["elapsed_seconds"] = round(float(elapsed_seconds), 6)
    stage["error_message"] = error_message
    manifest["status"] = "running" if optional else "failed"
    return stage


def copy_manifest_to_metadata(manifest_path: Path, metadata_dir: Path) -> Path:
    """Copies the checkpoint manifest beside the other metadata artifacts."""

    metadata_dir.mkdir(parents=True, exist_ok=True)
    destination = metadata_dir / CHECKPOINT_MANIFEST_FILE_NAME
    shutil.copy2(manifest_path, destination)
    return destination


def _safe_checkpoint_dir(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    raw_dir = manifest.get("checkpoints_dir") or manifest_path.parent
    checkpoints_dir = Path(str(raw_dir)).resolve()
    if checkpoints_dir != manifest_path.parent.resolve():
        raise ValueError(
            "Checkpoint manifest directory does not match the requested manifest path."
        )
    return checkpoints_dir


def _safe_context_path(value: Any, checkpoints_dir: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value)).resolve()
    if candidate.parent != checkpoints_dir:
        raise ValueError(f"Refusing to prune checkpoint outside {checkpoints_dir}: {candidate}")
    if candidate.suffix != CHECKPOINT_CONTEXT_SUFFIX:
        raise ValueError(f"Refusing to prune non-context checkpoint file: {candidate}")
    return candidate


def _context_checkpoint_candidates(
    checkpoints_dir: Path,
    manifest: dict[str, Any],
) -> set[Path]:
    candidates = {
        path.resolve()
        for path in checkpoints_dir.glob(f"*{CHECKPOINT_CONTEXT_SUFFIX}")
        if path.is_file()
    }
    initial_context_path = _safe_context_path(manifest.get("initial_context_path"), checkpoints_dir)
    if initial_context_path is not None:
        candidates.add(initial_context_path)
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        context_path = _safe_context_path(stage.get("context_path"), checkpoints_dir)
        if context_path is not None:
            candidates.add(context_path)
    return candidates


def _mark_pruned_context_references(
    manifest: dict[str, Any],
    deleted_paths: list[Path],
    pruned_at: str,
    checkpoints_dir: Path,
) -> None:
    deleted_paths_resolved = {path.resolve() for path in deleted_paths}
    initial_context_path = _safe_context_path(manifest.get("initial_context_path"), checkpoints_dir)
    if initial_context_path in deleted_paths_resolved:
        manifest["initial_context_pruned_at_utc"] = pruned_at
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        context_path = _safe_context_path(stage.get("context_path"), checkpoints_dir)
        if context_path in deleted_paths_resolved:
            stage["context_pruned_at_utc"] = pruned_at


def _update_retention_manifest(
    manifest: dict[str, Any],
    *,
    retained_path: Path | None,
    pruned_at: str,
    deleted_paths: list[Path],
) -> None:
    retention = manifest.setdefault("checkpoint_retention", {})
    prior_count = int(retention.get("pruned_context_count") or 0)
    retention["latest_context_retained"] = str(retained_path) if retained_path else ""
    if pruned_at:
        retention["last_pruned_at_utc"] = pruned_at
        retention["pruned_context_count"] = prior_count + len(deleted_paths)


def _stage_by_id(manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in manifest.get("stages", []):
        if stage.get("stage_id") == stage_id:
            return stage
    raise KeyError(f"Checkpoint manifest does not contain stage '{stage_id}'.")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
