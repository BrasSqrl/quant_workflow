"""Safe joblib serialization helpers with SHA-256 sidecar verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib

from quant_pd_framework.logging import get_logger

LOGGER = get_logger(__name__)
HASH_SUFFIX = ".sha256"


class HashVerificationError(ValueError):
    """Raised when a serialized artifact hash sidecar is missing or invalid."""


JoblibHashVerificationError = HashVerificationError


def sha256_sidecar_path(path: str | Path) -> Path:
    """Returns the SHA-256 sidecar path for a serialized artifact."""

    resolved = Path(path)
    return resolved.with_name(f"{resolved.name}{HASH_SUFFIX}")


def compute_file_sha256(path: str | Path) -> str:
    """Computes the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: str | Path) -> Path:
    """Writes the sidecar hash file and returns its path."""

    artifact_path = Path(path)
    digest = compute_file_sha256(artifact_path)
    sidecar_path = sha256_sidecar_path(artifact_path)
    sidecar_path.write_text(f"{digest}  {artifact_path.name}\n", encoding="utf-8")
    LOGGER.debug("Wrote SHA-256 sidecar for %s", artifact_path)
    return sidecar_path


def verify_sha256_sidecar(
    path: str | Path,
    *,
    allow_missing: bool = False,
    trusted_legacy_root: str | Path | None = None,
) -> bool:
    """Verifies a sidecar hash before loading a serialized artifact."""

    artifact_path = Path(path)
    sidecar_path = sha256_sidecar_path(artifact_path)
    if not sidecar_path.exists():
        if allow_missing and _is_within_root(artifact_path, trusted_legacy_root):
            LOGGER.warning(
                "Loading legacy joblib artifact without SHA-256 sidecar: %s",
                artifact_path,
            )
            return False
        message = f"Missing SHA-256 sidecar for serialized artifact: {sidecar_path}"
        LOGGER.error(message)
        raise HashVerificationError(message)
    expected = sidecar_path.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = compute_file_sha256(artifact_path).lower()
    if expected != actual:
        message = (
            f"SHA-256 mismatch for serialized artifact {artifact_path}: "
            f"expected {expected}, observed {actual}"
        )
        LOGGER.error(message)
        raise HashVerificationError(message)
    LOGGER.debug("Verified SHA-256 sidecar for %s", artifact_path)
    return True


def dump_joblib_with_hash(
    value: Any,
    path: str | Path,
    *,
    compress: int | bool | tuple[str, int] = 0,
) -> Path:
    """Dumps a joblib artifact and writes its SHA-256 sidecar."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, artifact_path, compress=compress)
    write_sha256_sidecar(artifact_path)
    return artifact_path


def load_joblib_verified(
    path: str | Path,
    *,
    allow_missing_sidecar: bool = False,
    trusted_legacy_root: str | Path | None = None,
) -> Any:
    """Verifies and loads a joblib artifact."""

    artifact_path = Path(path)
    verify_sha256_sidecar(
        artifact_path,
        allow_missing=allow_missing_sidecar,
        trusted_legacy_root=trusted_legacy_root,
    )
    return joblib.load(artifact_path)


def _is_within_root(path: Path, root: str | Path | None) -> bool:
    if root is None:
        return False
    try:
        path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True
