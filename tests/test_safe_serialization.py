"""Security tests for hashed joblib serialization."""

from __future__ import annotations

import pytest

from quant_pd_framework.safe_serialization import (
    HashVerificationError,
    dump_joblib_with_hash,
    load_joblib_verified,
    sha256_sidecar_path,
)


def test_dump_joblib_with_hash_writes_sidecar_and_loads(tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"

    dump_joblib_with_hash({"value": 1}, artifact_path)

    assert artifact_path.exists()
    assert sha256_sidecar_path(artifact_path).exists()
    assert load_joblib_verified(artifact_path) == {"value": 1}


def test_tampered_joblib_fails_before_load(tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"
    dump_joblib_with_hash({"value": 1}, artifact_path)
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")

    with pytest.raises(HashVerificationError, match="SHA-256 mismatch"):
        load_joblib_verified(artifact_path)


def test_missing_sidecar_blocks_external_load(tmp_path) -> None:
    artifact_path = tmp_path / "legacy.joblib"
    dump_joblib_with_hash({"value": 1}, artifact_path)
    sha256_sidecar_path(artifact_path).unlink()

    with pytest.raises(HashVerificationError, match="Missing SHA-256 sidecar"):
        load_joblib_verified(artifact_path)


def test_missing_sidecar_allows_trusted_legacy_internal_load(tmp_path) -> None:
    artifact_path = tmp_path / "legacy.joblib"
    dump_joblib_with_hash({"value": 1}, artifact_path)
    sha256_sidecar_path(artifact_path).unlink()

    assert load_joblib_verified(
        artifact_path,
        allow_missing_sidecar=True,
        trusted_legacy_root=tmp_path,
    ) == {"value": 1}
