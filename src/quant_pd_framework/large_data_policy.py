"""Large-data model certification and execution-policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .config import (
    LargeDataBackend,
    LargeDataCertificationStatus,
    LargeDataFitCapabilityStatus,
    LargeDataModelPolicy,
    ModelType,
    PerformanceConfig,
)

FULL_DATA_EXACT_MODEL_TYPES = frozenset(
    {
        ModelType.LOGISTIC_REGRESSION,
        ModelType.DISCRETE_TIME_HAZARD_MODEL,
        ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
        ModelType.SCORECARD_LOGISTIC_REGRESSION,
        ModelType.LINEAR_REGRESSION,
        ModelType.RIDGE_REGRESSION,
        ModelType.LASSO_REGRESSION,
        ModelType.ELASTIC_NET_REGRESSION,
    }
)
FULL_DATA_INCREMENTAL_MODEL_TYPES = frozenset({ModelType.XGBOOST})
IN_MEMORY_ONLY_MODEL_TYPES = frozenset(
    {
        ModelType.RANDOM_FOREST,
        ModelType.EXTRA_TREES,
        ModelType.DECISION_TREE,
        ModelType.EXPLAINABLE_BOOSTING_MACHINE,
    }
)
FULL_DATA_CERTIFIED_MODEL_TYPES = FULL_DATA_EXACT_MODEL_TYPES | FULL_DATA_INCREMENTAL_MODEL_TYPES


@dataclass(frozen=True, slots=True)
class LargeDataFitCapability:
    """Resolved estimator capability for file-backed large-data model fitting."""

    model_type: ModelType
    status: LargeDataFitCapabilityStatus
    certified: bool
    fit_strategy: str
    scoring_strategy: str
    explanation: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "status": self.status.value,
            "certified": self.certified,
            "fit_strategy": self.fit_strategy,
            "scoring_strategy": self.scoring_strategy,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class LargeDataCertification:
    """Resolved large-data certification decision for one model run."""

    model_type: ModelType
    backend: LargeDataBackend
    policy: LargeDataModelPolicy
    status: LargeDataCertificationStatus
    execution_strategy: str
    is_full_data_certified_model: bool
    fit_capability: LargeDataFitCapability
    override_confirmed: bool
    override_reason: str
    recommendation: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "backend": self.backend.value,
            "policy": self.policy.value,
            "status": self.status.value,
            "execution_strategy": self.execution_strategy,
            "is_full_data_certified_model": self.is_full_data_certified_model,
            "fit_capability": self.fit_capability.to_metadata(),
            "fit_capability_status": self.fit_capability.status.value,
            "override_confirmed": self.override_confirmed,
            "override_reason": self.override_reason,
            "recommendation": self.recommendation,
        }


def resolve_large_data_certification(
    model_type: ModelType | str,
    performance: PerformanceConfig,
) -> LargeDataCertification:
    """Returns the large-data certification decision for a model/config pair."""

    resolved_model_type = model_type if isinstance(model_type, ModelType) else ModelType(model_type)
    backend = (
        performance.large_data_backend
        if isinstance(performance.large_data_backend, LargeDataBackend)
        else LargeDataBackend(performance.large_data_backend)
    )
    policy = (
        performance.large_data_model_policy
        if isinstance(performance.large_data_model_policy, LargeDataModelPolicy)
        else LargeDataModelPolicy(performance.large_data_model_policy)
    )
    override_reason = str(performance.large_data_override_reason or "").strip()
    fit_capability = resolve_large_data_fit_capability(resolved_model_type)
    is_certified_model = fit_capability.certified

    if is_certified_model:
        status = LargeDataCertificationStatus.FULL_DATA_CERTIFIED
        strategy = (
            "disk_backed_full_data_fit"
            if backend in {LargeDataBackend.AUTO, LargeDataBackend.DISK_BACKED}
            else "governed_sample_fit_full_score"
        )
        recommendation = (
            "Proceed with the certified large-data path where implemented. Keep "
            "Parquet staging, paged result views, and compact exports enabled."
        )
    elif policy == LargeDataModelPolicy.CERTIFIED_ONLY:
        status = LargeDataCertificationStatus.BLOCKED
        strategy = "blocked_uncertified_model"
        recommendation = (
            "Choose a full-data-certified model type or change the large-data model "
            "policy to allow governed sample fallback."
        )
    elif policy == LargeDataModelPolicy.FORCE_FULL_DATA_OVERRIDE:
        if performance.large_data_override_confirmed and override_reason:
            status = LargeDataCertificationStatus.EXPERIMENTAL_FULL_DATA_OVERRIDE
            strategy = "forced_experimental_full_data_fit"
            recommendation = (
                "Proceed only if the instance has been sized for the selected complex "
                "model. This override is audited and is not treated as certified."
            )
        else:
            status = LargeDataCertificationStatus.BLOCKED
            strategy = "blocked_missing_override_confirmation"
            recommendation = (
                "Confirm the force override and provide an override reason before "
                "running an uncertified complex model on a large file."
            )
    else:
        status = LargeDataCertificationStatus.SAMPLE_FIT_FULL_SCORE
        strategy = "governed_sample_fit_full_score"
        recommendation = (
            "Use governed sample development plus chunked full-file scoring, or "
            "switch to a certified model type for optimized full-data execution."
        )

    return LargeDataCertification(
        model_type=resolved_model_type,
        backend=backend,
        policy=policy,
        status=status,
        execution_strategy=strategy,
        is_full_data_certified_model=is_certified_model,
        fit_capability=fit_capability,
        override_confirmed=bool(performance.large_data_override_confirmed),
        override_reason=override_reason,
        recommendation=recommendation,
    )


def resolve_large_data_fit_capability(model_type: ModelType | str) -> LargeDataFitCapability:
    """Classifies model fit support without overstating out-of-core capabilities."""

    resolved_model_type = model_type if isinstance(model_type, ModelType) else ModelType(model_type)
    if resolved_model_type in FULL_DATA_EXACT_MODEL_TYPES:
        return LargeDataFitCapability(
            model_type=resolved_model_type,
            status=LargeDataFitCapabilityStatus.FULL_DATA_EXACT,
            certified=True,
            fit_strategy="certified_full_data_or_aggregate_fit",
            scoring_strategy="chunked_full_file_scoring",
            explanation=(
                "This model family is the priority path for exact or aggregate-safe "
                "large-data fitting as disk-backed execution matures."
            ),
        )
    if resolved_model_type in FULL_DATA_INCREMENTAL_MODEL_TYPES:
        return LargeDataFitCapability(
            model_type=resolved_model_type,
            status=LargeDataFitCapabilityStatus.FULL_DATA_INCREMENTAL,
            certified=True,
            fit_strategy="external_memory_or_cpu_parallel_fit_when_available",
            scoring_strategy="chunked_full_file_scoring",
            explanation=(
                "This model can use CPU parallelism and supported external-memory "
                "paths, but it still requires staged Parquet and guardrails."
            ),
        )
    if resolved_model_type in IN_MEMORY_ONLY_MODEL_TYPES:
        return LargeDataFitCapability(
            model_type=resolved_model_type,
            status=LargeDataFitCapabilityStatus.IN_MEMORY_ONLY,
            certified=False,
            fit_strategy="in_memory_fit_below_safety_threshold",
            scoring_strategy="chunked_full_file_scoring_after_fit",
            explanation=(
                "This estimator remains in-memory with current dependencies. Use "
                "sample fit/full score or an explicit override for large files."
            ),
        )
    return LargeDataFitCapability(
        model_type=resolved_model_type,
        status=LargeDataFitCapabilityStatus.SAMPLE_FIT_FULL_SCORE,
        certified=False,
        fit_strategy="governed_sample_fit",
        scoring_strategy="chunked_full_file_scoring_after_fit",
        explanation=(
            "This family is not truly out-of-core with the current stack. Quant "
            "Studio keeps it selectable through sample-fit/full-score policy."
        ),
    )


def build_large_data_override_audit(
    certification: LargeDataCertification,
    *,
    source_metadata: dict[str, Any],
    memory_estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds an auditable record for forced large-data model overrides."""

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "certification": certification.to_metadata(),
        "source": dict(source_metadata),
        "memory_estimate": dict(memory_estimate or {}),
    }
