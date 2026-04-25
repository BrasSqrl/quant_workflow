"""User-facing workflow error classification for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowErrorGuidance:
    """Structured failure message shown to users after a workflow run fails."""

    title: str
    likely_cause: str
    recommended_action: str
    technical_summary: str
    technical_details: str = ""


def classify_workflow_exception(
    exc: Exception,
    *,
    technical_details: str = "",
) -> WorkflowErrorGuidance:
    """Maps common workflow failures to practical recovery guidance."""

    summary = _normalize_exception_text(exc)
    lowered = summary.lower()

    if any(token in lowered for token in ("memoryerror", "unable to allocate", "out of memory")):
        return WorkflowErrorGuidance(
            title="The run exceeded available memory.",
            likely_cause=(
                "The selected data, diagnostics, or export surfaces require more memory "
                "than the current machine can safely provide."
            ),
            recommended_action=(
                "Use Large Data Mode with a smaller governed sample, reduce diagnostic "
                "sample rows, choose Parquet outputs, or move to a larger SageMaker instance."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    if any(token in lowered for token in ("target", "positive_values", "target-source")):
        return WorkflowErrorGuidance(
            title="The target definition could not be built.",
            likely_cause=(
                "The column designer does not currently define one valid target source, "
                "or the target values do not match the selected target mode."
            ),
            recommended_action=(
                "In Column Designer, mark exactly one source column as the target source. "
                "For binary models, confirm the positive target values match the data."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    if any(token in lowered for token in ("date", "datetime", "time-series", "panel")):
        return WorkflowErrorGuidance(
            title="The time or panel split settings are invalid.",
            likely_cause=(
                "A date column is missing, cannot be parsed as dates, or the selected "
                "data structure requires a date/entity role that is not configured."
            ),
            recommended_action=(
                "Mark the date column in Column Designer, verify date values parse cleanly, "
                "and mark an identifier column when panel modeling is selected."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    if any(
        token in lowered
        for token in ("nan", "infinity", "inf", "could not convert string to float")
    ):
        return WorkflowErrorGuidance(
            title="Model features contain values that cannot be fit as configured.",
            likely_cause=(
                "The model received missing, infinite, or non-numeric values after the "
                "current cleaning, imputation, and transformation settings were applied."
            ),
            recommended_action=(
                "Review imputation policies, missing-value handling, feature roles, and "
                "transformations for the affected columns before rerunning."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    if any(
        token in lowered
        for token in ("singular matrix", "perfect separation", "convergence", "did not converge")
    ):
        return WorkflowErrorGuidance(
            title="The model fit was numerically unstable.",
            likely_cause=(
                "The selected features may be collinear, perfectly predictive, sparse, "
                "or poorly scaled for the chosen model family."
            ),
            recommended_action=(
                "Use variable selection or feature policy checks, remove highly collinear "
                "features, simplify transformations, or try a regularized model variant."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    if any(token in lowered for token in ("parquet", "pyarrow", "duckdb", "csv")):
        return WorkflowErrorGuidance(
            title="The input file or large-data staging step failed.",
            likely_cause=(
                "The selected file could not be read or staged with the configured "
                "large-data intake settings."
            ),
            recommended_action=(
                "Confirm the file exists in Data_Load, try converting CSV to Parquet, "
                "verify required packages are installed, and rerun with a smaller chunk size."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    if "cannot export artifacts" in lowered or "diagnostics finish" in lowered:
        return WorkflowErrorGuidance(
            title="The run stopped before the artifact bundle was complete.",
            likely_cause=(
                "A required upstream step did not produce the model, diagnostics, or "
                "tables needed by the export step."
            ),
            recommended_action=(
                "Review the run debug trace if it was written, then rerun after resolving "
                "the first failed modeling or diagnostics step."
            ),
            technical_summary=summary,
            technical_details=technical_details,
        )

    return WorkflowErrorGuidance(
        title="The workflow stopped before completion.",
        likely_cause=(
            "An unexpected exception occurred in the configured workflow. The technical "
            "details below preserve the original error for debugging."
        ),
        recommended_action=(
            "Check the readiness panel, confirm the data roles and output settings, then "
            "use the technical details when debugging the affected step."
        ),
        technical_summary=summary,
        technical_details=technical_details,
    )


def _normalize_exception_text(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def guidance_to_dict(guidance: WorkflowErrorGuidance) -> dict[str, Any]:
    """Returns a serializable representation useful for tests and session state."""

    return {
        "title": guidance.title,
        "likely_cause": guidance.likely_cause,
        "recommended_action": guidance.recommended_action,
        "technical_summary": guidance.technical_summary,
        "technical_details": guidance.technical_details,
    }
