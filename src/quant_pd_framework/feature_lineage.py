"""Feature lineage helpers for model-development evidence packages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .context import PipelineContext


LINEAGE_COLUMNS = [
    "model_feature_name",
    "source_feature",
    "feature_type",
    "lineage_type",
    "source_columns",
    "transformation_type",
    "imputation_policy",
    "imputation_fill_value",
    "selection_status",
    "selection_reason",
    "importance_value",
    "coefficient",
    "abs_coefficient",
    "p_value",
    "expected_sign",
    "definition",
    "inclusion_rationale",
    "review_guidance",
]


def build_feature_lineage_table(context: PipelineContext) -> pd.DataFrame:
    """Builds a reviewer-facing lineage table for the final model feature surface."""

    model_feature_names = _model_feature_names(context)
    if not model_feature_names:
        return pd.DataFrame(columns=LINEAGE_COLUMNS)

    transformation_lookup = _table_lookup(
        context.diagnostics_tables.get("governed_transformations"),
        key_column="output_feature",
    )
    imputation_lookup = _table_lookup(
        context.diagnostics_tables.get("imputation_rules"),
        key_column="feature_name",
    )
    selection_lookup = _table_lookup(
        context.diagnostics_tables.get("variable_selection"),
        key_column="feature_name",
    )
    dictionary_lookup = _table_lookup(
        context.diagnostics_tables.get("feature_dictionary"),
        key_column="feature_name",
    )
    importance_lookup = _table_lookup(context.feature_importance, key_column="feature_name")

    rows: list[dict[str, Any]] = []
    for model_feature_name in model_feature_names:
        source_feature = _infer_source_feature(model_feature_name, context.feature_columns)
        transformation = transformation_lookup.get(source_feature, {})
        source_columns = _source_columns(source_feature, transformation)
        imputation = _first_lookup(imputation_lookup, source_feature, *source_columns)
        selection = selection_lookup.get(source_feature, {})
        dictionary = dictionary_lookup.get(source_feature, {})
        importance = importance_lookup.get(model_feature_name, {})

        rows.append(
            {
                "model_feature_name": model_feature_name,
                "source_feature": source_feature,
                "feature_type": _feature_type(context, model_feature_name, source_feature),
                "lineage_type": _lineage_type(context, model_feature_name, source_feature),
                "source_columns": ", ".join(source_columns) if source_columns else source_feature,
                "transformation_type": _clean_value(transformation.get("transform_type")),
                "imputation_policy": _clean_value(
                    imputation.get("applied_policy", imputation.get("configured_policy"))
                ),
                "imputation_fill_value": _clean_value(imputation.get("fill_value")),
                "selection_status": _clean_value(selection.get("selection_status", "selected")),
                "selection_reason": _clean_value(selection.get("selection_reason")),
                "importance_value": _numeric_or_none(importance.get("importance_value")),
                "coefficient": _numeric_or_none(importance.get("coefficient")),
                "abs_coefficient": _numeric_or_none(importance.get("abs_coefficient")),
                "p_value": _numeric_or_none(importance.get("p_value")),
                "expected_sign": _clean_value(dictionary.get("expected_sign")),
                "definition": _clean_value(dictionary.get("definition")),
                "inclusion_rationale": _clean_value(dictionary.get("inclusion_rationale")),
                "review_guidance": _review_guidance(
                    context,
                    model_feature_name=model_feature_name,
                    source_feature=source_feature,
                    transformation=transformation,
                    imputation=imputation,
                    dictionary=dictionary,
                ),
            }
        )

    frame = pd.DataFrame(rows, columns=LINEAGE_COLUMNS)
    if "importance_value" in frame.columns:
        return frame.sort_values(
            ["importance_value", "model_feature_name"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return frame


def summarize_feature_lineage(lineage_table: pd.DataFrame) -> dict[str, Any]:
    """Returns compact lineage counts for reports and UI cards."""

    if lineage_table.empty:
        return {
            "model_feature_count": 0,
            "source_feature_count": 0,
            "transformed_feature_count": 0,
            "imputed_feature_count": 0,
            "documented_feature_count": 0,
            "undocumented_feature_count": 0,
        }
    documented = lineage_table["definition"].astype("string").str.strip().ne("").fillna(False)
    transformed = (
        lineage_table["transformation_type"].astype("string").str.strip().ne("").fillna(False)
    )
    imputed = lineage_table["imputation_policy"].astype("string").str.strip().ne("").fillna(False)
    return {
        "model_feature_count": int(len(lineage_table)),
        "source_feature_count": int(lineage_table["source_feature"].nunique(dropna=True)),
        "transformed_feature_count": int(transformed.sum()),
        "imputed_feature_count": int(imputed.sum()),
        "documented_feature_count": int(documented.sum()),
        "undocumented_feature_count": int((~documented).sum()),
    }


def _model_feature_names(context: PipelineContext) -> list[str]:
    importance = context.feature_importance
    if (
        isinstance(importance, pd.DataFrame)
        and not importance.empty
        and "feature_name" in importance.columns
    ):
        return [str(value) for value in importance["feature_name"].dropna().tolist()]
    return [str(value) for value in context.feature_columns]


def _table_lookup(table: pd.DataFrame | None, *, key_column: str) -> dict[str, dict[str, Any]]:
    if table is None or table.empty or key_column not in table.columns:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in table.iterrows():
        key = str(row.get(key_column, "")).strip()
        if key:
            lookup[key] = row.to_dict()
    return lookup


def _first_lookup(
    lookup: dict[str, dict[str, Any]],
    *keys: str,
) -> dict[str, Any]:
    for key in keys:
        if key in lookup:
            return lookup[key]
    return {}


def _infer_source_feature(model_feature_name: str, feature_columns: list[str]) -> str:
    if model_feature_name in {"const", "intercept", "Intercept"}:
        return "intercept"
    normalized = model_feature_name
    for prefix in ("num__", "cat__", "remainder__"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if normalized in feature_columns:
        return normalized

    # One-hot encoded categoricals usually look like cat__region_west.
    matching_candidates = [
        feature_name
        for feature_name in feature_columns
        if normalized.startswith(f"{feature_name}_") or normalized.startswith(f"{feature_name}=")
    ]
    if matching_candidates:
        return max(matching_candidates, key=len)
    return normalized


def _source_columns(source_feature: str, transformation: dict[str, Any]) -> list[str]:
    if not transformation:
        return [] if source_feature == "intercept" else [source_feature]
    raw_sources = [
        str(transformation.get("source_feature", "")).strip(),
        str(transformation.get("secondary_feature", "")).strip(),
    ]
    source_columns: list[str] = []
    for raw_source in raw_sources:
        if not raw_source:
            continue
        source_columns.extend(part.strip() for part in raw_source.split(",") if part.strip())
    return list(dict.fromkeys(source_columns))


def _feature_type(
    context: PipelineContext,
    model_feature_name: str,
    source_feature: str,
) -> str:
    if source_feature == "intercept":
        return "intercept"
    if source_feature in context.numeric_features or model_feature_name.startswith("num__"):
        return "numeric"
    if source_feature in context.categorical_features or model_feature_name.startswith("cat__"):
        return "categorical"
    return "derived"


def _lineage_type(
    context: PipelineContext,
    model_feature_name: str,
    source_feature: str,
) -> str:
    if source_feature == "intercept":
        return "model_intercept"
    generated_indicators = set(context.metadata.get("generated_missing_indicator_columns", []))
    if source_feature in generated_indicators or model_feature_name in generated_indicators:
        return "missing_indicator"
    created_date_parts = set(
        context.metadata.get("date_feature_engineering", {}).get("created_date_part_features", [])
    )
    if source_feature in created_date_parts or model_feature_name in created_date_parts:
        return "date_part"
    hazard_features = set(context.metadata.get("hazard_time_features", []))
    if source_feature in hazard_features or model_feature_name in hazard_features:
        return "hazard_time"
    transformation_table = context.diagnostics_tables.get("governed_transformations")
    if (
        isinstance(transformation_table, pd.DataFrame)
        and not transformation_table.empty
        and "output_feature" in transformation_table.columns
        and source_feature in set(transformation_table["output_feature"].astype(str))
    ):
        return "governed_transformation"
    if model_feature_name.startswith(("cat__", "num__")):
        return "preprocessed_model_term"
    return "raw_input_feature"


def _review_guidance(
    context: PipelineContext,
    *,
    model_feature_name: str,
    source_feature: str,
    transformation: dict[str, Any],
    imputation: dict[str, Any],
    dictionary: dict[str, Any],
) -> str:
    if source_feature == "intercept":
        return "Intercept term; review only with coefficient-level model output."
    guidance: list[str] = []
    if not dictionary.get("definition"):
        guidance.append("Add business definition.")
    if transformation:
        guidance.append("Confirm transformation rationale and train-only fit.")
    if imputation:
        guidance.append("Review imputation policy and missingness meaning.")
    if model_feature_name.startswith("cat__"):
        guidance.append("Verify encoded category is stable and not high-cardinality leakage.")
    if not guidance:
        guidance.append("Lineage is documented; review sign and performance contribution.")
    return " ".join(guidance)


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric
