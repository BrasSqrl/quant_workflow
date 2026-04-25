"""Helpers that keep the Streamlit UI thin and the framework reusable."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import CategoricalDtype

from .config import (
    AdvancedImputationConfig,
    ArtifactConfig,
    CalibrationConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    ComparisonConfig,
    CreditRiskDiagnosticConfig,
    CrossValidationConfig,
    DataStructure,
    DiagnosticConfig,
    DocumentationConfig,
    ExecutionConfig,
    ExecutionMode,
    ExplainabilityConfig,
    FeatureDictionaryConfig,
    FeatureDictionaryEntry,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FeatureReviewDecision,
    FeatureReviewDecisionType,
    FeatureSubsetSearchConfig,
    FrameworkConfig,
    ImputationSensitivityConfig,
    ManualReviewConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    PerformanceConfig,
    PresetName,
    RegulatoryReportConfig,
    ReproducibilityConfig,
    RobustnessConfig,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    ScorecardBinOverride,
    ScorecardConfig,
    ScorecardWorkbenchConfig,
    SplitConfig,
    SuitabilityCheckConfig,
    TargetConfig,
    TargetMode,
    TransformationConfig,
    TransformationSpec,
    TransformationType,
    VariableSelectionConfig,
    WorkflowGuardrailConfig,
)
from .presets import get_preset_definition, list_preset_definitions

EDITOR_COLUMNS = [
    "enabled",
    "source_name",
    "name",
    "role",
    "dtype",
    "missing_value_policy",
    "missing_value_fill_value",
    "missing_value_group_columns",
    "create_missing_indicator",
    "create_if_missing",
    "default_value",
    "keep_source",
]
FEATURE_DICTIONARY_COLUMNS = [
    "enabled",
    "feature_name",
    "business_name",
    "definition",
    "source_system",
    "unit",
    "allowed_range",
    "missingness_meaning",
    "expected_sign",
    "inclusion_rationale",
    "notes",
]
TRANSFORMATION_EDITOR_COLUMNS = [
    "enabled",
    "transform_type",
    "source_feature",
    "secondary_feature",
    "categorical_value",
    "output_feature",
    "lower_quantile",
    "upper_quantile",
    "parameter_value",
    "window_size",
    "lag_periods",
    "bin_edges",
    "generated_automatically",
    "notes",
]
FEATURE_REVIEW_COLUMNS = [
    "feature_name",
    "decision",
    "rationale",
]
SCORECARD_OVERRIDE_COLUMNS = [
    "feature_name",
    "bin_edges",
    "rationale",
]
SUPPORTED_DTYPES = ["auto", "string", "category", "float", "int", "bool", "datetime"]
SUPPORTED_MISSING_VALUE_POLICIES = [policy.value for policy in MissingValuePolicy]
SUPPORTED_TRANSFORMATION_TYPES = [transform_type.value for transform_type in TransformationType]
SUPPORTED_FEATURE_REVIEW_DECISIONS = [decision.value for decision in FeatureReviewDecisionType]


@dataclass(slots=True)
class GUIBuildInputs:
    """Captures the non-tabular settings collected by the GUI."""

    preset_name: PresetName | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    subset_search: FeatureSubsetSearchConfig = field(default_factory=FeatureSubsetSearchConfig)
    feature_policy: FeaturePolicyConfig = field(default_factory=FeaturePolicyConfig)
    feature_dictionary: FeatureDictionaryConfig = field(default_factory=FeatureDictionaryConfig)
    advanced_imputation: AdvancedImputationConfig = field(default_factory=AdvancedImputationConfig)
    transformations: TransformationConfig = field(default_factory=TransformationConfig)
    manual_review: ManualReviewConfig = field(default_factory=ManualReviewConfig)
    suitability_checks: SuitabilityCheckConfig = field(default_factory=SuitabilityCheckConfig)
    workflow_guardrails: WorkflowGuardrailConfig = field(default_factory=WorkflowGuardrailConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scorecard: ScorecardConfig = field(default_factory=ScorecardConfig)
    scorecard_workbench: ScorecardWorkbenchConfig = field(default_factory=ScorecardWorkbenchConfig)
    imputation_sensitivity: ImputationSensitivityConfig = field(
        default_factory=ImputationSensitivityConfig
    )
    variable_selection: VariableSelectionConfig = field(default_factory=VariableSelectionConfig)
    documentation: DocumentationConfig = field(default_factory=DocumentationConfig)
    regulatory_reporting: RegulatoryReportConfig = field(default_factory=RegulatoryReportConfig)
    scenario_testing: ScenarioTestConfig = field(default_factory=ScenarioTestConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    credit_risk: CreditRiskDiagnosticConfig = field(default_factory=CreditRiskDiagnosticConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    data_structure: DataStructure = DataStructure.CROSS_SECTIONAL
    train_size: float = 0.6
    validation_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    execution_mode: ExecutionMode = ExecutionMode.FIT_NEW_MODEL
    existing_model_path: Path | None = None
    existing_config_path: Path | None = None
    target_mode: TargetMode = TargetMode.BINARY
    target_output_column: str = "default_flag"
    positive_values_text: str = ""
    drop_target_source_column: bool = False
    pass_through_unconfigured_columns: bool = True
    output_root: Path = Path("artifacts")
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def build_column_editor_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates the editable schema table shown in the GUI."""

    rows: list[dict[str, Any]] = []
    for column in dataframe.columns:
        rows.append(
            {
                "enabled": True,
                "source_name": column,
                "name": column,
                "role": ColumnRole.FEATURE.value,
                "dtype": infer_dtype_label(dataframe[column]),
                "missing_value_policy": MissingValuePolicy.INHERIT_DEFAULT.value,
                "missing_value_fill_value": "",
                "missing_value_group_columns": "",
                "create_missing_indicator": False,
                "create_if_missing": False,
                "default_value": "",
                "keep_source": False,
            }
        )

    return pd.DataFrame(rows, columns=EDITOR_COLUMNS)


def build_column_editor_frame_from_schema(schema: SchemaConfig) -> pd.DataFrame:
    """Builds the schema editor from a resolved schema config."""

    rows = [
        {
            "enabled": spec.enabled,
            "source_name": spec.source_name or spec.name,
            "name": spec.name,
            "role": spec.role.value,
            "dtype": spec.dtype or "auto",
            "missing_value_policy": spec.missing_value_policy.value,
            "missing_value_fill_value": (
                "" if spec.missing_value_fill_value is None else spec.missing_value_fill_value
            ),
            "missing_value_group_columns": ", ".join(spec.missing_value_group_columns),
            "create_missing_indicator": spec.create_missing_indicator,
            "create_if_missing": spec.create_if_missing,
            "default_value": "" if spec.default_value is None else spec.default_value,
            "keep_source": spec.keep_source,
        }
        for spec in schema.column_specs
    ]
    return pd.DataFrame(rows, columns=EDITOR_COLUMNS)


def build_feature_dictionary_editor_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates a business-metadata table for the current dataframe columns."""

    rows = [
        {
            "enabled": True,
            "feature_name": column_name,
            "business_name": "",
            "definition": "",
            "source_system": "",
            "unit": "",
            "allowed_range": "",
            "missingness_meaning": "",
            "expected_sign": "",
            "inclusion_rationale": "",
            "notes": "",
        }
        for column_name in dataframe.columns
    ]
    return pd.DataFrame(rows, columns=FEATURE_DICTIONARY_COLUMNS)


def build_transformation_editor_frame() -> pd.DataFrame:
    """Creates the governed-transformation editor shown in the GUI."""

    return pd.DataFrame(columns=TRANSFORMATION_EDITOR_COLUMNS)


def build_feature_review_editor_frame() -> pd.DataFrame:
    """Creates the manual feature-review editor shown in the GUI."""

    return pd.DataFrame(columns=FEATURE_REVIEW_COLUMNS)


def build_scorecard_override_editor_frame() -> pd.DataFrame:
    """Creates the scorecard override editor shown in the GUI."""

    return pd.DataFrame(columns=SCORECARD_OVERRIDE_COLUMNS)


def frames_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compares editor frames by value while ignoring dtype-only differences."""

    if left.shape != right.shape:
        return False
    if list(left.columns) != list(right.columns):
        return False
    return _frame_records_for_compare(left) == _frame_records_for_compare(right)


def infer_dtype_label(series: pd.Series) -> str:
    """Maps pandas dtypes into the small set of dtypes the framework exposes."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if isinstance(series.dtype, CategoricalDtype):
        return "category"
    return "string"


def build_framework_config_from_editor(
    editor_frame: pd.DataFrame,
    inputs: GUIBuildInputs,
    *,
    validate: bool = True,
) -> FrameworkConfig:
    """Converts GUI inputs into the core framework configuration."""

    normalized_editor = normalize_editor_frame(editor_frame)
    column_specs = build_column_specs_from_editor(normalized_editor)
    using_prior_run_config = (
        inputs.execution_mode == ExecutionMode.SCORE_EXISTING_MODEL
        and inputs.existing_config_path is not None
    )
    target_rows = [
        row
        for _, row in normalized_editor.iterrows()
        if str(row["role"]).strip().lower() == ColumnRole.TARGET_SOURCE.value
        and bool(row["enabled"])
    ]
    if not using_prior_run_config and len(target_rows) != 1:
        raise ValueError("Mark exactly one enabled row as the target source in the schema editor.")

    if using_prior_run_config:
        target_source_column = inputs.target_output_column or "target_placeholder"
    else:
        target_row = target_rows[0]
        target_source_column = _clean_text(target_row["name"])
        if not target_source_column:
            raise ValueError("The target-source row must have a valid output column name.")

    date_column = _first_role_column(normalized_editor, ColumnRole.DATE)
    entity_column = _first_role_column(normalized_editor, ColumnRole.IDENTIFIER)

    if (
        not using_prior_run_config
        and inputs.data_structure in {DataStructure.TIME_SERIES, DataStructure.PANEL}
        and not date_column
    ):
        raise ValueError("Time-series and panel runs require one enabled row with role='date'.")
    if (
        not using_prior_run_config
        and inputs.data_structure == DataStructure.PANEL
        and not entity_column
    ):
        raise ValueError("Panel runs require one enabled row with role='identifier'.")

    split_config = SplitConfig(
        data_structure=inputs.data_structure,
        train_size=inputs.train_size,
        validation_size=inputs.validation_size,
        test_size=inputs.test_size,
        random_state=inputs.random_state,
        stratify=inputs.stratify
        if inputs.data_structure == DataStructure.CROSS_SECTIONAL
        else False,
        date_column=date_column,
        entity_column=entity_column,
    )
    split_config.validate()

    config = FrameworkConfig(
        schema=SchemaConfig(
            column_specs=column_specs,
            pass_through_unconfigured_columns=inputs.pass_through_unconfigured_columns,
        ),
        cleaning=inputs.cleaning,
        feature_engineering=inputs.feature_engineering,
        target=TargetConfig(
            source_column=target_source_column,
            mode=inputs.target_mode,
            output_column=inputs.target_output_column,
            positive_values=parse_positive_values(inputs.positive_values_text),
            drop_source_column=inputs.drop_target_source_column,
        ),
        split=split_config,
        preset_name=inputs.preset_name,
        execution=ExecutionConfig(
            mode=inputs.execution_mode,
            existing_model_path=inputs.existing_model_path,
            existing_config_path=inputs.existing_config_path,
        ),
        model=inputs.model,
        comparison=inputs.comparison,
        subset_search=inputs.subset_search,
        feature_policy=inputs.feature_policy,
        feature_dictionary=inputs.feature_dictionary,
        advanced_imputation=inputs.advanced_imputation,
        transformations=inputs.transformations,
        manual_review=inputs.manual_review,
        suitability_checks=inputs.suitability_checks,
        workflow_guardrails=inputs.workflow_guardrails,
        explainability=inputs.explainability,
        calibration=inputs.calibration,
        scorecard=inputs.scorecard,
        scorecard_workbench=inputs.scorecard_workbench,
        imputation_sensitivity=inputs.imputation_sensitivity,
        variable_selection=inputs.variable_selection,
        documentation=inputs.documentation,
        regulatory_reporting=inputs.regulatory_reporting,
        scenario_testing=inputs.scenario_testing,
        diagnostics=inputs.diagnostics,
        credit_risk=inputs.credit_risk,
        robustness=inputs.robustness,
        cross_validation=inputs.cross_validation,
        reproducibility=inputs.reproducibility,
        performance=inputs.performance,
        artifacts=replace(inputs.artifacts, output_root=inputs.output_root),
    )
    if validate:
        config.validate()
    return config


def normalize_editor_frame(editor_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the schema editor into a predictable tabular format."""

    working = editor_frame.copy(deep=True)
    for column in EDITOR_COLUMNS:
        if column not in working.columns:
            working[column] = (
                ""
                if column
                not in {
                    "enabled",
                    "create_if_missing",
                    "keep_source",
                    "create_missing_indicator",
                }
                else False
            )

    working = working.loc[:, EDITOR_COLUMNS].fillna("")
    for boolean_column in [
        "enabled",
        "create_if_missing",
        "keep_source",
        "create_missing_indicator",
    ]:
        working[boolean_column] = working[boolean_column].astype(bool)
    return working


def build_column_specs_from_editor(editor_frame: pd.DataFrame) -> list[ColumnSpec]:
    """Builds framework column specs from the editable schema table."""

    specs: list[ColumnSpec] = []
    seen_names: set[str] = set()

    for _, row in editor_frame.iterrows():
        source_name = _clean_text(row["source_name"])
        output_name = _clean_text(row["name"]) or source_name
        if not output_name:
            continue
        if output_name in seen_names:
            raise ValueError(
                f"Duplicate output column name '{output_name}' found in schema editor."
            )

        seen_names.add(output_name)
        dtype = _clean_text(row["dtype"])
        specs.append(
            ColumnSpec(
                name=output_name,
                source_name=source_name,
                enabled=bool(row["enabled"]),
                dtype=None if dtype in {"", "auto"} else dtype,
                role=ColumnRole(str(row["role"]).strip().lower()),
                missing_value_policy=MissingValuePolicy(
                    str(row["missing_value_policy"]).strip().lower()
                    or MissingValuePolicy.INHERIT_DEFAULT.value
                ),
                missing_value_fill_value=_normalize_default_value(row["missing_value_fill_value"]),
                missing_value_group_columns=_parse_text_list(row["missing_value_group_columns"]),
                create_missing_indicator=bool(row["create_missing_indicator"]),
                create_if_missing=bool(row["create_if_missing"]),
                default_value=_normalize_default_value(row["default_value"]),
                keep_source=bool(row["keep_source"]),
            )
        )

    if not specs:
        raise ValueError("The schema editor does not define any usable columns.")
    return specs


def normalize_feature_dictionary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the feature-dictionary editor into a predictable shape."""

    working = frame.copy(deep=True)
    for column in FEATURE_DICTIONARY_COLUMNS:
        if column not in working.columns:
            working[column] = False if column == "enabled" else ""
    working = working.loc[:, FEATURE_DICTIONARY_COLUMNS].fillna("")
    working["enabled"] = working["enabled"].astype(bool)
    return working


def normalize_transformation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the transformation editor into a predictable shape."""

    working = frame.copy(deep=True)
    for column in TRANSFORMATION_EDITOR_COLUMNS:
        if column not in working.columns:
            working[column] = False if column in {"enabled", "generated_automatically"} else ""
    working = working.loc[:, TRANSFORMATION_EDITOR_COLUMNS].fillna("")
    working["enabled"] = working["enabled"].astype(bool)
    working["generated_automatically"] = working["generated_automatically"].astype(bool)
    return working


def build_subset_search_feature_options(
    schema_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
) -> list[str]:
    """Returns feature-subset-search options including governed transformation outputs."""

    schema_features = (
        schema_frame.loc[
            schema_frame["enabled"]
            & (
                schema_frame["role"].astype(str).str.strip().str.lower()
                == ColumnRole.FEATURE.value
            ),
            "name",
        ]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )
    transformed_features = _resolve_subset_search_transformation_outputs(
        transformation_frame,
    )
    return list(dict.fromkeys([*schema_features, *transformed_features]))


def normalize_feature_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the manual feature-review editor into a predictable shape."""

    working = frame.copy(deep=True)
    for column in FEATURE_REVIEW_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working.loc[:, FEATURE_REVIEW_COLUMNS].fillna("")


def normalize_scorecard_override_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the scorecard-override editor into a predictable shape."""

    working = frame.copy(deep=True)
    for column in SCORECARD_OVERRIDE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working.loc[:, SCORECARD_OVERRIDE_COLUMNS].fillna("")


def parse_feature_dictionary_frame(frame: pd.DataFrame) -> FeatureDictionaryConfig:
    """Converts the feature-dictionary editor into typed config entries."""

    normalized = normalize_feature_dictionary_frame(frame)
    entries = [
        FeatureDictionaryEntry(
            feature_name=str(row["feature_name"]).strip(),
            business_name=str(row["business_name"]).strip(),
            definition=str(row["definition"]).strip(),
            source_system=str(row["source_system"]).strip(),
            unit=str(row["unit"]).strip(),
            allowed_range=str(row["allowed_range"]).strip(),
            missingness_meaning=str(row["missingness_meaning"]).strip(),
            expected_sign=str(row["expected_sign"]).strip(),
            inclusion_rationale=str(row["inclusion_rationale"]).strip(),
            notes=str(row["notes"]).strip(),
            enabled=bool(row["enabled"]),
        )
        for _, row in normalized.iterrows()
        if str(row["feature_name"]).strip()
    ]
    enabled_entries = [entry for entry in entries if entry.enabled]
    return FeatureDictionaryConfig(
        enabled=bool(enabled_entries),
        entries=entries,
    )


def parse_transformation_frame(frame: pd.DataFrame) -> TransformationConfig:
    """Converts the transformation editor into typed transformation config."""

    normalized = normalize_transformation_frame(frame)
    transformations: list[TransformationSpec] = []
    for _, row in normalized.iterrows():
        if not bool(row["enabled"]):
            continue
        source_feature = str(row["source_feature"]).strip()
        if not source_feature:
            continue
        transformations.append(
            TransformationSpec(
                enabled=True,
                transform_type=TransformationType(str(row["transform_type"]).strip()),
                source_feature=source_feature,
                secondary_feature=_clean_text(row["secondary_feature"]),
                categorical_value=_clean_text(row["categorical_value"]),
                output_feature=_clean_text(row["output_feature"]),
                lower_quantile=_coerce_optional_float(row["lower_quantile"]),
                upper_quantile=_coerce_optional_float(row["upper_quantile"]),
                parameter_value=_coerce_optional_float(row["parameter_value"]),
                window_size=_coerce_optional_int(row["window_size"]),
                lag_periods=_coerce_optional_int(row["lag_periods"]),
                bin_edges=_parse_float_list(row["bin_edges"]),
                generated_automatically=bool(row["generated_automatically"]),
                notes=str(row["notes"]).strip(),
            )
        )
    return TransformationConfig(
        enabled=bool(transformations),
        transformations=transformations,
    )


def parse_manual_review_frames(
    feature_review_frame: pd.DataFrame,
    scorecard_override_frame: pd.DataFrame,
    *,
    reviewer_name: str = "",
    require_review_complete: bool = False,
) -> ManualReviewConfig:
    """Converts review editors into the typed manual-review config."""

    normalized_feature_review = normalize_feature_review_frame(feature_review_frame)
    normalized_scorecard_override = normalize_scorecard_override_frame(scorecard_override_frame)
    feature_decisions = [
        FeatureReviewDecision(
            feature_name=str(row["feature_name"]).strip(),
            decision=FeatureReviewDecisionType(str(row["decision"]).strip()),
            rationale=str(row["rationale"]).strip(),
        )
        for _, row in normalized_feature_review.iterrows()
        if str(row["feature_name"]).strip() and str(row["decision"]).strip()
    ]
    scorecard_bin_overrides = [
        ScorecardBinOverride(
            feature_name=str(row["feature_name"]).strip(),
            bin_edges=_parse_float_list(row["bin_edges"]),
            rationale=str(row["rationale"]).strip(),
        )
        for _, row in normalized_scorecard_override.iterrows()
        if str(row["feature_name"]).strip() and str(row["bin_edges"]).strip()
    ]
    return ManualReviewConfig(
        enabled=bool(feature_decisions or scorecard_bin_overrides),
        reviewer_name=reviewer_name.strip(),
        require_review_complete=require_review_complete,
        feature_decisions=feature_decisions,
        scorecard_bin_overrides=scorecard_bin_overrides,
    )


def parse_positive_values(raw_text: str) -> list[str] | None:
    """Parses comma-separated positive target labels entered by the user."""

    values = [value.strip() for value in raw_text.split(",") if value.strip()]
    return values or None


def _parse_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _first_role_column(editor_frame: pd.DataFrame, role: ColumnRole) -> str | None:
    matching_rows = editor_frame.loc[
        editor_frame["enabled"]
        & (editor_frame["role"].astype(str).str.strip().str.lower() == role.value)
    ]
    if matching_rows.empty:
        return None
    return _clean_text(matching_rows.iloc[0]["name"])


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_default_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)
    if pd.isna(value):
        return None
    return float(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return int(float(stripped))
    if pd.isna(value):
        return None
    return int(value)


def _frame_records_for_compare(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [
        tuple(_normalize_compare_value(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]


def _normalize_compare_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def _parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _resolve_subset_search_transformation_outputs(frame: pd.DataFrame) -> list[str]:
    normalized = normalize_transformation_frame(frame)
    output_names: list[str] = []
    for _, row in normalized.iterrows():
        if not bool(row["enabled"]):
            continue
        source_feature = str(row["source_feature"]).strip()
        if not source_feature:
            continue
        transform_name = str(row["transform_type"]).strip()
        if not transform_name:
            continue
        try:
            transform_type = TransformationType(transform_name)
        except ValueError:
            continue
        configured_output = _clean_text(row["output_feature"])
        resolved_output = configured_output or _resolve_transformation_output_name_for_ui(
            transform_type=transform_type,
            source_feature=source_feature,
            secondary_feature=_clean_text(row["secondary_feature"]),
            categorical_value=_clean_text(row["categorical_value"]),
            parameter_value=_coerce_optional_float(row["parameter_value"]),
            window_size=_coerce_optional_int(row["window_size"]),
            lag_periods=_coerce_optional_int(row["lag_periods"]),
        )
        if transform_type == TransformationType.NATURAL_SPLINE:
            spline_df = 4 if _coerce_optional_float(row["parameter_value"]) is None else int(
                _coerce_optional_float(row["parameter_value"])
            )
            output_names.extend(
                f"{resolved_output}_basis_{index + 1}" for index in range(spline_df)
            )
            continue
        if resolved_output:
            output_names.append(resolved_output)
    return output_names


def _resolve_transformation_output_name_for_ui(
    *,
    transform_type: TransformationType,
    source_feature: str,
    secondary_feature: str | None,
    categorical_value: str | None,
    parameter_value: float | None,
    window_size: int | None,
    lag_periods: int | None,
) -> str:
    if transform_type == TransformationType.MANUAL_BINS:
        return f"{source_feature}_binned"
    if transform_type == TransformationType.RATIO:
        return f"{source_feature}_over_{secondary_feature}"
    if transform_type == TransformationType.INTERACTION:
        if (categorical_value or "").strip():
            return (
                f"{source_feature}_x_{secondary_feature}_"
                f"{_sanitize_transformation_token(categorical_value or '')}"
            )
        return f"{source_feature}_x_{secondary_feature}"
    if transform_type == TransformationType.YEO_JOHNSON:
        return f"{source_feature}_yeo_johnson"
    if transform_type == TransformationType.BOX_COX:
        return f"{source_feature}_box_cox"
    if transform_type == TransformationType.NATURAL_SPLINE:
        spline_df = 4 if parameter_value is None else int(parameter_value)
        return f"{source_feature}_spline_df_{spline_df}"
    if transform_type == TransformationType.CAPPED_ZSCORE:
        return f"{source_feature}_zscore"
    if transform_type == TransformationType.PIECEWISE_LINEAR:
        hinge_value = "hinge" if parameter_value is None else str(parameter_value)
        return f"{source_feature}_piecewise_{_sanitize_transformation_token(hinge_value)}"
    if transform_type == TransformationType.LAG:
        return f"{source_feature}_lag_{1 if lag_periods is None else lag_periods}"
    if transform_type == TransformationType.DIFFERENCE:
        return f"{source_feature}_diff_{1 if lag_periods is None else lag_periods}"
    if transform_type == TransformationType.EWMA:
        return f"{source_feature}_ewma_{3 if window_size is None else window_size}"
    if transform_type == TransformationType.ROLLING_MEAN:
        return f"{source_feature}_rollmean_{3 if window_size is None else window_size}"
    if transform_type == TransformationType.ROLLING_MEDIAN:
        return f"{source_feature}_rollmedian_{3 if window_size is None else window_size}"
    if transform_type == TransformationType.ROLLING_MIN:
        return f"{source_feature}_rollmin_{3 if window_size is None else window_size}"
    if transform_type == TransformationType.ROLLING_MAX:
        return f"{source_feature}_rollmax_{3 if window_size is None else window_size}"
    if transform_type == TransformationType.ROLLING_STD:
        return f"{source_feature}_rollstd_{3 if window_size is None else window_size}"
    if transform_type == TransformationType.PCT_CHANGE:
        return f"{source_feature}_pct_change_{1 if lag_periods is None else lag_periods}"
    return source_feature


def _sanitize_transformation_token(value: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in value)
    return sanitized.strip("_").lower() or "value"


def build_gui_inputs_from_preset(preset_name: PresetName) -> GUIBuildInputs:
    """Builds a GUI input bundle from a named preset."""

    preset = get_preset_definition(preset_name)
    return GUIBuildInputs(
        preset_name=preset_name,
        model=preset.model,
        feature_engineering=preset.feature_engineering,
        comparison=preset.comparison,
        subset_search=preset.subset_search,
        feature_policy=preset.feature_policy,
        feature_dictionary=preset.feature_dictionary,
        advanced_imputation=preset.advanced_imputation,
        transformations=preset.transformations,
        manual_review=preset.manual_review,
        suitability_checks=preset.suitability_checks,
        workflow_guardrails=preset.workflow_guardrails,
        explainability=preset.explainability,
        calibration=preset.calibration,
        scorecard=preset.scorecard,
        scorecard_workbench=preset.scorecard_workbench,
        imputation_sensitivity=preset.imputation_sensitivity,
        variable_selection=preset.variable_selection,
        documentation=preset.documentation,
        regulatory_reporting=preset.regulatory_reporting,
        scenario_testing=preset.scenario_testing,
        diagnostics=preset.diagnostics,
        credit_risk=preset.credit_risk,
        robustness=preset.robustness,
        cross_validation=preset.cross_validation,
        performance=preset.performance,
        data_structure=preset.data_structure,
        target_mode=preset.target_mode,
        target_output_column=preset.target_output_column,
        positive_values_text=preset.positive_values_text,
    )


def list_gui_preset_options() -> list[tuple[PresetName, str, str]]:
    """Returns preset name, label, and description tuples for the GUI."""

    return [
        (definition.name, definition.label, definition.description)
        for definition in list_preset_definitions()
    ]


def parse_expected_signs(raw_text: str) -> dict[str, str]:
    """Parses feature:direction pairs entered into the GUI."""

    result: dict[str, str] = {}
    for item in raw_text.split(","):
        if ":" not in item:
            continue
        feature_name, direction = item.split(":", 1)
        feature_name = feature_name.strip()
        direction = direction.strip()
        if feature_name and direction:
            result[feature_name] = direction
    return result


def parse_scenario_rows(rows: list[dict[str, Any]]) -> ScenarioTestConfig:
    """Converts a simple editable row payload into typed scenario config objects."""

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario_name = str(row.get("scenario_name", "")).strip()
        if not scenario_name:
            continue
        grouped_rows.setdefault(scenario_name, []).append(row)

    scenarios: list[ScenarioConfig] = []
    for scenario_name, scenario_rows in grouped_rows.items():
        shocks: list[ScenarioFeatureShock] = []
        description = ""
        enabled = True
        for row in scenario_rows:
            description = str(row.get("description", description)).strip()
            enabled = bool(row.get("enabled", enabled))
            feature_name = str(row.get("feature_name", "")).strip()
            operation = str(row.get("operation", ScenarioShockOperation.SET.value)).strip()
            value = row.get("value")
            if not feature_name:
                continue
            shocks.append(
                ScenarioFeatureShock(
                    feature_name=feature_name,
                    operation=ScenarioShockOperation(operation),
                    value=value,
                )
            )
        if shocks:
            scenarios.append(
                ScenarioConfig(
                    name=scenario_name,
                    description=description,
                    feature_shocks=shocks,
                    enabled=enabled,
                )
            )

    return ScenarioTestConfig(enabled=bool(scenarios), scenarios=scenarios)


def build_feature_dictionary_frame_from_config(
    feature_dictionary: FeatureDictionaryConfig,
    feature_names: list[str],
) -> pd.DataFrame:
    """Builds an editor-like feature dictionary frame from config entries."""

    entry_map = {entry.feature_name: entry for entry in feature_dictionary.entries}
    rows = []
    for feature_name in list(dict.fromkeys(feature_names)):
        entry = entry_map.get(feature_name)
        rows.append(
            {
                "enabled": True if entry is None else entry.enabled,
                "feature_name": feature_name,
                "business_name": "" if entry is None else entry.business_name,
                "definition": "" if entry is None else entry.definition,
                "source_system": "" if entry is None else entry.source_system,
                "unit": "" if entry is None else entry.unit,
                "allowed_range": "" if entry is None else entry.allowed_range,
                "missingness_meaning": "" if entry is None else entry.missingness_meaning,
                "expected_sign": "" if entry is None else entry.expected_sign,
                "inclusion_rationale": "" if entry is None else entry.inclusion_rationale,
                "notes": "" if entry is None else entry.notes,
            }
        )
    for feature_name, entry in entry_map.items():
        if feature_name in feature_names:
            continue
        rows.append(
            {
                "enabled": entry.enabled,
                "feature_name": feature_name,
                "business_name": entry.business_name,
                "definition": entry.definition,
                "source_system": entry.source_system,
                "unit": entry.unit,
                "allowed_range": entry.allowed_range,
                "missingness_meaning": entry.missingness_meaning,
                "expected_sign": entry.expected_sign,
                "inclusion_rationale": entry.inclusion_rationale,
                "notes": entry.notes,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_DICTIONARY_COLUMNS)


def build_transformation_frame_from_config(
    transformation_config: TransformationConfig,
) -> pd.DataFrame:
    """Builds an editor-like transformation frame from typed config."""

    rows = [
        {
            "enabled": transformation.enabled,
            "transform_type": transformation.transform_type.value,
            "source_feature": transformation.source_feature,
            "secondary_feature": transformation.secondary_feature or "",
            "categorical_value": transformation.categorical_value or "",
            "output_feature": transformation.output_feature or "",
            "lower_quantile": (
                transformation.lower_quantile if transformation.lower_quantile is not None else ""
            ),
            "upper_quantile": (
                transformation.upper_quantile if transformation.upper_quantile is not None else ""
            ),
            "parameter_value": (
                transformation.parameter_value if transformation.parameter_value is not None else ""
            ),
            "window_size": (
                transformation.window_size if transformation.window_size is not None else ""
            ),
            "lag_periods": (
                transformation.lag_periods if transformation.lag_periods is not None else ""
            ),
            "bin_edges": ", ".join(str(value) for value in transformation.bin_edges),
            "generated_automatically": transformation.generated_automatically,
            "notes": transformation.notes,
        }
        for transformation in transformation_config.transformations
    ]
    return pd.DataFrame(rows, columns=TRANSFORMATION_EDITOR_COLUMNS)


def build_feature_review_frame_from_config(
    manual_review: ManualReviewConfig,
) -> pd.DataFrame:
    """Builds an editor-like feature-review frame from typed config."""

    rows = [
        {
            "feature_name": decision.feature_name,
            "decision": decision.decision.value,
            "rationale": decision.rationale,
        }
        for decision in manual_review.feature_decisions
    ]
    return pd.DataFrame(rows, columns=FEATURE_REVIEW_COLUMNS)


def build_scorecard_override_frame_from_config(
    manual_review: ManualReviewConfig,
) -> pd.DataFrame:
    """Builds an editor-like scorecard override frame from typed config."""

    rows = [
        {
            "feature_name": override.feature_name,
            "bin_edges": ", ".join(str(value) for value in override.bin_edges),
            "rationale": override.rationale,
        }
        for override in manual_review.scorecard_bin_overrides
    ]
    return pd.DataFrame(rows, columns=SCORECARD_OVERRIDE_COLUMNS)


def build_template_workbook_bytes(
    *,
    schema_frame: pd.DataFrame,
    feature_dictionary_frame: pd.DataFrame,
    transformation_frame: pd.DataFrame,
    feature_review_frame: pd.DataFrame,
    scorecard_override_frame: pd.DataFrame,
) -> bytes:
    """Serializes the editable governance surfaces into an Excel workbook."""

    buffer = BytesIO()
    instructions = pd.DataFrame(
        [
            {
                "sheet_name": "schema",
                "purpose": (
                    "Column toggles, roles, dtypes, missing-value policy, "
                    "and create-if-missing rules."
                ),
            },
            {
                "sheet_name": "feature_dictionary",
                "purpose": (
                    "Business definitions, source lineage, expected sign, and inclusion rationale."
                ),
            },
            {
                "sheet_name": "transformations",
                "purpose": (
                    "Governed feature transformations such as winsorization, "
                    "log1p, ratio, interaction, and manual bins."
                ),
            },
            {
                "sheet_name": "feature_review",
                "purpose": "Manual approve/reject/force decisions for modeled features.",
            },
            {
                "sheet_name": "scorecard_overrides",
                "purpose": "Optional manual scorecard bin-edge overrides for numeric features.",
            },
        ]
    )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="instructions", index=False)
        normalize_editor_frame(schema_frame).to_excel(writer, sheet_name="schema", index=False)
        normalize_feature_dictionary_frame(feature_dictionary_frame).to_excel(
            writer, sheet_name="feature_dictionary", index=False
        )
        normalize_transformation_frame(transformation_frame).to_excel(
            writer, sheet_name="transformations", index=False
        )
        normalize_feature_review_frame(feature_review_frame).to_excel(
            writer, sheet_name="feature_review", index=False
        )
        normalize_scorecard_override_frame(scorecard_override_frame).to_excel(
            writer, sheet_name="scorecard_overrides", index=False
        )

    return buffer.getvalue()


def load_template_workbook(source: str | Path | BytesIO | Any) -> dict[str, pd.DataFrame]:
    """Loads an exported workbook template back into editable dataframes."""

    workbook = pd.read_excel(source, sheet_name=None)
    return {
        "schema": normalize_editor_frame(workbook.get("schema", pd.DataFrame())),
        "feature_dictionary": normalize_feature_dictionary_frame(
            workbook.get("feature_dictionary", pd.DataFrame())
        ),
        "transformations": normalize_transformation_frame(
            workbook.get("transformations", pd.DataFrame())
        ),
        "feature_review": normalize_feature_review_frame(
            workbook.get("feature_review", pd.DataFrame())
        ),
        "scorecard_overrides": normalize_scorecard_override_frame(
            workbook.get("scorecard_overrides", pd.DataFrame())
        ),
    }


def default_challengers_for_target_mode(target_mode: TargetMode) -> list[ModelType]:
    """Provides sensible challenger defaults when no preset is selected."""

    if target_mode == TargetMode.BINARY:
        return [
            ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
            ModelType.SCORECARD_LOGISTIC_REGRESSION,
            ModelType.PROBIT_REGRESSION,
            ModelType.XGBOOST,
        ]
    return [
        ModelType.BETA_REGRESSION,
        ModelType.TWO_STAGE_LGD_MODEL,
        ModelType.QUANTILE_REGRESSION,
        ModelType.XGBOOST,
    ]
