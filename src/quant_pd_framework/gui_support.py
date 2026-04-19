"""Helpers that keep the Streamlit UI thin and the framework reusable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import CategoricalDtype

from .config import (
    ArtifactConfig,
    CleaningConfig,
    ColumnRole,
    ColumnSpec,
    ComparisonConfig,
    DataStructure,
    DiagnosticConfig,
    ExecutionConfig,
    ExecutionMode,
    ExplainabilityConfig,
    FeatureEngineeringConfig,
    FeaturePolicyConfig,
    FrameworkConfig,
    MissingValuePolicy,
    ModelConfig,
    ModelType,
    PresetName,
    ScenarioConfig,
    ScenarioFeatureShock,
    ScenarioShockOperation,
    ScenarioTestConfig,
    SchemaConfig,
    SplitConfig,
    TargetConfig,
    TargetMode,
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
    "create_if_missing",
    "default_value",
    "keep_source",
]
SUPPORTED_DTYPES = ["auto", "string", "category", "float", "int", "bool", "datetime"]
SUPPORTED_MISSING_VALUE_POLICIES = [policy.value for policy in MissingValuePolicy]


@dataclass(slots=True)
class GUIBuildInputs:
    """Captures the non-tabular settings collected by the GUI."""

    preset_name: PresetName | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    feature_policy: FeaturePolicyConfig = field(default_factory=FeaturePolicyConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    scenario_testing: ScenarioTestConfig = field(default_factory=ScenarioTestConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
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
                "create_if_missing": False,
                "default_value": "",
                "keep_source": False,
            }
        )

    return pd.DataFrame(rows, columns=EDITOR_COLUMNS)


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
        feature_policy=inputs.feature_policy,
        explainability=inputs.explainability,
        scenario_testing=inputs.scenario_testing,
        diagnostics=inputs.diagnostics,
        artifacts=ArtifactConfig(output_root=inputs.output_root),
    )
    config.validate()
    return config


def normalize_editor_frame(editor_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the schema editor into a predictable tabular format."""

    working = editor_frame.copy(deep=True)
    for column in EDITOR_COLUMNS:
        if column not in working.columns:
            working[column] = (
                "" if column not in {"enabled", "create_if_missing", "keep_source"} else False
            )

    working = working.loc[:, EDITOR_COLUMNS].fillna("")
    for boolean_column in ["enabled", "create_if_missing", "keep_source"]:
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
                missing_value_fill_value=_normalize_default_value(
                    row["missing_value_fill_value"]
                ),
                create_if_missing=bool(row["create_if_missing"]),
                default_value=_normalize_default_value(row["default_value"]),
                keep_source=bool(row["keep_source"]),
            )
        )

    if not specs:
        raise ValueError("The schema editor does not define any usable columns.")
    return specs


def parse_positive_values(raw_text: str) -> list[str] | None:
    """Parses comma-separated positive target labels entered by the user."""

    values = [value.strip() for value in raw_text.split(",") if value.strip()]
    return values or None


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


def build_gui_inputs_from_preset(preset_name: PresetName) -> GUIBuildInputs:
    """Builds a GUI input bundle from a named preset."""

    preset = get_preset_definition(preset_name)
    return GUIBuildInputs(
        preset_name=preset_name,
        model=preset.model,
        feature_engineering=preset.feature_engineering,
        comparison=preset.comparison,
        feature_policy=preset.feature_policy,
        explainability=preset.explainability,
        scenario_testing=preset.scenario_testing,
        diagnostics=preset.diagnostics,
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
